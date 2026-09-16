"""CPU integration contracts. SensorBase/PhysX are explicit doubles, NOT Isaac execution."""
from __future__ import annotations

from copy import deepcopy
import importlib
import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scentience_isaaclab._torch import indices
from scentience_isaaclab.backends import (
    DeviceInputAdapter, NumpyDeviceBatch, NumpyPlumeBatch, WarpPlumeBatch,
    build_backends, canonical_species, plume_config, resolve_device_config,
)
from scentience_isaaclab.imu import read_imu, read_sim_imu_frame
from scentience_isaaclab import mdp
from scentience_isaaclab.tasks import RUNNER_KEYS, register_task, task_kwargs
from scentience_olfaction.plume.filament import FilamentPlumeConfig
from scentience_olfaction.emitters.emitters import PointEmitter


def cfg(**kwargs):
    defaults = dict(species=("ethanol",), plume=None, transport_backend="numpy",
                    device_backend="numpy", device_profile="scentience_v1",
                    sensor_profile="packaged_slow", randomize_per_episode=False,
                    seed=7, device_config=None, torch_device_factory=None,
                    numpy_device_factory=None, channel_names=None)
    return SimpleNamespace(**(defaults | kwargs))


@pytest.fixture
def sensor_module(monkeypatch):
    """Small, declared Lab 3 timestamp/config/view doubles; production class executes."""
    class Base:
        def __init__(self, config):
            self.cfg, self.is_initialized = config, False

        def _initialize_impl(self):
            self._num_envs, self._device, self._sim_physics_dt = 3, "cpu", 0.01
            self._timestamp = torch.zeros(3)
            self._timestamp_last_update = torch.zeros(3)
            self._is_outdated = torch.ones(3, dtype=torch.bool)

        def _resolve_rigid_body_ancestor_expr(self):
            return "env_.*/Robot/base", None, None

        def update(self, dt, force_recompute=False):
            if not self.is_initialized:
                return
            self._timestamp += dt
            self._is_outdated |= self._timestamp - self._timestamp_last_update + 1e-6 >= self.cfg.update_period
            if force_recompute:
                self._update_outdated_buffers()

        def _update_outdated_buffers(self):
            self._update_buffers_impl(self._is_outdated)
            self._timestamp_last_update[self._is_outdated] = self._timestamp[self._is_outdated]
            self._is_outdated[:] = False

        def reset(self, env_ids=None, env_mask=None):
            ids = indices(3, "cpu", env_ids, env_mask)
            self._timestamp[ids] = 0
            self._timestamp_last_update[ids] = 0
            self._is_outdated[ids] = True

    class BaseCfg:
        prim_path: str = "env_.*/Robot/base"
        update_period: float = .05
        debug_vis: bool = False

    def configclass(cls):
        def init(self, **kwargs):
            for base in reversed(type(self).__mro__):
                for name in getattr(base, "__annotations__", {}):
                    setattr(self, name, deepcopy(getattr(base, name)))
            for name, value in kwargs.items():
                setattr(self, name, value)
        cls.__init__ = init
        return cls

    transforms = torch.tensor([[.2, 0, 1, 0, 0, 0, 1],
                               [20.2, 0, 1, 0, 0, 0, 1],
                               [40.2, 0, 1, 0, 0, 0, 1]])
    view = SimpleNamespace(count=3, get_transforms=lambda: transforms)

    class PhysxManager:
        @classmethod
        def get_physics_sim_view(cls):
            return SimpleNamespace(create_rigid_body_view=lambda expr: view)

    modules = {name: ModuleType(name) for name in (
        "isaaclab", "isaaclab.sensors", "isaaclab.utils", "isaaclab.utils.configclass",
        "isaaclab.sim", "isaaclab_physx", "isaaclab_physx.physics")}
    modules["isaaclab.sensors"].SensorBase = Base
    modules["isaaclab.sensors"].SensorBaseCfg = BaseCfg
    modules["isaaclab.utils.configclass"].configclass = configclass
    modules["isaaclab.sim"].SimulationContext = SimpleNamespace(
        instance=lambda: SimpleNamespace(physics_manager=PhysxManager))
    modules["isaaclab_physx.physics"].PhysxManager = PhysxManager
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    name = "scentience_isaaclab.olfactory_sensor"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    yield module, transforms, view
    sys.modules.pop(name, None)


def make_sensor(sensor_module, **kwargs):
    module, _, _ = sensor_module
    config = module.OlfactorySensorCfg(**(dict(
        transport_backend="numpy", device_backend="numpy", update_period=.05,
        expose_ground_truth=True, randomize_per_episode=False,
        plume=FilamentPlumeConfig(max_filaments=64, meander_std_rad=0.)) | kwargs))
    sensor = module.OlfactorySensor(config)
    sensor.set_env_origins(torch.tensor([[0., 0, 0], [20., 0, 0], [40., 0, 0]]))
    sensor._initialize_impl()
    sensor.is_initialized = True
    return sensor


def test_species_alias_and_duplicate_validation():
    assert canonical_species(("CO2", "EtOH")) == ("carbon_dioxide", "ethanol")
    with pytest.raises(ValueError):
        canonical_species(("CO2", "carbon_dioxide"))
    with pytest.raises(KeyError):
        canonical_species(("unknown-gas",))


def test_plume_identity_and_background_rejection():
    with pytest.raises(ValueError, match="absent"):
        plume_config(FilamentPlumeConfig(species="methane"), ("ethanol",))
    with pytest.raises(ValueError, match="absent"):
        plume_config(FilamentPlumeConfig(background_ppm={"CO2": 420}), ("ethanol",))
    config = FilamentPlumeConfig(species="CO2", background_ppm={"CO2": 420})
    result = plume_config(config, ("carbon_dioxide",))
    assert result.species == "carbon_dioxide"
    assert config.species == "CO2"  # caller-owned config untouched


def test_numpy_species_order_stereo_and_environment_isolation():
    config = FilamentPlumeConfig(emitters=[
        PointEmitter(position=(0, 0, 1), species="ethanol"),
        PointEmitter(position=(1, 0, 1), species="carbon_dioxide"),
    ], max_filaments=32, background_ppm={"carbon_dioxide": 410})
    species = ("ethanol", "carbon_dioxide", "ammonia")
    batch = NumpyPlumeBatch(config, species, 3, "cpu", 0)
    batch.step(.1)
    ids = torch.tensor([2, 0])
    points = torch.tensor([[[0., 0, 1], [1., 0, 1]], [[0., 0, 1], [1., 0, 1]]])
    concentration, wind = batch.sample(points, ids)
    assert concentration.shape == (2, 2, 3) and wind.shape == (2, 3)
    assert torch.all(concentration[:, :, 1] >= 410)
    assert torch.count_nonzero(concentration[:, :, 2]) == 0
    for row, env_id in enumerate(ids):
        p = batch.plumes[env_id]
        expected = p.sample_species(points[row].numpy())
        np.testing.assert_allclose(concentration[row, :, :2].numpy(), expected[:, [1, 0]])
    before = batch.plumes[1].alive.copy()
    batch.reset(torch.tensor([0]))
    assert not batch.plumes[0].alive.any()
    np.testing.assert_array_equal(batch.plumes[1].alive, before)


def test_warp_rejects_unsupported_mixtures_before_import():
    with pytest.raises(ValueError, match="passive"):
        WarpPlumeBatch(FilamentPlumeConfig(), ("ethanol", "ammonia"), 2, "cpu", 0)


def test_warp_full_batch_probe_scatter_and_nonaliased_stereo():
    pytest.importorskip("warp")
    config = FilamentPlumeConfig(max_filaments=32, meander_std_rad=0,
                                 turbulence_intensity=0, sigma_u_floor=0)
    batch = WarpPlumeBatch(config, ("ethanol",), 3, "cpu", 0)
    batch.step(.1)
    points = torch.tensor([[[.1, 0, 1], [5, 0, 1]], [[.1, 0, 1], [5, 0, 1]]])
    conc, _ = batch.sample(points, torch.tensor([2, 0]))
    assert conc.shape == (2, 2, 1)
    assert bool((conc[:, 0, 0] > 0).all()) and bool((conc[:, 1, 0] == 0).all())
    assert torch.equal(batch._probes[1], torch.zeros(3))


def test_indices_mask_precedence_empty_and_invalid():
    assert indices(3, "cpu", [0], torch.tensor([False, True, False])).tolist() == [1]
    assert indices(3, "cpu", []).numel() == 0
    for invalid in ([0, 0], [-1], [3], [True], [1.5]):
        with pytest.raises(ValueError):
            indices(3, "cpu", invalid)
    with pytest.raises(ValueError):
        indices(3, "cpu", env_mask=torch.ones(3))


def test_warp_mask_boundary():
    wp = pytest.importorskip("warp")
    mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    assert indices(3, "cpu", env_mask=mask).tolist() == [1]


def test_three_dimensional_mount_child_frame_and_stereo(sensor_module):
    module, transforms, _ = sensor_module
    s = math.sqrt(.5)
    transforms[:, 3:] = torch.tensor([s, 0, 0, s])  # parent rolls 90 degrees
    sensor = make_sensor(sensor_module,
                         offset=module.OlfactorySensorCfg.OffsetCfg(pos=(0, 0, 1), rot=(0, 0, s, s)),
                         left_probe_pos=(0, .02, 0), right_probe_pos=(0, -.02, 0))
    sensor._resolve_rigid_body_ancestor_expr = lambda: ("body", (1, 0, 0), (0, 0, 0, 1))
    sensor._initialize_impl()
    data = sensor.data  # reset read: pose available; device remains invalid
    assert torch.allclose(data.pos_w[0], torch.tensor([1.2, -1., 1.]), atol=1e-6)
    # Child yaw maps left probe to -x; parent roll preserves x.
    assert torch.allclose(data.probe_pos_w[0, 0], torch.tensor([1.18, -1., 1.]), atol=1e-6)
    assert torch.allclose(data.probe_pos_w[0, 1], torch.tensor([1.22, -1., 1.]), atol=1e-6)
    assert not bool(data.valid.any())
    assert torch.count_nonzero(data.channels) == 0


def test_view_count_must_match_clone_count(sensor_module):
    sensor_module[2].count = 6
    with pytest.raises(RuntimeError, match="one rigid body"):
        make_sensor(sensor_module)


def test_origins_required_and_shape_checked(sensor_module):
    sensor = make_sensor(sensor_module)
    sensor._env_origins_input = None
    with pytest.raises(RuntimeError, match="env_origins"):
        _ = sensor.data
    with pytest.raises(ValueError, match="expected 3"):
        sensor.set_env_origins(torch.zeros(2, 3))


def test_lazy_timing_and_empty_mask_no_state_advance(sensor_module):
    sensor = make_sensor(sensor_module)
    _ = sensor.data
    sensor.update(.03)
    assert not bool(sensor.data.valid.any())
    sensor.update(.03)
    data = sensor.data
    assert torch.allclose(data.sample_dt, torch.full((3,), .06))  # NOT nominal .05
    before = data.channels.clone()
    times = data.timestamp.clone()
    for _ in range(3):
        assert torch.equal(sensor.data.channels, before)
    assert torch.equal(sensor.data.timestamp, times)
    # Transport progresses even if no data consumer asks for a measurement.
    sensor.update(.02)
    assert sensor._plume.plumes[0].t == pytest.approx(.08)
    assert torch.equal(sensor.data.channels, before)
    with pytest.raises(ValueError):
        sensor.update(float("nan"))


def test_partial_reset_isolation_masks_clocks_and_stale_data(sensor_module):
    sensor = make_sensor(sensor_module)
    sensor.update(.08)
    initial = sensor.data.channels.clone()
    mask = torch.tensor([False, True, False])
    sensor.reset(env_ids=[0], env_mask=mask)  # mask takes precedence
    assert not sensor.data.valid[1]
    assert sensor.data.timestamp[1] == 0
    assert torch.count_nonzero(sensor.data.channels[1]) == 0
    assert torch.equal(sensor.data.channels[[0, 2]], initial[[0, 2]])
    assert sensor._plume.plumes[1].t == 0
    assert sensor._plume.plumes[0].t == pytest.approx(.08)
    sensor.update(.02)
    assert not sensor.data.valid[1]  # t=0 read started next sample period
    sensor.update(.04)
    assert bool(sensor.data.valid.all())
    assert torch.allclose(sensor.data.sample_dt, torch.full((3,), .06))
    unchanged = sensor.data.channels.clone()
    sensor.reset(env_ids=[])
    assert torch.equal(sensor.data.channels, unchanged)


def test_selected_different_elapsed_times_and_stereo_dispatch():
    class Device:
        cfg = SimpleNamespace(co2=None)
        def __init__(self):
            self.calls = []
        def step(self, conc, dt, conc_ppm_2, env_ids):
            self.calls.append((env_ids.clone(), dt))
            return torch.cat((conc, conc_ppm_2), -1)
    device = Device()
    adapter = DeviceInputAdapter(device, ("left", "right"), ("ethanol",), {})
    left = torch.tensor([[1.], [2.], [3.]])
    right = left + 10
    out = adapter.step(left, torch.tensor([.1, .2, .1]), right, torch.tensor([2, 0, 1]))
    assert torch.equal(out, torch.cat((left, right), -1))
    assert len(device.calls) == 2
    assert device.calls[0][0].tolist() == [2, 1]


@pytest.mark.parametrize("profile", ["scd30", "scd40", "scd41"])
def test_numpy_manufacturer_factory_is_directly_selectable(profile):
    batch = NumpyDeviceBatch(device_profile=profile, sensor_profile="packaged_slow",
                            n_envs=3, device="cpu", randomize=False,
                            species_names=("carbon_dioxide",))
    assert batch.channel_names == ("co2_ppm",)
    output = batch.step(torch.tensor([[100.]]), .1, env_ids=torch.tensor([1]))
    assert output.shape == (1, 1) and torch.isfinite(output).all()
    assert batch.units[0].channel._y == 420


def test_background_config_is_absolute_and_nonmutating():
    config = cfg(device_profile="scd41", device_config={"tau63_s": 2.})
    result = resolve_device_config(config, {"carbon_dioxide": 500.})
    assert result["concentration_mode"] == "absolute" and result["ambient_baseline_ppm"] == 500
    assert "concentration_mode" not in config.device_config
    with pytest.raises(ValueError, match="absolute"):
        resolve_device_config(cfg(device_profile="scd41", device_config={"concentration_mode": "excess"}),
                              {"carbon_dioxide": 500.})


@pytest.mark.parametrize("backend,profile", [("numpy", "scentience_v1"), ("torch", "scentience_v1"), ("numpy", "scd41")])
def test_background_co2_not_doubled(backend, profile):
    config = cfg(species=("carbon_dioxide",), device_backend=backend, device_profile=profile,
                 plume=FilamentPlumeConfig(species="carbon_dioxide", release_rate_hz=0,
                                           background_ppm={"carbon_dioxide": 500.}, max_filaments=16))
    plume, model, _ = build_backends(config, 3, "cpu")
    points = torch.zeros(3, 2, 3)
    conc, _ = plume.sample(points, torch.arange(3))
    assert torch.all(conc == 500.)
    # Device has a slow CO2 lag: ambient initial state must match the reservoir too.
    reading = model.step(conc[:, 0], .1, conc[:, 1], torch.arange(3))
    assert torch.allclose(reading[:, model.channel_names.index("co2_ppm")], torch.full((3,), 500.))


def test_channel_schema_mismatch_rejected():
    with pytest.raises(ValueError, match="schema"):
        build_backends(cfg(channel_names=("invented",)), 3, "cpu")


def test_imu_proxy_frames_gravity_timestamps_snapshot():
    force = torch.tensor([[0., 0., 9.81], [0., 0., 0.]])
    gyro = torch.tensor([[1., 0, 0], [0., 0, 0]])
    def proxy(value):
        return SimpleNamespace(torch=value)
    sensor = SimpleNamespace(data=SimpleNamespace(lin_acc_b=proxy(force), ang_vel_b=proxy(gyro)),
                             _timestamp_last_update=torch.tensor([.2, 0.]))
    sample = read_imu(sensor, gravity_w=(0., 0., -9.81), orientation_w=(0., 0., 0., 1.))
    assert torch.allclose(sample.linear_acceleration[0], torch.zeros(3))
    assert sample.valid.tolist() == [True, False]
    assert sample.timestamp.tolist() == pytest.approx([.2, 0.])
    s = math.sqrt(.5)
    rotated = read_imu(sensor, rotation_target_from_imu=(0, 0, s, s), frame="robot")
    assert torch.allclose(rotated.angular_velocity[0], torch.tensor([0., 1., 0.]), atol=1e-6)
    force[:] = 0
    assert sample.specific_force[0, 2] == pytest.approx(9.81)
    with pytest.raises(ValueError, match="both"):
        read_imu(sensor, gravity_w=(0, 0, -9.81))
    with pytest.raises(ValueError, match="name"):
        read_imu(sensor, rotation_target_from_imu=(0, 0, 0, 1))


def test_imu_no_fabricated_timestamp():
    sensor = SimpleNamespace(data=SimpleNamespace(lin_acc_b=torch.zeros(1, 3),
                                                   ang_vel_b=torch.zeros(1, 3)))
    with pytest.raises(ValueError, match="timestamp"):
        read_imu(sensor)


def test_mdp_channels_frames_privileged_optin_and_age(sensor_module):
    sensor = make_sensor(sensor_module, expose_ground_truth=False)
    sensor.update(.03)
    env = SimpleNamespace(scene={"nose": sensor})
    assert mdp.gas_channels(env, channel_names=("ec2", "co2_ppm")).shape == (3, 2)
    assert mdp.wind_body(env) is sensor.data.wind_b
    with pytest.raises(ValueError, match="privileged"):
        mdp.gas_ground_truth(env)
    with pytest.raises(ValueError, match="expose_ground_truth"):
        mdp.gas_ground_truth(env, privileged=True)
    sensor.update(.01)
    assert torch.allclose(mdp.gas_sample_age(env), torch.full((3, 1), .01))


def test_runner_registry_keys_and_no_placeholder_task():
    gym = pytest.importorskip("gymnasium")
    assert "Isaac-PlumeNav-Scentience-v0" not in gym.registry
    kwargs = task_kwargs("robot.env:Cfg", {key: "robot.agents:config.yaml" for key in RUNNER_KEYS})
    assert set(kwargs) == {"env_cfg_entry_point", "sb3_cfg_entry_point", "rsl_rl_cfg_entry_point",
                           "skrl_cfg_entry_point", "rl_games_cfg_entry_point"}
    with pytest.raises(ValueError, match="unknown"):
        task_kwargs("robot.env:Cfg", {"unknown": "robot.agents:Cfg"})
    task_id = "Isaac-Scentience-Contract-Test-v0"
    try:
        register_task(task_id, env_entry_point="robot.env:Env", env_cfg_entry_point="robot.env:Cfg",
                      runner_cfgs={"sb3": "robot.agents:ppo.yaml"})
        assert gym.spec(task_id).kwargs["sb3_cfg_entry_point"] == "robot.agents:ppo.yaml"
        with pytest.raises(ValueError, match="already"):
            register_task(task_id, env_entry_point="robot.env:Env", env_cfg_entry_point="robot.env:Cfg",
                          runner_cfgs={"sb3": "robot.agents:ppo.yaml"})
    finally:
        gym.registry.pop(task_id, None)



@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_actual_torch_device_sensor_selective_reset(sensor_module, dtype):
    from scentience_olfaction.sensors.device_np import DeviceConfig

    sensor = make_sensor(sensor_module, device_backend="torch", device_config=DeviceConfig(dtype=dtype))
    sensor.update(.07)
    before = sensor.data.channels.clone()
    assert before.dtype == getattr(torch, dtype) and torch.isfinite(before).all()
    sensor.reset(env_ids=[2])
    assert torch.equal(sensor.data.channels[:2], before[:2])
    sensor.update(.06)
    assert bool(sensor.data.valid.all()) and torch.isfinite(sensor.data.channels).all()


def test_lazy_skipped_reads_use_elapsed_not_nominal_period(sensor_module):
    sensor = make_sensor(sensor_module)
    _ = sensor.data
    for _ in range(10):
        sensor.update(.01)
    assert torch.allclose(sensor.data.sample_dt, torch.full((3,), .1), atol=1e-6)


def test_world_wind_rotates_to_sensor_frame(sensor_module):
    module = sensor_module[0]
    s = math.sqrt(.5)
    sensor = make_sensor(sensor_module, offset=module.OlfactorySensorCfg.OffsetCfg(rot=(0, 0, s, s)))
    sensor.update(.1)
    assert torch.allclose(sensor.data.wind_w[0], torch.tensor([1., 0., 0.]))
    assert torch.allclose(sensor.data.wind_b[0], torch.tensor([0., -1., 0.]), atol=1e-6)




@pytest.mark.parametrize("read_gravity", [False, True])
def test_native_sim6_dict_frames_gravity_and_clock(read_gravity):
    s = math.sqrt(.5)
    frame = dict(time=12.5, physics_step=1500,
                 linear_acceleration=[0, 9.81 if read_gravity else 0., 0],
                 angular_velocity=[1, 2, 3], orientation=[s, s, 0, 0])  # native wxyz
    sample = read_sim_imu_frame(frame, read_gravity=read_gravity, is_valid=True,
                                 gravity_w=(0, 0, -9.81))
    assert torch.allclose(sample.orientation_w[0], torch.tensor([s, 0, 0, s]))
    assert torch.allclose(sample.linear_acceleration, torch.zeros(1, 3), atol=1e-5)
    assert torch.allclose(sample.specific_force, torch.tensor([[0, 9.81, 0]]), atol=1e-5)
    assert sample.timestamp.item() == 12.5
    held = read_sim_imu_frame(frame, read_gravity=read_gravity, is_valid=False)
    assert not held.valid.item()  # finite held data must not imply valid/fresh
    with pytest.raises(ValueError, match="missing"):
        read_sim_imu_frame({}, read_gravity=True, is_valid=False)


def test_numpy_reset_without_randomization_preserves_calibration():
    batch = NumpyDeviceBatch(device_profile="scentience_v1", sensor_profile="packaged_slow",
                            n_envs=2, device="cpu", randomize=True, species_names=("ethanol",))
    before = [m.r0 for m in batch.units[0].device.mox]
    batch.reset([0], randomize=False)
    assert [m.r0 for m in batch.units[0].device.mox] == before
