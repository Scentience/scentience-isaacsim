"""Scientific and state-isolation contracts for the existing sensor models."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from scentience_olfaction.sensors.device_np import CHANNELS, DeviceConfig, ScentienceV1
from scentience_olfaction.sensors.electrochemical import EC_CO, ECChannel
from scentience_olfaction.sensors.mox import MOX_RED, MOX_OX, MoxChannel, MoxChannelConfig
from scentience_olfaction.sensors.pid import PIDChannel, PIDConfig


def quiet_mox(cfg, **kwargs):
    return replace(cfg, white_noise_frac=0., flicker_noise_frac=0.,
                   drift_sigma_per_sqrt_s=0., humidity_coeff=0., adc_bits=24, **kwargs)


def quiet_device(**kwargs):
    base = DeviceConfig()
    return replace(base, mox_channels=tuple(quiet_mox(c) for c in base.resolved_mox()),
                   ec_channels=tuple(replace(c, noise_na=0., drift_na_per_sqrt_s=0.)
                                     for c in base.ec_channels),
                   co2=replace(base.co2, repeatability_ppm=0., asc_enabled=False),
                   dtype="float64", **kwargs)


@pytest.mark.parametrize("cfg,gas,concentration", [
    (MOX_RED, "ethanol", 80.), (MOX_OX, "nitrogen_dioxide", 2.)])
def test_exposure_and_recovery_use_correct_polarity(cfg, gas, concentration):
    cfg = quiet_mox(cfg, tau_rise_s=2., tau_fall_s=8.)
    ch = MoxChannel(cfg, np.random.default_rng(0), randomize=False)
    target = cfg.sensitivity[gas][0] * concentration ** (-cfg.sensitivity[gas][1])
    ch.step({gas: concentration}, 2.)
    expected = target + (cfg.rs_r0_clean_air - target) / math.e
    assert ch._y == pytest.approx(expected, rel=1e-13)
    ch.step({}, 8.)
    assert ch._y == pytest.approx(1 + (expected - 1) / math.e, rel=1e-13)


def test_delay_uses_time_and_clean_prehistory_with_variable_dt():
    cfg = quiet_mox(MOX_RED, dead_volume_delay_s=1., tau_rise_s=2., tau_fall_s=2.)
    ch = MoxChannel(cfg, np.random.default_rng(0), randomize=False)
    target = cfg.sensitivity["ethanol"][0] * 80 ** (-cfg.sensitivity["ethanol"][1])
    elapsed = 0.
    for dt in (0.2, 0.35, 0.4, 0.1, 0.45, 1.8):
        elapsed += dt
        reading = ch.step({"ethanol": 80.}, dt)
        expected = target + (1 - target) * math.exp(-max(0., elapsed - 1) / 2)
        assert reading["rs_true"] / ch.r0 == pytest.approx(expected, rel=1e-13)
    ch.reset(randomize=False)
    assert ch.step({"ethanol": 80.}, 0.5)["rs_true"] / ch.r0 == 1.


def test_delayed_pulse_preserves_old_input_after_new_input_arrives():
    cfg = quiet_mox(MOX_RED, dead_volume_delay_s=1., tau_rise_s=2., tau_fall_s=2.)
    ch = MoxChannel(cfg, np.random.default_rng(0), randomize=False)
    target = cfg.sensitivity["ethanol"][0] * 80 ** (-cfg.sensitivity["ethanol"][1])
    ch.step({"ethanol": 80.}, 0.5)
    reading = ch.step({}, 0.8)  # t=1.3: must reproduce exposure at t=0.3
    assert reading["rs_true"] / ch.r0 == pytest.approx(target + (1-target)*math.exp(-0.3/2))
    reading = ch.step({}, 0.7)  # t=2: recovery at t=1
    peak = target + (1-target)*math.exp(-0.5/2)
    assert reading["rs_true"] / ch.r0 == pytest.approx(1 + (peak-1)*math.exp(-0.5/2))


def test_fixed_baseline_feature_exposes_drift_without_redefining_legacy_ratio():
    ch = MoxChannel(quiet_mox(MOX_RED), np.random.default_rng(0), randomize=False)
    ch.ln_drift = math.log(1.5)
    out = ch.step({}, 0.1)
    assert out["ratio_measured"] == pytest.approx(1., abs=2e-6)
    assert out["ratio_baseline"] == pytest.approx(1.5, abs=2e-6)
    assert out["ratio_baseline"] / out["ratio_measured"] == pytest.approx(1.5)
    dev = ScentienceV1(config=quiet_device(ratio_feature="ratio_baseline"), randomize_unit=False)
    dev.mox[0].ln_drift = math.log(1.5)
    out = dev.step({}, 0.1)
    assert out[CHANNELS[0]] == dev.last_mox_readings[0]["ratio_baseline"]


def test_shared_json_config_calibration_and_defensive_copy():
    cfg = quiet_device()
    channels = list(cfg.mox_channels)
    channels[0] = replace(channels[0], sensitivity={"custom_species": (0.3, 0.5)},
                          r0_nominal=123456., r_load=54321.)
    cfg = replace(cfg, mox_channels=tuple(channels))
    restored = DeviceConfig.from_dict(json.loads(json.dumps(asdict(cfg))))
    dev = ScentienceV1(config=restored, randomize_unit=False)
    restored.mox_channels[0].sensitivity["custom_species"] = (9., 0.5)
    assert dev.mox[0].cfg.sensitivity["custom_species"] == (0.3, 0.5)
    assert dev.mox[0].r0 == 123456.
    assert MOX_RED.sensitivity["ethanol"] == (1.31, 0.645)
    for _ in range(30):
        out = dev.step({"custom_species": 100.}, 1.)
    assert out[CHANNELS[0]] < 0.04
    assert out[CHANNELS[3]] > 0.99


@pytest.mark.parametrize("payload", [
    {"unknown": 1}, {"co2": {"bogus": 1}}, {"ec_channels": []},
    {"mox_channels": []}, {"dtype": "float16"}, {"ratio_feature": "volts"},
    {"flow_mps": -1}, {"ambient_rh": 101}, {"co2": {"concentration_mode": "invalid"}}])
def test_invalid_json_calibration_rejected(payload):
    with pytest.raises(ValueError):
        DeviceConfig.from_dict(payload)


@pytest.mark.parametrize("overrides", [
    {"tau_rise_s": 0}, {"tau_fall_s": -1}, {"dead_volume_delay_s": -0.1},
    {"adc_bits": 2.5}, {"adc_bits": True}, {"r0_range": (10., 1.)},
    {"r_load": 0}, {"white_noise_frac": -1}, {"humidity_coeff": float("nan")},
    {"sensitivity": {"x": (-1., 0.5)}}, {"sensitivity": {"x": (1., 0.)}},
    {"sensitivity": {"x": (1., 0.5), "y": (1., -0.5)}},
])
def test_invalid_mox_calibration_rejected(overrides):
    with pytest.raises(ValueError):
        replace(MOX_RED, **overrides)


def test_mixed_polarity_calibration_requires_explicit_kinetic_choice():
    cfg = MoxChannelConfig("mixed", sensitivity={"x": (1., 0.5), "y": (1., -0.5)},
                           response_polarity=1)
    assert cfg.polarity == 1


@pytest.mark.parametrize("factory", [
    lambda: MoxChannel(MOX_RED, np.random.default_rng(0)),
    lambda: ECChannel(EC_CO, np.random.default_rng(0)),
    lambda: PIDChannel(PIDConfig(), np.random.default_rng(0)),
    lambda: ScentienceV1(seed=0),
])
@pytest.mark.parametrize("conc,dt", [({}, -1.), ({}, 0.), ({}, float("nan")),
                                        ({"ethanol": -1.}, 0.1),
                                        ({"unknown": float("inf")}, 0.1)])
def test_invalid_step_inputs_do_not_advance_rng_or_state(factory, conc, dt):
    model, control = factory(), factory()
    with pytest.raises(ValueError):
        model.step(conc, dt)
    assert model.step({"ethanol": 10.}, 0.1) == control.step({"ethanol": 10.}, 0.1)


def torch_device(n=3, cfg=None, seed=77):
    pytest.importorskip("torch")
    from scentience_olfaction.sensors.scentience_v1 import build_device
    return build_device("scentience_v1", "packaged_slow", n, "cpu", seed=seed,
                        species_names=("ethanol", "nitrogen_dioxide", "carbon_dioxide"),
                        config=cfg)


def test_torch_subset_reset_and_step_leave_other_envs_and_rng_untouched():
    torch = pytest.importorskip("torch")
    dev, control = torch_device(), torch_device()
    c = torch.tensor([[30., 2., 500.]] * 3)
    assert torch.equal(dev.step(c, 0.1), control.step(c, 0.1))
    dev.reset([1], randomize=True)
    for _ in range(5):
        dev.step(c[1:2], 0.13, env_ids=[1])
    a, b = dev.step(c[[2, 0]], 5.1, env_ids=[2, 0]), control.step(c[[2, 0]], 5.1, env_ids=[2, 0])
    assert torch.equal(a, b)
    assert dev._elapsed[0] == control._elapsed[0]


def test_torch_batch_size_and_subset_order_do_not_change_random_streams():
    torch = pytest.importorskip("torch")
    small, large = torch_device(n=2), torch_device(n=4)
    c = torch.tensor([[30., 2., 500.], [50., 1., 250.]])
    for _ in range(3):
        small.reset([1, 0], randomize=True)
        large.reset([0, 1], randomize=True)
        a = small.step(c.flip(0), 0.2, env_ids=[1, 0])
        b = large.step(c, 0.2, env_ids=[0, 1])
        assert torch.equal(a.flip(0), b)


def test_torch_reset_false_restores_nominal_channels_and_clears_delay():
    torch = pytest.importorskip("torch")
    cfg = replace(quiet_device(), mox_channels=tuple(
        replace(c, dead_volume_delay_s=1.) for c in quiet_device().mox_channels))
    dev = torch_device(cfg=cfg)
    dev.step(torch.full((3, 3), 20., dtype=torch.float64), 0.8)
    dev.reset([1], randomize=False)
    assert torch.equal(dev.r0[1], torch.tensor([c.r0_nominal for c in cfg.mox_channels], dtype=torch.float64))
    assert dev._elapsed[1] == 0 and not dev._history[1]
    out = dev.step(torch.full((1, 3), 20., dtype=torch.float64), 0.2, env_ids=[1])
    assert torch.allclose(out[0, :6], torch.ones(6, dtype=torch.float64), atol=2e-6)


@pytest.mark.parametrize("env_ids", [[0, 0], [-1], [3], [1.0], [True]])
def test_torch_invalid_env_ids_rejected_without_reset(env_ids):
    torch = pytest.importorskip("torch")
    dev = torch_device()
    before = dev.r0.clone()
    with pytest.raises(ValueError):
        dev.reset(env_ids, randomize=True)
    assert torch.equal(before, dev.r0)


def test_torch_input_shape_dtype_stereo_validation_and_empty_selection():
    torch = pytest.importorskip("torch")
    dev, control = torch_device(), torch_device()
    c = torch.ones(3, 3)
    for invalid in (torch.ones(3, 1), c.double(), c * float("nan"), -c):
        with pytest.raises(ValueError):
            dev.step(invalid, 0.1)
    with pytest.raises(ValueError):
        dev.step(c, 0.1, conc_ppm_2=-c)
    assert torch.equal(dev.step(c, 0.1), control.step(c, 0.1))
    dev.reset([], randomize=True)
    assert dev.step(c[:0], 0.1, env_ids=[]).shape == (0, 11)


def test_torch_global_rng_is_unchanged_and_seed_changes_unit():
    torch = pytest.importorskip("torch")
    before = torch.random.get_rng_state().clone()
    dev = torch_device()
    dev.step(torch.ones(3, 3), 0.2)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not torch.equal(dev.r0, torch_device(seed=78).r0)


def test_torch_warns_about_float32_high_resolution_adc():
    from scentience_olfaction.sensors.scentience_v1 import build_device
    pytest.importorskip("torch")
    with pytest.warns(UserWarning, match="ADC reconstruction"):
        build_device("scentience_v1", "fast_modulated", 1, "cpu")
