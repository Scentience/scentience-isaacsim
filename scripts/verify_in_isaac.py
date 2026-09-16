"""Live Sim 6 / Lab 3 sensor-mount smoke test. New robot runs are PENDING.

./isaaclab.sh -p /repo/scripts/verify_in_isaac.py --headless --robot go2 --num-envs 3
Robots are held on a fixed-root fixture for sensor verification, not controlled
by a locomotion/flight policy. Asset downloads require access to NVIDIA assets.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Paths checked against official v3.0.0-beta2 task configs. --body-path can adapt
# a locally modified USD. The actual prim must still resolve at runtime.
ROBOTS = {
    "go2": ("isaaclab_assets.robots.unitree:UNITREE_GO2_CFG", "base"),
    "h1": ("isaaclab_assets.robots.unitree:H1_CFG", "torso_link"),
    "crazyflie": ("isaaclab_assets.robots.quadcopter:CRAZYFLIE_CFG", "body"),
}


def run(args):
    import numpy as np
    import torch
    import warp as wp
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ImuCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

    from scentience_isaaclab._torch import as_torch
    from scentience_isaaclab.imu import read_imu
    from scentience_isaaclab.olfactory_sensor import OlfactorySensorCfg
    from scentience_olfaction.plume.filament import FilamentPlumeConfig
    from validate_install import run_checks

    if run_checks():
        raise RuntimeError("target version/binding checks failed")
    if args.robot == "cuboid":
        robot_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0., 1.)),
        )
        body = ""
    else:
        target, body = ROBOTS[args.robot]
        module, name = target.split(":")
        robot_cfg = getattr(importlib.import_module(module), name).copy()
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.init_state.pos = (0.3, 0., 1.)
        # This is deliberately a fixed fixture; no unimplemented action model.
        robot_cfg.spawn.articulation_props.fix_root_link = True
        if args.robot == "crazyflie":
            robot_cfg.init_state.joint_vel = {".*": 0.0}
    if args.body_path is not None:
        body = args.body_path.strip("/")
    mount_path = "{ENV_REGEX_NS}/Robot" + ("/" + body if body else "")
    # Nonzero 3-D translation/roll; identity rotation is xyzw in Lab 3.
    offset_pos = (0.04, 0.01, 0.06)
    offset_rot = (0.2588190451, 0., 0., 0.9659258263)

    @configclass
    class SmokeSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
        robot = robot_cfg
        nose = OlfactorySensorCfg(
            prim_path=mount_path, update_period=0.05,
            offset=OlfactorySensorCfg.OffsetCfg(pos=offset_pos, rot=offset_rot),
            left_probe_pos=(0., 0.02, 0.), right_probe_pos=(0., -0.02, 0.),
            transport_backend=args.transport_backend, device_backend=args.device_backend,
            device_profile=args.device_profile, species=(args.species,),
            plume=FilamentPlumeConfig(species=args.species, max_filaments=2048,
                                      source_pos=(0., 0., 1.), max_age_s=10.),
            expose_ground_truth=True, randomize_per_episode=False, seed=0,
        )
        imu = ImuCfg(prim_path=mount_path, update_period=0.,
                     offset=ImuCfg.OffsetCfg(pos=offset_pos, rot=offset_rot))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    scene = InteractiveScene(SmokeSceneCfg(num_envs=args.num_envs, env_spacing=20.))
    nose = scene["nose"]
    nose.set_env_origins(scene.env_origins)
    sim.reset()
    if nose.num_instances != args.num_envs:
        raise AssertionError("wrong rigid-body view count")
    dt = sim.get_physics_dt()
    records = {key: [] for key in ("t", "channels", "ground_truth", "ground_truth_right",
                                  "pos_w", "probe_pos_w", "quat_w", "capture_time",
                                  "sample_dt", "valid", "imu_force", "imu_gyro", "imu_time")}
    subset_checks = 0
    for step in range(args.steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        data, imu = nose.data, read_imu(scene["imu"])
        for value in (data.channels, data.concentration_gt, data.pos_w, data.quat_w,
                      data.probe_pos_w, imu.specific_force, imu.angular_velocity):
            if not bool(torch.isfinite(value).all()):
                raise AssertionError(f"nonfinite output at physics step {step}")
        ready = as_torch(nose._timestamp) + 1e-6 >= nose.cfg.update_period
        if not bool(data.valid[ready].all()) or not bool(imu.valid.all()):
            raise AssertionError("sensors did not produce valid post-step captures")
        if not torch.allclose((data.probe_pos_w[:, 0] - data.probe_pos_w[:, 1]).norm(dim=-1),
                              torch.full((args.num_envs,), .04, device=args.device), atol=1e-4):
            raise AssertionError("stereo baseline was not preserved by mount transforms")
        held = data.channels.clone()
        if not torch.equal(nose.data.channels, held):
            raise AssertionError("repeated lazy read mutated device state")
        values = (data.channels, data.concentration_gt, data.concentration_right_gt,
                  data.pos_w, data.probe_pos_w, data.quat_w, data.timestamp,
                  data.sample_dt, data.valid, imu.specific_force, imu.angular_velocity, imu.timestamp)
        records["t"].append((step + 1) * dt)
        for key, value in zip(tuple(records)[1:], values):
            records[key].append(value.detach().cpu().numpy().copy())
        if step in (args.steps // 3, 2 * args.steps // 3):
            untouched = torch.arange(1, args.num_envs, device=args.device)
            gas_before = data.channels[untouched].clone()
            time_before = data.timestamp[untouched].clone()
            imu_before = imu.specific_force[untouched].clone()
            reset_ids = torch.tensor([0], device=args.device)
            if subset_checks == 0:
                nose.reset(env_ids=reset_ids)
                scene["imu"].reset(env_ids=reset_ids)
            else:
                mask = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
                mask[0] = True
                env_mask = wp.from_torch(mask, dtype=wp.bool)
                nose.reset(env_mask=env_mask)
                scene["imu"].reset(env_mask=env_mask)
            reset_data, reset_imu = nose.data, read_imu(scene["imu"])
            assert not reset_data.valid[0] and reset_data.timestamp[0] == 0
            assert not reset_imu.valid[0] and reset_imu.timestamp[0] == 0
            assert torch.equal(reset_data.channels[untouched], gas_before)
            assert torch.equal(reset_data.timestamp[untouched], time_before)
            assert torch.equal(reset_imu.specific_force[untouched], imu_before)
            assert as_torch(nose._timestamp)[0] == 0
            subset_checks += 1
    assert subset_checks == 2
    assert bool(nose.data.valid.all()), "nose did not resume captures after subset reset"
    result = {key: np.asarray(value) for key, value in records.items()}
    # No-positive-plume can be legitimate at a remote mount; report it explicitly.
    metadata = {"robot": args.robot, "body_path": body, "fixed_root_fixture": True,
                "steps": args.steps, "num_envs": args.num_envs,
                "device": args.device, "transport_backend": args.transport_backend,
                "device_backend": args.device_backend, "species": list(data.species_names),
                "channel_names": list(data.channel_names), "quaternion_order": "xyzw",
                "subset_reset_checks": subset_checks,
                "max_ground_truth_ppm": float(result["ground_truth"].max())}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".npz", **result)
    Path(str(out) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("[PASS] live sensor mounting, finite captures, lazy reads, index and Warp-mask resets")
    print(json.dumps(metadata, indent=2))
    if metadata["max_ground_truth_ppm"] <= 0:
        print("[NOTE] no gas reached this mount; this run proves binding, not chemical response")
    return 0


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("cuboid", *ROBOTS), default="cuboid")
    parser.add_argument("--body-path", help="override rigid body/child frame relative to Robot")
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--transport-backend", choices=("numpy", "warp"), default="warp")
    parser.add_argument("--device-backend", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--device-profile", default="scentience_v1")
    parser.add_argument("--species", default="ethanol")
    parser.add_argument("--out", default="runs/isaac_smoke")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps < 36:
        parser.error("select at least two environments and 36 physics steps for subset reset/recovery")
    launcher = AppLauncher(args)
    try:
        return run(args)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
