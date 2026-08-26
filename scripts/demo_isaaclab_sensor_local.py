"""Run the Isaac Lab olfactory sensor's FULL runtime loop -- on a machine
that cannot run Isaac Sim.

Isaac Lab itself cannot run without a GPU: it requires Isaac Sim, whose
renderer requires RT cores. What CAN run is everything this wrapper adds on
top of it. This demo executes the sensor's complete Isaac Lab lifecycle --
construction against the real `SensorBase`, `_initialize_impl`, per-step
`update()` with the real lazy-evaluation timestamp machinery, `data` reads,
`reset()` -- driven by genuine isaaclab 2.3.2 code, with exactly three
things faked, all listed here:

  1. the Omniverse kit runtime (carb/omni/isaacsim) -- inert stubs
     (scripts/isaaclab_kit_stubs.py, same as the validation harness);
  2. `SimulationContext.instance()` -- a namespace giving device/dt;
  3. the PhysX rigid-body view -- a scripted trajectory: a robot flying
     upwind through the plume at 1 m/s.

The plume transport (Warp, CPU or CUDA), the torch device model, and every
line of isaaclab SensorBase code are real. What this does NOT prove: PhysX,
USD prims, rendering -- see ISAAC_COMPATIBILITY.md tier 3.

Usage (needs the harness venv -- deps listed in check_isaaclab_binding.py):
    python scripts/demo_isaaclab_sensor_local.py [--steps 400] [--speed 1.0]
"""
import argparse
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from isaaclab_kit_stubs import add_isaaclab_paths, install_stubs  # noqa: E402

add_isaaclab_paths()
install_stubs()

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.sensors.sensor_base as sensor_base_mod  # noqa: E402

from scentience_isaaclab.olfactory_sensor import (  # noqa: E402
    OlfactorySensor, OlfactorySensorCfg)


class FakeRigidBodyView:
    """Stands in for the PhysX view: (n, 7) transforms, pos + quat(xyzw)."""

    def __init__(self, n: int, start, device: str):
        self.count = n
        self._tf = torch.zeros(n, 7, device=device)
        self._tf[:, :3] = torch.tensor(start, device=device)
        self._tf[:, 6] = 1.0            # identity quaternion, xyzw

    def get_transforms(self) -> torch.Tensor:
        return self._tf.clone()

    def advance(self, vel, dt: float) -> None:
        self._tf[:, :3] += torch.tensor(vel, device=self._tf.device) * dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--envs", type=int, default=4)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[demo] torch device: {device} (works on either)")

    # --- the three declared fakes -----------------------------------------
    sim_ns = types.SimpleNamespace(device=device, backend="torch",
                                   get_physics_dt=lambda: args.dt)
    sim_utils.SimulationContext.instance = staticmethod(lambda: sim_ns)
    fake_prims = [object()] * args.envs
    sensor_base_mod.sim_utils.find_matching_prims = lambda expr: fake_prims

    view = FakeRigidBodyView(args.envs, start=(12.0, 0.0, 1.0), device=device)
    import isaacsim.core.simulation_manager as sm
    sm.SimulationManager.get_physics_sim_view = staticmethod(
        lambda: types.SimpleNamespace(create_rigid_body_view=lambda expr: view))

    # --- from here on, everything is the real wrapper on real isaaclab ----
    cfg = OlfactorySensorCfg(prim_path="/World/envs/env_.*/Robot/base",
                             update_period=args.dt,
                             sensor_profile="fast_modulated",
                             expose_ground_truth=True)   # verification use
    sensor = OlfactorySensor(cfg)
    sensor._initialize_impl()
    sensor._is_initialized = True
    print(f"[demo] sensor initialized: {sensor._num_envs} envs, "
          f"{len(cfg.channel_names)} channels, plume on '{device}'")

    print(f"[demo] flying upwind from x=12 m at {args.speed} m/s "
          f"({args.steps} steps x {args.dt} s)")
    header = f"{'t [s]':>6} {'x [m]':>6} {'gt ppm':>8} {'left_red':>9} {'right_red':>10}"
    print(header + "\n" + "-" * len(header))

    peak_defl, peak_gt = 0.0, 0.0
    for i in range(args.steps):
        view.advance((-args.speed, 0.0, 0.0), args.dt)
        sensor._step_plume(args.dt)
        sensor.update(args.dt, force_recompute=True)
        d = sensor.data                     # real lazy-eval path
        gt = float(d.concentration_gt[0, 0])
        lr = float(d.channels[0, 0])
        rr = float(d.channels[0, 3])
        peak_gt = max(peak_gt, gt)
        peak_defl = max(peak_defl, 1.0 - lr)
        if i % max(1, args.steps // 10) == 0:
            print(f"{i*args.dt:6.1f} {float(d.pos_w[0,0]):6.2f} {gt:8.3f} "
                  f"{lr:9.4f} {rr:10.4f}")

    sensor.reset()                          # real reset path incl. plume/device
    print("-" * len(header))
    print(f"[demo] peak ground truth {peak_gt:.3f} ppm | "
          f"peak chem_left_red deflection {peak_defl:.4f}")
    if peak_gt <= 0.0:
        print("[demo] FAIL: sensor never smelled the plume")
        return 1
    if peak_defl <= 0.0:
        print("[demo] FAIL: device channels never responded")
        return 1
    print("[demo] PASS: full Isaac Lab sensor lifecycle ran without Isaac. "
          "PhysX/USD/rendering remain live-install territory "
          "(ISAAC_COMPATIBILITY.md tier 3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
