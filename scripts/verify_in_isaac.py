"""In-Isaac runtime verification: log the olfactory sensor against ground
truth inside a LIVE Isaac Sim, and plot the comparison.

The runtime companion to `validate_install.py` (API checks) -- the analogue
of the verification-script + plot workflow vendor sensor repos use to show
clean vs realistic output side by side. Run it INSIDE Isaac:

    ./isaaclab.sh -p /path/to/scripts/verify_in_isaac.py --steps 2000
    # or: ./python.sh ... on a raw Isaac Sim install with Isaac Lab

It builds a minimal scene (ground plane + one rigid cuboid standing in for a
robot base), attaches an OlfactorySensor to the cuboid, steps physics, and
records device channels alongside ground-truth concentration. Outputs:

    verify_in_isaac.npz            time, channels, ground truth, positions
    verify_in_isaac.png            channels vs truth (needs matplotlib)

STATUS: UNVALIDATED -- this script has never executed in a live install
(see ISAAC_COMPATIBILITY.md; this machine cannot run Isaac Sim). What IS
verified: every isaaclab symbol used here exists in the isaaclab==2.3.2
wheel (scripts/check_isaaclab_contract.py, "verify_in_isaac surface"
checks), and the sensor/cfg classes it drives execute correctly under
genuine isaaclab code (scripts/check_isaaclab_binding.py, 10/10). When this
script runs live, paste its console output into ISAAC_COMPATIBILITY.md.
"""

import argparse

# AppLauncher must run before ANY other isaaclab/omni import -- it boots the
# kit runtime. This ordering is the Isaac Lab standard pattern, and it is why
# the imports below are not at the top of the file.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=2000, help="physics steps")
parser.add_argument("--out", default="verify_in_isaac", help="output basename")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from scentience_isaaclab.olfactory_sensor import OlfactorySensorCfg  # noqa: E402


@configclass
class VerifySceneCfg(InteractiveSceneCfg):
    """Ground + one cuboid 'robot base' + the olfactory sensor on it."""

    ground = sim_utils.GroundPlaneCfg()

    robot = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 1.0)),
    )

    nose = OlfactorySensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=0.05,
        sensor_profile="fast_modulated",
        expose_ground_truth=True,   # verification IS the sanctioned use
    )


def main() -> int:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
    scene = InteractiveScene(VerifySceneCfg(num_envs=1, env_spacing=20.0))
    sim.reset()
    print("[verify_in_isaac] scene ready; stepping", args.steps, "physics steps")

    t, chans, gts = [], [], []
    dt = sim.get_physics_dt()
    for i in range(args.steps):
        sim.step()
        scene.update(dt)
        sensor = scene["nose"]
        t.append(i * dt)
        chans.append(sensor.data.channels[0].cpu().numpy().copy())
        gts.append(sensor.data.concentration_gt[0].cpu().numpy().copy())

    t = np.asarray(t)
    chans = np.asarray(chans)
    gts = np.asarray(gts)
    np.savez_compressed(args.out + ".npz", t=t, channels=chans, ground_truth=gts)
    print("[verify_in_isaac] wrote", args.out + ".npz",
          "| channels", chans.shape, "| gt", gts.shape)

    # Sanity summary a human can eyeball in the console log
    defl = np.maximum(1.0 - chans[:, 0], 0.0)     # chem_left_red deflection
    print("[verify_in_isaac] gt ppm: max %.3f mean %.3f | chem_left_red "
          "deflection: max %.4f" % (gts.max(), gts.mean(), defl.max()))
    if gts.max() <= 0.0:
        print("[verify_in_isaac] WARNING: ground truth never nonzero -- "
              "is the plume configured and inside the domain?")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        a0.plot(t, gts[:, 0], lw=0.7, color="0.25")
        a0.set_ylabel("ground truth [ppm]")
        a0.set_title("In-Isaac verification: air vs device")
        a1.plot(t, defl, lw=0.7, label="chem_left_red deflection")
        a1.set_xlabel("time [s]")
        a1.set_ylabel("deflection [frac Rs/R0]")
        a1.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out + ".png", dpi=130)
        print("[verify_in_isaac] wrote", args.out + ".png")
    except ImportError:
        print("[verify_in_isaac] matplotlib absent; skipped plot")

    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
