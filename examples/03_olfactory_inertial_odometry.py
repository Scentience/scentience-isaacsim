"""
Olfactory Inertial Odometry on four robot platforms.

Reference implementation of the OIO concept from:
  France, Daescu -- "Olfactory Inertial Odometry" (arXiv:2506.04539,
    IEEE INERTIAL 2025)
  France et al. -- "Chasing Ghosts" (arXiv:2602.19577), whose dual-timescale
    EMA bout detector is used verbatim in spirit.

A robot transects a live filament plume while dead-reckoning on a biased IMU.
Odor bouts + the anemometer bound heading and crosswind drift.

Run:  python examples/03_olfactory_inertial_odometry.py --platform quadruped
      (choices: uav | quadruped | biped | arm)

The 'arm' platform is the degenerate case: a manipulator base does not
translate, so OIO reduces to end-effector concentration mapping -- the script
sweeps the plume cross-section and prints a coarse concentration profile
instead of odometry error.
"""
import sys, pathlib, argparse, math
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import numpy as np
from scentience_olfaction import FilamentPlume, FilamentPlumeConfig
from scentience_olfaction.sensors.device_np import ScentienceV1, DeviceState
from scentience_olfaction.oio.oio import (OlfactoryInertialOdometry, OIOConfig,
                                          PLATFORMS, BoutDetectorConfig)

def run(platform: str, seed: int = 0, t_end: float = 60.0, verbose=True):
    dt = 0.02
    spec = PLATFORMS[platform]
    plume = FilamentPlume(FilamentPlumeConfig(
        source_pos=(0, 0, 1), release_rate_hz=40, ppm_center_initial=300.0,
        max_filaments=5000, max_age_s=40), seed=seed)
    for _ in range(int(20 / dt)):
        plume.step(dt)

    if platform == "arm":
        # end-effector sweep across the plume at x=3 m: concentration mapping
        ys = np.linspace(-2, 2, 21)
        prof = []
        for y in ys:
            acc = 0.0
            for _ in range(25):
                plume.step(dt)
                acc += plume.sample(np.array([[3.0, y, 1.0]]))[0]
            prof.append(acc / 25)
        if verbose:
            print("cross-section ppm @ x=3 m:")
            for y, c in zip(ys, prof):
                print(f"  y={y:+.1f}  {'#' * int(min(c, 30))}  {c:.2f}")
        return None

    dev = ScentienceV1("fast_modulated", seed=seed + 1)
    oio = OlfactoryInertialOdometry(OIOConfig(
        platform=platform, wind_world_bearing_rad=0.0,
        detector=BoutDetectorConfig(noise_sigma=2e-3)), seed=seed + 2)

    # weaving transect through the plume, downwind -> upwind
    true_p = np.array([10.0, 3.0]); true_h = math.pi  # facing upwind
    speed = 0.6 if spec.gait_hz else 1.2
    baseline = None
    rng = np.random.default_rng(seed + 3)
    for i in range(int(t_end / dt)):
        omega_true = 0.35 * math.cos(2 * math.pi * 0.05 * i * dt)
        true_h += omega_true * dt
        true_p += speed * np.array([math.cos(true_h), math.sin(true_h)]) * dt
        plume.step(dt)

        conc = dict(zip(plume.species_names,
                        plume.sample_species(np.array([[true_p[0], true_p[1], 1.0]]))[0]))
        r = dev.step(conc, dt, DeviceState(flow_mps=spec.sensor_flow_mps))
        mox = r["chem_left_red"]
        baseline = mox if baseline is None else baseline + 0.002 * (mox - baseline)
        deflection = max(baseline - mox, 0.0)

        wind_w = plume.airflow.velocity(np.array([[true_p[0], true_p[1], 1.0]]))[0][:2]
        c, s = math.cos(-true_h), math.sin(-true_h)
        wind_body = np.array([c * wind_w[0] - s * wind_w[1],
                              s * wind_w[0] + c * wind_w[1]]) \
            + 0.05 * rng.standard_normal(2)
        accel, omega = oio.simulate_imu(np.zeros(2), omega_true, dt)
        out = oio.step(accel, omega, wind_body, deflection, dt)

    e_dr = abs(((out["h_dr"] - true_h) + math.pi) % (2 * math.pi) - math.pi)
    e_oio = abs(((out["h_oio"] - true_h) + math.pi) % (2 * math.pi) - math.pi)
    if verbose:
        print(f"platform={platform}  bouts={oio.detector.n_bouts}")
        print(f"heading error: dead-reckoning {math.degrees(e_dr):6.2f} deg"
              f" | OIO {math.degrees(e_oio):6.2f} deg"
              f" | reduction {100 * (1 - e_oio / max(e_dr, 1e-9)):5.1f}%")
    return e_dr, e_oio

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="quadruped", choices=sorted(PLATFORMS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.platform, a.seed)
