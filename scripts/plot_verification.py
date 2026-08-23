"""Visual verification: what the DEVICE reports vs what the AIR contains.

The numeric twin of `validate_physics.py`. Two figures:

  sensor_bandwidth.png      ground-truth ppm at a fixed probe, with the slow
                            (packaged, tau_fall ~12 s) and fast (Dennler-class,
                            ~46 ms) device responses on the same timeline --
                            the visual form of the README's "sensor bandwidth
                            gates what a policy can see".
  stereo_lateralisation.png left vs right chemical-sensor deflection for a
                            robot parked off the plume centreline (meander off,
                            wide demo baseline): the left/right separation IS
                            the stereo cue `StereoCastAndSurge` steers on.

Usage:
    pip install "scentience-olfaction[viz]"     # matplotlib
    python scripts/plot_verification.py [--outdir runs/plots] [--seconds 120]

No GPU, no Isaac. Runs in well under a minute at the defaults.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")           # file output only; no display needed
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit("plot_verification needs matplotlib: "
                     'pip install "scentience-olfaction[viz]"') from e

from scentience_olfaction import OlfactionWorld
from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig

DT = 0.05
PROBE = (6.0, 0.0, 1.0)


def _plume_cfg(**kw):
    cfg = dict(source_pos=(0.0, 0.0, 1.0), wind_mean=(1.0, 0.0, 0.0),
               release_rate_hz=40.0, ppm_center_initial=300.0,
               turbulence_intensity=0.30, lagrangian_timescale=1.5,
               meander_std_rad=0.22, meander_timescale=15.0,
               gamma=2.0e-3, sigma0=0.05, max_filaments=5000, max_age_s=40.0)
    cfg.update(kw)
    return FilamentPlumeConfig(**cfg)


def fig_bandwidth(outdir: str, seconds: float, seed: int) -> str:
    n = int(seconds / DT)
    worlds = {p: OlfactionWorld(FilamentPlume(_plume_cfg(), seed=seed),
                                sensor_profile=p, seed=seed)
              for p in ("packaged_slow", "fast_modulated")}
    truth = np.zeros(n)
    defl = {p: np.zeros(n) for p in worlds}
    for i in range(n):
        for p, w in worlds.items():
            w.step(DT)
            r = w.read(PROBE, dt=DT)
            # deflection below clean-air ratio; RED channel of the left sensor
            defl[p][i] = max(0.0, 1.0 - r["chem_left_red"])
        truth[i] = worlds["packaged_slow"].truth(PROBE)["ethanol"]

    t = np.arange(n) * DT
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax0.plot(t, truth, lw=0.6, color="0.25")
    ax0.set_ylabel("ground truth [ppm]")
    ax0.set_title(f"What the air contains vs what the device reports "
                  f"({seconds:.0f} s at probe {PROBE})")
    ax1.plot(t, defl["packaged_slow"], lw=1.2, label="packaged_slow (tau_fall ~12 s)")
    ax1.plot(t, defl["fast_modulated"], lw=0.7, alpha=0.85,
             label="fast_modulated (~46 ms)")
    ax1.set_ylabel("MOX deflection [frac Rs/R0]")
    ax1.set_xlabel("time [s]")
    ax1.legend(loc="upper right", fontsize=8)
    for ax in (ax0, ax1):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(outdir, "sensor_bandwidth.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_stereo(outdir: str, seconds: float, seed: int) -> str:
    # Meander OFF so the time-averaged centreline is fixed and the left/right
    # asymmetry is pure geometry (same convention as examples/05).
    world = OlfactionWorld(
        FilamentPlume(_plume_cfg(meander_std_rad=0.0, turbulence_intensity=0.10),
                      seed=seed),
        sensor_profile="fast_modulated", seed=seed, stereo_baseline_m=0.5)
    n = int(seconds / DT)
    left = np.zeros(n)
    right = np.zeros(n)
    for i in range(n):
        world.step(DT)
        r = world.read((4.0, -0.25, 1.0), dt=DT, heading=0.0)
        left[i] = max(0.0, 1.0 - r["chem_left_red"])
        right[i] = max(0.0, 1.0 - r["chem_right_red"])

    t = np.arange(n) * DT
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, left, lw=0.8, label="chem_left (on centreline)")
    ax.plot(t, right, lw=0.8, alpha=0.85, label="chem_right (0.5 m off)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("deflection [frac Rs/R0]")
    ax.set_title("Stereo olfaction: the left/right difference is the "
                 "lateralisation cue (meander off, demo baseline 0.5 m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(outdir, "stereo_lateralisation.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None) -> list[str]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=os.path.join("runs", "plots"))
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    paths = [fig_bandwidth(args.outdir, args.seconds, args.seed),
             fig_stereo(args.outdir, args.seconds, args.seed)]
    for p in paths:
        print("wrote", p)
    return paths


if __name__ == "__main__":
    main()
