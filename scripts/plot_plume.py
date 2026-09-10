"""Render a reproducible plume slice and sensor trace, with a JSON run record."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scentience_olfaction import OlfactionWorld
from scentience_olfaction.config import load_world
from scentience_olfaction.recording.recorder import run_metadata
from scentience_olfaction.visualization import plot_plume


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe", type=float, nargs=3, default=(6.0, 0.0, 1.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--xlim", type=float, nargs=2, default=(-2.0, 16.0))
    parser.add_argument("--ylim", type=float, nargs=2, default=(-5.0, 5.0))
    parser.add_argument("--out", type=Path, default=Path("runs/plume"))
    args = parser.parse_args(argv)
    if not np.isfinite([args.seconds, args.dt]).all() or not 0 < args.dt <= args.seconds:
        parser.error("Require finite seconds >= dt > 0")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    world = load_world(args.config) if args.config else OlfactionWorld.simple(
        strength_ppm=300, sensor_profile="fast_modulated", seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    probe = np.asarray(args.probe)
    times, truth, measured = [], [], []
    for _ in range(int(args.seconds / args.dt)):
        world.step(args.dt)
        reading = world.read(probe, dt=args.dt)
        times.append(world.plume.t)
        truth.append(list(world.truth(probe).values())[0])
        measured.append(reading[world.channel_names[0]])
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), height_ratios=(2, 1, 1), layout="constrained")
    plot_plume(world.plume, ax=axes[0], height_m=float(probe[2]), xlim=args.xlim, ylim=args.ylim)
    axes[0].scatter(*probe[:2], color="#ff686b", edgecolors="white", s=45, label="probe")
    axes[0].legend(loc="lower left")
    axes[1].plot(times, truth, color="#247b92", linewidth=0.8)
    axes[1].set(ylabel="Air concentration [ppm]", xlabel="Simulation time [s]")
    axes[2].plot(times, measured, color="#a65121", linewidth=0.9)
    channel = world.channel_names[0]
    unit = "Rs/R0" if channel.startswith("chem_") else ("ppm" if channel == "co2_ppm" else "nA")
    axes[2].set(ylabel=f"{world.channel_names[0]} [{unit}]", xlabel="Simulation time [s]")
    for ax in axes[1:]:
        ax.grid(alpha=0.2)
    fig.suptitle("Scentience olfaction · transport and instrument response", fontsize=15)
    path = args.out / "plume.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    meta = run_metadata(world.configuration(), world._seed,
                        sensor_profile=world.sensor_profile, device_profile=world.device_profile,
                        channel_names=list(world.channel_names), dt_s=args.dt,
                        slice_xlim_m=list(args.xlim), slice_ylim_m=list(args.ylim),
                        duration_s=times[-1], probe_m=probe.tolist(),
                        mean_ppm=float(np.mean(truth)), peak_ppm=float(np.max(truth)))
    (args.out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(path)
    return path


if __name__ == "__main__":
    main()
