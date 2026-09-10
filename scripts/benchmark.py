"""Compare seeded sensor-only navigation baselines and save reproducible metrics."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scentience_olfaction.agents.baselines import CastAndSurge, RandomAgent, StereoCastAndSurge
from scentience_olfaction.config import config_dict, load_navigation
from scentience_olfaction.envs.plume_nav import PlumeNavConfig, PlumeNavEnv
from scentience_olfaction.recording.recorder import EpisodeRecorder, run_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--record", action="store_true", help="save aligned observation/action trajectories")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    cfg = load_navigation(args.config) if args.config else PlumeNavConfig()
    args.out.mkdir(parents=True, exist_ok=False)
    meta = run_metadata(config_dict(cfg), args.seed, benchmark="sensor-only-baselines")
    rows = []
    factories = {
        "random": lambda env, seed: RandomAgent(env.action_space, seed),
        "cast_and_surge": lambda env, seed: CastAndSurge(dt=cfg.dt, seed=seed),
        "stereo_cast_and_surge": lambda env, seed: StereoCastAndSurge(dt=cfg.dt, seed=seed),
    }
    for name, factory in factories.items():
        env = PlumeNavEnv(cfg)
        for episode in range(args.episodes):
            seed = args.seed + episode
            obs, _ = env.reset(seed=seed)
            agent = factory(env, seed)
            agent.reset()
            rec = EpisodeRecorder(args.out / name, [f"obs_{i}" for i in range(11)],
                                  env._plume.species_names, meta) if args.record else None
            total = 0.0
            while True:
                action = agent.act(obs)
                next_obs, reward, term, trunc, info = env.step(action)
                total += reward
                declared = bool(getattr(agent, "declared", False))
                if rec:
                    rec.log(obs=obs, action=action, next_obs=next_obs, reward=reward,
                            t=info["elapsed_s"], terminated=term, truncated=trunc,
                            declared=declared)
                obs = next_obs
                if term or trunc or declared:
                    success = info["dist"] < cfg.success_radius
                    row = {"agent": name, "seed": seed, "success": success,
                           "declared": declared, "false_declaration": declared and not success,
                           "return": total, **info}
                    rows.append(row)
                    if rec:
                        rec.end_episode(**row)
                    break
        env.close()
    with (args.out / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in factories:
        selected = [r for r in rows if r["agent"] == name]
        n, successes = len(selected), sum(r["success"] for r in selected)
        p, z = successes / n, 1.96
        centre = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        summary[name] = {"episodes": n, "success_rate": p,
                         "success_95pct_wilson": [centre-half, centre+half],
                         "mean_final_distance_m": float(np.mean([r["dist"] for r in selected])),
                         "mean_path_length_m": float(np.mean([r["path_length_m"] for r in selected])),
                         "false_declarations": sum(r["false_declaration"] for r in selected)}
    (args.out / "summary.json").write_text(json.dumps({**meta, "results": summary}, indent=2),
                                          encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
