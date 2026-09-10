"""Train/evaluate PPO, SAC or TD3 on the CPU reference navigation environment.

The same observation history and normalization are saved with each checkpoint.
This is a small reproducible baseline, not a robot locomotion controller.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scentience_olfaction.config import config_dict, load_navigation
from scentience_olfaction.envs.plume_nav import PlumeNavConfig, PlumeNavEnv
from scentience_olfaction.envs.wrappers import ObservationHistory
from scentience_olfaction.recording.recorder import run_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "sac", "td3"), default="ppo")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--out", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--evaluate", type=Path, help="directory containing model.zip and normalization.pkl")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args(argv)
    if min(args.steps, args.num_envs, args.history, args.episodes) < 1:
        parser.error("steps, num-envs, history and episodes must be positive")
    if not 0 <= args.gamma <= 1:
        parser.error("gamma must lie in [0,1]")
    try:
        from gymnasium.wrappers import RescaleAction
        from stable_baselines3 import PPO, SAC, TD3
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise SystemExit('Install training dependencies: pip install -e ".[train]"') from exc

    if args.evaluate:
        saved = json.loads((args.evaluate / "run.json").read_text())
        cfg = load_navigation(args.evaluate / "environment.json")
        args.algorithm, args.history = saved["algorithm"], saved["history"]
        args.num_envs = 1
    else:
        cfg = load_navigation(args.config) if args.config else PlumeNavConfig()
        cfg.shaping_gamma = args.gamma

    def make_env():
        env = ObservationHistory(PlumeNavEnv(cfg), args.history)
        return Monitor(RescaleAction(env, -1.0, 1.0), info_keywords=("is_success", "spl"))

    vec = DummyVecEnv([make_env for _ in range(args.num_envs)])
    vec.seed(args.seed)
    algorithms = {"ppo": PPO, "sac": SAC, "td3": TD3}
    if args.evaluate:
        vec = VecNormalize.load(args.evaluate / "normalization.pkl", vec)
        vec.training, vec.norm_reward = False, False
        model = algorithms[args.algorithm].load(args.evaluate / "model.zip", env=vec, device="cpu")
        obs, results = vec.reset(), []
        while len(results) < args.episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = vec.step(action)
            if done[0]:
                results.append({k: infos[0][k] for k in ("is_success", "dist", "spl", "path_length_m")})
        print(json.dumps({"seed": args.seed, "episodes": results}, indent=2))
        vec.close()
        return results

    args.out.mkdir(parents=True, exist_ok=False)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    kwargs = {"n_steps": 256, "batch_size": 64} if args.algorithm == "ppo" else {
        "buffer_size": 100_000, "learning_starts": 1000, "batch_size": 128}
    model = algorithms[args.algorithm]("MlpPolicy", vec, seed=args.seed, device="cpu",
                                       gamma=cfg.shaping_gamma, verbose=1, **kwargs)
    metadata = run_metadata(config_dict(cfg), args.seed, algorithm=args.algorithm,
                            history=args.history, num_envs=args.num_envs,
                            requested_timesteps=args.steps)
    (args.out / "environment.json").write_text(json.dumps(config_dict(cfg), indent=2), encoding="utf-8")
    model.learn(total_timesteps=args.steps)
    model.save(args.out / "model.zip")
    vec.save(args.out / "normalization.pkl")
    metadata["actual_timesteps"] = model.num_timesteps
    (args.out / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    vec.close()
    return args.out


if __name__ == "__main__":
    main()
