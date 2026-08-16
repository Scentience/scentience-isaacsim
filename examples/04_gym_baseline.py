"""
PlumeNav + the cast-and-surge baseline + episode logging.
This is the benchmark loop a learned policy must beat.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import numpy as np
from scentience_olfaction.envs.plume_nav import PlumeNavEnv, PlumeNavConfig
from scentience_olfaction.agents.baselines import CastAndSurge, run_episode
from scentience_olfaction.recording.recorder import EpisodeRecorder
from scentience_olfaction.sensors.device_np import CHANNELS

cfg = PlumeNavConfig()
env = PlumeNavEnv(cfg)
rec = EpisodeRecorder("runs/cast_and_surge", CHANNELS,
                      ["ethanol"], {"agent": "cast_and_surge"})
results = []
for seed in range(5):
    r = run_episode(env, CastAndSurge(detect_threshold=0.005, dt=cfg.dt, seed=seed),
                    seed=seed)
    results.append(r)
    print(f"seed {seed}: success={r['success']} final_dist={r['final_dist']:.2f} m"
          f" steps={r['steps']}")
print(f"success rate {np.mean([r['success'] for r in results]):.2f}, "
      f"mean final dist {np.mean([r['final_dist'] for r in results]):.2f} m")
