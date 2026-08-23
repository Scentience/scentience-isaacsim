"""
PlumeNav + two baselines + episode logging: the benchmark loop a learned
policy must beat.

  CastAndSurge        classic moth strategy, ignores the stereo cue
  StereoCastAndSurge  adds the kit's inter-sensor lateralisation, divergence
                      signal-line surge/cast switching, and the sensor-only
                      source-declaration stop (Chasing Ghosts: France et al.,
                      arXiv:2602.19577, Secs. III-E1, III-F, III-H)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import numpy as np
from scentience_olfaction.envs.plume_nav import PlumeNavEnv, PlumeNavConfig
from scentience_olfaction.agents.baselines import (CastAndSurge,
                                                   StereoCastAndSurge, run_episode)
from scentience_olfaction.recording.recorder import EpisodeRecorder
from scentience_olfaction.sensors.device_np import CHANNELS

cfg = PlumeNavConfig()
env = PlumeNavEnv(cfg)
rec = EpisodeRecorder("runs/cast_and_surge", CHANNELS,
                      ["ethanol"], {"agent": "cast_and_surge"})
for name, mk in (("cast_and_surge", CastAndSurge),
                 ("stereo_cast_and_surge", StereoCastAndSurge)):
    results = []
    for seed in range(5):
        r = run_episode(env, mk(detect_threshold=0.005, dt=cfg.dt, seed=seed),
                        seed=seed)
        results.append(r)
        extra = "  [declared]" if r.get("declared") else ""
        print(f"  seed {seed}: success={r['success']} "
              f"final_dist={r['final_dist']:.2f} m steps={r['steps']}{extra}")
    print(f"{name}: success rate {np.mean([r['success'] for r in results]):.2f}, "
          f"mean final dist {np.mean([r['final_dist'] for r in results]):.2f} m\n")
