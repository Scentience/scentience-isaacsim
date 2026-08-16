import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
gym = pytest.importorskip("gymnasium")
from scentience_olfaction.envs.plume_nav import PlumeNavEnv, PlumeNavConfig
from scentience_olfaction.agents.baselines import CastAndSurge, RandomAgent, run_episode
from scentience_olfaction.recording.recorder import EpisodeRecorder


def small_cfg():
    cfg = PlumeNavConfig()
    cfg.timeout_s = 60.0
    cfg.warmup_s = 15.0
    cfg.plume.max_filaments = 3000
    return cfg


def test_env_contract():
    env = PlumeNavEnv(small_cfg())
    obs, info = env.reset(seed=0)
    assert obs.shape == (11,) and obs.dtype == np.float32
    obs2, r, te, tr, info = env.step(env.action_space.sample())
    assert obs2.shape == (11,)
    # determinism under seed
    o1, _ = env.reset(seed=42)
    o2, _ = PlumeNavEnv(small_cfg()).reset(seed=42)
    assert np.allclose(o1, o2)


@pytest.mark.slow
def test_cast_and_surge_beats_random():
    cfg = small_cfg()
    cfg.timeout_s = 90.0
    cfg.warmup_s = 30.0
    env = PlumeNavEnv(cfg)
    cs = [run_episode(env, CastAndSurge(detect_threshold=0.005, dt=cfg.dt,
                                        seed=s), seed=s) for s in range(6)]
    rn = [run_episode(env, RandomAgent(env.action_space, seed=s), seed=s)
          for s in range(6)]
    d_cs = np.mean([r["final_dist"] for r in cs])
    d_rn = np.mean([r["final_dist"] for r in rn])
    assert d_cs < d_rn, f"cast&surge {d_cs:.1f} m vs random {d_rn:.1f} m"


def test_recorder_roundtrip(tmp_path):
    rec = EpisodeRecorder(tmp_path, ["a", "b"], ["ethanol"], {"agent": "x"})
    for i in range(10):
        rec.log(t=i * 0.1, obs=np.arange(3), reward=1.0)
    p = rec.end_episode(success=True)
    arrays, meta = EpisodeRecorder.load(p)
    assert arrays["obs"].shape == (10, 3)
    assert meta["channel_names"] == ["a", "b"] and meta["n_steps"] == 10
