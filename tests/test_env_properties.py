import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
gym = pytest.importorskip("gymnasium")
from scentience_olfaction.envs.plume_nav import PlumeNavEnv, PlumeNavConfig


def short_cfg():
    cfg = PlumeNavConfig()
    cfg.timeout_s = 10.0
    cfg.warmup_s = 5.0
    cfg.plume.max_filaments = 2000
    return cfg


def test_obs_always_finite_and_gt_free():
    """No NaN/inf over a full episode, and the observation dimension leaves no
    room for ground truth (6 defl + 1 ddt + 2 wind + 2 heading = 11)."""
    env = PlumeNavEnv(short_cfg())
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        obs, r, te, tr, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        if te or tr:
            break
    assert env.observation_space.shape == (11,)


def test_shaping_telescopes():
    """Potential-based shaping must sum to scale*(d0 - dT) over any rollout
    (Ng et al. 1999) -- the property that makes it optimum-preserving."""
    env = PlumeNavEnv(short_cfg())
    env.reset(seed=1)
    d0 = env._dist_to_source()
    total = 0.0
    for _ in range(150):
        obs, r, te, tr, info = env.step(np.array([1.0, 0.2], np.float32))
        total += r
        if te or tr:
            break
    dT = info["dist"]
    bonus = 10.0 if (te and dT < env.cfg.success_radius + 1e-6) else 0.0
    bonus -= 5.0 if te and dT >= env.cfg.success_radius else 0.0
    expected = env.cfg.shaping_scale * (d0 - dT) + bonus
    assert abs(total - expected) < 1e-6, (total, expected)


def test_out_of_bounds_terminates():
    env = PlumeNavEnv(short_cfg())
    env.reset(seed=2)
    env._pos = np.array([29.5, 0.0])
    env._heading = 0.0                       # drive straight at the boundary
    for _ in range(50):
        obs, r, te, tr, _ = env.step(np.array([1.0, 0.0], np.float32))
        if te:
            break
    assert te, "leaving the domain must terminate"
