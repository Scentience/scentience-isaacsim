import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scentience_olfaction.plume.filament import FilamentPlumeConfig
from scentience_olfaction.transport.filament_warp import WarpFilamentPlume


def test_pool_wraparound_is_bounded_and_stable():
    """Release far more filaments than capacity: the round-robin cursor must
    wrap, alive count must saturate at capacity, and sampling must stay
    finite. This is the failure mode of fixed pools -- overflow corrupting
    state -- so it gets its own test."""
    cfg = FilamentPlumeConfig(release_rate_hz=500.0, max_filaments=200,
                              max_age_s=1e9)
    w = WarpFilamentPlume(cfg, n_envs=2, seed=0)
    w.set_probes(np.tile([[1.0, 0.0, 1.0]], (2, 1)).astype(np.float32))
    for _ in range(400):                     # 4 s * 500/s = 2000 >> 200
        w.step(0.01)
    alive = w.n_alive
    assert (alive <= 200).all() and (alive >= 190).all(), alive
    c = w.sample()
    assert np.isfinite(c).all() and (c >= 0).all()


def test_multi_env_independent_sources():
    cfg = FilamentPlumeConfig(max_filaments=500, meander_std_rad=0.0)
    w = WarpFilamentPlume(cfg, n_envs=3, seed=0)
    src = np.array([[0, 0, 1], [0, 5, 1], [0, -5, 1]], np.float32)
    w.source.assign(src)
    w.set_probes((src + np.array([[3, 0, 0]], np.float32)))
    acc = np.zeros(3)
    for i in range(600):
        w.step(0.01)
        if i > 200:
            acc += w.sample()
    assert (acc > 0).all(), f"every env sees its own plume: {acc}"
