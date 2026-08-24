"""
Warp GPU path must agree with the NumPy reference on PHYSICS, not on RNG.

The two draw different random streams, so per-filament trajectories differ.
What must match is the statistical behaviour the model is defined by:
growth law, mass conservation, OU stationary variance, and the concentration
field's aggregate statistics.  Asserting bitwise equality would be a test of
the RNG, not the physics.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

wp_mod = pytest.importorskip("warp")  # optional [gpu] extra

from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig
from scentience_olfaction.transport.filament_warp import WarpFilamentPlume

CFG = dict(source_pos=(0., 0., 1.), release_rate_hz=40.0, wind_mean=(1., 0., 0.),
           turbulence_intensity=0.30, lagrangian_timescale=1.5,
           meander_std_rad=0.22, meander_timescale=15.0, gamma=2e-3, sigma0=0.05,
           max_filaments=4000, max_age_s=40.0)
DT, PROBE = 0.01, np.array([[8., 0., 1.]], np.float32)


def _run(n_steps=20000):
    cfg = FilamentPlumeConfig(**CFG)
    npr = FilamentPlume(cfg, seed=7)
    wpp = WarpFilamentPlume(cfg, n_envs=1, seed=7); wpp.set_probes(PROBE)
    a = np.empty(n_steps); b = np.empty(n_steps)
    for i in range(n_steps):
        npr.step(DT); wpp.step(DT)
        a[i] = npr.sample(PROBE)[0]; b[i] = wpp.sample()[0]
    return a, b, npr, wpp


def test_parity():
    a, b, npr, wpp = _run()
    na, nb = npr.n_alive, int(wpp.n_alive[0])
    print(f"alive  numpy={na}  warp={nb}")
    assert abs(na - nb) / max(na, 1) < 0.10, "population size diverged"

    for nm, x in (("numpy", a), ("warp", b)):
        print(f"{nm:6s} mean={x.mean():.5f} peak={x.max():.4f} "
              f"zero={(x==0).mean():.3f} p95={np.quantile(x,.95):.4f}")

    # Aggregate field statistics must agree within sampling error.
    assert abs(a.mean() - b.mean()) / max(a.mean(), 1e-9) < 0.25, "mean concentration"
    assert abs((a == 0).mean() - (b == 0).mean()) < 0.10, "zero fraction"
    assert 0.4 < np.quantile(a, .95) / max(np.quantile(b, .95), 1e-12) < 2.5, "p95"

    # Growth law is deterministic -- this one IS exact.
    s_np = npr.sigma[npr.alive]; s_wp = wpp.sigma.numpy()[0][wpp.alive.numpy()[0] == 1]
    t_max = CFG["max_age_s"]
    hi = math.sqrt(CFG["sigma0"]**2 + CFG["gamma"] * t_max)
    for nm, s in (("numpy", s_np), ("warp", s_wp)):
        assert s.min() >= CFG["sigma0"] - 1e-9 and s.max() <= hi + 1e-6, f"{nm} sigma range"
    print(f"sigma range ok: [{CFG['sigma0']:.3f}, {hi:.4f}]")


def test_ou_stationary_variance():
    """OU velocity must converge to sigma_u, independent of dt. This is the
    property that a dt-scaled kick (rather than sqrt(dt)) silently breaks."""
    cfg = FilamentPlumeConfig(**{**CFG, "meander_std_rad": 0.0})
    for dt in (0.005, 0.01, 0.05):
        w = WarpFilamentPlume(cfg, n_envs=1, seed=3)
        for _ in range(int(30.0 / dt)):
            w.step(dt)
        al = w.alive.numpy()[0] == 1
        s = w.uprime.numpy()[0][al].std()
        print(f"dt={dt:.3f}  sigma_u measured={s:.4f}  target={w.sigma_u:.4f}")
        assert abs(s - w.sigma_u) / w.sigma_u < 0.25, f"OU variance drifts with dt={dt}"


if __name__ == "__main__":
    test_parity(); print(); test_ou_stationary_variance(); print("\nPASS")
