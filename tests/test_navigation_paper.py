"""Chasing Ghosts navigation machinery (France et al., arXiv:2602.19577):
DivergenceSignal (Eqs. 5-7), SourceDeclaration (Eqs. 9-11), and the
StereoCastAndSurge baseline that composes them with the stereo cue.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

from scentience_olfaction.agents.declaration import SourceDeclaration
from scentience_olfaction.oio.oio import DivergenceSignal


# ---------------------------------------------------------------- declaration
def test_declaration_reproduces_paper_worked_example():
    """Paper Eq. 11: k=20, 95% -> point estimate 1.05 m, CI ~ [1.001m, 1.2025m].
    The max arrives EARLY and the signal then plateaus near it -- the profile
    of a robot sitting at the source, which is when declaring is legitimate."""
    d = SourceDeclaration(confidence=0.95, margin=0.25, min_samples=20)
    m = 3.7
    d.observe(m)
    for _ in range(19):
        d.observe(0.9 * m)          # >= m/1.25, so 'at the maximum' holds
    assert d.k == 20 and d.m == m
    assert d.point_estimate() == pytest.approx(1.05 * m, rel=1e-9)
    lo, hi = d.interval()
    assert lo == pytest.approx(m / 0.975 ** (1 / 20), rel=1e-9)
    assert hi == pytest.approx(m / 0.025 ** (1 / 20), rel=1e-9)
    assert hi == pytest.approx(1.2025 * m, rel=1e-3)   # the paper's number
    assert d.declared()                                 # 1.2025 <= 1.25


def test_declaration_refuses_while_signal_still_climbing_or_after_leaving():
    m = 2.0
    # still climbing: every sample is a new max -> plateau gate must refuse
    d = SourceDeclaration(min_samples=20)
    for i in range(40):
        d.observe(0.05 * (i + 1) * m)
    assert not d.declared()
    # left the plume: current reading far below the remembered max -> refuse
    d = SourceDeclaration(min_samples=20)
    d.observe(m)
    for _ in range(30):
        d.observe(0.1 * m)          # qualifying but nowhere near m
    assert not d.declared()


def test_declaration_needs_samples_and_signal():
    d = SourceDeclaration(min_samples=20, min_signal=0.01)
    assert not d.declared()                    # no data
    for _ in range(100):
        d.observe(0.0)                         # clean air must not count
    assert d.k == 0 and not d.declared()
    for _ in range(19):
        d.observe(1.0)
    assert not d.declared()                    # below min_samples
    d.observe(1.0)
    assert d.declared()


def test_declaration_interval_tightens_with_k():
    widths = []
    d = SourceDeclaration(min_samples=1)
    for k in (5, 20, 100):
        d.reset()
        for _ in range(k):
            d.observe(1.0)
        lo, hi = d.interval()
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]


# ----------------------------------------------------------- divergence/signal
def test_divergence_signal_orders_taus():
    with pytest.raises(AssertionError):
        DivergenceSignal(tau_fast_s=5.0, tau_slow_s=8.0, tau_signal_s=3.0)


def test_divergence_signal_onset_then_loss():
    """Odor step ON -> divergence above signal line (surge). Step OFF and
    decay -> divergence falls below the signal line (cast)."""
    f = DivergenceSignal()
    dt = 0.1
    surged = False
    for _ in range(50):                      # 5 s of odor
        out = f.step(1.0, dt)
        surged = surged or out["surging"]
    assert surged and out["divergence"] > 0.0 - 1e-12
    fell = False
    for _ in range(100):                     # 10 s of clean air
        out = f.step(0.0, dt)
        fell = fell or (not out["surging"])
    assert fell, "loss of odor never triggered the cast condition"


# ------------------------------------------------------------- stereo baseline
def _obs(left, right, wind=(-1.0, 0.0)):
    o = np.zeros(11, np.float32)
    o[:3], o[3:6] = left, right
    o[6] = 0.0
    o[7:9] = wind
    return o


def test_stereo_agent_steers_toward_earlier_onset():
    """Paper Eqs. 3-4: whiff hits the LEFT sensor first -> plume is to the
    left -> turn bias left (+w). Lag-based, so per-die calibration
    differences (here: very different amplitudes) must not matter."""
    from scentience_olfaction.agents.baselines import StereoCastAndSurge
    quiet, l_on, r_on = [0.0] * 3, [0.9, 0.0, 0.0], [0.05, 0.0, 0.0]
    ag = StereoCastAndSurge(dt=0.05, seed=0)
    ag.reset()
    ag.act(_obs(quiet, quiet))
    for _ in range(4):                    # left smells it first...
        ag.act(_obs(l_on, quiet))
    a = ag.act(_obs(l_on, r_on))          # ...right follows 200 ms later
    assert a[1] > 0.0, "left-first onset must bias the turn left (+w)"

    ag.reset()
    ag.act(_obs(quiet, quiet))
    for _ in range(4):                    # right first, amplitudes swapped
        ag.act(_obs(quiet, [0.9, 0.0, 0.0]))
    a = ag.act(_obs([0.05, 0.0, 0.0], [0.9, 0.0, 0.0]))
    assert a[1] < 0.0, "right-first onset must bias the turn right (-w)"


def test_stereo_agent_simultaneous_onsets_hold_wind_line():
    """Same-tick onsets carry no lateral information: the lag cue must stay
    zero and the agent holds the upwind line (the paper's deadband)."""
    from scentience_olfaction.agents.baselines import StereoCastAndSurge
    quiet, on = [0.0] * 3, [0.5, 0.0, 0.0]
    ag = StereoCastAndSurge(dt=0.05, seed=0)
    ag.reset()
    ag.act(_obs(quiet, quiet))
    for _ in range(10):
        a = ag.act(_obs(on, on))          # both sides at once, forever
    assert a[1] == 0.0, "simultaneous onsets must hold the wind line"
    assert ag._bias == 0.0


def test_stereo_agent_episode_and_declaration():
    """Full-loop smoke test on the real env; also checks run_episode's
    declared-stop plumbing scores by ground truth, not by trust."""
    pytest.importorskip("gymnasium")
    from scentience_olfaction.agents.baselines import (StereoCastAndSurge,
                                                       run_episode)
    from scentience_olfaction.envs.plume_nav import PlumeNavConfig, PlumeNavEnv
    env = PlumeNavEnv(PlumeNavConfig())
    r = run_episode(env, StereoCastAndSurge(detect_threshold=0.005,
                                            dt=env.cfg.dt, seed=0), seed=0)
    assert set(r) == {"success", "steps", "reward", "final_dist", "declared"}
    assert r["steps"] > 0 and np.isfinite(r["final_dist"])
    if r["declared"] and r["final_dist"] >= env.cfg.success_radius + 1e-6:
        assert not r["success"], "a wrong declaration must not score success"


if __name__ == "__main__":
    test_declaration_reproduces_paper_worked_example()
    test_declaration_needs_samples_and_signal()
    test_declaration_interval_tightens_with_k()
    test_divergence_signal_onset_then_loss()
    test_stereo_agent_steers_toward_earlier_onset()
    test_stereo_agent_simultaneous_onsets_hold_wind_line()
    print("PASS")
