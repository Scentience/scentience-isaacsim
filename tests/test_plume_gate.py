"""
The plume realism gate, as a test.

This is the load-bearing test in the repository. If it regresses, every
downstream result is invalid, and nothing else in the suite will notice.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig
from scentience_olfaction.validation import plume_stats as ps

DT = 0.01
PROBE = np.array([[8.0, 0.0, 1.0]])
CFG = dict(source_pos=(0., 0., 1.), release_rate_hz=40.0, wind_mean=(1., 0., 0.),
           turbulence_intensity=0.30, lagrangian_timescale=1.5,
           meander_std_rad=0.22, meander_timescale=15.0, gamma=2e-3, sigma0=0.05,
           max_filaments=8000, max_age_s=40.0)


def _record(seconds=600.0, **over):
    p = FilamentPlume(FilamentPlumeConfig(**{**CFG, **over}), seed=7)
    out = np.empty(int(seconds / DT))
    for i in range(out.size):
        p.step(DT)
        out[i] = p.sample(PROBE)[0]
    return out


@pytest.mark.slow
def test_gate_passes():
    x = _record()
    st = ps.summarize(x, DT, 0.10 * float(x[x > 0].mean()))
    ok, fails = ps.gate(st)
    assert ok, f"realism gate failed: {fails}"
    assert st["blank_cv"] > 1.0
    assert st["peak_to_mean"] > 3.0
    # Tail exponent should bracket the -3/2 first-return exponent.
    assert -2.5 < st["blank_tail_slope"] < -1.0, st["blank_tail_slope"]


@pytest.mark.slow
def test_meander_ablation_fails_the_gate():
    """Removing large-scale meander MUST fail. If this test starts passing,
    the gate has stopped discriminating and is no longer protecting anything."""
    x = _record(meander_std_rad=0.0)
    st = ps.summarize(x, DT, 0.10 * float(x[x > 0].mean()))
    ok, _ = ps.gate(st)
    assert not ok, "gate no longer detects a plume with no large-scale meander"
    assert st["blank_cv"] < 1.2


def test_deterministic_replay():
    a, b = (FilamentPlume(FilamentPlumeConfig(**CFG), seed=42) for _ in range(2))
    for _ in range(500):
        a.step(DT); b.step(DT)
    assert np.allclose(a.sample(PROBE), b.sample(PROBE))
