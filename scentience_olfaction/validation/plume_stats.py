"""Intermittency diagnostics and a regression gate for the reference scenario.

Threshold, probe location, sampling period, and finite-record censoring affect
all statistics. These checks detect regressions in the supplied benchmark;
they do not certify arbitrary flows against turbulent-plume theory.

Celani et al. (2014), https://doi.org/10.1103/PhysRevX.4.041015, predict a
-3/2 *probability density* exponent in a scaling regime. The corresponding
untruncated CCDF slope is -1/2. The empirical CCDF tail slopes below are
summaries of finite records with cutoffs, not estimates of that PDF exponent.
"""

from __future__ import annotations

import numpy as np


def whiff_blank_durations(
    signal: np.ndarray, dt: float, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Complete above/below-threshold runs; boundary-censored runs are excluded."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("signal must be a finite one-dimensional array")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(threshold):
        raise ValueError("dt must be positive and threshold finite")
    above = signal > threshold
    if above.size == 0:
        return np.array([]), np.array([])
    edges = np.flatnonzero(np.diff(above.astype(np.int8))) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [above.size]))
    lengths = (ends - starts) * dt
    states = above[starts]
    # Drop first and last runs: they are censored by the record boundary.
    lengths, states = lengths[1:-1], states[1:-1]
    return lengths[states], lengths[~states]


def ccdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    return xs, 1.0 - np.arange(xs.size) / xs.size


def tail_exponent(x: np.ndarray, lo_q: float = 0.75, hi_q: float = 0.99) -> float:
    """Empirical log-log CCDF slope over a declared quantile range.

    This descriptive slope includes finite-record and exponential-cutoff
    effects. It must not be compared directly with a theoretical PDF exponent.
    """
    xs, p = ccdf(x)
    if xs.size < 30:
        return float("nan")
    lo, hi = np.quantile(xs, lo_q), np.quantile(xs, hi_q)
    m = (xs >= lo) & (xs <= hi) & (p > 0) & (xs > 0)
    if m.sum() < 10:
        return float("nan")
    return float(np.polyfit(np.log(xs[m]), np.log(p[m]), 1)[0])


def exponentiality(x: np.ndarray) -> float:
    """
    Coefficient of variation.  CV == 1 for an exponential distribution;
    heavy-tailed intermittent plumes give CV substantially above 1.
    """
    if x.size < 10 or x.mean() <= 0:
        return float("nan")
    return float(x.std() / x.mean())


def summarize(signal: np.ndarray, dt: float, threshold: float) -> dict:
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        raise ValueError("summarize requires a nonempty signal")
    whiffs, blanks = whiff_blank_durations(signal, dt, threshold)
    nz = signal[signal > threshold]
    mean_all = float(signal.mean())
    return {
        "intermittency": float((signal > threshold).mean()),
        "mean_ppm": mean_all,
        "peak_ppm": float(signal.max()),
        "peak_to_mean": float(signal.max() / mean_all) if mean_all > 0 else float("nan"),
        "n_whiffs": int(whiffs.size),
        "whiff_median_s": float(np.median(whiffs)) if whiffs.size else float("nan"),
        "whiff_cv": exponentiality(whiffs),
        "whiff_tail_slope": tail_exponent(whiffs),
        "blank_median_s": float(np.median(blanks)) if blanks.size else float("nan"),
        "blank_cv": exponentiality(blanks),
        "blank_tail_slope": tail_exponent(blanks),
        "conditional_mean_ppm": float(nz.mean()) if nz.size else 0.0,
    }


def gate(stats: dict) -> tuple[bool, list[str]]:
    """Pass/fail against the literature targets. Returns (ok, failure reasons)."""
    fails = []
    if stats["n_whiffs"] < 30:
        fails.append(f"too few whiffs ({stats['n_whiffs']}): record longer or move closer")
    if stats["intermittency"] > 0.95:
        fails.append(
            f"intermittency {stats['intermittency']:.3f} > 0.95 -- signal is essentially "
            "always above threshold, so there is no intermittency to exploit and "
            "gradient ascent will solve the task"
        )
    elif stats["intermittency"] < 0.02:
        fails.append(f"intermittency {stats['intermittency']:.3f} < 0.02 -- probe is out of plume")
    cv = stats["blank_cv"]
    if not np.isfinite(cv):
        fails.append("insufficient complete blank durations to estimate CV")
    elif cv < 1.0:
        fails.append(
            f"blank-duration CV {cv:.2f} < 1.0 -- blanks are sub-exponential, "
            "plume has no large-scale meander and is too easy"
        )
    if stats["peak_to_mean"] < 3.0:
        fails.append(f"peak-to-mean {stats['peak_to_mean']:.1f} < 3 -- field is too smooth")
    return (len(fails) == 0), fails
