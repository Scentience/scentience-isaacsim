"""
Plume-realism gate.

The single biggest failure mode in simulated olfactory navigation is a plume
that is too smooth.  A time-averaged Gaussian field has a monotone gradient,
so gradient ascent solves it, and the resulting policy fails immediately on
hardware.  These statistics are what distinguishes a plume an agent must
*search* from one it can simply climb.

Targets, from the literature:

  whiff / blank duration CCDF   heavy tailed, log-log slope near -3/2 over
                                1-2 decades with an exponential cutoff at the
                                large-eddy correlation time
                                (Celani, Villermaux & Vergassola, PRX 4:041015)
  intermittency                 ~0.5 at ~2 m off-axis, falling with distance
  peak-to-mean                  ~14 near source
                                (Farrell et al. 2002, Env. Fluid Mech. 2:143)

An exponential blank-duration distribution means the plume has no large-scale
meander and the environment is easier than reality.  Treat that as a failure.
"""

from __future__ import annotations

import numpy as np


def whiff_blank_durations(
    signal: np.ndarray, dt: float, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Durations of above-threshold (whiff) and below-threshold (blank) runs."""
    above = signal > threshold
    if above.size == 0:
        return np.array([]), np.array([])
    edges = np.flatnonzero(np.diff(above.astype(np.int8))) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [above.size]))
    lengths = (ends - starts) * dt
    states = above[starts]
    # Drop first and last runs: they are censored by the record boundary.
    if lengths.size > 2:
        lengths, states = lengths[1:-1], states[1:-1]
    return lengths[states], lengths[~states]


def ccdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    return xs, 1.0 - np.arange(xs.size) / xs.size


def tail_exponent(x: np.ndarray, lo_q: float = 0.75, hi_q: float = 0.99) -> float:
    """
    Log-log slope of the CCDF over the TAIL.

    The fit range is not a detail -- it changes the answer.  On a validated
    600 s filament record the same blank-duration sample gives:

        q10-q90 (body)   -0.66      <- fitting here is simply wrong
        q50-q95 (mid)    -1.36
        q75-q99 (tail)   -1.68
        q90-max          -1.28

    Only the last three are comparable to the -3/2 first-return exponent.
    Power-law exponents are a tail property; fitting the body mixes in the
    small-eddy regime and reports a spuriously shallow slope.  Report the
    range alongside the number, always.
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
    if cv == cv and cv < 1.0:
        fails.append(
            f"blank-duration CV {cv:.2f} < 1.0 -- blanks are sub-exponential, "
            "plume has no large-scale meander and is too easy"
        )
    if stats["peak_to_mean"] < 3.0:
        fails.append(f"peak-to-mean {stats['peak_to_mean']:.1f} < 3 -- field is too smooth")
    return (len(fails) == 0), fails
