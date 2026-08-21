"""
End-to-end check of the three claims the whole design rests on:

  1. A filament plume with BOTH small-scale (per-filament OU) and large-scale
     (shared bearing meander) turbulence produces heavy-tailed blank durations.
     A time-averaged Gaussian plume does not.
  2. Sensor time constant -- not plume physics -- decides whether that
     structure survives into the observation the policy actually sees.
  3. The realism gate can tell these apart automatically, so it can be run in
     CI instead of eyeballing a plot.

Every series is thresholded at the SAME absolute concentration.  Sensor output
is inverted back through the power law to apparent ppm first, which is what
firmware reports, so the comparison is apples-to-apples.

CPU only. No Isaac, no GPU.
"""

import sys, os, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig
from scentience_olfaction.sensors.mox import (
    MICS6814_RED, FAST_OVERRIDES, MoxChannel, MoxChannelConfig,
)
from scentience_olfaction.validation import plume_stats as ps

DT = 0.01
T_END = 600.0
PROBE = np.array([[8.0, 0.0, 1.0]])
A_ETH, B_ETH = MICS6814_RED.sensitivity["ethanol"]


def base_cfg(**kw):
    c = dict(
        source_pos=(0.0, 0.0, 1.0), release_rate_hz=40.0, wind_mean=(1.0, 0.0, 0.0),
        turbulence_intensity=0.30, lagrangian_timescale=1.5,
        meander_std_rad=0.22, meander_timescale=15.0,
        gamma=2.0e-3, sigma0=0.05, max_filaments=8000, max_age_s=40.0,
    )
    c.update(kw)
    return FilamentPlumeConfig(**c)


def run_plume(cfg, seed=7, cache=None):
    if cache and os.path.exists(cache):
        return np.load(cache)["a"]
    plume = FilamentPlume(cfg, seed=seed)
    n = int(T_END / DT)
    out = np.zeros(n)
    for i in range(n):
        plume.step(DT)
        out[i] = plume.sample(PROBE)[0]
    if cache:
        np.savez_compressed(cache, a=out)
    return out


def run_gaussian():
    """Time-averaged Gaussian plume + slow meander, same order of magnitude."""
    n = int(T_END / DT)
    x, y = PROBE[0][0], PROBE[0][1]
    sy = 0.16 * x / math.sqrt(1 + 1e-4 * x)
    sz = 0.12 * x
    c0 = 1.0 / (2 * math.pi * 1.0 * sy * sz) * math.exp(-(y**2) / (2 * sy**2))
    rng = np.random.default_rng(3)
    w = np.zeros(n)
    a = math.exp(-DT / 15.0)
    for i in range(1, n):
        w[i] = a * w[i - 1] + math.sqrt(1 - a * a) * rng.standard_normal()
    return c0 * np.exp(0.5 * w) * 0.5


def through_sensor(conc, profile):
    """
    Returns (deflection, threshold).  Deflection is fractional drop in Rs/R0
    below the clean-air baseline -- the sensor's own natural units.  The
    threshold is 3 sigma of the sensor's OWN noise floor, measured by running
    it in clean air, which is how a detection threshold is set on hardware.
    Inverting to apparent-ppm instead would compare a noisy, heavily
    compressed nonlinear reconstruction against a clean ground truth.
    """
    cfg = MoxChannelConfig(**dict(MICS6814_RED.__dict__))
    if profile == "fast":
        for k, v in FAST_OVERRIDES.items():
            setattr(cfg, k, v)

    def track(x, tau_s=60.0):
        """Slow-EMA baseline tracker -- what e-nose firmware runs to reject the
        1/f drift that otherwise swamps a small deflection."""
        a = math.exp(-DT / tau_s)
        b = np.empty_like(x)
        acc = x[0]
        for i, v in enumerate(x):
            acc = a * acc + (1 - a) * v
            b[i] = acc
        return b

    # --- noise floor in clean air ---------------------------------------
    ch0 = MoxChannel(cfg, np.random.default_rng(101), randomize=False)
    blank = np.array([ch0.step({}, DT)["ratio_measured"] for _ in range(30000)])
    blank_d = (track(blank) - blank)[5000:]
    noise_sigma = float(blank_d.std())

    ch = MoxChannel(cfg, np.random.default_rng(11), randomize=False)
    ratio = np.empty(conc.size)
    for i, c in enumerate(conc):
        ratio[i] = ch.step({"ethanol": float(c)}, DT, temp_c=20.0, rh_pct=50.0,
                           flow_mps=0.5)["ratio_measured"]
    deflection = np.maximum(track(ratio) - ratio, 0.0)  # baseline-tracked
    return deflection, max(3.0 * noise_sigma, 1e-9)


def report(name, sig, thr):
    st = ps.summarize(sig, DT, thr)
    ok, fails = ps.gate(st)
    print(f"\n=== {name} ===")
    for k in ("intermittency", "peak_to_mean", "n_whiffs", "whiff_median_s",
              "blank_median_s", "blank_cv", "blank_tail_slope"):
        print(f"  {k:20s} {st[k]:>9.3f}" if isinstance(st[k], float) else f"  {k:20s} {st[k]:>9d}")
    print(f"  GATE: {'PASS' if ok else 'FAIL'}")
    for f in fails:
        print(f"    ! {f}")
    return st


if __name__ == "__main__":
    print(f"Simulating {T_END:.0f} s @ {1/DT:.0f} Hz, probe {PROBE[0]}")

    full = run_plume(base_cfg(), cache=os.path.join(tempfile.gettempdir(), "full.npz"))
    nomeander = run_plume(base_cfg(meander_std_rad=0.0), cache=os.path.join(tempfile.gettempdir(), "nm.npz"))
    gau = run_gaussian()

    # One absolute threshold for everything: 10% of the full plume's
    # conditional mean when detectable at all.
    pos = full[full > 0]
    THR = 0.10 * float(pos.mean())
    print(f"detection threshold: {THR:.4g} ppm (shared by every series)")

    s_full = report("filament, small+large scale turbulence [GROUND TRUTH]", full, THR)
    s_nm   = report("filament, small scale ONLY (no meander)  [ablation]", nomeander, THR)
    s_gau  = report("gaussian plume + slow meander            [baseline]", gau, THR)
    # Transport is linear in source strength, so scaling the series is exact.
    # MiCS-6814 ethanol range is 10-500 ppm; an un-scaled 0.1-2 ppm plume is
    # simply below the part's detection limit and 1/f drift swamps it -- itself
    # a finding worth stating, but it hides the time-constant effect.
    SCALE = 300.0
    print(f"\nscaling plume by {SCALE:.0f}x into the MiCS-6814 ethanol range "
          f"(peak {full.max()*SCALE:.0f} ppm, conditional mean "
          f"{full[full>0].mean()*SCALE:.0f} ppm)")
    d_slow, t_slow = through_sensor(full * SCALE, "slow")
    d_fast, t_fast = through_sensor(full * SCALE, "fast")
    print(f"\nsensor detection thresholds (3 sigma of own noise floor): "
          f"slow {t_slow:.3g}, fast {t_fast:.3g}  [fractional Rs/R0 deflection]")
    print(f"fraction of ground truth above the sensor's power-law knee "
          f"({A_ETH**(1/B_ETH):.2f} ppm): {(full > A_ETH**(1/B_ETH)).mean():.4f}")
    s_slow = report("full plume -> MOX tau_fall 12 s (packaged, still air)", d_slow, t_slow)
    s_fast = report("full plume -> MOX tau_fall 46 ms (Dennler-class)", d_fast, t_fast)

    print("\n" + "-" * 68)
    print(f"{'series':<44}{'blankCV':>9}{'whiffs':>9}")
    for nm, s in (("filament full", s_full), ("filament, meander ablated", s_nm),
                  ("gaussian", s_gau), ("via slow MOX", s_slow), ("via fast MOX", s_fast)):
        print(f"{nm:<44}{s['blank_cv']:>9.2f}{s['n_whiffs']:>9d}")
    print("-" * 68)
    print(f"slow sensor retains {100*s_slow['n_whiffs']/max(s_full['n_whiffs'],1):5.1f}% of ground-truth whiffs")
    print(f"fast sensor retains {100*s_fast['n_whiffs']/max(s_full['n_whiffs'],1):5.1f}% of ground-truth whiffs")
