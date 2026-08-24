"""Sensor-based task termination: when may a robot declare "source found"?

Implements the termination criterion of Chasing Ghosts (France et al.,
arXiv:2602.19577, Sec. III-H, Eqs. 9-11). The env's `success_radius` uses
ground-truth distance to the source -- fine for benchmarking, impossible on a
real robot, which knows only what it has smelled. This estimator gives the
robot an on-board stopping rule built purely from its own concentration
samples.

Treat the observed samples as draws from the plume's concentration
distribution, bounded above by the unknown true maximum C_max (found at the
source). With m the largest observation over k samples:

  point estimate (Eq. 9, the continuous 'German tank' MVUE):
      C_hat = m * (k + 1) / k

  confidence interval (Eq. 10), from the quantiles of the sample maximum
  (P[max <= x] = (x / C_max)^k for uniform order statistics):
      [ m / q^(1/k),  m / p^(1/k) ]      e.g. p, q = 0.025, 0.975 for 95%

The paper's worked example (Eq. 11): k=20 -> C_hat = 1.05 m and a 95% CI of
about [m, 1.2 m]. Declaration rule: stop once the CI upper bound is within
`margin` (paper: 25%) of the observed maximum -- i.e., with the chosen
confidence, whatever remains unfound is at most `margin` stronger than what
the robot has already smelled.

The uniform-order-statistic assumption is a modelling choice (samples are
neither independent nor uniform in a real plume); the paper uses it as a
practical bound and so do we. Evidence: the RULE is from the paper; its
calibration in this simulator is untested -- treat thresholds as starting
points.

Be aware of one property before tuning: under this model the RATIO of the CI
upper bound to m is p^(-1/k), a function of k alone. So for fixed confidence
and margin the rule is equivalent to "count at least k* qualifying samples"
(k* = ln(1/p)/ln(1+margin), ~17 for 95%/25%). The data enters through which
samples QUALIFY -- `min_signal` should be a real odor-contact threshold (e.g.
the bout-detector's), not zero, or the robot can 'declare' on k ticks of
clean air.
"""

from __future__ import annotations


class SourceDeclaration:
    """Track (m, k) over observed concentrations and expose the stopping rule.

    Feed it the same scalar the policy navigates on (e.g. max MOX deflection);
    it must NEVER see ground truth. `min_samples` guards the small-k regime
    where the m^(1/k) bound is vacuously wide.
    """

    def __init__(self, confidence: float = 0.95, margin: float = 0.25,
                 min_samples: int = 20, min_signal: float = 0.0,
                 sample_period_s: float = 1.0, plateau_samples: int = 5):
        assert 0.0 < confidence < 1.0 and margin > 0.0
        self.confidence = confidence
        self.margin = margin
        self.min_samples = min_samples
        self.min_signal = min_signal
        self.sample_period_s = sample_period_s
        """The paper samples the MOX pair at 1 Hz in flight (Sec. III-E1);
        feeding every 20 Hz control tick would count heavily-correlated reads
        as independent samples and reach k* in one second of noise. observe()
        decimates to this period when called with dt."""
        self.plateau_samples = plateau_samples
        """'Found the HIGHEST concentration' (Sec. III-H) means m has stopped
        improving, not that m merely exists: require this many qualifying
        samples since the last new maximum before declaring."""
        self.reset()

    def reset(self) -> None:
        self.m = 0.0        # largest concentration observed
        self.k = 0          # number of qualifying samples observed
        self.k_at_max = 0   # k when m last improved
        self.last = 0.0     # most recent qualifying concentration
        self._acc = 0.0     # decimation accumulator

    def observe(self, concentration: float, dt: float | None = None) -> None:
        """Feed the navigation signal. Pass your control `dt` and readings are
        decimated to `sample_period_s`; omit it to count every call (only
        correct if you already call at ~1 Hz)."""
        if dt is not None:
            self._acc += dt
            if self._acc < self.sample_period_s:
                return
            self._acc = 0.0
        if concentration > self.min_signal:
            self.k += 1
            self.last = concentration
            if concentration > self.m:
                self.m = concentration
                self.k_at_max = self.k

    # ------------------------------------------------------------- estimates
    def point_estimate(self) -> float:
        """Eq. 9: MVUE of the true maximum concentration."""
        if self.k == 0:
            return 0.0
        return self.m * (self.k + 1) / self.k

    def interval(self) -> tuple[float, float]:
        """Eq. 10: CI for the true maximum, from the sample-max quantiles."""
        if self.k == 0:
            return (0.0, float("inf"))
        p = (1.0 - self.confidence) / 2.0        # e.g. 0.025
        q = 1.0 - p                              # e.g. 0.975
        return (self.m / q ** (1.0 / self.k), self.m / p ** (1.0 / self.k))

    def declared(self) -> bool:
        """True when the robot may stop. All of (Sec. III-H, operationalised):

        1. enough qualifying samples (k >= min_samples at ~1 Hz);
        2. the CI upper bound says the unseen maximum is within `margin` of
           the observed one (Eq. 11 and the paper's 25%-of-max rule);
        3. m has PLATEAUED -- no new maximum for `plateau_samples` samples
           (still climbing means the source is still ahead);
        4. the robot is AT the maximum now: the latest qualifying reading is
           itself within `margin` of m. Remembering a strong whiff from 30
           metres ago is not finding the source.
        """
        if self.k < self.min_samples or self.m <= self.min_signal:
            return False
        if self.k - self.k_at_max < self.plateau_samples:
            return False
        if self.last < self.m / (1.0 + self.margin):
            return False
        _, hi = self.interval()
        return hi <= (1.0 + self.margin) * self.m
