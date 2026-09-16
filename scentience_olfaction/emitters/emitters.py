"""
Chemical emitters.

An emitter answers one question per step: how many filaments of which species
to release, and where. Emission strength is expressed the way the filament
model needs it -- a release rate [filaments/s] plus the initial centre
concentration and radius of each filament -- because that triple, not a
mass-flux scalar, is what fixes the moles carried per filament:

    N_fil = (ppm_center/1e6) * n_air * (2*pi)^{3/2} * sigma0^3     [mol]

so mass flux Q [mol/s] = release_rate_hz * N_fil.  `mass_flux_mol_s()` reports
it for anyone who needs the physical number.

All emitters are seedable and deterministic under seed. A moving emitter is
supported by mutating `position` between steps (the Isaac adapter binds it to
a prim's world pose).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def validate_dt(dt: float) -> None:
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")


@dataclass
class PointEmitter:
    position: tuple[float, float, float]
    species: str = "ethanol"
    release_rate_hz: float = 20.0
    ppm_center_initial: float = 20.0
    sigma0: float = 0.10
    t_start: float = 0.0
    t_stop: float = math.inf
    # Pulsed release: on for `pulse_on_s`, off for `pulse_off_s`, repeating.
    pulse_on_s: float = math.inf
    pulse_off_s: float = 0.0
    # Multiplicative stochastic modulation of the rate (lognormal, OU-driven);
    # 0 disables. Models a flickering/turbulent source, e.g. evaporation gusts.
    rate_modulation_std: float = 0.0
    rate_modulation_tau_s: float = 5.0
    # Optional physical source strength. Overrides ppm_center_initial when set.
    # release_rate_hz still controls packet resolution: N_fil = Q / rate.
    molar_rate_mol_s: float | None = None

    _accum: float = field(default=0.0, repr=False)
    _mod_state: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if np.asarray(self.position).shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError("position must be a finite 3-vector")
        for name in ("release_rate_hz", "ppm_center_initial", "rate_modulation_std"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("sigma0", "rate_modulation_tau_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.t_start) or math.isnan(self.t_stop) or self.t_stop < self.t_start:
            raise ValueError("require finite t_start <= t_stop (t_stop may be +inf)")
        if (math.isnan(self.pulse_on_s) or self.pulse_on_s <= 0 or
                not math.isfinite(self.pulse_off_s) or self.pulse_off_s < 0):
            raise ValueError("pulse_on_s must be positive; pulse_off_s finite and nonnegative")
        if self.molar_rate_mol_s is not None:
            if not math.isfinite(self.molar_rate_mol_s) or self.molar_rate_mol_s < 0:
                raise ValueError("molar_rate_mol_s must be finite and nonnegative")
            if self.molar_rate_mol_s > 0 and self.release_rate_hz <= 0:
                raise ValueError("a positive molar rate requires positive release_rate_hz")

    def active_duration(self, t: float, dt: float) -> float:
        """Exact overlap of [t, t+dt) with the source window and pulse train."""
        lo, hi = max(t, self.t_start), min(t + dt, self.t_stop)
        if hi <= lo:
            return 0.0
        if math.isinf(self.pulse_on_s) or self.pulse_off_s == 0:
            # Keep the original rate*dt arithmetic for wholly active steps.
            return dt if lo == t and hi == t + dt else hi - lo
        period = self.pulse_on_s + self.pulse_off_s

        def integral(x):
            cycles, phase = divmod(x - self.t_start, period)
            return cycles * self.pulse_on_s + min(phase, self.pulse_on_s)

        return min(dt, max(0.0, integral(hi) - integral(lo)))

    def active(self, t: float) -> bool:
        if not (self.t_start <= t < self.t_stop):
            return False
        if math.isinf(self.pulse_on_s):
            return True
        period = self.pulse_on_s + self.pulse_off_s
        return ((t - self.t_start) % period) < self.pulse_on_s

    def n_release(self, t: float, dt: float, rng: np.random.Generator) -> int:
        """Number of filaments to release this step. Fractional-rate exact via
        an accumulator, so release_rate_hz=2.5 at dt=0.1 releases 0.25/step on
        average with no long-run bias."""
        validate_dt(dt)
        if not math.isfinite(t) or not math.isfinite(t + dt):
            raise ValueError("emitter time must be finite")
        duration = self.active_duration(t, dt)
        if duration == 0:
            return 0
        rate = self.release_rate_hz
        if self.rate_modulation_std > 0.0:
            a = math.exp(-dt / self.rate_modulation_tau_s)  # exact OU update
            self._mod_state = a * self._mod_state + self.rate_modulation_std * math.sqrt(
                max(1.0 - a * a, 0.0)) * rng.standard_normal()
            rate *= math.exp(self._mod_state - 0.5 * self.rate_modulation_std**2)
        self._accum += rate * duration
        # Do not delay a whole packet because 10 * 0.1 rounded just below 1.
        # Window/pulse overlap also subtracts absolute timestamps: include that
        # rounding scale, especially when the source has been running a while.
        tol = 8 * (math.ulp(max(1.0, self._accum)) +
                   rate * math.ulp(max(abs(t), abs(t + dt))))
        n = math.floor(self._accum + tol)
        self._accum = max(0.0, self._accum - n)
        return n

    def sample_positions(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.tile(np.asarray(self.position, np.float64), (n, 1))

    def mass_flux_mol_s(self, n_air_mol_m3: float) -> float:
        """Nominal rate while ON; modulation and duty cycle are not averaged in."""
        if self.molar_rate_mol_s is not None:
            return self.molar_rate_mol_s
        n_fil = (self.ppm_center_initial / 1e6) * n_air_mol_m3 * \
            (2.0 * math.pi) ** 1.5 * self.sigma0 ** 3
        return self.release_rate_hz * n_fil

    def moles_per_filament(self, n_air_mol_m3: float) -> float:
        if self.molar_rate_mol_s is not None:
            return self.molar_rate_mol_s / self.release_rate_hz if self.release_rate_hz else 0.0
        return (self.ppm_center_initial / 1e6) * n_air_mol_m3 * (2 * math.pi)**1.5 * self.sigma0**3

    def reset(self) -> None:
        self._accum = 0.0
        self._mod_state = 0.0


@dataclass
class LineEmitter(PointEmitter):
    """Release along a segment [position, end] -- a leaking pipe or a doorway."""
    end: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        super().validate()
        if np.asarray(self.end).shape != (3,) or not np.isfinite(self.end).all():
            raise ValueError("end must be a finite 3-vector")

    def sample_positions(self, n: int, rng: np.random.Generator) -> np.ndarray:
        a = np.asarray(self.position, np.float64)
        b = np.asarray(self.end, np.float64)
        u = rng.random((n, 1))
        return a[None, :] + u * (b - a)[None, :]


@dataclass
class BoxEmitter(PointEmitter):
    """Uniform release inside an axis-aligned box -- an evaporating surface or
    a diffuse area source. `position` is the box minimum corner."""
    size: tuple[float, float, float] = (0.1, 0.1, 0.1)

    def validate(self) -> None:
        super().validate()
        if (np.asarray(self.size).shape != (3,) or not np.isfinite(self.size).all() or
                np.any(np.asarray(self.size) < 0)):
            raise ValueError("size must be a finite nonnegative 3-vector")

    def sample_positions(self, n: int, rng: np.random.Generator) -> np.ndarray:
        lo = np.asarray(self.position, np.float64)
        return lo[None, :] + rng.random((n, 3)) * np.asarray(self.size, np.float64)[None, :]
