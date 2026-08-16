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

    _accum: float = field(default=0.0, repr=False)
    _mod_state: float = field(default=0.0, repr=False)

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
        if not self.active(t):
            return 0
        rate = self.release_rate_hz
        if self.rate_modulation_std > 0.0:
            a = math.exp(-dt / self.rate_modulation_tau_s)  # exact OU update
            self._mod_state = a * self._mod_state + self.rate_modulation_std * math.sqrt(
                max(1.0 - a * a, 0.0)) * rng.standard_normal()
            rate *= math.exp(self._mod_state - 0.5 * self.rate_modulation_std**2)
        self._accum += rate * dt
        n = int(self._accum)
        self._accum -= n
        return n

    def sample_positions(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.tile(np.asarray(self.position, np.float64), (n, 1))

    def mass_flux_mol_s(self, n_air_mol_m3: float) -> float:
        n_fil = (self.ppm_center_initial / 1e6) * n_air_mol_m3 * \
            (2.0 * math.pi) ** 1.5 * self.sigma0 ** 3
        return self.release_rate_hz * n_fil

    def reset(self) -> None:
        self._accum = 0.0
        self._mod_state = 0.0


@dataclass
class LineEmitter(PointEmitter):
    """Release along a segment [position, end] -- a leaking pipe or a doorway."""
    end: tuple[float, float, float] = (0.0, 0.0, 0.0)

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

    def sample_positions(self, n: int, rng: np.random.Generator) -> np.ndarray:
        lo = np.asarray(self.position, np.float64)
        return lo[None, :] + rng.random((n, 3)) * np.asarray(self.size, np.float64)[None, :]
