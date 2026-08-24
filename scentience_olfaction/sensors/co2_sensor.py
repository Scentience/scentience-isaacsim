"""
Sensirion SCD4x-class CO2 sensor model.

Two facts drive the whole model, both DATASHEET evidence (SCD4x datasheet
v1.5) and both routinely gotten wrong:

  * It is PHOTOACOUSTIC, not NDIR -- that is how it fits in 10x10x6.5 mm.
  * tau63 = 60 s and the sample interval is 5 s: t90 ~ 138 s.  It cannot
    resolve a plume whiff; in an olfactory stack it is slow environmental
    context, not an olfactory channel.  France & Daescu's OIO calibration
    tables (arXiv:2506.04539, IEEE INERTIAL 2025) put Sensirion NDIR /
    photoacoustic sampling windows at 0.1-1.0 s with 30-50 ppm error --
    consistent readings, but the 60 s diffusion lag still gates the dynamics.

Also modelled: ASC (automatic self-calibration). The device assumes it sees
400 ppm at least weekly and pulls its rolling minimum to 400. In a
continuously occupied or sealed space this DRAGS THE WHOLE CALIBRATION DOWN --
a real deployment failure worth reproducing in sim, and off by default on the
real part only if you disable it explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CO2Config:
    tau63_s: float = 60.0
    sample_interval_s: float = 5.0
    accuracy_base_ppm: float = 50.0     # +/-(50 ppm + 2.5 %) band, 400-1000 ppm
    accuracy_frac: float = 0.025
    repeatability_ppm: float = 10.0
    ambient_baseline_ppm: float = 420.0
    asc_enabled: bool = True
    asc_window_s: float = 7 * 24 * 3600.0
    asc_gain: float = 0.2               # fraction of (400 - min) applied per window


class CO2Channel:
    def __init__(self, cfg: CO2Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        self._y = self.cfg.ambient_baseline_ppm
        self._held = self.cfg.ambient_baseline_ppm
        self._t_since_sample = 0.0
        self._asc_offset = 0.0
        self._window_min = math.inf
        self._t_window = 0.0

    def step(self, co2_excess_ppm: float, dt: float) -> dict:
        """co2_excess_ppm: plume CO2 above ambient at the sensor position."""
        cfg = self.cfg
        true_ppm = cfg.ambient_baseline_ppm + max(co2_excess_ppm, 0.0)

        # diffusion-membrane lag (first order, exact update)
        alpha = 1.0 - math.exp(-dt / cfg.tau63_s)
        self._y += alpha * (true_ppm - self._y)

        # ASC bookkeeping
        self._window_min = min(self._window_min, self._y)
        self._t_window += dt
        if cfg.asc_enabled and self._t_window >= cfg.asc_window_s:
            self._asc_offset += cfg.asc_gain * (400.0 - self._window_min)
            self._window_min = math.inf
            self._t_window = 0.0

        # zero-order hold at the 5 s sample interval
        self._t_since_sample += dt
        if self._t_since_sample >= cfg.sample_interval_s:
            self._t_since_sample = 0.0
            noise = (cfg.repeatability_ppm * self.rng.standard_normal())
            self._held = self._y + self._asc_offset + noise
        return {"co2_ppm": self._held,
                "asc_offset_ppm": self._asc_offset}
