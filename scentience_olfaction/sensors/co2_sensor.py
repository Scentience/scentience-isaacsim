"""Optional Sensirion CO2 profiles and the legacy CO2 channel interface.

SCD30 uses conventional NDIR; SCD40/41 use photoacoustic NDIR (PASens).
Manufacturer specifications and simulation assumptions are separated in
docs/SENSOR_PROFILES.md. This is a response model, not sensor firmware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np


__all__ = ["CO2Profile", "SCD30", "SCD40", "SCD41", "CO2_PROFILES",
           "CO2Config", "co2_config", "CO2Channel"]


@dataclass(frozen=True)
class CO2Profile:
    """Datasheet defaults; accuracy bands are (upper ppm, base ppm, fraction)."""

    name: str
    technology: str
    tau63_s: float
    sample_interval_s: float
    accuracy_bands: tuple[tuple[float, float, float], ...]
    accuracy_min_ppm: float = 400.0
    repeatability_ppm: float = 10.0
    output_range_ppm: tuple[float, float] = (0.0, 40000.0)
    resolution_ppm: float = 1.0
    low_power_interval_s: float | None = None
    supports_single_shot: bool = False


# SCD30 datasheet v1.0, May 2020, Table 1; SCD4x v1.7, April 2025,
# Tables 1/4. URLs and conditions are in docs/SENSOR_PROFILES.md.
SCD30 = CO2Profile("scd30", "NDIR", 20.0, 2.0,
                   ((10000.0, 30.0, 0.03),), resolution_ppm=0.0)
SCD40 = CO2Profile("scd40", "photoacoustic NDIR", 60.0, 5.0,
                   ((2000.0, 50.0, 0.05),), low_power_interval_s=30.0)
SCD41 = CO2Profile("scd41", "photoacoustic NDIR", 60.0, 5.0,
                   ((1000.0, 50.0, 0.025), (2000.0, 50.0, 0.03),
                    (5000.0, 40.0, 0.05)),
                   low_power_interval_s=30.0, supports_single_shot=True)
CO2_PROFILES = MappingProxyType({p.name: p for p in (SCD30, SCD40, SCD41)})


@dataclass
class CO2Config:
    """Legacy defaults remain excess-above-ambient ppm with ASC enabled.

Use :func:`co2_config` for manufacturer defaults. Accuracy is an envelope,
not a Gaussian sigma; ``accuracy_bias_fraction`` applies a chosen signed
fraction of that envelope. Repeatability is separately modeled as white noise.
"""

    tau63_s: float = 60.0
    sample_interval_s: float = 5.0
    accuracy_base_ppm: float = 50.0     # +/-(50 ppm + 2.5 %) band, 400-1000 ppm
    accuracy_frac: float = 0.025
    repeatability_ppm: float = 10.0
    ambient_baseline_ppm: float = 420.0
    asc_enabled: bool = True
    asc_window_s: float = 7 * 24 * 3600.0
    asc_gain: float = 0.2               # fraction of (400 - min) applied per window
    concentration_mode: str = "excess"
    accuracy_bands: tuple[tuple[float, float, float], ...] = ()
    accuracy_range_ppm: tuple[float, float] = (400.0, 1000.0)
    accuracy_bias_fraction: float = 0.0
    output_range_ppm: tuple[float, float] | None = None
    resolution_ppm: float = 0.0
    asc_target_ppm: float = 400.0

    def __post_init__(self) -> None:
        for name in ("tau63_s", "sample_interval_s", "asc_window_s"):
            _positive(name, getattr(self, name))
        for name in ("accuracy_base_ppm", "accuracy_frac", "repeatability_ppm",
                     "ambient_baseline_ppm", "resolution_ppm", "asc_target_ppm"):
            _nonnegative(name, getattr(self, name))
        if not isinstance(self.asc_enabled, bool):
            raise ValueError("asc_enabled must be a bool")
        if self.concentration_mode not in ("absolute", "excess"):
            raise ValueError("concentration_mode must be 'absolute' or 'excess'")
        if not math.isfinite(self.asc_gain) or not 0 <= self.asc_gain <= 1:
            raise ValueError("asc_gain must be between 0 and 1")
        if (not math.isfinite(self.accuracy_bias_fraction)
                or not -1 <= self.accuracy_bias_fraction <= 1):
            raise ValueError("accuracy_bias_fraction must be between -1 and 1")
        self.accuracy_range_ppm = _range("accuracy_range_ppm", self.accuracy_range_ppm)
        if self.output_range_ppm is not None:
            self.output_range_ppm = _range("output_range_ppm", self.output_range_ppm)
            if not self.output_range_ppm[0] <= self.ambient_baseline_ppm <= self.output_range_ppm[1]:
                raise ValueError("ambient_baseline_ppm must be inside output_range_ppm")
        self.accuracy_bands = tuple(tuple(b) for b in self.accuracy_bands)
        previous = self.accuracy_range_ppm[0]
        for band in self.accuracy_bands:
            if len(band) != 3:
                raise ValueError("accuracy bands must contain (upper_ppm, base_ppm, fraction)")
            upper, base, fraction = band
            if not math.isfinite(upper) or upper <= previous:
                raise ValueError("accuracy band upper limits must increase")
            _nonnegative("accuracy band base", base)
            _nonnegative("accuracy band fraction", fraction)
            previous = upper
        if self.accuracy_bands and previous != self.accuracy_range_ppm[1]:
            raise ValueError("accuracy bands must end at accuracy_range_ppm maximum")

    def accuracy_bound_ppm(self, reading_ppm: float) -> float | None:
        """Specified ± envelope, or None outside the specified accuracy range."""
        if not math.isfinite(reading_ppm):
            raise ValueError("reading_ppm must be finite")
        if not self.accuracy_range_ppm[0] <= reading_ppm <= self.accuracy_range_ppm[1]:
            return None
        for upper, base, fraction in self.accuracy_bands:
            if reading_ppm <= upper:
                return base + fraction * reading_ppm
        return self.accuracy_base_ppm + self.accuracy_frac * reading_ppm


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _range(name: str, value) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two limits")
    lo, hi = value
    _nonnegative(name, lo)
    _positive(name, hi)
    if lo >= hi:
        raise ValueError(f"{name} limits must increase")
    return (lo, hi)


def co2_config(profile: str | CO2Profile = SCD41, *,
               low_power: bool = False, **overrides) -> CO2Config:
    """Fresh tunable manufacturer config (absolute ppm, simplified ASC off).

    ``low_power=True`` selects SCD40/41's 30 s cadence. Other cadence/lag
    overrides are simulation parameters, not claims of hardware support.
    Overriding base/fraction replaces the piecewise band with a single band.
    """
    if isinstance(profile, str):
        try:
            profile = CO2_PROFILES[profile.lower()]
        except KeyError as exc:
            raise ValueError(f"unknown CO2 profile {profile!r}") from exc
    if not isinstance(profile, CO2Profile):
        raise TypeError("profile must be a CO2Profile or profile name")
    if not isinstance(low_power, bool):
        raise ValueError("low_power must be a bool")
    if low_power and profile.low_power_interval_s is None:
        raise ValueError(f"{profile.name} has no low-power periodic profile")
    _, base, fraction = profile.accuracy_bands[0]
    values = dict(tau63_s=profile.tau63_s,
                  sample_interval_s=(profile.low_power_interval_s if low_power
                                     else profile.sample_interval_s),
                  accuracy_base_ppm=base, accuracy_frac=fraction,
                  repeatability_ppm=profile.repeatability_ppm,
                  accuracy_bands=profile.accuracy_bands,
                  accuracy_range_ppm=(profile.accuracy_min_ppm, profile.accuracy_bands[-1][0]),
                  output_range_ppm=profile.output_range_ppm,
                  resolution_ppm=profile.resolution_ppm,
                  concentration_mode="absolute", asc_enabled=False)
    if "accuracy_base_ppm" in overrides or "accuracy_frac" in overrides:
        values["accuracy_bands"] = ()
    values.update(overrides)
    return CO2Config(**values)


class CO2Channel:
    def __init__(self, cfg: CO2Config, rng: np.random.Generator):
        self.cfg = replace(cfg)  # validate and keep configuration local to this unit
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
        """Advance with constant input over dt; return held absolute ppm.

        The parameter name is retained for keyword compatibility. Its units
        follow cfg.concentration_mode (legacy: excess, profiles: absolute).
        """
        cfg = self.cfg
        if not math.isfinite(co2_excess_ppm):
            raise ValueError("CO2 concentration must be finite")
        if cfg.concentration_mode == "excess":
            true_ppm = cfg.ambient_baseline_ppm + max(co2_excess_ppm, 0.0)
        else:
            _nonnegative("absolute CO2 concentration", co2_excess_ppm)
            true_ppm = co2_excess_ppm
        return self._step_absolute(true_ppm, dt)

    def step_absolute(self, co2_ppm: float, dt: float) -> dict:
        """Explicit absolute-ppm entry point independent of configured input mode."""
        _nonnegative("co2_ppm", co2_ppm)
        return self._step_absolute(co2_ppm, dt)

    def _step_absolute(self, true_ppm: float, dt: float) -> dict:
        _nonnegative("dt", dt)
        cfg = self.cfg
        # Split at sampling and ASC boundaries. This preserves residual time
        # and samples the lagged value at the actual boundary even for large dt.
        remaining = dt
        tolerance = 1e-12 * min(cfg.sample_interval_s, cfg.asc_window_s)
        while remaining > 0:
            segment = min(remaining, cfg.sample_interval_s - self._t_since_sample)
            if cfg.asc_enabled:
                segment = min(segment, cfg.asc_window_s - self._t_window)
            self._y += -math.expm1(-segment / cfg.tau63_s) * (true_ppm - self._y)
            self._t_since_sample += segment
            remaining = max(0.0, remaining - segment)
            if cfg.asc_enabled:
                self._window_min = min(self._window_min, self._y)
                self._t_window += segment
                if self._t_window >= cfg.asc_window_s - tolerance:
                    # Deliberately a simple drift experiment, not Sensirion ASC.
                    self._asc_offset += cfg.asc_gain * (cfg.asc_target_ppm - self._window_min)
                    self._window_min = math.inf
                    self._t_window = 0.0
            if self._t_since_sample >= cfg.sample_interval_s - tolerance:
                self._t_since_sample = 0.0
                envelope = cfg.accuracy_bound_ppm(self._y)
                bias = cfg.accuracy_bias_fraction * (envelope or 0.0)
                noise = cfg.repeatability_ppm * self.rng.standard_normal()
                reading = self._y + self._asc_offset + bias + noise
                if cfg.resolution_ppm:
                    reading = round(reading / cfg.resolution_ppm) * cfg.resolution_ppm
                if cfg.output_range_ppm is not None:
                    reading = float(np.clip(reading, *cfg.output_range_ppm))
                self._held = reading
        return {"co2_ppm": self._held,
                "asc_offset_ppm": self._asc_offset}
