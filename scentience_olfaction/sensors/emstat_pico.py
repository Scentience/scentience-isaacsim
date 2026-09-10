"""PalmSens EmStat Pico fixed-bias current readout with calibrated analyte cells.

Electronics limits come from PalmSens' EmStat Pico brochure Rev.11-2024-018.
Cell sensitivities, time constants and environmental coefficients must come
from the user's transducer calibration. No firmware, SDK, voltammetry or EIS
implementation is included. See docs/SENSOR_PROFILES.md for scope and sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from .co2_sensor import _nonnegative, _positive

__all__ = ["ElectrochemicalCellConfig", "EmStatPicoChannelConfig", "EmStatPicoConfig",
           "EmStatPicoChannel", "EMSTAT_CURRENT_RANGES_NA"]

EMSTAT_CURRENT_RANGES_NA = MappingProxyType({
    "low_speed": (100., 2000., 4000., 8000., 16000., 32000., 63000.,
                  125000., 250000., 500000., 1000000., 5000000.),
    "high_speed": (100., 1000., 6000., 13000., 25000., 50000., 100000.,
                   200000., 1000000., 5000000.),
    "max_range": (100., 1000., 6000., 13000., 25000., 50000., 100000.,
                  200000., 1000000., 5000000.),
})


@dataclass
class ElectrochemicalCellConfig:
    """User calibration at a fixed bias and electrode arrangement.

    Signed sensitivities include cross-sensitivity; concentrations are ppm.
    Defaults for other coefficients are idealized assumptions, not PalmSens
    cell specifications. ``calibration_id`` records the source/experiment.
    """

    sensitivity_na_per_ppm: dict[str, float]
    calibration_id: str
    tau_s: float = 1.0
    zero_current_na: float = 0.0
    reference_temp_c: float = 20.0
    zero_tempco_na_per_k: float = 0.0
    span_tempco_per_k: float = 0.0
    noise_na: float = 0.0
    drift_na_per_sqrt_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise ValueError("cell calibration_id must identify the calibration source")
        if not isinstance(self.sensitivity_na_per_ppm, dict) or not self.sensitivity_na_per_ppm:
            raise ValueError("cell sensitivity_na_per_ppm must be a nonempty mapping")
        self.sensitivity_na_per_ppm = dict(self.sensitivity_na_per_ppm)
        for species, sensitivity in self.sensitivity_na_per_ppm.items():
            if not isinstance(species, str) or not species:
                raise ValueError("calibration species names must be nonempty strings")
            _finite("cell sensitivity", sensitivity)
        for name in ("tau_s", "noise_na", "drift_na_per_sqrt_s"):
            _nonnegative(name, getattr(self, name))
        for name in ("zero_current_na", "reference_temp_c", "zero_tempco_na_per_k",
                     "span_tempco_per_k"):
            _finite(name, getattr(self, name))


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass
class EmStatPicoChannelConfig:
    """One potentiostat circuit attached to one separately calibrated cell.

    Two electrodes means RE and CE tied at the counter/reference electrode;
    three means separate WE/RE/CE. Electrode count and bias describe the
    calibration setup; changing them requires new cell calibration.
    """

    cell: ElectrochemicalCellConfig
    electrode_count: int = 3
    bias_v: float = 0.0
    mode: str = "low_speed"
    current_range_na: float = 100.0
    sample_interval_s: float = 0.1
    resolution_na: float | None = None
    max_current_na: float = 3_000_000.0
    gain_error_fraction: float = 0.0
    offset_na: float = 0.0
    noise_na: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.cell, ElectrochemicalCellConfig):
            raise TypeError("cell must be an ElectrochemicalCellConfig")
        self.cell = replace(self.cell)
        if type(self.electrode_count) is not int or self.electrode_count not in (2, 3):
            raise ValueError("electrode_count must be 2 or 3")
        if self.mode not in EMSTAT_CURRENT_RANGES_NA:
            raise ValueError(f"mode must be one of {tuple(EMSTAT_CURRENT_RANGES_NA)}")
        if self.current_range_na not in EMSTAT_CURRENT_RANGES_NA[self.mode]:
            raise ValueError(f"current_range_na is not supported in {self.mode} mode")
        _finite("bias_v", self.bias_v)
        lower = -1.2 if self.mode == "low_speed" else -1.7
        if not lower <= self.bias_v <= 2.0:
            raise ValueError(f"bias_v must be between {lower} and 2.0 V")
        _positive("sample_interval_s", self.sample_interval_s)
        max_rate = 1000 if self.mode == "high_speed" else 100
        if self.sample_interval_s < 1.0 / max_rate:
            raise ValueError(f"{self.mode} acquisition rate cannot exceed {max_rate} Hz")
        _positive("max_current_na", self.max_current_na)
        if self.max_current_na > 3_000_000.0:
            raise ValueError("EmStat Pico maximum current is 3 mA, including on the 5 mA range")
        if self.resolution_na is not None:
            _positive("resolution_na", self.resolution_na)
            if self.resolution_na > self.current_limit_na:
                raise ValueError("resolution_na must not exceed the current limit")
        _finite("gain_error_fraction", self.gain_error_fraction)
        if self.gain_error_fraction <= -1:
            raise ValueError("gain_error_fraction must be greater than -1")
        _finite("offset_na", self.offset_na)
        _nonnegative("noise_na", self.noise_na)

    @property
    def current_limit_na(self) -> float:
        return min(self.current_range_na, self.max_current_na)

    @property
    def quantum_na(self) -> float:
        if self.resolution_na is not None:
            return self.resolution_na
        # PalmSens states 0.006% of range and separately 5.5 pA at 100 nA.
        return 0.0055 if self.current_range_na == 100.0 else 0.00006 * self.current_range_na

    def accuracy_bound_na(self, current_na: float) -> float:
        """Electronics envelope only; does not include cell calibration error."""
        _finite("current_na", current_na)
        fraction = 0.005 if self.mode == "low_speed" else 0.01
        return fraction * abs(current_na) + 0.001 * self.current_range_na


@dataclass
class EmStatPicoConfig:
    """One or two independent cells; simultaneous dual readout is low-speed only."""

    channels: tuple[EmStatPicoChannelConfig, ...]

    def __post_init__(self) -> None:
        if len(self.channels) not in (1, 2):
            raise ValueError("EmStat Pico requires one or two independent cells")
        if any(not isinstance(ch, EmStatPicoChannelConfig) for ch in self.channels):
            raise TypeError("channels must contain EmStatPicoChannelConfig objects")
        self.channels = tuple(replace(ch) for ch in self.channels)
        if len(self.channels) == 2 and any(ch.mode != "low_speed" for ch in self.channels):
            raise ValueError("simultaneous independent cells require low_speed on both channels")


class EmStatPicoChannel:
    """Single-cell response followed by sample/hold, electronics errors and ADC.

    ``current_na`` and saturation diagnostics are held until a new sample.
    ``signal_na`` is the instantaneous lagged analyte contribution. Reset
    clears dynamics and hold state without rewinding the caller's RNG.
    """

    def __init__(self, cfg: EmStatPicoChannelConfig, rng: np.random.Generator):
        self.cfg = replace(cfg)
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        self._signal_na = 0.0
        self._drift_na = 0.0
        self._t_since_sample = 0.0
        self._held = {"current_na": 0.0, "unclipped_current_na": 0.0,
                      "saturated": False, "counts": 0, "sampled": False}

    def step(self, conc_ppm: dict[str, float], dt: float, temp_c: float = 20.0) -> dict:
        _nonnegative("dt", dt)
        _finite("temp_c", temp_c)
        for value in conc_ppm.values():
            _nonnegative("concentration", value)
        cfg, cell = self.cfg, self.cfg.cell
        delta_temp = temp_c - cell.reference_temp_c
        span = 1.0 + cell.span_tempco_per_k * delta_temp
        if span <= 0:
            raise ValueError("temperature lies outside positive cell span calibration")
        target = span * sum(sensitivity * conc_ppm.get(gas, 0.0)
                            for gas, sensitivity in cell.sensitivity_na_per_ppm.items())
        _finite("calibrated target current", target)
        remaining = dt
        while remaining > 0:
            segment = min(remaining, cfg.sample_interval_s - self._t_since_sample)
            alpha = -math.expm1(-segment / cell.tau_s) if cell.tau_s else 1.0
            self._signal_na += alpha * (target - self._signal_na)
            self._t_since_sample += segment
            remaining = max(0.0, remaining - segment)
            if self._t_since_sample >= cfg.sample_interval_s * (1 - 1e-12):
                self._t_since_sample = 0.0
                # Drift is evolved on the acquisition clock to make seeded
                # output independent of the caller's subdivision of dt.
                self._drift_na += (cell.drift_na_per_sqrt_s * math.sqrt(cfg.sample_interval_s)
                                   * self.rng.standard_normal())
                current = (self._signal_na + cell.zero_current_na
                           + cell.zero_tempco_na_per_k * delta_temp + self._drift_na
                           + cell.noise_na * self.rng.standard_normal())
                current = (current * (1.0 + cfg.gain_error_fraction) + cfg.offset_na
                           + cfg.noise_na * self.rng.standard_normal())
                _finite("measured current", current)
                q, limit = cfg.quantum_na, cfg.current_limit_na
                max_count = math.floor(limit / q)
                count = max(-max_count, min(max_count, round(current / q)))
                self._held = {"current_na": count * q, "unclipped_current_na": current,
                              "saturated": abs(current) > limit, "counts": count,
                              "sampled": True}
        return {**self._held, "signal_na": self._signal_na}
