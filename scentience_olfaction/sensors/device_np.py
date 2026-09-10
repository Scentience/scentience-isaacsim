"""
Scentience V1 device -- single-instance NumPy implementation.

This is the device model for standalone Python, the Gymnasium environment,
and single-robot use: one virtual unit, stepped scalar-wise, no torch, no
Isaac.  The vectorised torch twin (sensors/scentience_v1.py) is for Isaac Lab
RL at scale; tests/test_device_parity.py holds the two to the same step
response when torch is available.

Channel schema follows the Scentience BLE/Sockets ordering. Matching a schema
does not establish hardware accuracy or transferability of a learned policy.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from .electrochemical import EC_CO, EC_H2S, ECChannel, ECChannelConfig
from .mox import (FAST_OVERRIDES, MOX_NH3, MOX_OX, MOX_RED,
                  MoxChannel, MoxChannelConfig, finite_value,
                  validate_concentrations, validate_environment)
from .co2_sensor import CO2Channel, CO2Config

CHANNELS = (
    "chem_left_red", "chem_left_nh3", "chem_left_ox",
    "chem_right_red", "chem_right_nh3", "chem_right_ox",
    "co2_ppm", "temperature_c", "relative_humidity", "ec1", "ec2",
)

SENSOR_PROFILES = ("packaged_slow", "fast_modulated")


@dataclass
class DeviceState:
    temp_c: float = 20.0
    rh_pct: float = 50.0
    flow_mps: float = 0.3
    heater_level: float = 1.0

    def __post_init__(self):
        validate_environment(self.temp_c, self.rh_pct, self.flow_mps, self.heater_level)


@dataclass
class DeviceConfig:
    """Shared calibration for both backends; explicit channels take precedence
    over profile presets. Scalar MOX overrides are applied last. NumPy uses
    float64; ``dtype`` selects torch arithmetic without implicit downcasting.
    """

    sensor_profile: str = "packaged_slow"
    mox_channels: tuple[MoxChannelConfig, ...] | None = None
    co2: CO2Config = field(default_factory=CO2Config)
    ec_channels: tuple[ECChannelConfig, ...] = field(
        default_factory=lambda: deepcopy((EC_CO, EC_H2S)))
    ambient_temp_c: float = 20.0
    ambient_rh: float = 50.0
    flow_mps: float = 0.3
    heater_level: float = 1.0
    ratio_feature: str = "ratio_measured"
    dtype: str = "float32"
    # Legacy torch constructor fields, now explicit overrides of shared dies.
    r0_range: tuple[float, float] | None = None
    rs_r0_clean_air: float | None = None
    drift_sigma_per_sqrt_s: float | None = None
    white_noise_frac: float | None = None
    flicker_noise_frac: float | None = None
    humidity_coeff: float | None = None

    def __post_init__(self):
        if self.sensor_profile not in SENSOR_PROFILES:
            raise ValueError(f"sensor_profile must be one of {SENSOR_PROFILES}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.ratio_feature not in ("ratio_measured", "ratio_baseline"):
            raise ValueError("ratio_feature must be 'ratio_measured' or 'ratio_baseline'")
        self.default_state()
        if not isinstance(self.co2, CO2Config):
            raise ValueError("co2 must be a CO2Config")
        if hasattr(self.co2, "__post_init__"):
            self.co2.__post_init__()
        # Keep the CO2Config backward interface; profile extensions belong to
        # co2_sensor.py and are not inferred or silently substituted here.
        for name in ("tau63_s", "sample_interval_s", "asc_window_s"):
            finite_value(name, getattr(self.co2, name), positive=True)
        for name in ("accuracy_base_ppm", "accuracy_frac", "repeatability_ppm",
                     "ambient_baseline_ppm", "asc_gain"):
            finite_value(name, getattr(self.co2, name), nonnegative=True)
        if not isinstance(self.co2.asc_enabled, bool) or self.co2.asc_gain > 1:
            raise ValueError("CO2 asc_enabled must be boolean and asc_gain in [0, 1]")
        if not isinstance(self.ec_channels, (tuple, list)) or len(self.ec_channels) != 2 or not all(isinstance(c, ECChannelConfig) for c in self.ec_channels):
            raise ValueError("ec_channels must contain two ECChannelConfig objects")
        for cfg in self.ec_channels:
            cfg.__post_init__()
            if 1 + cfg.span_tempco_per_k * (self.ambient_temp_c - 20) <= 0:
                raise ValueError("ambient temperature lies outside positive EC span calibration")
        self.resolved_mox()

    def default_state(self) -> DeviceState:
        return DeviceState(self.ambient_temp_c, self.ambient_rh, self.flow_mps, self.heater_level)

    def resolved_mox(self) -> tuple[MoxChannelConfig, ...]:
        if self.mox_channels is None:
            channels = (MOX_RED, MOX_NH3, MOX_OX) * 2
            if self.sensor_profile == "fast_modulated":
                channels = tuple(replace(c, **FAST_OVERRIDES) for c in channels)
        else:
            channels = self.mox_channels
        if not isinstance(channels, (tuple, list)) or len(channels) != 6 or not all(isinstance(c, MoxChannelConfig) for c in channels):
            raise ValueError("mox_channels must contain six MoxChannelConfig objects")
        overrides = {name: getattr(self, name) for name in (
            "r0_range", "rs_r0_clean_air", "drift_sigma_per_sqrt_s",
            "white_noise_frac", "flicker_noise_frac", "humidity_coeff")
            if getattr(self, name) is not None}
        return tuple(replace(deepcopy(c), **overrides) for c in channels)

    @classmethod
    def from_dict(cls, payload: Mapping) -> "DeviceConfig":
        """Decode JSON-compatible calibration; unknown keys raise ValueError."""
        if not isinstance(payload, Mapping):
            raise ValueError("device config must be a dictionary")
        data = deepcopy(dict(payload))
        try:
            if data.get("mox_channels") is not None:
                data["mox_channels"] = tuple(replace(c) if isinstance(c, MoxChannelConfig)
                                             else MoxChannelConfig(**c) for c in data["mox_channels"])
            if "co2" in data:
                c = data["co2"]
                data["co2"] = replace(c) if isinstance(c, CO2Config) else CO2Config(**c)
            if "ec_channels" in data:
                data["ec_channels"] = tuple(replace(c) if isinstance(c, ECChannelConfig)
                                            else ECChannelConfig(**c) for c in data["ec_channels"])
            return cls(**data)
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"invalid device config: {exc}") from exc


class ScentienceV1:
    """
    One virtual Reconnaisscent-class unit: 2x MiCS-6814 (RED/NH3/OX each),
    an SCD4x CO2 channel, and 2 electrochemical cells, per the stack described
    in France et al. (arXiv:2602.19577) and the Scentience product docs.

    ``sensor_profile`` selects a packaged (12 s recovery) or experimental
    fast (46 ms recovery) regime, before flow/heater corrections. Event
    retention depends on the plume, sampling and detection method.
    """

    def __init__(self, sensor_profile: str | None = None,
                 seed: int | None = 0, randomize_unit: bool = True,
                 config: DeviceConfig | Mapping | None = None):
        if isinstance(config, Mapping):
            config = DeviceConfig.from_dict({"sensor_profile": sensor_profile or "packaged_slow", **config})
        self.cfg = deepcopy(config) if config is not None else DeviceConfig()
        if not isinstance(self.cfg, DeviceConfig):
            raise ValueError("config must be a DeviceConfig; use DeviceConfig.from_dict for JSON")
        if sensor_profile is not None:
            if config is not None and sensor_profile != config.sensor_profile:
                raise ValueError("sensor_profile conflicts with config.sensor_profile")
            self.cfg = replace(self.cfg, sensor_profile=sensor_profile)
        self.cfg.__post_init__()
        self.sensor_profile = self.cfg.sensor_profile
        self.rng = np.random.default_rng(seed)
        self.mox = [MoxChannel(c, self.rng, randomize=randomize_unit)
                    for c in self.cfg.resolved_mox()]
        self.co2 = CO2Channel(self.cfg.co2, self.rng)
        self.ec = [ECChannel(c, self.rng) for c in self.cfg.ec_channels]
        self.last_mox_readings: tuple[dict, ...] = ()

    def reset(self, randomize: bool | None = None) -> None:
        for m in self.mox:
            m.reset(randomize=randomize)
        self.co2.reset()
        for e in self.ec:
            e.reset()
        self.last_mox_readings = ()

    def step(self, conc_ppm: dict[str, float], dt: float,
             state: DeviceState | None = None,
             conc_ppm_2: dict[str, float] | None = None) -> dict[str, float]:
        """One device tick.

        `conc_ppm` feeds MiCS die 1 (`chem_left_*`), the SCD4x and the EC cells.
        `conc_ppm_2`, when given, feeds MiCS die 2 (`chem_right_*`) -- this is the
        stereo-olfaction path: the two dies sit at different points on the
        board, so sampling the plume at two positions and passing both here
        reproduces the inter-sensor concentration difference the hardware
        exists to measure. Omitted, die 2 sees the same air as die 1 (mono),
        which is the pre-stereo behaviour, unchanged.
        """
        finite_value("dt", dt, positive=True)
        validate_concentrations(conc_ppm)
        if conc_ppm_2 is not None:
            validate_concentrations(conc_ppm_2)
        if state is not None and not isinstance(state, DeviceState):
            raise ValueError("state must be a DeviceState or None")
        st = state or self.cfg.default_state()
        st.__post_init__()
        if any(1 + e.cfg.span_tempco_per_k * (st.temp_c - 20) <= 0 for e in self.ec):
            raise ValueError("temperature lies outside positive EC span calibration")
        out: dict[str, float] = {}
        diagnostics = []
        for i, (name, ch) in enumerate(zip(CHANNELS[:6], self.mox)):
            c = conc_ppm if (i < 3 or conc_ppm_2 is None) else conc_ppm_2
            r = ch.step(c, dt, temp_c=st.temp_c, rh_pct=st.rh_pct,
                        flow_mps=st.flow_mps, heater_level=st.heater_level)
            diagnostics.append(r)
            out[name] = r[self.cfg.ratio_feature]
        self.last_mox_readings = tuple(diagnostics)
        # A missing species means ambient air, not zero absolute CO2.
        cc = self.co2.cfg
        missing = cc.ambient_baseline_ppm if cc.concentration_mode == "absolute" else 0.0
        out["co2_ppm"] = self.co2.step(conc_ppm.get("carbon_dioxide", missing), dt)["co2_ppm"]
        out["temperature_c"] = st.temp_c
        out["relative_humidity"] = st.rh_pct
        out["ec1"] = self.ec[0].step(conc_ppm, dt, temp_c=st.temp_c)["current_na"]
        out["ec2"] = self.ec[1].step(conc_ppm, dt, temp_c=st.temp_c)["current_na"]
        return out

    def observation_vector(self, reading: dict[str, float]) -> np.ndarray:
        return np.array([reading[c] for c in CHANNELS], dtype=np.float64)
