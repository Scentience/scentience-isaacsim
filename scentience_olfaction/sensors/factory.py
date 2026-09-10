"""NumPy sensor factory for world/config integration; no torch or vendor SDK.

All devices expose step(conc_ppm, dt, state=None, conc_ppm_2=None), reset(),
channel_names and observation_vector(). Import Scentience only when selected.
"""

from __future__ import annotations

import math
import types
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from numbers import Real
from typing import TYPE_CHECKING, Literal, Protocol, Union, get_args, get_origin, get_type_hints

import numpy as np

from .co2_sensor import CO2Channel, CO2Config, CO2_PROFILES, co2_config
from .emstat_pico import EmStatPicoChannel, EmStatPicoConfig

if TYPE_CHECKING:
    from .device_np import DeviceConfig, DeviceState

__all__ = ["DEVICE_PROFILES", "Sensor", "create_sensor"]

DEVICE_PROFILES = ("scentience_v1", "scd30", "scd40", "scd41", "emstat_pico")


class Sensor(Protocol):
    @property
    def configuration(self): ...

    @property
    def channel_names(self) -> tuple[str, ...]: ...

    @property
    def last_reading(self) -> dict: ...

    def reset(self) -> None: ...

    def step(self, conc_ppm: Mapping[str, float], dt: float,
             state: DeviceState | None = None,
             conc_ppm_2: Mapping[str, float] | None = None) -> dict[str, float]: ...

    def observation_vector(self, reading: Mapping[str, float]) -> np.ndarray: ...


def _dataclass_config(cls, value, path="config"):
    """Construct and validate a fresh typed config; reject unknown JSON keys."""
    if isinstance(value, cls):
        value = {f.name: getattr(value, f.name) for f in fields(cls) if f.init}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be {cls.__name__} or a mapping")
    allowed = {f.name for f in fields(cls) if f.init}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown, key=str)!r}")
    hints = get_type_hints(cls)
    parsed = {name: _config_value(hints[name], item, f"{path}.{name}")
              for name, item in value.items()}
    try:
        return cls(**parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _config_value(annotation, value, path):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        for choice in args:
            try:
                return _config_value(choice, value, path)
            except (TypeError, ValueError):
                pass
        raise ValueError(f"{path} does not match {annotation}")
    if annotation is type(None):
        if value is not None:
            raise ValueError(f"{path} must be null")
        return None
    if is_dataclass(annotation):
        return _dataclass_config(annotation, value, path)
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a mapping")
        return {_config_value(args[0], key, path): _config_value(args[1], item, f"{path}.{key}")
                for key, item in value.items()}
    if origin in (tuple, list):
        if not isinstance(value, (tuple, list)):
            raise ValueError(f"{path} must be an array")
        if origin is list or (len(args) == 2 and args[1] is Ellipsis):
            parsed = [_config_value(args[0], item, f"{path}[{i}]")
                      for i, item in enumerate(value)]
        else:
            if len(value) != len(args):
                raise ValueError(f"{path} must have {len(args)} elements")
            parsed = [_config_value(kind, item, f"{path}[{i}]")
                      for i, (kind, item) in enumerate(zip(args, value))]
        return tuple(parsed) if origin is tuple else parsed
    if origin is Literal:
        if value not in args:
            raise ValueError(f"{path} must be one of {args}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number")
        return float(value)
    if annotation in (str, bool, int):
        if type(value) is not annotation:
            raise ValueError(f"{path} must be {annotation.__name__}")
        return value
    raise TypeError(f"{path}: unsupported configuration type {annotation}")


class _SensorBase:
    @property
    def configuration(self):
        """Resolved calibration snapshot for recording; modifying it has no effect."""
        return deepcopy(self._configuration)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self._channel_names

    @property
    def last_reading(self) -> dict:
        """Copy of the last model diagnostics; empty until step() or after reset()."""
        return deepcopy(self._last_reading)

    def observation_vector(self, reading: Mapping[str, float]) -> np.ndarray:
        return np.asarray([reading[name] for name in self.channel_names], dtype=np.float64)


class _ScentienceSensor(_SensorBase):
    def __init__(self, device, channel_names):
        self.device = device
        self._configuration = deepcopy(device.cfg)
        self._channel_names = tuple(channel_names)
        self._last_reading = {}

    def reset(self, randomize=None) -> None:
        self.device.reset(randomize=randomize)
        self._last_reading = {}

    def step(self, conc_ppm, dt, state=None, conc_ppm_2=None) -> dict[str, float]:
        reading = self.device.step(conc_ppm, dt, state=state, conc_ppm_2=conc_ppm_2)
        self._last_reading = {"channels": dict(reading), "mox": self.device.last_mox_readings}
        return reading


class _CO2Sensor(_SensorBase):
    _channel_names = ("co2_ppm",)

    def __init__(self, cfg: CO2Config, seed):
        self.channel = CO2Channel(cfg, np.random.default_rng(seed))
        self._configuration = deepcopy(self.channel.cfg)
        self._last_reading = {}

    def reset(self) -> None:
        self.channel.reset()
        self._last_reading = {}

    def step(self, conc_ppm, dt, state=None, conc_ppm_2=None) -> dict[str, float]:
        # Missing species means no plume excess, or ambient in absolute mode.
        cfg = self.channel.cfg
        missing = cfg.ambient_baseline_ppm if cfg.concentration_mode == "absolute" else 0.0
        reading = self.channel.step(conc_ppm.get("carbon_dioxide", missing), dt)
        self._last_reading = reading
        return {"co2_ppm": reading["co2_ppm"]}


class _EmStatSensor(_SensorBase):
    def __init__(self, cfg: EmStatPicoConfig, seed):
        self.cfg = cfg
        self._configuration = deepcopy(cfg)
        streams = np.random.SeedSequence(seed).spawn(len(cfg.channels))
        self.channels = tuple(EmStatPicoChannel(ch, np.random.default_rng(stream))
                              for ch, stream in zip(cfg.channels, streams))
        self._channel_names = tuple(f"ec{i + 1}" for i in range(len(self.channels)))
        self._last_reading = {}

    def reset(self) -> None:
        for channel in self.channels:
            channel.reset()
        self._last_reading = {}

    def step(self, conc_ppm, dt, state=None, conc_ppm_2=None) -> dict[str, float]:
        temp_c = 20.0 if state is None else state.temp_c
        self._last_reading = {
            name: channel.step(conc_ppm if i == 0 or conc_ppm_2 is None else conc_ppm_2,
                               dt, temp_c=temp_c)
            for i, (name, channel) in enumerate(zip(self.channel_names, self.channels))}
        return {name: reading["current_na"] for name, reading in self._last_reading.items()}


def create_sensor(device_profile: str = "scentience_v1",
                  sensor_profile: str = "packaged_slow", seed: int | None = 0,
                  config: DeviceConfig | CO2Config | EmStatPicoConfig | Mapping | None = None,
                  randomize: bool = True,
                  ) -> Sensor:
    """Create a standard scalar sensor from a typed config or strict JSON mapping.

    Manufacturer CO2 defaults here consume plume EXCESS ppm. Explicit configs
    control their own concentration_mode. An EmStat cell calibration is required.
    ``sensor_profile`` selects only the Scentience MOX dynamic regime.
    """
    if device_profile not in DEVICE_PROFILES:
        raise ValueError(f"unknown device_profile {device_profile!r}; choose {DEVICE_PROFILES}")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    if not isinstance(randomize, bool):
        raise ValueError("randomize must be a bool")
    if sensor_profile not in ("packaged_slow", "fast_modulated"):
        raise ValueError("sensor_profile must be 'packaged_slow' or 'fast_modulated'")
    if device_profile == "scentience_v1":
        from .device_np import CHANNELS, ScentienceV1

        kwargs = {}
        if config is not None:
            from .device_np import DeviceConfig

            if isinstance(config, Mapping):
                config = {"sensor_profile": sensor_profile, **config}
            kwargs["config"] = _dataclass_config(DeviceConfig, config)
        device = ScentienceV1(sensor_profile=sensor_profile, seed=seed,
                             randomize_unit=randomize, **kwargs)
        return _ScentienceSensor(device, CHANNELS)
    if device_profile in CO2_PROFILES:
        if isinstance(config, CO2Config):
            cfg = _dataclass_config(CO2Config, config)
        else:
            defaults = co2_config(device_profile, concentration_mode="excess")
            if config is None:
                cfg = defaults
            else:
                if not isinstance(config, Mapping):
                    raise TypeError("CO2 config must be CO2Config or a mapping")
                values = {f.name: getattr(defaults, f.name) for f in fields(CO2Config)}
                if "accuracy_base_ppm" in config or "accuracy_frac" in config:
                    values["accuracy_bands"] = ()
                values.update(config)
                cfg = _dataclass_config(CO2Config, values)
        return _CO2Sensor(cfg, seed)
    if config is None:
        raise ValueError("emstat_pico requires EmStatPicoConfig with calibrated analyte cells")
    return _EmStatSensor(_dataclass_config(EmStatPicoConfig, config), seed)
