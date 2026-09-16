"""Convert NumPy observations to Scentience SDK reading dictionaries.

Compound keys follow ScentienceDevice.sample_ble(). MOX values are static
primary-analyte equivalents, not firmware predictions or mixture separation.
Lag, environmental effects and noise remain in the inferred signal.
"""
from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping

from ..sensors.device_np import DeviceConfig
from ..sensors.mox import finite_value

# Stable array order; packet key insertion order is not an API.
COMPOUND_FIELDS = (
    "CO2", "NH3", "NO", "NO2", "CO", "C2H5OH", "H2", "CH4",
    "C3H8", "C4H10", "H2S", "HCHO", "SO2", "VOC",
)
_PRIMARY = (("red", "C2H5OH", "ethanol"), ("nh3", "NH3", "ammonia"),
            ("ox", "NO2", "nitrogen_dioxide"))


def _invert_power_law(ratio: float, A: float, beta: float,
                      clean_air_ratio: float = 1.0) -> float:
    """Invert Rs/R0 = A C**(-beta), respecting either response polarity."""
    finite_value("ratio", ratio, nonnegative=True)
    finite_value("A", A, positive=True)
    finite_value("beta", beta)
    finite_value("clean_air_ratio", clean_air_ratio, positive=True)
    if beta == 0:
        raise ValueError("beta must be nonzero for concentration inversion")
    if ratio == 0 or (beta > 0 and ratio >= clean_air_ratio) or (beta < 0 and ratio <= clean_air_ratio):
        return 0.0
    try:
        value = float((ratio / A) ** (-1.0 / beta))
    except OverflowError as exc:
        raise ValueError("calibration inverse exceeds finite concentration range") from exc
    finite_value("equivalent concentration", value, nonnegative=True)
    return value


def _validate_frame_options(uid, timestamp, pressure_hpa, side, include_sim_metadata):
    """Validate the envelope before a world read advances instrument state."""
    if not isinstance(uid, str) or not uid:
        raise ValueError("uid must be a nonempty string")
    if timestamp is not None and (not isinstance(timestamp, str) or not timestamp):
        raise ValueError("timestamp must be a nonempty string or None")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if not isinstance(include_sim_metadata, bool):
        raise ValueError("include_sim_metadata must be boolean")
    finite_value("pressure_hpa", pressure_hpa, positive=True)


def ble_frame(reading: Mapping[str, float], uid: str = "SIM001",
              timestamp: str | None = None, pressure_hpa: float = 1010.0, *,
              device_config: DeviceConfig | None = None,
              side: str = "left", include_sim_metadata: bool = True) -> dict:
    """Map one ScentienceV1 reading to an SDK-shaped dictionary.

    Pass the device's resolved configuration for custom calibrations. The SDK
    compound schema has no stereo axis; side selects one MOX die. CO2 is the
    device's absolute ppm reading. EC currents have no unambiguous compound
    equivalent and are not mapped. No battery model is implied.
    """
    if not isinstance(reading, Mapping):
        raise ValueError("reading must be a ScentienceV1 channel mapping")
    _validate_frame_options(uid, timestamp, pressure_hpa, side, include_sim_metadata)
    cfg = device_config if device_config is not None else DeviceConfig()
    if not isinstance(cfg, DeviceConfig):
        raise ValueError("device_config must be a DeviceConfig")
    cfg.__post_init__()
    temp, rh, co2 = (reading[k] for k in ("temperature_c", "relative_humidity", "co2_ppm"))
    for name, value in (("temperature_c", temp), ("relative_humidity", rh), ("co2_ppm", co2)):
        finite_value(name, value)
    if not 0 <= rh <= 100 or co2 < 0:
        raise ValueError("require relative_humidity in [0, 100] and co2_ppm >= 0")
    frame = {
        "UID": uid,
        "TIMESTAMP": timestamp if timestamp is not None else _dt.datetime.now(_dt.timezone.utc)
            .isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "ENV_temperatureC": float(temp), "ENV_humidity": float(rh),
        "ENV_pressureHpa": float(pressure_hpa),
    }
    if include_sim_metadata:
        frame.update(_sim_units="ppm", _sim_attribution="primary_analyte_equivalent",
                     _sim_mox_side=side)
    channels = cfg.resolved_mox()[0:3] if side == "left" else cfg.resolved_mox()[3:6]
    for (suffix, key, species), channel in zip(_PRIMARY, channels):
        if species not in channel.sensitivity:
            continue  # custom dies need an explicit application-specific inverse
        A, beta = channel.sensitivity[species]
        ppm = round(_invert_power_law(reading[f"chem_{side}_{suffix}"], A, beta,
                                     channel.rs_r0_clean_air), 3)
        if ppm > 0:
            frame[key] = ppm
    if round(co2, 1) > 0:
        frame["CO2"] = round(float(co2), 1)
    return frame
