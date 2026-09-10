"""NumPy interoperability with the optional PyPI scentience SDK.

Only the OVL helpers import the SDK. They call local mapping functions;
no helper connects to Bluetooth, sends requests or downloads weights.
"""
from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

import numpy as np

from .ble_schema import COMPOUND_FIELDS


def _frames(readings) -> list[Mapping]:
    if isinstance(readings, Mapping):
        return [readings]
    try:
        frames = list(readings)
    except TypeError as exc:
        raise ValueError("readings must be one SDK dictionary or an iterable of dictionaries") from exc
    if not all(isinstance(row, Mapping) for row in frames):
        raise ValueError("each SDK reading must be a dictionary")
    return frames


def readings_to_numpy(readings, *, fields=COMPOUND_FIELDS, missing: float = 0.0) -> np.ndarray:
    """Convert SDK readings or exported JSON rows to float64 [N,F].

    Column order is explicit. Missing compounds default to zero, following BLE
    omission rules; missing=np.nan preserves missingness. Present malformed or
    nonfinite values raise. Unselected metadata is ignored. Values retain the
    source packet's units; this function does not convert units.
    """
    if isinstance(fields, str):
        raise ValueError("fields must be a sequence of unique numeric field names")
    fields = tuple(fields)
    if not fields or not all(isinstance(k, str) and k for k in fields) or len(set(fields)) != len(fields):
        raise ValueError("fields must be nonempty, unique strings")
    missing = float(missing)
    if np.isinf(missing):
        raise ValueError("missing must be finite or NaN")
    frames = _frames(readings)
    out = np.full((len(frames), len(fields)), missing, dtype=np.float64)
    for i, row in enumerate(frames):
        for j, key in enumerate(fields):
            if key not in row:
                continue
            value = row[key]
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"reading {i} field {key!r} must be numeric") from exc
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(number):
                raise ValueError(f"reading {i} field {key!r} must be finite numeric data")
            out[i, j] = number
    return out


def sample_numpy(device, *, fields=COMPOUND_FIELDS, missing: float = 0.0) -> np.ndarray:
    """Sample an already connected ScentienceDevice through its public API.

    Returns [1,F] for one device and [N,F] for a multi-device connection. Use
    readings_to_numpy(device.sample_ble()) yourself when retaining UID and
    timestamp dictionaries alongside the numeric array.
    """
    return readings_to_numpy(device.sample_ble(), fields=fields, missing=missing)


def _sdk():
    try:
        sdk = import_module("scentience")
    except ModuleNotFoundError as exc:
        if exc.name != "scentience":
            raise
        raise ImportError('Install the SDK bridge with: pip install "scentience-olfaction[bridge]"') from exc
    if not hasattr(sdk, "OVL_SENSOR_CHANNELS") or not hasattr(sdk, "OVLClient"):
        raise ImportError("OVL integration requires scentience>=2.2.2,<3")
    return sdk


def ovl_sensor_channels() -> tuple[str, ...]:
    """The installed SDK's authoritative OVL sensor column order."""
    return tuple(_sdk().OVL_SENSOR_CHANNELS)


def ovl_sensor_window(readings) -> np.ndarray:
    """Map one device's ordered SDK frames to an OVL [T,6] NumPy window.

    Delegates to OVLClient.device_reading_to_ovl. That experimental mapping
    does not calibrate the encoder to simulated MOX/EC channels. Split mixed
    streams by UID before constructing a temporal window. No inference is run.
    """
    frames = _frames(readings)
    if not frames:
        raise ValueError("an OVL window needs at least one reading")
    uids = [frame.get("UID") for frame in frames]
    if any(uid is not None and (not isinstance(uid, str) or not uid) for uid in uids):
        raise ValueError("UID must be a nonempty string when supplied")
    if len(set(uids)) > 1:
        raise ValueError("an OVL time window must contain one UID; split device streams first")
    readings_to_numpy(frames)  # reject invalid values before the SDK's zero fallback
    sdk = _sdk()
    window = np.asarray([sdk.OVLClient.device_reading_to_ovl(dict(row)) for row in frames],
                        dtype=np.float64)
    if window.shape != (len(frames), len(sdk.OVL_SENSOR_CHANNELS)) or not np.isfinite(window).all():
        raise ValueError("SDK returned an invalid OVL sensor window")
    return window
