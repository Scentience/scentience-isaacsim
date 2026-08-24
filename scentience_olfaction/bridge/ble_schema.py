"""
Bridge to the Scentience client packages (PyPI `scentience`, and the C++/Rust/
NPM siblings): emit simulator readings as the EXACT dict schema
`ScentienceDevice.sample_ble()` produces, so downstream code written against
the real hardware client consumes simulator output unchanged.

Schema (scentience PyPI v2.2.2, BLE API docs): UID, TIMESTAMP,
ENV_temperatureC / ENV_humidity / ENV_pressureHpa, BATT_*, STATUS_opuA, and
the 14 compounds CO2 NH3 NO NO2 CO C2H5OH H2 CH4 C3H8 C4H10 H2S HCHO SO2 VOC
-- zero-magnitude compounds omitted from the frame, as on hardware.

Attribution is deliberately firmware-naive in v0.1: each MOX die inverts its
own power law assuming its PRIMARY analyte (RED -> C2H5OH-equivalent,
NH3 die -> NH3, OX -> NO2); CO2 comes from the SCD4x channel. A mixed plume
is therefore mis-attributed exactly the way cheap single-die firmware
mis-attributes it. That is a feature for sim-to-real honesty, and a
documented limitation for analytics.

KNOWN SCHEMA GAP (upstream): the BLE schema publishes NO units. This module
emits ppm and says so in the frame under `_sim_units` (an extension key
hardware does not send). See docs/UPSTREAM_REQUESTS.md.
"""

from __future__ import annotations

import datetime as _dt

from ..sensors.mox import MOX_NH3, MOX_OX, MOX_RED

_PRIMARY = {  # die -> (BLE compound key, (A, beta) of the primary analyte)
    "chem_left_red": ("C2H5OH", MOX_RED.sensitivity["ethanol"]),
    "chem_left_nh3": ("NH3", MOX_NH3.sensitivity["ammonia"]),
    "chem_left_ox": ("NO2", MOX_OX.sensitivity["nitrogen_dioxide"]),
}


def _invert_power_law(ratio: float, A: float, beta: float) -> float:
    """C = (ratio/A)^(-1/beta); 0 below the clean-air knee."""
    if ratio >= 1.0 or ratio <= 0.0:
        return 0.0
    return float((ratio / A) ** (-1.0 / beta))


def ble_frame(reading: dict[str, float], uid: str = "SIM001",
              timestamp: str | None = None,
              pressure_hpa: float = 1010.0) -> dict:
    """Map one ScentienceV1 reading dict -> a sample_ble()-shaped frame."""
    frame = {
        "UID": uid,
        "TIMESTAMP": timestamp or _dt.datetime.now(_dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ENV_temperatureC": reading["temperature_c"],
        "ENV_humidity": reading["relative_humidity"],
        "ENV_pressureHpa": pressure_hpa,
        "STATUS_opuA": 1,
        "BATT_health": 100, "BATT_v": 4.2, "BATT_charge": 95, "BATT_time": 8,
        "_sim_units": "ppm",   # extension key; real hardware omits units entirely
    }
    for ch, (key, (A, beta)) in _PRIMARY.items():
        ppm = _invert_power_law(reading[ch], A, beta)
        if ppm > 0.0:
            frame[key] = round(ppm, 3)
    co2 = reading["co2_ppm"]
    if co2 > 0:
        frame["CO2"] = round(co2, 1)
    return frame
