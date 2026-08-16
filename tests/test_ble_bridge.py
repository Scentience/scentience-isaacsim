import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scentience_olfaction.bridge.ble_schema import ble_frame, _invert_power_law
from scentience_olfaction.sensors.mox import MICS6814_RED


def test_inversion_roundtrip():
    A, beta = MICS6814_RED.sensitivity["ethanol"]
    c = 50.0
    ratio = A * c ** (-beta)
    assert abs(_invert_power_law(ratio, A, beta) - c) / c < 1e-9
    assert _invert_power_law(1.2, A, beta) == 0.0   # above baseline -> 0


def test_frame_schema_and_zero_omission():
    reading = {"mics1_red": 1.0, "mics1_nh3": 1.0, "mics1_ox": 1.0,
               "mics2_red": 1.0, "mics2_nh3": 1.0, "mics2_ox": 1.0,
               "co2_ppm": 420.0, "temperature_c": 21.0,
               "relative_humidity": 45.0, "ec1": 0.0, "ec2": 0.0}
    f = ble_frame(reading, timestamp="2026-01-01T00:00:00Z")
    assert "C2H5OH" not in f and "NH3" not in f     # clean air -> omitted
    assert f["ENV_temperatureC"] == 21.0 and f["UID"] == "SIM001"
    reading["mics1_red"] = 0.2                       # strong ethanol response
    f2 = ble_frame(reading, timestamp="2026-01-01T00:00:00Z")
    assert f2["C2H5OH"] > 0
