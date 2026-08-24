import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from scentience_olfaction.sensors.mox import (MOX_RED, MoxChannel,
                                              absolute_humidity)
from scentience_olfaction.sensors.electrochemical import ECChannel, EC_CO
from scentience_olfaction.sensors.co2_sensor import CO2Channel, CO2Config
from scentience_olfaction.sensors.pid import PIDChannel, PIDConfig
from scentience_olfaction.sensors.device_np import ScentienceV1, CHANNELS

def quiet(cfg):
    from dataclasses import replace
    return replace(cfg, white_noise_frac=0.0, flicker_noise_frac=0.0,
                   drift_sigma_per_sqrt_s=0.0, adc_bits=24)


def test_mox_asymmetric_tau():
    ch = MoxChannel(quiet(MOX_RED), np.random.default_rng(0), randomize=False)
    on = [ch.step({"ethanol": 100.0}, 0.1)["ratio_measured"] for _ in range(300)]
    off = [ch.step({}, 0.1)["ratio_measured"] for _ in range(600)]
    on, off = np.array(on), np.array(off)
    t_on = 0.1 * np.argmax(on < on[0] - 0.9 * (on[0] - on[-1]))
    t_off = 0.1 * np.argmax(off > off[0] + 0.9 * (off[-1] - off[0]))
    assert t_off > 2.0 * t_on, f"recovery {t_off}s should be slower than onset {t_on}s"


def test_mox_power_law_monotone_and_humidity():
    ch = MoxChannel(quiet(MOX_RED), np.random.default_rng(0), randomize=False)
    r = []
    for c in (10.0, 30.0, 100.0, 300.0):
        ch.reset()
        for _ in range(500):
            last = ch.step({"ethanol": c}, 0.1)["ratio_measured"]
        r.append(last)
    assert all(a > b for a, b in zip(r, r[1:])), "more gas -> lower Rs/R0"
    ch.reset()
    for _ in range(500):
        dry = ch.step({"ethanol": 50.0}, 0.1, rh_pct=20.0)["ratio_measured"]
    ch.reset()
    for _ in range(500):
        wet = ch.step({"ethanol": 50.0}, 0.1, rh_pct=80.0)["ratio_measured"]
    assert wet < dry, "humidity lowers Rs (false-reducing-gas direction)"
    assert absolute_humidity(20.0, 50.0) == pytest.approx(8.65, abs=0.5)


def test_adc_resolution_collapse_at_high_rs():
    """The (Rs+RL)^2 effect: at high Rs (clean air on a megohm die) a 12-bit
    ADC cannot distinguish neighbouring Rs values; at low Rs it can."""
    from dataclasses import replace
    cfg = replace(quiet(MOX_RED), adc_bits=12, r0_nominal=1.5e6)
    ch = MoxChannel(cfg, np.random.default_rng(0), randomize=False)
    def counts_for(c):
        ch.reset()
        for _ in range(400):
            out = ch.step({"ethanol": c} if c else {}, 0.1)
        return out["counts"]
    # equal MULTIPLICATIVE concentration steps -> equal Rs ratios; the ADC
    # resolves far fewer counts per step at high Rs than at low Rs.
    d_hi = abs(counts_for(2.0) - counts_for(3.0))      # high-Rs regime
    d_lo = abs(counts_for(100.0) - counts_for(150.0))  # low-Rs regime
    assert d_lo > 2 * d_hi, f"resolution should collapse at high Rs "\
        f"(hi step {d_hi} counts, lo step {d_lo} counts)"


def test_ec_linearity_and_cottrell():
    ec = ECChannel(EC_CO, np.random.default_rng(0))
    def settle(c):
        ec.reset()
        for _ in range(2000):
            out = ec.step({"carbon_monoxide": c}, 0.1)
        return out["signal_na"]
    s10, s20, s40 = settle(10), settle(20), settle(40)
    assert s20 == pytest.approx(2 * s10, rel=0.05)
    assert s40 == pytest.approx(4 * s10, rel=0.05)
    t = np.array([1.0, 4.0, 16.0])
    i = ec.cottrell_current(1e-9, t)
    assert i[0] / i[1] == pytest.approx(2.0, rel=1e-3)   # t^-1/2 slope
    assert i[1] / i[2] == pytest.approx(2.0, rel=1e-3)


def test_scd4x_slow_and_asc():
    co2 = CO2Channel(CO2Config(), np.random.default_rng(0))
    # step of +500 ppm: at t=60s the response should be ~63% (tau63)
    for _ in range(600):
        r = co2.step(500.0, 0.1)
    got = r["co2_ppm"] - 420.0
    assert 0.5 * 500 < got < 0.75 * 500, "tau63=60 s behaviour"
    # ASC drags calibration down in never-clean air
    cfg = CO2Config(asc_window_s=100.0)
    c2 = CO2Channel(cfg, np.random.default_rng(0))
    for _ in range(3000):
        r2 = c2.step(600.0, 0.1)   # 300 s continuously elevated
    assert r2["asc_offset_ppm"] < -50.0, "ASC should have pulled baseline down"


def test_pid_correction_factors():
    pid = PIDChannel(PIDConfig(lamp_ev=10.6), np.random.default_rng(0))
    assert pid.correction_factor("ethanol") == pytest.approx(3.1)   # NOT 10.47
    assert math.isinf(pid.correction_factor("methane"))             # PID-blind
    assert math.isinf(pid.correction_factor("carbon_monoxide"))
    for _ in range(400):
        r = pid.step({"ethanol": 31.0}, 0.1, rh_pct=0.0)
    assert r["ppm_isobutylene_equiv"] == pytest.approx(10.0, rel=0.1)


def test_device_schema_and_reproducibility():
    d1 = ScentienceV1(seed=7)
    d2 = ScentienceV1(seed=7)
    r1 = [d1.step({"ethanol": 50.0}, 0.1) for _ in range(50)]
    r2 = [d2.step({"ethanol": 50.0}, 0.1) for _ in range(50)]
    assert tuple(r1[-1].keys()) == CHANNELS
    assert r1[-1] == r2[-1], "same seed -> identical trajectory"
    assert ScentienceV1(seed=8).step({"ethanol": 50.0}, 0.1) != r1[0]
