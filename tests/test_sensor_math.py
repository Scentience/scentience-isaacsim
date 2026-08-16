"""
Sensor models vs closed-form predictions -- every configured constant is
verified to actually govern the output it claims to govern.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace
import numpy as np
import pytest
from scentience_olfaction.sensors.mox import (MICS6814_OX, MICS6814_RED,
                                              MoxChannel, absolute_humidity)
from scentience_olfaction.sensors.electrochemical import EC_CO, ECChannel
from scentience_olfaction.sensors.pid import PIDChannel, PIDConfig
from scentience_olfaction.sensors.scd4x import SCD4xChannel, SCD4xConfig


def quiet(cfg, **kw):
    return replace(cfg, white_noise_frac=0, flicker_noise_frac=0,
                   drift_sigma_per_sqrt_s=0, adc_bits=24, **kw)


def fit_tau(y, dt):
    y = np.asarray(y)
    yinf, y0 = y[-1], y[0]
    z = (y - yinf) / (y0 - yinf)
    m = (z > 0.05) & (z < 0.95)
    t = np.arange(len(y)) * dt
    return -1.0 / np.polyfit(t[m], np.log(z[m]), 1)[0]


def test_mox_steady_state_closed_form():
    ch = MoxChannel(quiet(MICS6814_RED), np.random.default_rng(0), randomize=False)
    C, T, RH = 80.0, 20.0, 50.0
    for _ in range(3000):
        out = ch.step({"ethanol": C}, 0.1, temp_c=T, rh_pct=RH)
    A, B = MICS6814_RED.sensitivity["ethanol"]
    pred = A * C ** (-B) * math.exp(
        MICS6814_RED.humidity_coeff * absolute_humidity(T, RH))
    assert abs(out["ratio_measured"] - pred) < 2e-3


def test_mox_fitted_tau_matches_config():
    cfg = quiet(MICS6814_RED, tau_rise_s=2.0, tau_fall_s=8.0)
    ch = MoxChannel(cfg, np.random.default_rng(0), randomize=False)
    dt = 0.02
    rise = [ch.step({"ethanol": 100.0}, dt)["ratio_measured"] for _ in range(2000)]
    fall = [ch.step({}, dt)["ratio_measured"] for _ in range(4000)]
    assert abs(fit_tau(rise, dt) - 2.0) < 0.15
    assert abs(fit_tau(fall, dt) - 8.0) < 0.5


def test_mox_flow_correction_exponent():
    cfg = quiet(MICS6814_RED, tau_rise_s=2.0, tau_fall_s=8.0)
    ch = MoxChannel(cfg, np.random.default_rng(0), randomize=False)
    dt = 0.02
    r1 = [ch.step({"ethanol": 100.0}, dt, flow_mps=0.1)["ratio_measured"]
          for _ in range(2000)]
    ch.reset()
    r2 = [ch.step({"ethanol": 100.0}, dt, flow_mps=2.0)["ratio_measured"]
          for _ in range(2000)]
    ratio = fit_tau(r2, dt) / fit_tau(r1, dt)
    assert abs(ratio - (2.0 / 0.1) ** -0.5) < 0.08


def test_oxidizing_gas_raises_resistance():
    ch = MoxChannel(quiet(MICS6814_OX), np.random.default_rng(0), randomize=False)
    for _ in range(3000):
        out = ch.step({"nitrogen_dioxide": 2.0}, 0.1)
    assert out["ratio_measured"] > 3.0


def test_adc_quantisation_exact_and_monotone():
    cfg = replace(quiet(MICS6814_RED), adc_bits=12)
    ch = MoxChannel(cfg, np.random.default_rng(0), randomize=False)
    q = 3.3 / 2 ** 12
    prev = -1
    for c in (5, 20, 50, 100, 200, 400):
        ch.reset()
        for _ in range(1500):
            out = ch.step({"ethanol": c}, 0.1)
        assert abs(out["volts"] - out["counts"] * q) < 1e-12
        assert out["counts"] > prev, "reducing gas must raise divider counts"
        prev = out["counts"]


def test_ec_span_tempco_exact():
    ec = ECChannel(EC_CO, np.random.default_rng(0))
    def settle(T):
        ec.reset()
        for _ in range(3000):
            out = ec.step({"carbon_monoxide": 10.0}, 0.1, temp_c=T)
        return out["signal_na"]
    assert abs(settle(30.0) / settle(20.0) - 1.08) < 1e-3


def test_scd4x_t63_and_hold_cadence():
    co2 = SCD4xChannel(SCD4xConfig(), np.random.default_rng(0))
    vals = []
    for _ in range(1200):
        vals.append(co2.step(500.0, 0.1)["co2_ppm"])
    v60 = (vals[599] - 420.0) / 500.0
    assert abs(v60 - (1 - math.e ** -1)) < 0.05
    changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
    assert 20 <= changes <= 28   # 5 s zero-order hold over 120 s


def test_pid_humidity_quench_exact():
    pid = PIDChannel(PIDConfig(noise_ppm=0.0), np.random.default_rng(0))
    for _ in range(500):
        dry = pid.step({"ethanol": 31.0}, 0.1, rh_pct=0.0)["ppm_isobutylene_equiv"]
    pid.reset()
    for _ in range(500):
        wet = pid.step({"ethanol": 31.0}, 0.1, rh_pct=90.0)["ppm_isobutylene_equiv"]
    assert abs(wet / dry - 0.70) < 1e-3
