import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scentience_olfaction.oio.oio import (BoutDetector, BoutDetectorConfig,
                                          OlfactoryInertialOdometry, OIOConfig)


def test_bout_detector_counts_pulses():
    det = BoutDetector(BoutDetectorConfig(noise_sigma=0.01, tau_fast_s=0.2,
                                          tau_slow_s=10.0))
    dt, n = 0.01, 0
    for i in range(6000):   # 60 s, 1 s pulses every 10 s
        x = 1.0 if (i // 100) % 10 == 0 else 0.0
        det.step(x, dt)
    assert det.n_bouts == 6, f"expected 6 bouts, got {det.n_bouts}"


def test_oio_heading_beats_dead_reckoning():
    wins = 0
    for seed in range(5):
        oio = OlfactoryInertialOdometry(OIOConfig(platform="quadruped"), seed=seed)
        rng = np.random.default_rng(seed + 100)
        true_h, dt = 0.0, 0.01
        for i in range(3000):
            omega_true = 0.2 * math.sin(0.02 * i)
            true_h += omega_true * dt
            accel, omega = oio.simulate_imu(np.zeros(2), omega_true, dt)
            c, s = math.cos(-true_h), math.sin(-true_h)
            wb = np.array([c, s]) + 0.05 * rng.standard_normal(2)  # world wind (1,0)
            defl = 0.05 if (i % 300) < 30 else 0.0                 # periodic bouts
            out = oio.step(accel, omega, wb, defl, dt)
        e_dr = abs((out["h_dr"] - true_h + math.pi) % (2 * math.pi) - math.pi)
        e_oio = abs((out["h_oio"] - true_h + math.pi) % (2 * math.pi) - math.pi)
        wins += e_oio < e_dr
    assert wins >= 4, f"OIO heading won only {wins}/5 seeds"


def test_oio_crosswind_corrected_downwind_not():
    """The honest asymmetry: bouts observe the crosswind coordinate (plume
    axis) but carry no downwind information. The estimator must reflect that,
    not pretend otherwise."""
    oio = OlfactoryInertialOdometry(OIOConfig(platform="uav"), seed=0)
    dt = 0.01
    # true motion: straight upwind along -x at y=0 (on the plume axis)
    for i in range(4000):
        accel, omega = oio.simulate_imu(np.zeros(2), 0.0, dt)
        wb = np.array([1.0, 0.0])
        defl = 0.05 if (i % 200) < 40 else 0.0
        out = oio.step(accel, omega, wb, defl, dt)
    # crosswind error corrected toward axis estimate; downwind untouched
    assert abs(out["p_oio"][1]) <= abs(out["p_dr"][1]) + 1e-9
    assert out["p_oio"][0] == out["p_dr"][0], "downwind must be identical -- "\
        "bouts alone cannot observe it"
