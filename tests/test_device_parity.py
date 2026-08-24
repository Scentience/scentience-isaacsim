import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_numpy_torch_mox_step_response_agree():
    """The torch (Isaac-scale) and NumPy (standalone) device models must show
    the same primary-channel step response when noise and unit variation are
    disabled. Guards against the two implementations drifting apart."""
    from dataclasses import replace
    from scentience_olfaction.sensors.mox import MOX_RED, MoxChannel
    from scentience_olfaction.sensors.scentience_v1 import (DeviceConfig,
                                                            ScentienceV1Device)
    dt, c_ppm, n = 0.1, 100.0, 400

    np_cfg = replace(MOX_RED, white_noise_frac=0.0, flicker_noise_frac=0.0,
                     drift_sigma_per_sqrt_s=0.0, adc_bits=24, humidity_coeff=0.0)
    ch = MoxChannel(np_cfg, np.random.default_rng(0), randomize=False)
    np_tr = [ch.step({"ethanol": c_ppm}, dt, rh_pct=50.0)["ratio_measured"]
             for _ in range(n)]

    t_cfg = DeviceConfig(white_noise_frac=0.0, drift_sigma_per_sqrt_s=0.0,
                         humidity_coeff=0.0)
    dev = ScentienceV1Device(t_cfg, n_envs=1, device="cpu", randomize=False)
    conc = torch.full((1, 1), c_ppm)
    t_tr = [float(dev.step(conc, dt)[0, 0]) for _ in range(n)]

    np_tr, t_tr = np.array(np_tr), np.array(t_tr)
    # same steady state and same time constant (within flow-correction delta:
    # numpy applies flow correction only when flow>0; both called with defaults)
    assert abs(np_tr[-1] - t_tr[-1]) < 0.02, (np_tr[-1], t_tr[-1])
    t63_np = dt * int(np.argmax(np_tr <= np_tr[0] + 0.63 * (np_tr[-1] - np_tr[0])))
    t63_t = dt * int(np.argmax(t_tr <= t_tr[0] + 0.63 * (t_tr[-1] - t_tr[0])))
    assert abs(t63_np - t63_t) <= 2 * dt, (t63_np, t63_t)
