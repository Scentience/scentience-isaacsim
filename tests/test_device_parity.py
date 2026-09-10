"""Deterministic physics/readout parity; backend RNG sequences are not equal."""
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scentience_olfaction.sensors.device_np import CHANNELS, DeviceConfig, ScentienceV1
from scentience_olfaction.sensors.scentience_v1 import build_device

SPECIES = ("carbon_dioxide", "nitrogen_dioxide", "ammonia", "hydrogen_sulfide",
           "carbon_monoxide", "ethanol", "hydrogen", "methane", "unmodeled")


def matched_config(profile="packaged_slow", dtype="float64"):
    base = DeviceConfig(sensor_profile=profile, dtype=dtype)
    channels = tuple(replace(c, white_noise_frac=0., flicker_noise_frac=0.,
                             drift_sigma_per_sqrt_s=0.) for c in base.resolved_mox())
    return replace(base, mox_channels=channels,
                   co2=replace(base.co2, repeatability_ppm=0., asc_enabled=False),
                   ec_channels=tuple(replace(c, noise_na=0., drift_na_per_sqrt_s=0.)
                                     for c in base.ec_channels))


@pytest.mark.parametrize("profile", ["packaged_slow", "fast_modulated"])
def test_numpy_torch_all_channels_stereo_variable_dt_parity(profile):
    cfg = matched_config(profile)
    left = np.array([500., 2., 20., 1., 10., 80., 5., 50., 1.e4])
    right = np.array([1000., 0.5, 10., 0.3, 3., 20., 1., 5., 0.])
    ndev = ScentienceV1(config=cfg, randomize_unit=False)
    tdev = build_device("scentience_v1", profile, 1, "cpu", randomize=False,
                        species_names=SPECIES, seed=77, config=cfg)
    # Includes sub-delay steps, large steps, sample-boundary crossings and recovery.
    for tick, dt in enumerate((0.002, 0.001, 0.02, 0.1, 0.37, 1.2, 4.6, 0.03, 7.3) * 3):
        left_ppm, right_ppm = (left, right) if tick < 15 else (left * 0, right * 0)
        expected = ndev.observation_vector(ndev.step(dict(zip(SPECIES, left_ppm)), dt,
                                                     conc_ppm_2=dict(zip(SPECIES, right_ppm))))
        actual = tdev.step(torch.tensor(left_ppm[None]), dt, conc_ppm_2=torch.tensor(right_ppm[None]))[0].numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-10)
        np.testing.assert_array_equal(tdev.last_mox_readings["counts"][0].numpy(),
                                      [d["counts"] for d in ndev.last_mox_readings])
    assert actual[6] > 420 and actual[9] != 0 and actual[10] != 0


def test_numpy_torch_custom_environment_calibration_and_asc_parity():
    cfg = matched_config()
    cfg = replace(cfg, ambient_temp_c=29., ambient_rh=75., flow_mps=1.2, heater_level=1.5,
                  mox_channels=tuple(replace(c, humidity_sensitivity_coeff=0.002,
                                            activation_energy_ev=0.05, dead_volume_delay_s=(i+1)*0.04)
                                     for i, c in enumerate(cfg.mox_channels)),
                  co2=replace(cfg.co2, asc_enabled=True, asc_window_s=1.3, sample_interval_s=0.7,
                              accuracy_bias_fraction=0.2, resolution_ppm=1.,
                              output_range_ppm=(0., 40000.)))
    ndev = ScentienceV1(config=cfg, randomize_unit=False)
    tdev = build_device("scentience_v1", cfg.sensor_profile, 1, "cpu", randomize=False,
                        species_names=SPECIES, config=cfg)
    conc = np.array([600., 1., 5., 1., 10., 20., 0., 0., 0.])
    for dt in (0.03, 0.01, 0.07, 0.05, 0.12, 0.6, 0.4, 4., 1., 5.):
        expected = ndev.observation_vector(ndev.step(dict(zip(SPECIES, conc)), dt))
        actual = tdev.step(torch.tensor(conc[None]), dt)[0].numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_species_order_is_explicit_and_channels_are_not_ethanol_copies():
    cfg = matched_config()
    names = ("ethanol", "ammonia", "nitrogen_dioxide", "carbon_dioxide", "carbon_monoxide")
    conc = torch.tensor([[80., 20., 2., 500., 10.]], dtype=torch.float64)
    a = build_device("scentience_v1", cfg.sensor_profile, 1, "cpu", randomize=False,
                     species_names=names, config=cfg)
    b = build_device("scentience_v1", cfg.sensor_profile, 1, "cpu", randomize=False,
                     species_names=names[::-1], config=cfg)
    for _ in range(20):
        x, y = a.step(conc, 1.), b.step(conc.flip(-1), 1.)
        torch.testing.assert_close(x, y, rtol=1e-13, atol=1e-12)
    assert x[0, 0] < 1 and x[0, 1] < 1 and x[0, 2] > 1
    assert x[0, 0] != x[0, 1]
    assert x[0, 6] > 420 and x[0, 9] > 0
    assert len(CHANNELS) == x.shape[1] == 11


def test_float32_matches_quantized_reference_with_declared_tolerance():
    cfg = matched_config(dtype="float32")
    ndev = ScentienceV1(config=cfg, randomize_unit=False)
    tdev = build_device("scentience_v1", cfg.sensor_profile, 1, "cpu", randomize=False,
                        species_names=SPECIES, config=cfg)
    conc = torch.tensor([[600., 1., 5., 1., 10., 20., 0., 0., 0.]])
    for _ in range(100):
        expected = ndev.observation_vector(ndev.step(dict(zip(SPECIES, conc[0].tolist())), 0.1))
        actual = tdev.step(conc, 0.1)[0].numpy()
        # One voltage LSB can create a larger ratio difference at high Rs.
        # This tolerance is for this trajectory, not a universal precision claim.
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-4)


@pytest.mark.parametrize("mode,value", [("absolute", 900.), ("excess", 500.)])
def test_co2_modes_have_no_double_counted_background(mode, value):
    cfg = matched_config()
    cfg = replace(cfg, co2=replace(cfg.co2, concentration_mode=mode,
                                  ambient_baseline_ppm=400.))
    nd = ScentienceV1(config=cfg, randomize_unit=False)
    td = build_device("scentience_v1", cfg.sensor_profile, 1, "cpu", randomize=False,
                      species_names=("carbon_dioxide",), config=cfg)
    expected_ppm = 400. + 500. * (1. - np.exp(-60./cfg.co2.tau63_s))
    nr = nd.step({"carbon_dioxide": value}, 60.)
    tr = td.step(torch.tensor([[value]], dtype=torch.float64), 60.)
    assert nr["co2_ppm"] == pytest.approx(expected_ppm)
    assert float(tr[0, 6]) == pytest.approx(expected_ppm)


@pytest.mark.parametrize("mode", ["absolute", "excess"])
def test_missing_co2_species_means_ambient_and_mapping_config_is_supported(mode):
    payload = {"dtype": "float64", "co2": {"concentration_mode": mode,
               "ambient_baseline_ppm": 400., "asc_enabled": False, "repeatability_ppm": 0.}}
    nd = ScentienceV1(config=payload, randomize_unit=False)
    td = build_device("scentience_v1", "packaged_slow", 1, "cpu", randomize=False,
                      species_names=("ethanol",), config=payload)
    assert nd.step({}, 60.)["co2_ppm"] == 400.
    assert td.step(torch.zeros(1, 1, dtype=torch.float64), 60.)[0, 6] == 400.
