"""Manufacturer profiles, calibrated current readout and the world factory contract."""

from dataclasses import asdict, replace
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from scentience_olfaction.sensors.co2_sensor import (
    CO2Channel, CO2Config, SCD30, SCD40, SCD41, co2_config,
)
from scentience_olfaction.sensors.emstat_pico import (
    ElectrochemicalCellConfig, EmStatPicoChannel, EmStatPicoChannelConfig, EmStatPicoConfig,
)
from scentience_olfaction.sensors.factory import create_sensor


def cell_channel(**overrides):
    cell = ElectrochemicalCellConfig({"test_analyte": 10.0}, "synthetic test calibration", tau_s=0)
    return EmStatPicoChannelConfig(cell=cell, **overrides)


@pytest.mark.parametrize("profile,tau,interval", [(SCD30, 20, 2), (SCD40, 60, 5), (SCD41, 60, 5)])
def test_co2_profile_response_and_hold(profile, tau, interval):
    cfg = co2_config(profile, repeatability_ppm=0, resolution_ppm=0)
    channel = CO2Channel(cfg, np.random.default_rng(1))
    assert channel.step(1420, interval / 2)["co2_ppm"] == 420
    channel.step(1420, interval / 2)
    out = channel.step(1420, tau - interval)
    assert out["co2_ppm"] == pytest.approx(420 + 1000 * (1 - math.exp(-1)))
    channel.reset()
    assert channel.step(1420, 0) == {"co2_ppm": 420, "asc_offset_ppm": 0}


def test_co2_clock_preserves_residual_and_samples_at_boundaries():
    cfg = co2_config(SCD30, repeatability_ppm=0)
    channel = CO2Channel(cfg, np.random.default_rng(1))
    at_two = 420 + 1000 * (1 - math.exp(-2 / 20))
    assert channel.step(1420, 3)["co2_ppm"] == pytest.approx(at_two)
    assert channel.step(1420, .5)["co2_ppm"] == pytest.approx(at_two)
    assert channel.step(1420, .5)["co2_ppm"] == pytest.approx(
        420 + 1000 * (1 - math.exp(-4 / 20)))
    a = CO2Channel(co2_config(SCD30), np.random.default_rng(9))
    b = CO2Channel(co2_config(SCD30), np.random.default_rng(9))
    whole = a.step(1420, 10.5)
    for _ in range(105):
        pieces = b.step(1420, .1)
    assert whole == pytest.approx(pieces)
    assert a.step(1420, 1.5) == pytest.approx(b.step(1420, 1.5))


def test_co2_concentration_modes_factory_and_legacy_keywords():
    legacy = CO2Channel(CO2Config(repeatability_ppm=0, asc_enabled=False), np.random.default_rng(0))
    absolute = CO2Channel(co2_config(SCD41, repeatability_ppm=0, resolution_ppm=0),
                          np.random.default_rng(0))
    assert legacy.step(co2_excess_ppm=500, dt=60) == pytest.approx(absolute.step(920, 60))
    factory = create_sensor("scd41", config={"repeatability_ppm": 0, "resolution_ppm": 0})
    assert factory.channel_names == ("co2_ppm",)
    assert factory.step({}, 5)["co2_ppm"] == 420
    explicit = create_sensor("scd41", config=co2_config(SCD41, repeatability_ppm=0))
    assert explicit.step({}, 5)["co2_ppm"] == 420
    explicit.channel.reset()
    assert explicit.channel.step_absolute(420, 5)["co2_ppm"] == 420


@pytest.mark.parametrize("ppm,bound", [(399, None), (400, 60), (1000, 75),
                                      (1000.5, 80.015), (2000, 110),
                                      (2000.5, 140.025), (5000, 290), (5001, None)])
def test_scd41_accuracy_is_piecewise_and_range_limited(ppm, bound):
    result = co2_config(SCD41).accuracy_bound_ppm(ppm)
    assert result is None if bound is None else result == pytest.approx(bound)


def test_co2_bias_quantization_and_output_range_are_distinct_from_accuracy():
    cfg = co2_config(SCD41, repeatability_ppm=0, resolution_ppm=0, accuracy_bias_fraction=.5)
    channel = CO2Channel(cfg, np.random.default_rng(0))
    assert channel.step(420, 5)["co2_ppm"] == pytest.approx(450.25)
    cfg = co2_config(SCD41, repeatability_ppm=0, tau63_s=.001)
    channel = CO2Channel(cfg, np.random.default_rng(0))
    assert channel.step(3000.4, 5)["co2_ppm"] == 3000  # 2000 is not an output cap
    assert channel.step(50000, 5)["co2_ppm"] == 40000
    assert channel.step(0, 5)["co2_ppm"] == 0
    assert co2_config(SCD30).accuracy_bound_ppm(10000) == 330
    assert co2_config(SCD40).accuracy_bound_ppm(2000) == 150
    assert co2_config(SCD41, accuracy_base_ppm=20, accuracy_frac=.01).accuracy_bound_ppm(3000) == 50


def test_low_power_is_cadence_not_response_time():
    cfg = co2_config(SCD41, low_power=True, repeatability_ppm=0, resolution_ppm=0)
    channel = CO2Channel(cfg, np.random.default_rng(0))
    assert channel.step(1420, 29)["co2_ppm"] == 420
    assert channel.step(1420, 1)["co2_ppm"] == pytest.approx(420 + 1000 * (1 - math.exp(-.5)))
    with pytest.raises(ValueError):
        co2_config(SCD30, low_power=True)


def test_emstat_cell_calibration_temperature_and_electronics():
    cell = ElectrochemicalCellConfig({"test_analyte": 10, "interferent": -3}, "test",
                                    tau_s=0, zero_current_na=5, span_tempco_per_k=.01,
                                    zero_tempco_na_per_k=2)
    cfg = EmStatPicoChannelConfig(cell, electrode_count=2, resolution_na=.01,
                                 gain_error_fraction=.1, offset_na=1)
    channel = EmStatPicoChannel(cfg, np.random.default_rng(0))
    reading = channel.step({"test_analyte": 2, "interferent": 1}, .1, temp_c=30)
    assert reading["signal_na"] == pytest.approx(18.7)
    assert reading["current_na"] == pytest.approx(49.07)
    assert reading["current_na"] == reading["counts"] * cfg.quantum_na
    assert not reading["saturated"]
    held = channel.step({}, .05)
    assert held["current_na"] == reading["current_na"]
    channel.reset()
    assert channel.step({}, 0)["current_na"] == 0


def test_emstat_lag_and_seeded_sampling_are_partition_invariant():
    cell = ElectrochemicalCellConfig({"test_analyte": 10}, "test", tau_s=2)
    cfg = EmStatPicoChannelConfig(cell, resolution_na=.001)
    channel = EmStatPicoChannel(cfg, np.random.default_rng(0))
    assert channel.step({"test_analyte": 2}, 2)["current_na"] == pytest.approx(
        20 * (1 - math.exp(-1)), abs=.001)
    cfg = replace(cfg, cell=replace(cell, noise_na=.02, drift_na_per_sqrt_s=.01), noise_na=.01)
    a, b = [EmStatPicoChannel(cfg, np.random.default_rng(10)) for _ in range(2)]
    whole = a.step({"test_analyte": 2}, 1.05)
    for _ in range(105):
        pieces = b.step({"test_analyte": 2}, .01)
    assert whole == pytest.approx(pieces)


@pytest.mark.parametrize("current_range,concentration,limit", [(100, 20, 100),
                                                            (100, -20, 100),
                                                            (5_000_000, 400000, 3_000_000)])
def test_emstat_quantization_and_bipolar_saturation(current_range, concentration, limit):
    cfg = cell_channel(current_range_na=current_range)
    if concentration < 0:
        cfg.cell.sensitivity_na_per_ppm["test_analyte"] *= -1
    reading = EmStatPicoChannel(cfg, np.random.default_rng(0)).step(
        {"test_analyte": abs(concentration)}, .1)
    assert reading["saturated"]
    assert abs(reading["current_na"]) <= limit
    assert abs(reading["current_na"]) == pytest.approx(limit, abs=cfg.quantum_na)
    assert reading["current_na"] == reading["counts"] * cfg.quantum_na
    assert np.sign(reading["current_na"]) == np.sign(concentration)
    assert cell_channel().quantum_na == .0055


def test_factory_dual_cells_have_independent_inputs_state_and_random_streams():
    one = cell_channel(electrode_count=2, noise_na=.1)
    two = cell_channel(electrode_count=3, noise_na=.1)
    dual_config = EmStatPicoConfig((one, two))
    a, b = [create_sensor("emstat_pico", seed=8, config=dual_config) for _ in range(2)]
    assert a.channel_names == ("ec1", "ec2")
    for _ in range(5):
        first = a.step({"test_analyte": 2}, .1, conc_ppm_2={"test_analyte": 3})
        second = b.step({"test_analyte": 2}, .1, conc_ppm_2={"test_analyte": 7})
        assert first["ec1"] == second["ec1"]
        assert first["ec2"] != second["ec2"]
    np.testing.assert_array_equal(a.observation_vector(first), [first["ec1"], first["ec2"]])
    a.reset()
    assert a.step({}, 0) == {"ec1": 0, "ec2": 0}
    single = create_sensor("emstat_pico", config=EmStatPicoConfig((one,)))
    assert single.channel_names == ("ec1",)


def test_factory_json_roundtrip_and_scentience_configuration():
    from scentience_olfaction.sensors.device_np import CHANNELS, DeviceConfig, ScentienceV1

    configs = [("emstat_pico", EmStatPicoConfig((cell_channel(),))),
               ("scd30", co2_config(SCD30)), ("scentience_v1", DeviceConfig())]
    for profile, config in configs:
        typed = create_sensor(profile, config=config)
        mapped = create_sensor(profile, config=json.loads(json.dumps(asdict(config))))
        assert typed.step({}, .1) == mapped.step({}, .1)
    wrapped = create_sensor()
    assert wrapped.channel_names == CHANNELS
    assert wrapped.step({}, .1) == ScentienceV1(seed=0).step({}, .1)
    fast = create_sensor(sensor_profile="fast_modulated", config={"white_noise_frac": 0})
    assert fast.device.sensor_profile == "fast_modulated"
    fixed = create_sensor(randomize=False)
    expected = ScentienceV1(seed=0, randomize_unit=False).step({}, .1)
    assert fixed.step({}, .1) == expected


def test_factory_diagnostics_expose_saturation_and_are_copied():
    sensor = create_sensor("emstat_pico", config=EmStatPicoConfig((cell_channel(),)))
    sensor.step({"test_analyte": 20}, .1)
    diagnostics = sensor.last_reading
    assert diagnostics["ec1"]["saturated"]
    assert diagnostics["ec1"]["unclipped_current_na"] == 200
    diagnostics["ec1"]["saturated"] = False
    assert sensor.last_reading["ec1"]["saturated"]
    sensor.reset()
    assert sensor.last_reading == {}


@pytest.mark.parametrize("profile", ["scd30", "scd40", "scd41", "scentience_v1"])
def test_world_background_is_not_added_twice(profile):
    from scentience_olfaction.api import OlfactionWorld
    from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig

    config = {"repeatability_ppm": 0}
    if profile == "scentience_v1":
        config = {"co2": config}
    plume = FilamentPlume(FilamentPlumeConfig(emitters=[],
                          background_ppm={"carbon_dioxide": 420}), seed=0)
    world = OlfactionWorld(plume, device_profile=profile, device_config=config)
    assert world.read((0, 0, 1), dt=60)["co2_ppm"] == 420


@pytest.mark.parametrize("profile,config", [
    ("scd30", {"typo": 1}), ("scd41", {"sample_interval_s": 0}),
    ("scd41", {"asc_enabled": "false"}), ("scd41", {"accuracy_frac": float("nan")}),
    ("scd41", {"repeatability_ppm": True}), ("emstat_pico", None),
    ("emstat_pico", {"channels": []}),
    ("emstat_pico", {"channels": [{"cell": {"calibration_id": "test", "typo": 1}}]}),
    ("scentience_v1", {"co2": {"tau63_s": "60"}}),
])
def test_factory_rejects_invalid_config(profile, config):
    with pytest.raises((ValueError, TypeError)):
        create_sensor(profile, config=config)


def test_emstat_rejects_unsupported_hardware_and_invalid_input():
    for overrides in ({"electrode_count": 4}, {"current_range_na": 1000},
                      {"sample_interval_s": .001}, {"max_current_na": 5_000_000},
                      {"bias_v": 3}, {"resolution_na": 0}):
        with pytest.raises(ValueError):
            cell_channel(**overrides)
    with pytest.raises(ValueError):
        EmStatPicoConfig((cell_channel(), cell_channel(mode="high_speed")))
    channel = EmStatPicoChannel(cell_channel(), np.random.default_rng(0))
    for conc, dt in [({"test_analyte": -1}, .1), ({}, -1), ({}, float("nan"))]:
        with pytest.raises(ValueError):
            channel.step(conc, dt)
    co2 = CO2Channel(co2_config(SCD41), np.random.default_rng(0))
    for conc, dt in [(-1, .1), (420, -1), (float("nan"), 1)]:
        with pytest.raises(ValueError):
            co2.step(conc, dt)


def test_optional_factory_does_not_import_scentience_torch_or_vendor_sdks():
    code = """
import sys
import importlib
# Package initialization imports DeviceState through api.py. Isolate the
# factory's own import/creation boundary from those package-level imports.
import scentience_olfaction
sys.modules.pop('scentience_olfaction.sensors.device_np', None)
from scentience_olfaction.sensors import factory
importlib.reload(factory)
factory.create_sensor('scd30')
assert 'scentience_olfaction.sensors.device_np' not in sys.modules
assert 'torch' not in sys.modules
assert 'warp' not in sys.modules
assert not any(name.lower().startswith(('palmsens', 'pspython')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
