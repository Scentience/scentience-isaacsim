"""SDK data contracts; optional real client tests replace only BLE transport."""
import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scentience_olfaction import OlfactionWorld
from scentience_olfaction.bridge import (
    COMPOUND_FIELDS, ble_frame, ovl_sensor_channels, ovl_sensor_window,
    readings_to_numpy, sample_numpy,
)
from scentience_olfaction.bridge.ble_schema import _invert_power_law
from scentience_olfaction.sensors.device_np import CHANNELS, DeviceConfig
from scentience_olfaction.sensors.mox import MOX_OX


def test_oxidizing_inverse_and_custom_stereo_calibration():
    A, beta = MOX_OX.sensitivity['nitrogen_dioxide']
    assert _invert_power_law(A * 20 ** -beta, A, beta) == pytest.approx(20)
    assert _invert_power_law(.8, A, beta) == 0
    cfg = DeviceConfig()
    channels = list(cfg.resolved_mox())
    channels[5] = replace(channels[5], sensitivity={'nitrogen_dioxide': (2.0, -.5)})
    cfg = replace(cfg, mox_channels=tuple(channels))
    reading = dict.fromkeys(CHANNELS, 1.)
    reading.update(chem_right_ox=10., co2_ppm=420., temperature_c=20., relative_humidity=50.)
    right = ble_frame(reading, side='right', device_config=cfg, include_sim_metadata=False)
    assert right['NO2'] == 25.
    assert 'NO2' not in ble_frame(reading, device_config=cfg)
    assert not any(k.startswith(('_sim', 'BATT', 'STATUS')) for k in right)
    json.dumps(right, allow_nan=False)


def test_array_order_missingness_and_multiple_devices():
    frames = [{'UID': 'A', 'NH3': 3., 'CO2': 420.}, {'UID': 'B', 'CO2': 550.}]
    out = readings_to_numpy(frames)
    assert out.shape == (2, 14) and out.dtype == np.float64
    np.testing.assert_array_equal(out[:, :2], [[420, 3], [550, 0]])
    assert np.isnan(readings_to_numpy(frames, missing=np.nan)[1, 1])
    assert readings_to_numpy([]).shape == (0, len(COMPOUND_FIELDS))
    assert readings_to_numpy(frames[0]).shape == (1, 14)
    np.testing.assert_array_equal(
        sample_numpy(SimpleNamespace(sample_ble=lambda: frames), fields=('NH3', 'CO2')),
        [[3, 420], [0, 550]])


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf'), 'invalid'])
def test_bad_present_values_are_not_silently_zeros(value):
    with pytest.raises(ValueError):
        readings_to_numpy({'CO2': value})
    with pytest.raises(ValueError):
        ovl_sensor_window({'CO2': value})


def test_world_sdk_reading_ticks_once_and_replays():
    a, b = OlfactionWorld.simple(seed=31), OlfactionWorld.simple(seed=31)
    a.step(.037)
    b.step(.037)
    with pytest.raises(ValueError, match='uid'):
        a.read_ble((1, 0, 1), uid='')
    raw = b.read((1, 0, 1), heading=.3)
    frame = a.read_ble((1, 0, 1), heading=.3)
    expected = ble_frame(raw, timestamp=frame['TIMESTAMP'])
    assert {k: v for k, v in frame.items() if k != '_sim_time_s'} == expected
    assert frame['TIMESTAMP'] == '1970-01-01T00:00:00.037000Z'
    assert frame['_sim_time_s'] == .037
    assert a.read((1, 0, 1)) == b.read((1, 0, 1))  # no double tick in conversion
    a.reset(seed=31)
    a.step(.037)
    assert a.read_ble((1, 0, 1), heading=.3) == frame
    with pytest.raises(ValueError, match='scentience_v1'):
        OlfactionWorld.simple(device_profile='scd41').read_ble((1, 0, 1))


def test_ovl_delegates_to_installed_sdk_without_inference(monkeypatch):
    sdk = pytest.importorskip('scentience')
    frames = [{'UID': 'A', 'NO2': 2., 'C2H5OH': 3., 'C3H8': 5., 'C4H10': 7.}]
    mapper = sdk.OVLClient.device_reading_to_ovl
    calls = []

    def convert(row):
        calls.append(row)
        return mapper(row)

    monkeypatch.setattr(sdk.OVLClient, 'device_reading_to_ovl', staticmethod(convert))
    np.testing.assert_array_equal(ovl_sensor_window(frames), [[2, 3, 0, 0, 3, 12]])
    assert calls == frames
    assert ovl_sensor_channels() == tuple(sdk.OVL_SENSOR_CHANNELS)
    with pytest.raises(ValueError, match='one UID'):
        ovl_sensor_window([frames[0], {'UID': 'B'}])
    with pytest.raises(ValueError, match='at least one'):
        ovl_sensor_window([])


@pytest.mark.parametrize('multi', [False, True])
def test_real_sdk_sampling_with_fake_gatt_transport(multi):
    sdk = pytest.importorskip('scentience')
    pytest.importorskip('bleak')
    world = OlfactionWorld.simple()
    world.step(.05)
    frames = [world.read_ble((1, 0, 1), name=n, uid=n) for n in ('A', 'B')]

    class Gatt:
        is_connected = True

        def __init__(self, frame):
            self.raw = json.dumps(frame, allow_nan=False).encode()

        async def read_gatt_char(self, uuid):
            return self.raw

    # Bypass BLE connection/thread creation; execute the real public sampling,
    # JSON decoder and logging paths against an in-memory GATT byte source.
    device = object.__new__(sdk.ScentienceDevice)
    selected = frames if multi else frames[:1]
    device._clients = {f['UID']: Gatt(f) for f in selected}
    device._char_uuid, device._multi, device._log = 'test-characteristic', multi, []
    device._submit = asyncio.run
    np.testing.assert_array_equal(sample_numpy(device), readings_to_numpy(selected))
    assert device.log == selected


def test_core_bridge_import_and_arrays_work_without_sdk():
    code = '''
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('scentience', 'bleak', 'torch', 'warp'):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockOptional())
from scentience_olfaction import OlfactionWorld
from scentience_olfaction.bridge import readings_to_numpy, ovl_sensor_window
world = OlfactionWorld.simple()
world.step(.05)
assert readings_to_numpy(world.read_ble((1,0,1))).shape == (1,14)
try:
    ovl_sensor_window({'CO2':420})
except ImportError as error:
    assert 'scentience-olfaction[bridge]' in str(error)
else:
    raise AssertionError('SDK was expected to be absent')
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
