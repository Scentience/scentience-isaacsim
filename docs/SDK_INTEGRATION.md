# Scentience SDK and NumPy workflows

Install the optional Python integration from the repository root:

```bash
python -m pip install -e ".[bridge]"
python examples/07_scentience_sdk.py --ovl
```

The extra selects `scentience>=2.2.2,<3`. The example runs locally without a
Bluetooth device, API key, inference request or model download. The core
simulation and dictionary/array conversions still require only NumPy.

## Simulated readings

```python
from scentience_olfaction import OlfactionWorld
from scentience_olfaction.bridge import COMPOUND_FIELDS, readings_to_numpy

world = OlfactionWorld.simple(seed=7)
world.step(0.05)
frame = world.read_ble((2.0, 0.0, 1.0), uid="SIM001", heading=0.0)
values = readings_to_numpy(frame)  # float64 [1, 14], columns = COMPOUND_FIELDS
```

`read_ble` advances one named sensor exactly as `read` does; use one or the
other per capture. Its default timestamp is the plume's elapsed time relative
to 1970-01-01 UTC, with microsecond resolution. This synthetic epoch is not a
wall-clock acquisition date. Pass `timestamp=` when aligning another clock.
Step transport before each capture. Seeded resets replay timestamps and noise.

To convert a reading you already captured, including direct NumPy device use:

```python
from scentience_olfaction.sensors.device_np import DeviceConfig, ScentienceV1
from scentience_olfaction.bridge import ble_frame

device = ScentienceV1(config=DeviceConfig(), seed=7)
reading = device.step({"ethanol": 10.0}, dt=0.05)
frame = ble_frame(reading, device_config=device.cfg, timestamp="2026-09-09T12:00:00Z")
```

Pass the actual calibration to `ble_frame`; `world.read_ble` supplies it
automatically. `side="left"` is the default; select `"right"` for the other MOX
die. Convert the same captured reading twice when exporting both sides, with
separate UIDs so temporal windows stay independent. CO₂ and environmental
channels remain the shared device channels. Standalone SCD/EmStat profiles
retain their own schemas and do not impersonate Scentience hardware.

The SDK reports compounds while this simulator exposes 11 instrument channels.
The bridge performs a static primary-analyte inversion for RED → ethanol,
NH₃ → ammonia and OX → NO₂, including the opposite OX response polarity.
This is an equivalent-concentration feature: lag, drift, environmental
response and cross-sensitivity can bias it. It does not reproduce proprietary
firmware or recover the composition of a mixture. A custom MOX calibration
without the respective primary analyte omits that compound. EC current cannot
be identified as a unique compound and remains in the raw instrument reading.

Frames label their simulated compound units and attribution using `_sim_*`
metadata. Use `include_sim_metadata=False` for consumers that accept only SDK
fields, and retain provenance separately. The bridge omits zero compounds after
rounding and does not invent battery or operational status measurements. It
creates dictionaries; it is not a BLE peripheral or network server.

## Hardware and recorded readings

Use the real [PyPI SDK](https://pypi.org/project/scentience/) to own connections,
credentials, device discovery and streaming:

```python
import os
import scentience as scn
from scentience_olfaction.bridge import readings_to_numpy

with scn.ScentienceDevice(api_key=os.environ["SCENTIENCE_API_KEY"]) as device:
    device.connect_ble(char_uuid=os.environ["SCENTIENCE_CHAR_UUID"])
    packets = device.sample_ble()
    values = readings_to_numpy(packets)
```

`sample_numpy(device)` combines those last two lines when the original UID and
timestamp are not needed. Both converters return `[N, F]`, including `[1, F]`
for one device. Multi-device packet order is preserved; retain the packets to
associate rows with UIDs. `readings_to_numpy(json.load(file))` also accepts SDK
JSON exports. Pass `fields=("CO2", "ENV_temperatureC")` for another explicit
column selection. No code sorts a changing set of packet keys into features.

Missing compounds default to zero according to the documented BLE omission
rule. Set `missing=np.nan` to preserve missingness instead. Present invalid
values raise. Conversion retains numeric units from the source; the hardware
schema does not declare compound units, so confirm them for your firmware
before comparing hardware values to simulated ppm. These compound arrays are
not interchangeable with the 11-channel raw observation used by the Gym task.

## OVL preprocessing

```python
from scentience_olfaction.bridge import ovl_sensor_channels, ovl_sensor_window

window = ovl_sensor_window(frames)  # ordered frames from one UID, [T, 6]
channel_names = ovl_sensor_channels()
# In an application that opts into hosted inference:
# result = ovl_client.embed_sensor(window.tolist())
```

The converter calls the installed SDK's public
`OVLClient.device_reading_to_ovl` and reads `OVL_SENSOR_CHANNELS` directly.
It performs no inference or I/O. Split multi-device streams by UID and order
each stream by acquisition time before calling it; mixed UIDs are rejected.
The SDK describes its device-to-encoder mapping as experimental. The bridge
preserves that mapping, and does not establish calibration or inference quality.

`ColipModel` uses a different 138-value descriptor input. Raw instrument or
compound arrays are not that descriptor; no automatic zero-padding adapter is
provided. Model weights and hosted inference are separate application choices.
`run_metadata` records the installed `scentience` version with experiment logs.

## Other Scentience SDKs

| Ecosystem | Package reference | Use with this repository |
|---|---|---|
| Python | [PyPI scentience](https://pypi.org/project/scentience/) | Tested optional NumPy/SDK integration |
| Rust | [Scentience crates](https://crates.io/search?q=scentience) | JSON compound dictionaries at an application boundary |
| C++ | [Conan package](https://scentience.jfrog.io/ui/packages/conan:%2F%2Fscentience/) | JSON compound dictionaries at an application boundary |
| JavaScript / React Native | [NPM scentience](https://www.npmjs.com/package/scentience) | JSON compound dictionaries at an application boundary |

The [Scentience BLE API documentation](https://scentience.github.io/docs-api/ble-api)
describes the shared payload. This repository links the native SDKs without
adding language runtimes, bindings or claiming their connection APIs were tested.
Choose and validate a transport in your robot/application when crossing languages.

## Verification

The integration was checked against the actual PyPI 2.2.2 wheel. Focused tests
exercise its public sampling, JSON decoding and logging with in-memory GATT
transport, its local OVL mapping, NumPy ordering, custom calibration, simulation
replay and missing-SDK behavior. They do not establish a live BLE connection.

```bash
python -m pytest -q tests/test_ble_bridge.py tests/test_sdk_bridge.py
```

The SDK wheel contains an Apache-2.0 license; see
[dependency provenance](LICENSES_AND_PROVENANCE.md). Existing repository and
sensor model license files remain unchanged.
