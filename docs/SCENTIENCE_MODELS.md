# Scentience sensor models: contract and scientific limits

The scalar NumPy and batched torch devices use the same `DeviceConfig`,
`MoxChannelConfig`, `CO2Config`, and `ECChannelConfig`. Both model the six
MOX readouts, CO₂, ambient temperature/humidity, and two electrochemical
currents. Schema compatibility and deterministic numerical agreement are
software properties; neither establishes accuracy against physical hardware.

## Device API

```python
from scentience_olfaction.sensors.device_np import DeviceConfig, ScentienceV1
from scentience_olfaction.sensors.scentience_v1 import build_device, CHANNELS

species = ("ethanol", "ammonia", "nitrogen_dioxide", "carbon_dioxide")
config = DeviceConfig.from_dict({
    "sensor_profile": "packaged_slow",
    "dtype": "float32",
    "co2": {"concentration_mode": "absolute", "ambient_baseline_ppm": 400.0},
})
scalar = ScentienceV1(seed=7, randomize_unit=False, config=config)
batch = build_device(
    "scentience_v1", "packaged_slow", n_envs=32, device="cpu",
    randomize=False, species_names=species, seed=7, config=config,
)
```

The batched signature is:

```python
build_device(device_profile, sensor_profile, n_envs, device,
             randomize=True, *, species_names=("ethanol",), seed=0, config=None)
device.step(conc_ppm, dt, conc_ppm_2=None, env_ids=None)
device.reset(env_ids=None, randomize=False)
```

`species_names` must be a nonempty tuple of unique nonempty names. Columns are
matched by name, never assumed to be ethanol. A species without a configured
sensitivity contributes zero to that sensor's target; this means unmodeled
sensitivity, not demonstrated physical blindness.

Concentrations are finite nonnegative torch tensors `(K, S)` on the declared
device and in the configured dtype. `S = len(species_names)`; `K = n_envs` for
all environments, or `len(env_ids)` for a subset. Subset rows and output rows
follow the supplied index order. `env_ids` accepts unique integer indices, a
one-dimensional torch boolean mask of length `n_envs`, or a slice. An empty
selection returns `(0, 11)` and changes nothing. Full-batch input with a
smaller selection is rejected. `dt` is a finite positive scalar in seconds;
per-environment time intervals must be grouped by the caller.

Left concentrations drive left MOX, CO₂, and both EC channels. Optional
`conc_ppm_2` drives only right MOX; omission gives both triplets the same air.
Subset steps and resets change only selected environments, including their
clocks, held samples, delay history, drift and random generators.

NumPy retains its dictionary API:

```python
scalar.step(conc_ppm, dt, state=None, conc_ppm_2=None)  # dict in CHANNELS order
scalar.observation_vector(reading)                   # float64 array, shape (11,)
scalar.reset(randomize=None)
```

`DeviceState(temp_c=20, rh_pct=50, flow_mps=0.3, heater_level=1)` overrides
NumPy ambient conditions per step. Without a state argument, both backends
use the ambient conditions in `DeviceConfig`. The low-level `MoxChannel`
default flow remains zero; comparing it to a full device requires matching
flow explicitly.

| Indices | Channels | Units / interpretation |
| --- | --- | --- |
| 0–2 | `chem_left_red`, `chem_left_nh3`, `chem_left_ox` | Reconstructed resistance ratios |
| 3–5 | `chem_right_red`, `chem_right_nh3`, `chem_right_ox` | Same, from right input |
| 6 | `co2_ppm` | Held absolute CO₂ reading in ppm |
| 7–8 | `temperature_c`, `relative_humidity` | Configured °C and %RH; no temperature/RH sensor dynamics |
| 9–10 | `ec1`, `ec2` | Current in nA, including configured zero and drift |

## CO₂ input convention

`config.co2.concentration_mode` controls both backends:

- `"excess"` (legacy default): add `ambient_baseline_ppm` to plume CO₂.
- `"absolute"`: input already includes background; add nothing.

Missing CO₂ means ambient air in either mode. A supplied absolute value of
zero means zero, not missing data. This distinction prevents double-counting
background from plume fields that already return total concentration.
The device does not infer the plume's convention: integration layers must
configure it consistently. Explicit calibration is never silently rewritten.

The implementation follows the CO₂ module's sample-boundary integration,
residual sample time, optional simplified ASC, systematic accuracy-bias
fraction, resolution and output limits. Large steps can cross multiple
sampling/ASC boundaries. Manufacturer profiles are selected through
`co2_config(...)` and passed as `DeviceConfig(co2=...)`; they retain their
configured absolute/excess convention. See `SENSOR_PROFILES.md` for the CO₂
and optional EmStat models maintained separately.

## Calibration and defaults

`DeviceConfig` is defined in `sensors/device_np.py` and re-exported by
`sensors/scentience_v1.py`, so NumPy configuration does not import torch.
Both device entry points accept this object or a mapping.
`DeviceConfig.from_dict(mapping)` decodes nested `mox_channels`, `co2`, and
`ec_channels`, validates parameters, and rejects unknown keys.

| Configuration | Meaning |
| --- | --- |
| `mox_channels=None` | Six independent copies of RED/NH3/OX defaults in schema order |
| `mox_channels=(...)` | Exactly six complete `MoxChannelConfig` calibrations |
| `ec_channels=(EC_CO, EC_H2S)` | Two generic illustrative amperometric cells; independently configurable |
| `co2=CO2Config()` | Legacy 60 s lag, 5 s sampling, 420 ppm baseline and simplified ASC |
| `ambient_temp_c`, `ambient_rh` | 20 °C and 50 %RH |
| `flow_mps`, `heater_level` | 0.3 m/s and 1.0 |
| `ratio_feature` | `"ratio_measured"` by default; `"ratio_baseline"` exposes baseline drift |
| `dtype` | Torch `"float32"` by default; optional `"float64"`. NumPy uses float64 |

The profile applies only when `mox_channels` is absent. Explicit channel
calibrations are used as supplied, including their time constants, ADC,
load resistance and species sensitivities. Legacy scalar MOX overrides
`r0_range`, `rs_r0_clean_air`, `drift_sigma_per_sqrt_s`, `white_noise_frac`,
`flicker_noise_frac`, and `humidity_coeff` are optional (`None` means leave
channel settings alone), and apply last to all six channels.
The factory's `sensor_profile` must agree with an explicitly supplied config.
Models copy device configuration at construction; modifying the original
object cannot alter a running device or module defaults. Rebuild to apply a
new calibration; direct mutation of internal model configuration is unsupported.

## MOX science and readout

For a calibrated species, the target ratio uses `r = A * C**(-beta)`.
Reducing species (`beta > 0`) subtract `max(base - r, 0)` from the clean-air
ratio. Oxidizing species (`beta < 0`) add `max(r - base, 0)`. Contributions
are combined in resistance space and floored at `1e-3` before environmental
modulation. This mixture rule is an empirical approximation, not a validated
competitive adsorption model. Species at or below `1e-9` ppm are treated as
absent by the inherited power-law model.

Absolute humidity, the optional humidity/concentration cross term, and the
optional Arrhenius temperature term modulate the target in log resistance.
Magnus humidity is an approximation; accepting an input does not establish
calibration at that temperature or humidity.

The lag uses an exact exponential update for input held constant over each
step. `tau_rise_s` means movement in the exposure direction: decreasing
resistance for RED/NH3 and increasing resistance for OX. `tau_fall_s` means
recovery. Polarity is inferred from sensitivity exponents. A calibration
with both reducing and oxidizing exponents must set `response_polarity` to
`-1` or `+1`: a single lag state cannot identify separate opposing kinetics.
Flow and heater settings rescale the configured time constant.

Transport delay evaluates the historical lag trajectory at `t - delay`,
using its piecewise exponential segments. Pre-reset history is the configured
clean-air ratio. No early exposure leaks through during the first delay
interval. History uses timestamps, so changing `dt` does not reinterpret a
queue's sample count as time. The delay remains after the lag, preserving the
existing model's signal-chain order.

Multiplicative log-baseline drift, white noise, and a four-pole AR(1) flicker
approximation precede the divider. Voltage is quantized with
`q = v_ref / 2**adc_bits`, integer counts are floored and clipped to the ADC
range, then resistance is reconstructed from quantized voltage. Quantizing
concentration or reporting an unquantized target would omit this readout loss.
The inherited 1 Ω resistance floor and half-LSB reconstruction denominator at
zero counts are numerical regularizations, not hardware specifications.

Low-level readings expose `rs_true`, `rs_measured`, `counts`, `volts`,
`r0_current`, and both ratios:

- `ratio_measured = rs_measured / r0_current` retains the legacy feature.
  It uses a simulated instantaneous baseline, generally unavailable to real
  firmware. Drift cancels ideally, but can still change quantization/clipping.
- `ratio_baseline = rs_measured / r0_at_episode_reset` exposes drift relative
  to a fixed calibration. Select it explicitly for device channel output.

NumPy keeps these diagnostics in `last_mox_readings`, a tuple of six reading
dictionaries, empty after reset. Torch uses a dictionary of `(n_envs, 6)`
tensors with the same keys; unstepped/reset rows are NaN (counts: `-1`).
Diagnostics are model internals, not a hardware promise or additional policy
observation unless the caller deliberately includes them.

## Evidence and fidelity

[SGX's MiCS-6814 documentation](https://www.sgxsensortech.com/sensor/mics-6814)
describes three separate sensing dies. The inherited RED/NH3/OX resistance
ranges and distinct load resistances remain separate. The power-law
coefficients are illustrative inversions of digitized driver fits, with
some synthesized cross terms; they are not per-unit Scentience calibration.
The legacy provenance registry records source references, not certification
of the whole model or a subsequently supplied calibration.

The packaged 3 s onset / 12 s recovery defaults are assumptions. The fast
profile retains the existing 38 ms / 46 ms lag, 5 ms delay, and 24-bit ADC
preset inspired by
[Dennler et al., *High-speed odour sensing using miniaturised electronic nose*](https://arxiv.org/abs/2406.01904).
Those experimental conditions do not establish the response of a packaged
Scentience device. Converting reported onset/recovery times to these
first-order constants is a modeling choice. No universal whiff-retention
percentage is implied by either profile.

EC concentration mixing is linear and preserves signed cross sensitivities,
including negative NO₂ response on the illustrative H₂S channel. Current
includes membrane lag, span and zero temperature coefficients, drift and
noise. The default generic EC calibration is not validated against a specific
ionic-liquid cell or EmStat acquisition. `cottrell_current` is a separate
ideal potential-step helper with a legacy 1 µs time floor; the ordinary EC
step response is not a Cottrell transient. Electrode kinetics, double-layer
charging and front-end saturation require additional models.

PID remains a standalone model, not an extra channel in the 11-channel
Scentience device. Its empirical correction-factor table is retained from
[RAE TN-106, revised August 2010](https://clu-in.org/download/contaminantfocus/dnapl/Detection_and_Site_Characterization/TN-106_Correction_Factors.pdf).
For ethanol at 10.6 eV, CF is 3.1; 10.47 is its ionization energy, not the
missing 11.7 eV CF. An absent table entry is uncalibrated in this model,
not proof that the species is physically invisible. Published weak responses
are preserved instead of applying an idealized lamp-energy cutoff to the
empirical table.

## Reproducibility, precision and verification

Torch uses a private generator for each environment, seeded by root seed and
environment index. Randomized resets draw each die's R₀ log-uniformly from
its configured range. No undocumented A/beta perturbations are added.
Batch size, unrelated subset steps/resets, and subset ordering do not change
a given environment's random stream. The global torch RNG is untouched.

`reset(randomize=False)` restores nominal R₀ and clears dynamics without
rewinding random streams. `True` samples fresh R₀ from that environment's
continuing stream. Recreate the same backend/device with the same seed and
call sequence to replay. NumPy's argument-free reset retains its original
constructor randomization choice; pass `randomize=False` to restore nominal
R₀. Randomization flags affect unit variation, not configured noise or drift.

NumPy and torch use different random generators. Equal seeds do not promise
identical noisy trajectories across backends or CPU/CUDA/MPS. Deterministic
parity disables all white/flicker noise, baseline drift, CO₂ repeatability,
and EC noise/drift, and uses nominal R₀ and matched configuration. Matching
only the primary ethanol channel's asymptote is insufficient.

Torch rejects input device/dtype mismatches instead of silently casting.
Float64 is explicit; MPS rejects it rather than falling back to CPU/float32.
Unsupported random-generator backends also fail rather than silently using
a host generator. Float32 with an ADC of 24 bits or higher emits a precision
warning. Near ADC thresholds, roundoff can change an integer count; at high
resistance, one count can cause a substantial reconstructed-ratio difference.
There is no universal cross-backend tolerance or exact ADC-precision claim.

Signal arithmetic is implemented in torch. Validation, private RNG dispatch,
sample scheduling, and nonzero-delay history require Python bookkeeping and
can synchronize an accelerator. This implementation is not advertised as a
fused GPU kernel or as having no host round trips. GPU throughput and
cross-device numerical behavior require separate measurements.

The focused regression tests cover analytical MOX polarity/lag, real delay
with variable steps and pulsed input, accessible drift, calibration isolation,
invalid-input rejection, per-environment reset/RNG isolation, species ordering,
stereo input, CO₂ mode/ambient semantics, sample timing, and full deterministic
NumPy/torch channel and ADC-count parity. Float32 is tested separately from
float64 with a declared trajectory-specific tolerance. These are numerical
model checks, not hardware validation.

```sh
.venv/bin/python -m pytest tests/test_sensor_contract.py tests/test_device_parity.py \
    tests/test_sensors.py tests/test_sensor_math.py -q
.venv/bin/python -m pytest -m 'not isaac' -q
```
