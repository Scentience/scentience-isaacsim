# Optional manufacturer sensor profiles

These are standalone NumPy **simulation models**. No device connection,
manufacturer firmware, MethodSCRIPT interpreter, PalmSens SDK, or new dependency
is included. Manufacturer specifications set defaults; the approximations below
are this project's model choices. They do not constitute manufacturer validation.

## Public API and world integration

```python
from scentience_olfaction.sensors.factory import create_sensor

sensor = create_sensor(
    device_profile="scd41",       # scentience_v1, scd30, scd40, scd41, emstat_pico
    sensor_profile="packaged_slow",
    seed=0,
    config={"repeatability_ppm": 0.0},
    randomize=True,
)
reading = sensor.step({"carbon_dioxide": 500.0}, dt=5.0)
vector = sensor.observation_vector(reading)
diagnostics = sensor.last_reading
sensor.reset()
```

`create_sensor(device_profile="scentience_v1", sensor_profile="packaged_slow",
seed=0, config=None, randomize=True)` accepts a typed configuration or a JSON
mapping. It rejects unknown fields, wrong scalar types, nonfinite numbers and
invalid nested configurations. JSON arrays become typed tuples. Mappings overlay
the selected CO₂ defaults; complete `CO2Config` objects specify all values.
`sensor_profile` selects Scentience's MOX regime (`packaged_slow` or
`fast_modulated`); it does not change optional sensor physics. `randomize` controls
Scentience unit variation, not optional-channel noise. Set the noise coefficients
to zero for deterministic optional models.

| Device | Typed config | `channel_names` | Output units |
| --- | --- | --- | --- |
| `scentience_v1` | `device_np.DeviceConfig` | Existing 11-channel tuple | Existing schema |
| `scd30`, `scd40`, `scd41` | `co2_sensor.CO2Config` | `("co2_ppm",)` | Absolute ppm |
| `emstat_pico` | `emstat_pico.EmStatPicoConfig` | `("ec1",)` or `("ec1", "ec2")` | nA |

Every result has a defensive `.configuration` calibration snapshot,
`.step(conc_ppm, dt, state=None, conc_ppm_2=None)`, `.reset()`,
`.channel_names`, `.observation_vector(reading)` and `.last_reading`.
`state` follows `device_np.DeviceState`; EmStat uses `temp_c`. CO₂ ignores the
environmental state because environmental compensation is not modeled.
`conc_ppm_2` supplies the second independent EmStat cell or Scentience's right
MOX die; it has no effect for a single CO₂ sensor. Omitted, both locations see
`conc_ppm`. Factory Scentience imports are lazy.

`last_reading` returns a defensive copy of diagnostics, or `{}` after reset.
For CO₂ it contains the channel's `co2_ppm` and `asc_offset_ppm`. For EmStat it
maps `ec1`/`ec2` to current, unclipped current, signed quantizer counts,
`saturated`, `sampled`, and instantaneous `signal_na`. `sampled` means that at
least one acquisition has occurred since reset, not freshness on the latest
call. For Scentience it contains `channels` and `mox` diagnostics.

**Concentration convention:** bare `CO2Config()` and factory CO₂ profiles consume
**excess above ambient**. The input in the example means a 920 ppm target with
the default 420 ppm baseline. Direct `co2_config(SCD41)` instead consumes
**absolute ppm**. For plume truth that already includes a CO₂ background, pass
`concentration_mode="absolute"` and the matching `ambient_baseline_ppm`; adding
the baseline again would double count it. World/Isaac integrations can supply
these explicit fields when a background reservoir is configured. An omitted CO₂
key at the factory means ambient (zero excess). An explicit zero in absolute
mode means zero ppm. Legacy excess inputs below zero are clamped to zero;
absolute inputs below zero are rejected.

## Sensirion SCD30, SCD40 and SCD41

Exported names in `sensors.co2_sensor` are `CO2Profile`, immutable `SCD30`,
`SCD40`, `SCD41`, read-only `CO2_PROFILES`, `co2_config`, `CO2Config` and
`CO2Channel`. Existing positional configuration fields, keyword arguments
`step(co2_excess_ppm=..., dt=...)`, and the two reading keys remain available.
`step_absolute(co2_ppm, dt)` bypasses the configured input convention.

| Profile | Principle | Default cadence | τ63 | Specified accuracy (±) |
| --- | --- | --- | --- | --- |
| SCD30 | Conventional NDIR | 2 s | 20 s | 30 ppm + 3% of reading, 400–10,000 ppm |
| SCD40 | Photoacoustic NDIR / PASens | 5 s | 60 s | 50 ppm + 5%, 400–2,000 ppm |
| SCD41 | Photoacoustic NDIR / PASens | 5 s | 60 s | 50 ppm + 2.5%, 400–1,000; 50 ppm + 3%, 1,001–2,000; 40 ppm + 5%, 2,001–5,000 ppm |

Sources: [SCD30 datasheet v1.0, May 2020, Table 1](https://sensirion.com/en/media/documents/4EAF6AF8/61652C3C/Sensirion_CO2_Sensors_SCD30_Datasheet.pdf)
and [SCD4x datasheet v1.7, April 2025, Tables 1/4](https://sensirion.com/media/documents/48C4B7FB/67FE0194/CD_DS_SCD4x_Datasheet_D1.pdf).
Both specify a 0–40,000 ppm digital output range, distinct from the accuracy
range. Reference conditions include 25 °C, 50% RH, 1013 mbar, 3.3 V and continuous
operation. Response depends on packaging and environment. SCD30 repeatability
is 10 ppm RMS; SCD4x lists typical ±10 ppm. Field calibration and appropriate
exposure conditions are necessary to maintain specified accuracy.

SCD40/41 support approximately 30 s low-power periodic updates; SCD41 also
supports single-shot operation. Use `co2_config("scd41", low_power=True)` for
the 30 s approximation; power cycling and single-shot commands are not emulated.
SCD4x digital CO₂ readings are integer ppm. These details are in
[SCD4x §§3.6, 3.9, 3.11](https://sensirion.com/media/documents/48C4B7FB/67FE0194/CD_DS_SCD4x_Datasheet_D1.pdf).
SCD30 allows integer measurement intervals from 2–1800 s and returns floating
point digital readings; its profile imposes no uniform ppm quantization.
See [SCD30 interface description v1.0, §§1.4.3/1.5](https://sensirion.com/de/media/documents/D7CEEF4A/6165372F/Sensirion_CO2_Sensors_SCD30_Interface_Description.pdf).

Simulation assumptions:

- A first-order lag evolves under piecewise constant input: `dy/dt = (C-y)/tau`.
  Thus t90 is `ln(10)*tau`, about 46 s or 138 s for these profiles. Cadence is
  separate from lag. The held output changes only at acquisition boundaries;
  large `dt` processes every boundary and preserves leftover time.
- Initial and reset readings equal the configured ambient baseline. They are
  initialized state, not a simulated hardware measurement. No startup warm-up,
  pressure/RH compensation, enclosure flow solver or lifetime model is included.
- `repeatability_ppm` is modeled as per-acquisition Gaussian sigma, default 10.
  This distribution is an assumption, especially for SCD4x's typical ± value.
  The accuracy envelope is **not** another white-noise sigma or a guarantee
  that individual simulated readings stay inside a band.
- `cfg.accuracy_bound_ppm(C)` returns the specified envelope or `None` outside
  its concentration range. To avoid gaps for continuous simulated ppm, SCD41's
  next band applies immediately above 1000 and 2000. `accuracy_bias_fraction`
  defaults to zero; a fixed value in [-1,1] applies that signed envelope fraction
  at the lagged noiseless reading. Outside the specified range it adds no bias;
  this does not imply accuracy there. ADC clipping uses the output range.
- `co2_config(..., **overrides)` returns a fresh configuration. Cadence, lag,
  output range, resolution, noise, baseline and accuracy are tunable. Overriding
  `accuracy_base_ppm` or `accuracy_frac` replaces piecewise bands with one formula;
  explicit `accuracy_bands` can override that. Arbitrary timing overrides are
  simulator experiments, not additional manufacturer-supported modes.
- The legacy ASC approximation remains enabled for bare `CO2Config()`, and is
  **disabled** in manufacturer configs. When enabled it periodically moves an
  offset toward `asc_target_ppm` using a window minimum and `asc_gain`. This is a
  qualitative calibration-drift experiment, not Sensirion's proprietary ASC
  algorithm or a reproduction of its scheduling and convergence rules.

`reset()` clears dynamics, held values, clocks and ASC offset, without rewinding
the supplied RNG. Construct another channel with the same seed to replay noise.

## PalmSens EmStat Pico electronics and attached cells

EmStat Pico is a potentiostat, not an analyte-selective sensor. Its circuitry
supports two low-speed potentiostat channels; the high-speed resource is shared.
See [PalmSens hardware datasheet Rev.7-2023-011, pp.1/11](https://www.palmsens.com/app/uploads/2023/04/PSDAT-ESP-EmStat-Pico-Datasheet.pdf)
and [PalmSens developer specifications](https://dev.palmsens.com/espico/espico_specs.html).
The software-oriented brochure distinguishes sequential independent cells from
shared-reference bipotentiostat measurements. This model uses independent
fixed-bias cells in low-speed dual mode, and does not reproduce firmware
scheduling or bipotentiostat coupling.

| Mode | Nominal current ranges, in µA | Maximum acquisition rate |
| --- | --- | --- |
| `low_speed` | 0.1, 2, 4, 8, 16, 32, 63, 125, 250, 500, 1000, 5000 | 100 samples/s |
| `high_speed` / `max_range` | 0.1, 1, 6, 13, 25, 50, 100, 200, 1000, 5000 | 1000 / 100 samples/s |

The hardware maximum is ±3 mA, including on the nominal 5 mA range. Published
resolution is approximately 0.006% of selected range, with 5.5 pA specifically
listed at 100 nA. Electronics accuracy is <0.5% of current plus 0.1% of range in
low-speed mode, and <1% plus 0.1% otherwise. Full bias ranges are -1.2 to +2 V
(low speed) and -1.7 to +2 V (other modes). Source:
[PalmSens brochure Rev.11-2024-018, pp.4–5](https://assets.palmsens.com/app/uploads/2024/11/EmStat-Pico-Brochure.pdf).

```python
from scentience_olfaction.sensors.emstat_pico import (
    ElectrochemicalCellConfig, EmStatPicoChannelConfig, EmStatPicoConfig,
)
from scentience_olfaction.sensors.factory import create_sensor

# Synthetic demonstration coefficients; replace with measured cell calibration.
cell = ElectrochemicalCellConfig(
    sensitivity_na_per_ppm={"carbon_monoxide": 10.0, "hydrogen": 2.0},
    calibration_id="synthetic example; not a PalmSens gas-cell specification",
    tau_s=5.0, zero_current_na=1.0,
)
config = EmStatPicoConfig(channels=(
    EmStatPicoChannelConfig(cell=cell, electrode_count=2, bias_v=0.0,
                           current_range_na=2000.0),
    EmStatPicoChannelConfig(cell=cell, electrode_count=3, bias_v=0.0,
                           current_range_na=4000.0),
))
sensor = create_sensor("emstat_pico", config=config, seed=7)
reading = sensor.step({"carbon_monoxide": 20.0}, 0.1,
                      conc_ppm_2={"carbon_monoxide": 30.0})
```

`EmStatPicoConfig.channels` contains one or two `EmStatPicoChannelConfig`s.
Each owns a cell calibration, electrode count, fixed `bias_v`, mode, current
range, cadence, optional `resolution_na`, current limit and electronics error
coefficients. A two-electrode configuration represents a combined
counter/reference electrode; three electrodes represent separate WE/RE/CE.
The calibration must match this arrangement, bias, electrolyte and analyte.
Changing electrode count or bias does not invent a new transducer calibration.
Two simultaneous independent cells require low-speed mode in this model.

The cell signal is a lagged, signed linear sensitivity sum in nA/ppm, including
cross terms. `tau_s=0` gives an instantaneous calibrated response. Temperature
scales span about `reference_temp_c` and shifts zero. Cell drift and Gaussian
noise are separate from electronics gain, offset and Gaussian noise. None of
these cell coefficients are supplied by PalmSens; `calibration_id` is required
to identify their provenance, not to certify their accuracy. Concentrations
must be finite and nonnegative; sensitivities and measured currents can be signed.

The readout assumes a symmetric nominal range capped by `max_current_na` (at
most 3,000,000 nA). Current is rounded to the chosen quantum and constrained to
representable counts within that limit. `saturated` compares the pre-quantized
current with the limit; the last representable value may sit just below the
rail. `quantum_na` uses 0.0055 nA at 100 nA, and 0.00006 times range elsewhere,
unless overridden. This approximates the calibrated readout; it is not a raw
ADC register model inferred from the 16-bit converter.

`accuracy_bound_na()` reports the electronics envelope separately. Gain/offset
errors default to zero; neither they nor noise are randomly drawn from the
published accuracy specification. Each cell has separate lag, drift, hold,
saturation status and RNG stream. Cadence is independent of caller `dt`, with
the initial current held at zero until the first acquisition.

This is fixed-bias amperometric response plus electronics, **not full
voltammetry, chronoamperometric diffusion kinetics or EIS**. It does not solve
electrode potentials, cell impedance, reference polarization, compliance,
electrolyte transport, potential-dependent kinetics or channel coupling.
In particular, the documented channel-2 series resistance is not simulated;
calibrate the installed cell/readout together when that affects the experiment.
See [PalmSens developer documentation](https://dev.palmsens.com/espico/espico_main.html)
for actual hardware operation.
