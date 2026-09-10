# Sensor models

The instrument chain remains separate from air concentration. A virtual sensor
has response dynamics, environmental dependence, readout limits and calibration
uncertainty; a ppm field is not itself a sensor observation.

- [Scentience models](SCENTIENCE_MODELS.md): shared NumPy/Torch calibration for
  stereo MiCS-6814 channels, CO₂ and electrochemical channels, including the
  precise parity scope and observable baseline drift.
- [Manufacturer profiles](SENSOR_PROFILES.md): SCD30/40/41 and EmStat Pico
  electronics with independently calibrated one/two-cell setups.
- [Configuration](CONFIGURATION.md): Python dataclasses, strict JSON, mounting,
  timing, and concentration conventions.

`packaged_slow` and `fast_modulated` select different MOX response regimes. The
fast constants come from Dennler et al., *Science Advances* 10, eadp1764 (2024).
Their transfer is a simulation assumption, not an implementation of that
hardware or heater modulation method. Whiff-retention percentages vary with
exposure, sampling and detector settings; record the protocol with each result.

The standalone EC module retains linear cross-sensitivity and Cottrell-response
utilities for the existing chronoamperometry work. EmStat Pico is an optional
fixed-bias electronics/readout model and does not implement those diffusion
kinetics automatically. The PID module uses correction-factor-weighted
concentration and humidity quenching; it is separate from the Scentience device
channel schema.

Olfactory inertial odometry remains in `scentience_olfaction/oio/oio.py` with the
existing bout detection and platform presets. Its observability assumptions
remain: bout events alone do not correct downwind drift. See
[Isaac integration](ISAAC_INTEGRATION.md) for IMU frames and capture timestamps,
and [sim-to-real evaluation](SIM_TO_REAL.md) for calibration requirements.
