# Changelog

## 0.1.0 (2026-08-15) -- initial release ("timestamp release")

First olfactory and chemical sensing package for NVIDIA Isaac Sim / Isaac
Lab, to our knowledge. Highlights:

* Filament plume transport (Farrell 2002), multi-species, multi-emitter,
  obstacle-aware (occupancy grid + line-of-sight + slide collision), with
  two-scale turbulence (per-filament OU + shared bearing meander). NumPy
  reference = specification; Warp GPU twin, parity-tested.
* CI-enforced plume realism gate (blank-CV, tail exponents, intermittency,
  peak-to-mean) with a meander ablation that MUST fail.
* Sensor suite: MiCS-6814-class MOX (power law, asymmetric lag, drift, 1/f,
  ADC divider quantisation; fast profile time constants credited to Dennler
  et al. 2024), electrochemical (linear + Cottrell), SCD4x CO2 (photoacoustic,
  ASC drag), PID (TN-106 CFs). Scentience V1 device in the BLE channel schema.
* Olfactory Inertial Odometry reference implementation (bout detection +
  heading/crosswind drift correction) with uav/quadruped/biped/arm presets.
* Gymnasium PlumeNavEnv + cast-and-surge and random baselines + recorder.
* Evidence provenance on every physical constant; claim_check() gating.
* Isaac Lab SensorBase integration (WRITTEN, UNVALIDATED in a live install --
  see docs/ISAAC_COMPATIBILITY.md) + Kit extension scaffold.

Known limitations: GPU path is single-species/point-source/no-occupancy;
buoyancy off by default; sensitivity coefficients DIGITIZED not MEASURED;
COLIP-2 / vision-language integration deliberately deferred to a later
release.
