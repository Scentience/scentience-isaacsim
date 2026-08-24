# Changelog

## Unreleased (release-validation pass, 2026-08-21..23)

* **Isaac Lab wrapper completed and wrapper-wide validated.** The
  executed-binding harness now covers the WHOLE wrapper (10 checks): mdp
  observation terms, the DirectRLEnv task cfg constructing under the real
  `DirectRLEnvCfg`, and gym registration with resolving entry points --
  which caught and fixed an empty `mdp/__init__.py` that made
  `mdp.gas_channels` unreachable. Isaac-absent fallbacks now record their
  `IMPORT_ERROR` instead of silently binding None. New:
  `scripts/verify_in_isaac.py` (in-Isaac runtime verification: minimal
  scene, channels vs ground truth, npz + plot; UNVALIDATED live, but its
  full isaaclab API surface is contract-checked -- static checks now 34/34)
  and `docs/ISAAC_USAGE.md` (install -> validate -> attach -> verify
  walkthrough).

* **Repo-hygiene parity with vendor sensor repos** (after review of ST's
  `st-mems-isaac-sim2real`): `SECURITY.md` (private vulnerability reporting),
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `BRANCHING.md` (release /
  support / experimental branch policy for Isaac version targets),
  `docs/TROUBLESHOOTING.md` (12 entries, each a failure mode actually
  reproduced during validation), and `scripts/plot_verification.py` (+ `[viz]`
  extra) producing clean-vs-device and stereo-cue verification figures, with
  a test pinning that both render.

* **Beginner-facing sensor naming.** Part numbers no longer appear in any
  identifier or data key a developer types: the two metal-oxide sensors are
  `chem_left_*` / `chem_right_*` (matching their stereo roles), the CO2
  channel class is `CO2Channel` (`sensors/co2_sensor.py`), and the MOX
  configs are `MOX_RED`/`MOX_NH3`/`MOX_OX`. Part numbers (MiCS-6814, SCD4x)
  remain in docstrings and provenance records, where they are facts about
  the modelled hardware rather than API surface.
* **Chasing Ghosts navigation machinery** (France et al., arXiv:2602.19577):
  `DivergenceSignal` (Eqs. 5-7 dual-timescale divergence + signal line),
  `SourceDeclaration` (Eqs. 9-11 sensor-only "source found" stopping rule --
  hardened beyond the paper with sampling-cadence decimation, a plateau
  requirement and an at-the-maximum gate, all documented in the module),
  and `StereoCastAndSurge`, a baseline steering on the inter-sensor ONSET
  LAG (Eqs. 3-4). Lag rather than amplitude on purpose: per-die calibration
  spread swamps the amplitude cue at small baselines -- measured here, and
  the reason the paper's formulation is lag-based. `run_episode` now ends an
  episode when an agent declares, scoring the declaration against ground
  truth. 9 new tests reproduce the paper's Eq. 11 worked example among
  others.

* **Stereo olfaction** -- the reason the dev kit carries two MiCS-6814 dies.
  `ScentienceV1.step(..., conc_ppm_2=)` feeds the second die independently;
  `OlfactionWorld.read(..., heading=)` samples the two dies
  `stereo_baseline_m` apart perpendicular to the heading (`chem_left_*`,
  `chem_right_*`); `PlumeNavConfig.stereo_baseline_m` (default 0.04 m,
  evidence ASSUMED -- measure your kit) puts the lateralisation cue in the
  PlumeNav observation. Mono behaviour unchanged when no heading /
  second-concentration is given; `tests/test_stereo.py` pins back-compat,
  L/R physics, and the observation contract. `examples/05_stereo_olfaction.py`.
* **Isaac Lab 2.3.2 API-contract validation without a GPU**:
  `scripts/check_isaaclab_contract.py` (22 static checks against the real
  wheel) and `scripts/check_isaaclab_binding.py` (our classes executed by
  genuine isaaclab code, kit runtime stubbed) -- see
  docs/ISAAC_COMPATIBILITY.md tiers. Caught and fixed a real bug:
  `OlfactorySensorCfg.class_type` was bound after `@configclass` had already
  baked the `None` default into `__init__`, so instances always had
  `class_type=None`; now bound at definition, upstream-style.
* Fixes from the release validation pass: pytest no longer aborts without
  warp (`importorskip`); provenance readable without torch (lazy import);
  Kit extension `__init__` literal-`\n` corruption; `validate_physics.py`
  bad kwarg + POSIX-only cache path; `.gitignore` `*_*.md` no longer
  swallows docs; lint contract pinned (`ruff check .` green); sdist ships
  examples/scripts/docs; full Apache-2.0 text + NOTICE; new docs
  LICENSES_AND_PROVENANCE / CHEMICAL_MODEL / UPSTREAM_REQUESTS /
  RELEASE_VALIDATION.

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
