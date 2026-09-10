# Calibration and sim-to-real evaluation

Preserve calibration data and scientific conditions with every result. The
shipped default coefficients include digitized, synthesized and assumed values;
software tests establish mathematical behavior, not absolute device accuracy.
The existing Scentience research references remain the basis of these models.

1. Split calibration and evaluation by exposure session and physical device to
   prevent temporal leakage. Record analyte mixtures, concentration range,
   temperature, humidity, airflow, heater settings, ADC and sampling interval.
2. Fit static sensitivity first, then rise/recovery constants and transport
   delay. Inspect residuals and saturation. Compare simulated and held-out
   instrument traces with RMSE, correlation, response lag and event statistics.
3. Load per-unit settings through `DeviceConfig` or the manufacturer/cell config.
   Record calibration IDs and units. A potentiostat's current accuracy does not
   imply a gas cell's analyte accuracy or a transferable bias/electrode setup.
4. Randomize only parameters with defensible ranges. Default unit variation
   samples R0; additional sensitivity and kinetics variations require explicit
   configuration/calibration. Run held-out winds, source fluxes, mounting offsets
   and device calibrations rather than relying on one seed.
5. Evaluate source declarations independently of privileged success-radius
   scoring. Check sampling timestamps, coordinate transforms and IMU gravity
   conventions before interpreting odometry drift improvements.

The evidence registry supports scoped claim checks on registered constants;
it does not automatically register every custom coefficient. A published
measurement on another instrument is not measured Scentience calibration.
See [SCENTIENCE_MODELS.md](SCENTIENCE_MODELS.md) and
[RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md).
