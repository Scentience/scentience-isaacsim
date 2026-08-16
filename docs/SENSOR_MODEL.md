# Sensor models

## MOX (MiCS-6814 class) -- sensors/mox.py

Chain: power-law steady state (superposed across species in RESISTANCE
space, oxidizing gases with negative beta raising Rs) -> absolute-humidity and
temperature modulation in log space -> asymmetric first-order lag (tau_rise !=
tau_fall; flow- and heater-corrected) -> transport delay -> baseline random
walk + 1/f + white noise -> voltage divider -> ADC quantisation.

Two dynamic profiles:
* `packaged_slow` (DEFAULT): simple MiCS-6814, tau_fall ~12 s. Retains ~19%
  of plume whiff events.
* `fast_modulated`: time constants taken from the measured performance of the
  high-speed e-nose of **Dennler, Rastogi, Fonollosa, van Schaik & Schmuker,
  "High-speed odor sensing using miniaturized electronic nose", Science
  Advances 10, eadp1764 (2024)** (onset 87 +/- 20 ms, recovery 106 +/- 24 ms).
  We do NOT implement their heater-modulation method -- only the demonstrated
  time-constant regime, with credit. Retains ~97% of whiff events.

Sensitivity (A, beta) values are DIGITIZED evidence (inverted from open-source
driver fits; the SGX datasheet publishes log-log graphs only). Replace via
calibration before absolute-ppm claims; provenance.py enforces the labelling.

## Electrochemical -- sensors/electrochemical.py
Linear in C (I = nFAD/delta * C); mixing-matrix cross-sensitivity (correct for
EC, wrong for MOX); Cottrell transient I ~ t^{-1/2} for chronoamperometry,
matching the accelerated-chronoamperometry line of France & Daescu
(arXiv:2506.04540). Default cell follows the Chasing Ghosts ItalSens setup
(A = 2.25 cm^2).

## CO2 -- sensors/scd4x.py
SCD4x-class, photoacoustic NOT NDIR; tau63 = 60 s (t90 ~ 138 s -- cannot see a
whiff; environmental context only); 5 s zero-order hold; ASC drag modelled.

## PID -- sensors/pid.py
CF-weighted sum per RAE TN-106; PID-blind gases encoded as CF = inf; humidity
quench. Ethanol at 10.6 eV uses CF = 3.1 (a known incumbent bug ships 10.47,
which is ethanol's ionisation ENERGY; test_pid_correction_factors guards it).

## Device -- sensors/device_np.py (NumPy) + sensors/scentience_v1.py (torch)
Channel schema == Scentience BLE/Sockets ordering. The torch twin is the
Isaac-scale path; test_device_parity holds them together (torch optional).

## OIO -- oio/oio.py
Dual-timescale EMA bout detection (Chasing Ghosts, arXiv:2602.19577) and
olfactory drift correction (Olfactory Inertial Odometry, arXiv:2506.04539):
anemometer-referenced heading correction + bout-anchored crosswind correction.
The estimator is honest about observability: downwind drift is NOT corrected
by bouts alone, and a test asserts that asymmetry. Platform presets: uav,
quadruped, biped, arm.
