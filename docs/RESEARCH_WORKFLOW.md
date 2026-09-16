# Reproducible olfactory robotics experiments

The package separates air concentration, instrument observations, and privileged
evaluation data. This makes it possible to identify whether a failed policy
lost the plume, failed to resolve a whiff, or failed to control its robot.

## Visual inspection

`python scripts/plot_plume.py --out runs/plume` produces a labeled XY
concentration slice, wind vectors, a probe location, and aligned air/device
traces. Its JSON sidecar records parameters, software versions, the seed and
configuration hash. Use `--config configs/scd41.json` to view a CO₂ scenario.
The logarithmic colorbar remains in ppm; particle density is not substituted
for concentration. `visualization.concentration_slice` also returns arrays for
your own visualization pipeline without importing Matplotlib.

`plot_verification.py` compares the slow/fast MOX responses and separated
stereo probes. Its wide demonstration baseline is explicitly labeled; do not
interpret that plot as a measured device geometry. `PlumeNavEnv` supports
`render_mode="rgb_array"` for Gymnasium video wrappers. Rendering is optional
and should be disabled during throughput measurements.

## Benchmarks and metrics

```bash
python scripts/benchmark.py --config configs/navigation.json --episodes 20 --seed 100 --record --out runs/eval
```

Each baseline receives the same episode seeds and configuration. Output includes
per-episode CSV, an aggregate JSON report, and optional trajectories. Report:

- Success rate with its Wilson 95% interval and the number of episodes.
- Final source distance, elapsed time and path length in metres.
- Sensor-only declaration and false-declaration counts, separately from the
  privileged source-radius termination used to evaluate the search.
- SPL in the environment info: success weighted by the shortest distance to
  the success-radius boundary divided by travelled distance. This planar task
  uses straight-line distance and has no robot collision/path planner.
- Sensor contact fraction: steps whose maximum MOX deflection exceeds 0.01.
  This is an instrument threshold, not ground-truth plume intermittency.

`EpisodeRecorder` saves compressed numeric NPZ plus JSON with explicit column
names. It copies samples at log time, rejects changing field sets/shapes, and
allocates a new episode filename after restart. Record `obs`, `action`,
`next_obs`, `terminated`, and `truncated` to preserve transition semantics.
`run_metadata` adds Python/library versions and a stable SHA-256 configuration
fingerprint. Include your Git commit, calibration dataset ID and asset versions
in `extra_meta` when preparing a publication. Configuration hashing is not a
complete dataset or source-code provenance system.

## Statistical interpretation

The supplied long-record gate guards a particular intermittent-plume benchmark.
Vary sensor height, wind, thresholds and source geometry when evaluating a new
scene. Record the sampling interval and threshold with whiff/blank statistics.
The first and last runs are censored by the observation window and excluded
from duration summaries, including very short records.

The -3/2 exponent in Celani et al. describes a probability density in a scaling
regime; a corresponding untruncated CCDF has slope -1/2. A finite-record CCDF
tail fit with an exponential cutoff cannot be compared directly with that PDF
exponent. The code reports an empirical CCDF slope as a diagnostic. See
[Celani et al., PRX 4, 041015](https://doi.org/10.1103/PhysRevX.4.041015).

Use held-out seeds and entire held-out exposure sessions for evaluation. Vary
wind, source flux, sensor response, mounting and calibration across trials.
Compare against the included sensor-only baselines. Short training and smoke
runs establish software functionality; they do not establish navigation quality,
sensor accuracy, sim-to-real transfer, or calibrated odometry performance.
