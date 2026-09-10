# Contributing

Changes should make olfactory experiments easier to reproduce and inspect.
Keep the NumPy core independent of Isaac, Torch, GPU support and training tools.

## Review expectations

1. Preserve the NumPy reference model and check the implemented Warp subset.
   Reject unsupported options explicitly instead of silently changing physics.
2. Explain numerical/scientific changes and validate a relevant property:
   conservation, response time, source timing, coordinate transform or replay.
   Do not replace meaningful assertions with implementation snapshots.
3. Identify evidence and conditions for new physical coefficients. A transferred
   calibration names its source and remains a transfer assumption.
4. Keep ground truth separate from actor observations. Privileged critic/reward
   data and logged labels must be clearly named.
5. Keep statistical benchmark thresholds stable unless a scientific change
   justifies revisiting them. A passed gate does not validate arbitrary scenes.
6. State whether Isaac changes were executed in a live installation and record
   exact versions in [compatibility notes](docs/ISAAC_COMPATIBILITY.md).

## Local checks

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -m "not isaac and not slow"
python -m pytest -m slow
python -m build
```

Include a changelog entry, updated configuration/example documentation and only
the tests needed to guard meaningful behavior. Report new calibration data,
asset requirements, backend limitations and any observation-schema changes.

Existing repository and sensor license terms apply; see
[LICENSES_AND_PROVENANCE.md](docs/LICENSES_AND_PROVENANCE.md). Add only
permissively licensed software dependencies with attribution. Do not vendor
copyleft simulator implementations or manufacturer firmware/SDKs. Do not change
license files as part of a technical improvement without explicit authorization.
