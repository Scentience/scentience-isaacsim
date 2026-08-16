# Contributing

Bi-weekly release cadence; `main` is always releasable.

## Ground rules (the ones that are enforced)

1. **The NumPy transport is the specification.** Any change to
   `plume/filament.py` must update `transport/filament_warp.py` and keep
   `tests/test_warp_parity.py` green in the same PR.
2. **The realism gate is load-bearing.** If `tests/test_plume_gate.py` fails,
   the physics changed in a way that makes the environment less real. Fix the
   physics, never the thresholds, unless you bring literature.
3. **Every physical constant carries provenance** (`provenance.py`). New
   constants declare an evidence level and a source; "it works" is not a
   source. Synthesized coefficients name their donor.
4. **Ground truth never enters a policy observation.** Reviewers reject PRs
   that route `concentration_gt` or `truth()` into an actor.
5. **Closed-form tests over snapshots.** Prefer asserting an analytic
   property (mass flux, fitted tau, quantisation exactness) over golden
   values -- see tests/test_physics_rigor.py and tests/test_sensor_math.py
   for the house style.
6. **Isaac claims require Isaac evidence.** Anything touching
   `scentience_isaaclab/` or `isaac_extension/` states in the PR whether it
   was executed in a live install; `docs/ISAAC_COMPATIBILITY.md` is updated
   accordingly. Never claim an Isaac test passed when Isaac was unavailable.

## Workflow

`pip install -e ".[dev]"` -> branch -> change + tests -> `ruff check .` ->
`pytest -m "not isaac"` (fast) and `pytest -m slow` before pushing ->
PR with the CHANGELOG entry included.

Licensing: Apache-2.0 only (or MIT/BSD dependencies). No LGPL/GPL code may be
vendored or transcribed -- GADEN in particular is cited, never read into this
codebase. See docs/LICENSES_AND_PROVENANCE.md.
