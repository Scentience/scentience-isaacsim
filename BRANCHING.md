# Branching and versioning

The project releases roughly bi-weekly. This file states which branch means
what, so a user landing on the repo knows what they can trust -- the pattern
follows the version-disciplined layout used by vendor sensor repos such as
ST's `st-mems-isaac-sim2real`.

## Branches

| Branch | Meaning |
|---|---|
| `main` | The release line. Everything on it passes the full CPU suite (`pytest -m "not isaac"`), `ruff`, the plume realism gate, and builds a clean sdist/wheel. Isaac Lab code on `main` targets **Isaac Lab 2.3.x / Isaac Sim 5.1** and is API-contract validated (`docs/ISAAC_COMPATIBILITY.md`). |
| `support/isaac-lab-2.3` | Created the day `main` moves to a newer Isaac Lab target. Bug fixes only; no new features. Does not exist until then. |
| `experimental/isaac-lab-3.0` | Where the `env_mask` port of `scentience_isaaclab/` will live (Isaac Lab 3.0 changes `_update_buffers_impl(env_ids)` to `(env_mask)` and `data.field` to `data.field.torch`). Isolated until `scripts/validate_install.py` passes on a live 3.0 install; never merged before that. |

Branch names encode the compatibility they affect, so a future
`experimental/isaac-sim-6.0-python-3.12` is self-describing.

## Releases

* Tags are `vMAJOR.MINOR.PATCH`; version lives in `pyproject.toml` and
  `CITATION.cff` (keep them in lockstep -- CI does not check this yet).
* Every release updates `CHANGELOG.md`. Claims that require hardware
  (live-Isaac validation, measured coefficients) enter the changelog only
  with their evidence attached, per the provenance policy.

## Rules that protect users

1. **`main` is never red.** If the realism gate fails, the physics changed;
   that is a release blocker, not a flaky test.
2. **API contract before merge**: anything touching `scentience_isaaclab/`
   must pass `scripts/check_isaaclab_contract.py` and
   `scripts/check_isaaclab_binding.py` against the pinned wheel.
3. **Unvalidated integrations stay labelled.** The Isaac path keeps its
   UNVALIDATED notices until a live run is pasted into
   `docs/ISAAC_COMPATIBILITY.md`, regardless of branch.
