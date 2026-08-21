# Release validation

Independent pre-publication check of the packaged distribution.
Date: 2026-08-21. Host: Windows 10, i7-1065G7, GTX 1650 Max-Q.

## Result

The package is functional for its two supported paths (standalone Python,
Gymnasium) on every Python version it claims. Seven defects were found and
fixed; three items are left for the maintainer to decide (bottom of file).

## What was verified

| Check | Result |
|---|---|
| `pip install` from built wheel, **numpy only** | core + 31 submodules import |
| README "Five lines to smell", copied verbatim | runs, correct output |
| `pytest -m "not isaac"` on Python **3.10** | 62 passed |
| `pytest -m "not isaac"` on Python **3.11** | 62 passed |
| `pytest -m "not isaac"` on Python **3.12** | 62 passed |
| `pytest -m "not isaac"` with **no torch, no warp, no GPU** | 57 passed, 3 skipped |
| CI step 1 `ruff check .` | All checks passed |
| CI step 2 `pytest -m "not isaac and not slow"` | 57 passed |
| CI step 3 plume realism gate (`-m slow`) | 2 passed |
| `twine check` on sdist + wheel | both PASSED |
| End-to-end: unpack sdist, install, run all 4 examples | all 4 correct |
| `gymnasium.utils.env_checker.check_env(PlumeNavEnv)` | PASSED |
| Warp GPU path on GTX 1650 (sm_75) | `cuda:0` available, parity tests pass |

Results were byte-identical between the development checkout and a clean
install from the sdist, on all four examples.

## Defects found and fixed

1. **Kit extension could not load.** Four files in `isaac_extension/` were
   written with the literal two-character sequence `\n` instead of newlines.
   Both `__init__.py` were hard `SyntaxError`s
   (`unexpected character after line continuation character`).
2. **`pytest -m "not isaac"` aborted without warp.** `test_warp_parity.py` and
   `test_warp_pool.py` imported `filament_warp` at module scope, so collection
   died with 2 errors and *zero* tests ran -- exactly the path the README
   advertises. Added `pytest.importorskip("warp")`, matching the idiom already
   used in `test_device_parity.py` and `test_env_*.py`.
3. **Provenance unreadable in a numpy-only install.** `sensors/scentience_v1.py`
   imported torch at module scope, so `register_coefficients()` -- pure
   metadata, no tensors -- was unreachable, breaking ARCHITECTURE.md invariant 4
   and the README's provenance claim. torch is now lazy; the batched device
   still raises a clear, actionable ImportError.
4. **`scripts/validate_physics.py` did not run at all.** Passed
   `specific_gravity=` to `FilamentPlumeConfig`, which has no such field (it
   belongs to `Species`) -> `TypeError` on every invocation. Also cached to
   hardcoded `/tmp/...`, which does not resolve on Windows.
5. **CI had never passed.** `ruff check .` reported 100+ errors. The lint set
   was implicit, so it also widened on every ruff upgrade. Now pinned in both
   directions: explicit `[tool.ruff.lint]` select/ignore encoding the repo's
   real style, and `ruff>=0.16,<0.17` in the dev extra. 12 genuinely dead
   imports removed; the deliberate Isaac availability probe kept with a `noqa`.
6. **`.gitignore` silently deleted documentation.** The pattern `*_*.md`
   matches *any* markdown file with an underscore. It is why
   `docs/LICENSES_AND_PROVENANCE.md` -- which README's "Honesty policy" cites
   for the GADEN/LGPL position -- and `docs/CHEMICAL_MODEL.md` are referenced
   but absent from the repo. It would also have swallowed `CODE_OF_CONDUCT.md`
   and `PULL_REQUEST_TEMPLATE.md`. Narrowed to `*SKILL.md` + `notes/`.
7. **sdist was not self-contained.** Shipped no `examples/`, `scripts/`,
   `docs/` or `isaac_extension/`, though the README tells users to run
   `python examples/01_minimal.py`. Added `MANIFEST.in`.

## Reproducibility note on the README's two headline numbers

The README states blank-duration CV "2.31 with meander, 0.96 without" without
naming a seed or a detection threshold. Measured across 5 seeds at the stated
conditions (600 s @ 100 Hz, 8 m downwind), using `validate_physics.py`'s own
threshold convention (10% of the full plume's conditional mean):

| series | mean | sd | range |
|---|---|---|---|
| full plume | 1.72 | 0.41 | 1.40 - 2.40 |
| meander ablated | 0.950 | 0.020 | 0.923 - 0.968 |

Both README figures are attainable -- 2.31 sits near the top of the full-plume
range (seed 11 gives 2.40), and 0.96 is squarely typical. But the full-plume CV
varies ~24% relative across seeds, so quoting a bare "2.31" implies precision
the estimator does not have; a reader reproducing it will usually see ~1.7. The
ablation *direction* -- the actual scientific claim -- is robust: meander
roughly doubles CV, and every ablated seed lands below the gate's 1.2 ceiling.

Recommend quoting mean +/- sd, or stating seed and threshold.

## Left for the maintainer

- **`CITATION.cff` points at the wrong repository.** It says
  `github.com/scentience/scentience-isaac-olfaction`; `git remote` says
  `github.com/Scentience/scentience-isaacsim`. Citations would 404.
  `[project.urls]` was set from the git remote; reconcile the two.
- **Docs referenced but never written**: `docs/LICENSES_AND_PROVENANCE.md`,
  `docs/CHEMICAL_MODEL.md`, `docs/UPSTREAM_REQUESTS.md`. Now that `.gitignore`
  no longer eats them, they need to exist -- the first is load-bearing for the
  GADEN/LGPL claim in the README.
- **Isaac Lab integration remains unvalidated** and cannot be validated on this
  hardware. See `docs/ISAAC_COMPATIBILITY.md`.
