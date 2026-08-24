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

## Second pass (2026-08-23): licensing and documentation

Resolved since the first pass:

- **`CITATION.cff` repository URL** corrected by the maintainer.
- **`LICENSE` was a stub.** It contained only the Apache header plus a note
  reading "ACTION REQUIRED BEFORE PUBLIC RELEASE: replace this file with the
  full, verbatim Apache License 2.0 text". Replaced with the verbatim 202-line
  text from apache.org plus the filled-in appendix boilerplate. The licensing
  rationale that was in the stub moved to `LICENSES_AND_PROVENANCE.md`, since a
  LICENSE file should not carry prose that could read as modifying terms.
- **`NOTICE` added.** Apache-2.0 section 4(c) obliges redistributors to
  propagate NOTICE into derivative works, so this is the mechanism that gives
  the citation request real reach. Verified present in both the sdist and the
  wheel's `dist-info/licenses/`.
- **Three referenced-but-missing docs written**: `LICENSES_AND_PROVENANCE.md`,
  `CHEMICAL_MODEL.md`, `UPSTREAM_REQUESTS.md`. All five code sites that defer
  to `CHEMICAL_MODEL.md` are now actually answered by it. Every
  `[A-Z_]+\.md` cross-reference in the repo resolves.
- **`_paper/` exclusion did not work.** The added pattern `*/_paper/*` requires
  a leading directory component and so never matched root-level `_paper/`, and
  the two files were already tracked -- `.gitignore` cannot untrack. Changed to
  `/_paper/` and ran `git rm --cached -r _paper`. Confirmed absent from the
  built sdist.
- **`SETUP.md` pointed at an excluded file.** The new `*.ps1` rule excludes
  `run.ps1`, so published instructions referenced a script users would not
  receive. `SETUP.md` now gives direct interpreter commands and notes the
  Linux/macOS path form.

Licensing decision: **Apache-2.0 retained.** Citation is expressed as a
strongly-worded norm via `CITATION.cff` + `NOTICE` rather than as a license
condition, and research-status/no-warranty language is stated plainly without
restricting use. This keeps OSI compliance, keeps the
`License :: OSI Approved` classifier truthful, and preserves the commercial
robotics adoption the LICENSE rationale names as the strategic goal.

## Left for the maintainer

- **Isaac Lab integration remains unvalidated** and cannot be validated on this
  hardware. See `ISAAC_COMPATIBILITY.md`. Disclosed in README, NOTICE and
  `LICENSES_AND_PROVENANCE.md`, which is the honest handling for v0.1.
- **Coefficient calibration** is the highest-value provenance upgrade
  available: it would promote the MiCS-6814 constants from DIGITIZED to
  MEASURED and unlock absolute-ppm claims. Tracked as UR-3 in
  `UPSTREAM_REQUESTS.md`.
- **Legal review.** The licensing documents here were written by engineers.
  Have counsel read `LICENSE`, `NOTICE` and `LICENSES_AND_PROVENANCE.md` before
  the public debut.

## Third pass (2026-08-23): kit fidelity and Isaac API validation

Release intent (stated by the maintainer): model the Scentience olfactory dev
kit -- two MiCS-6814 sensors for STEREO olfaction plus one SCD-41 -- and
first-order scent navigation.

Audit against that intent found one substantive gap: **stereo was not
modelled**. `ScentienceV1.step()` fed one concentration dict to all six MOX
channels, so both dies always sampled the same point and the inter-die
difference -- the cue the second sensor exists to provide -- was structurally
zero. Fixed end-to-end, mono-back-compatible (see CHANGELOG); 6 new tests pin
back-compat bit-identity, L/R physics, and the observation contract.
SCD-41 was already faithful (SCD4x datasheet constants, tau63=60 s, ASC);
navigation was already present (PlumeNavEnv + cast-and-surge baseline).

Isaac Lab integration was promoted from "written, unvalidated" to
"API-contract validated" without GPU hardware: 22/22 static checks against
the real `isaaclab==2.3.2` wheel and 7/7 checks executing our classes under
genuine isaaclab code (kit runtime stubbed). This caught a real
`class_type`-binding bug that would have broken sensor construction in a live
install. Live-install execution (tier 3) remains open and is disclosed. See
docs/ISAAC_COMPATIBILITY.md.

Full verification after all changes: `ruff check .` clean, 68 passed
(62 prior + 6 stereo), all 5 examples run, sdist/wheel rebuild + twine PASS.

## Fourth pass (2026-08-23): naming accessibility and paper alignment

Two maintainer requests:

**1. No part numbers in identifiers.** Every identifier and data key a
developer touches is now beginner-readable: `chem_left_*`/`chem_right_*`
channels (named for their stereo roles), `CO2Channel`/`CO2Config`
(`sensors/co2_sensor.py`), `MOX_RED`/`MOX_NH3`/`MOX_OX`. Part numbers stay in
docstrings and provenance records as facts about the modelled hardware.
Verified zero stale identifiers repo-wide after the rename; full suite green.

**2. Alignment with Chasing Ghosts (France et al., arXiv:2602.19577).** The
repo already carried the paper's bout detection, accelerated
chronoamperometry, OIO, and (since the third pass) stereo sampling. Three
genuine gaps were implemented:

- `DivergenceSignal` (Eqs. 5-7): dual-timescale divergence + signal line for
  surge/cast switching, timestep-invariant form of the paper's
  alpha=3, beta=8, rho=5.
- `SourceDeclaration` (Eqs. 9-11): the sensor-only stopping rule. The paper's
  worked example (k=20 -> point estimate 1.05m, CI upper ~1.2025m) is
  reproduced numerically in tests. Operationalising it honestly required
  three additions the paper leaves implicit: decimation to the paper's 1 Hz
  sampling cadence (at 20 Hz control rate the k* threshold is otherwise
  reached in one second of correlated noise -- observed), a plateau
  requirement (still-climbing m means the source is still ahead), and an
  at-the-maximum gate (remembering a strong whiff is not finding the source).
- `StereoCastAndSurge`: steering from the inter-sensor ONSET LAG (Eqs. 3-4).
  An amplitude-difference formulation was tried first and measurably failed
  (0/5 success): per-die calibration spread (the simulator randomises R0 and
  sensitivity per unit, as hardware varies) injects a constant steering bias
  that swamps the true stereo signal at a 0.04 m baseline. The lag cue is
  calibration-invariant -- which is precisely why the paper formulates it as
  a time delay. The failure and the fix are both documented in the class.

20-seed benchmark after the fix (die-scale 0.04 m / antenna-scale 0.30 m):
mono 0.45/0.40 success, stereo 0.40/0.40 -- parity within noise, and the
declaration rule produced ZERO false "source found" claims in 40 episodes.
At these baselines and 20 Hz the lag frequently quantises to zero, so the
stereo agent degrades gracefully to upwind cast-and-surge; that resolution
limit is stated in the class docstring rather than papered over.

Full verification after both changes: ruff clean, 77 tests passed, all 5
examples run, build + twine PASS, isaaclab binding harness 7/7.

## Fifth pass (2026-08-23): benchmark against ST's Isaac sensor repository

Reviewed `STMicroelectronics/st-mems-isaac-sim2real` (IMU sim2real extension
for Isaac Sim) at the maintainer's request, as a credibility benchmark.

**Where this repo was already ahead** (nothing adopted): CI with an enforced
physics-realism gate (ST has no CI), test breadth (79 tests vs 2 files),
provenance/evidence system, citation metadata (CITATION.cff + NOTICE; ST has
neither), physics documentation depth, and a pip-installable pure-Python core
(ST ships pre-compiled .so binaries, Ubuntu-only).

**Adopted from ST's practice**:
- Community-health files: `SECURITY.md`, `CODE_OF_CONDUCT.md`.
- `BRANCHING.md`: their version-disciplined branch layout (stable /
  support / experimental per Isaac target) fits the planned bi-weekly
  cadence and the coming Isaac Lab 3.0 port.
- `docs/TROUBLESHOOTING.md`: ST's README carries 11 concrete
  symptom->fix entries; ours now carries 12, each one a failure mode
  actually reproduced during these validation passes (including the
  verified fact that the Warp twin runs CPU-only when no CUDA device
  exists).
- `scripts/plot_verification.py`: ST's strongest user-facing practice is
  plots comparing clean vs realistic sensor output. Ours renders (a) ground
  truth vs slow/fast device response -- the visual form of the README's
  whiff-retention claim -- and (b) the stereo left/right cue. Figures
  inspected and correct; a test pins that both render.

**Deliberately NOT adopted, with reasons**:
- Pre-compiled native backends: contradicts the "core imports with NumPy
  only" invariant and Windows/macOS support; Warp covers the fast path.
- In-Isaac menu/UI integration (Create -> Sensors): requires live-Isaac
  iteration this hardware cannot do; the Kit extension scaffold and the
  API-contract harness are the honest current ceiling. Roadmap item.
- JSON sensor-profile files: our configs are typed dataclasses with
  provenance attached; externalising them would detach constants from their
  evidence records. May revisit for the `[yaml]` extra later.

Final state after this pass: 78 tests passing, ruff clean, build + twine
PASS, all examples + plot script run.

## Sixth pass (2026-08-23): the Isaac Lab wrapper itself

Request: everything needed for a VALID Isaac Lab wrapper, with ST-repo
structural parity where it applies. Extended the executed-binding harness
from the sensor alone to the whole wrapper -- mdp terms, the DirectRLEnv
task cfg under the real `DirectRLEnvCfg`, and gym registration with
entry-point resolution (10/10). That extension caught one real bug:
`scentience_isaaclab/mdp/__init__.py` was EMPTY, so `mdp.gas_channels` --
the exact path Isaac Lab observation configs reference terms by -- was
unreachable. Fixed with explicit re-exports.

Also hardened: the env/cfg modules' Isaac-absent fallbacks now record WHY
they are unavailable (`IMPORT_ERROR`) instead of silently binding None --
during harness work those silent Nones cost real debugging time, which is
exactly what a user would hit.

ST-parity artifacts added for the Isaac path:
- `scripts/verify_in_isaac.py` -- the runtime companion to
  `validate_install.py` (minimal scene, sensor on a rigid body, channels
  logged vs ground truth, npz + plot), mirroring ST's verification-script +
  plot workflow. It cannot be executed on this hardware and says so in its
  banner; every isaaclab symbol it touches is contract-checked against the
  real wheel (the "verify_in_isaac surface" checks; static total now 34/34)
  and it compiles clean.
- `docs/ISAAC_USAGE.md` -- ST-README-style walkthrough: install, validate
  BEFORE use, attach to a robot, observation terms, RL task id, runtime
  verification, and an explicit table of what is and is not validated today.

Final: 78 tests, ruff clean, contract 34/34, binding 10/10, build + twine
PASS.
