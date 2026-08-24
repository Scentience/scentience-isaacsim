# Setup (verified on this machine, 2026-08-21)

Standalone Python + Gymnasium paths are installed and passing. The Isaac Sim
path is blocked by hardware -- see `docs/ISAAC_COMPATIBILITY.md`.

## Environment

| | |
|---|---|
| Python | 3.11.9 (`.venv/` at repo root) |
| Install | `pip install -e ".[dev]"` |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| Warp | 1.16.0, **CUDA `cuda:0` sm_75 available** (GTX 1650, 4 GiB) |
| Torch | 2.13.0 |
| Gymnasium | 1.3.0 |

Python 3.11 was chosen deliberately: it satisfies `requires-python >=3.10` and
is also the interpreter Isaac Sim 5.1 / Isaac Lab 2.3.x expect, so this same
venv can be reused if the Isaac path is ever unblocked.

## Create the environment from scratch

```powershell
winget install --id Python.Python.3.11
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Run

Windows (PowerShell) -- substitute `.venv/bin/python` on Linux/macOS:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not isaac"          # 62 passed
.\.venv\Scripts\python.exe examples\01_minimal.py
.\.venv\Scripts\python.exe examples\02_walls_and_wind.py
.\.venv\Scripts\python.exe examples\03_olfactory_inertial_odometry.py --platform quadruped
.\.venv\Scripts\python.exe examples\04_gym_baseline.py
.\.venv\Scripts\python.exe scripts\validate_physics.py       # the realism gate
.\.venv\Scripts\python.exe scripts\provenance_demo.py        # evidence levels
```

The Isaac integration check runs inside Isaac Sim's own interpreter, not this
venv, and is expected to fail on hardware without RT cores:

```powershell
$env:PYTHONPATH = $PWD; C:\isaacsim\python.bat scripts\validate_install.py
```

## Verified output (2026-08-21)

```
pytest -m "not isaac"          62 passed in 248.44s
01_minimal.py                  11 device channels + ground truth
02_walls_and_wind.py           upwind 4.860 ppm | behind wall 0.000 (blocked)
03_..._odometry.py quadruped   bouts=7  dead-reckoning 169.56 deg -> OIO 14.62 deg (-91.4%)
04_gym_baseline.py             cast-and-surge success rate 0.40, mean final dist 4.52 m
```

The realism gate (`tests/test_plume_gate.py`) and `test_physics_rigor.py` are
inside that 62 -- the plume statistics still bracket published turbulence
theory on this install.

## Realism gate (`scripts/validate_physics.py`, 600 s @ 100 Hz, probe 8 m downwind)

| series | blank CV | whiffs | gate |
|---|---|---|---|
| filament, full | 1.78 | 318 | |
| filament, meander ablated | 0.97 | 448 | |
| gaussian | nan | 1 | **FAIL (by design)** |
| via slow MOX (tau_fall 12 s) | 1.99 | 75 | PASS |
| via fast MOX (46 ms, Dennler-class) | 1.79 | 327 | PASS |

Slow sensor retains 23.6% of ground-truth whiffs; fast retains 102.8%. Both
README mechanisms hold: meander is what makes blank durations heavy-tailed
(CV 1.78 -> 0.97 when ablated), and sensor bandwidth gates what a policy can
see. The Gaussian series failing the gate is the intended negative control.

The README's specific figure of 2.31 for the full plume is seed-dependent --
across 5 seeds the full-plume CV is 1.72 +/- 0.41 (range 1.40-2.40), so this
run's 1.78 is typical and 2.31 is a high draw. The ablated 0.96 is robust
(0.950 +/- 0.020). See docs/RELEASE_VALIDATION.md.

## Fixes applied during setup

`scripts/validate_physics.py` did not run as shipped. Two pre-existing bugs,
both fixed:

1. `base_cfg()` passed `specific_gravity=1.0` to `FilamentPlumeConfig`, which
   has no such field -- `specific_gravity` belongs to `Species`
   (`chemistry/registry.py:30`). Raised `TypeError` on every invocation.
   Removed; behaviour-identical because `buoyancy_model` defaults to `"none"`,
   so the term is never applied.
2. Cache paths were hardcoded to `/tmp/full.npz` and `/tmp/nm.npz`, which do
   not resolve on Windows. Now `tempfile.gettempdir()`.

No other repo source was modified.
