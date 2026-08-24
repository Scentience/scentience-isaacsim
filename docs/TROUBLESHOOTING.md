# Troubleshooting

Every entry below is a failure mode actually reproduced during release
validation, with the exact symptom and the fix.

---

**1. `ImportError: ... needs torch for the batched device`**
You imported `sensors/scentience_v1.py` (the vectorised torch device) in a
numpy-only install. The metadata (`CHANNELS`, `PROFILES`,
`register_coefficients`) still works without torch; only constructing the
batched device needs it. Fix: `pip install torch`, or use the NumPy device
(`sensors/device_np.py`) -- it is the supported standalone path.

**2. `ImportError: PlumeNav needs gymnasium`**
The RL environment is an optional extra. Fix: `pip install "scentience-olfaction[envs]"`
(or `pip install gymnasium`).

**3. `ModuleNotFoundError: No module named 'warp'` when running your own
scripts against the GPU twin**
The GPU transport twin is the `[gpu]` extra: `pip install "scentience-olfaction[gpu]"`.
The test suite itself skips warp tests cleanly when warp is absent.

**4. No CUDA device / laptop without NVIDIA GPU**
Warp falls back to CPU and the twin still runs (verified: with CUDA hidden,
`WarpFilamentPlume` steps correctly on the `cpu` device). Expect the FIRST
run to pause a few seconds while kernels compile; they cache afterwards.
The CPU path exists for CI parity, not throughput -- the NumPy reference is
the specification and is what you should benchmark against anyway.

**5. First device reading looks like clean air even inside a plume**
Sensor dynamics are real: the packaged profile has tau_rise ~3 s, so a single
`read()` after one step shows ~no deflection. Step the world (and keep calling
`read(dt=...)` with your control period) for several time constants, or use
`sensor_profile="fast_modulated"`. This is the README's "sensor bandwidth
gates what a policy can see" point, not a bug.

**6. `pytest -m "not isaac"` suddenly fails the plume gate after you edited
transport or airflow code**
Working as designed: the realism gate (`tests/test_plume_gate.py`) fails the
build when plume statistics regress (e.g. blank-duration CV drops below its
floor because meander was weakened). See `CHEMICAL_MODEL.md` before tuning
constants.

**7. `examples/05_stereo_olfaction.py` says LEFT but your own variant says
RIGHT sometimes**
The shipped example disables meander so geometry is deterministic. With
meander ON, the plume snakes across the robot and the correct answer flips
with it on a ~15 s timescale -- that is the search problem, not an error.

**8. Stereo cue reads zero almost always**
At the default 0.04 m die baseline, 20 Hz sampling and ~1 m/s wind, the
inter-sensor onset lag is usually below one control tick, so lag-based
lateralisation quantises to zero and behaviour degrades to plain upwind
cast-and-surge (documented in `StereoCastAndSurge`). Widen
`stereo_baseline_m`, slow the wind, or raise the sampling rate.

**9. BLE frames are missing compounds you expected**
`bridge/ble_schema.py` omits zero-magnitude compounds from the frame because
the hardware does. The `_sim_units: "ppm"` key is a simulator-only extension
(hardware sends no units at all -- see `UPSTREAM_REQUESTS.md`, UR-1).

**10. Isaac: `validate_install.py` fails with `_update_buffers_impl takes
'env_mask', not 'env_ids'`**
You are on Isaac Lab 3.0 / Isaac Sim 6.x. This release targets Isaac Lab
2.3.x / Isaac Sim 5.1 and refuses the 3.0 API on purpose rather than half
working. Install the 2.3.x/5.1 pairing, or wait for the
`experimental/isaac-lab-3.0` branch (see `BRANCHING.md`).

**11. Isaac Sim itself will not start: "Skipping unsupported non-RTX GPU" /
"No device could be created"**
Isaac Sim's renderer requires RT cores (minimum spec RTX 4080-class, 16 GB
VRAM, driver >= 580.88). No driver update fixes a GPU without RT cores. The
standalone and Gymnasium paths -- including Warp compute -- do not need any of
that and run fine on such machines. See `ISAAC_COMPATIBILITY.md`.

**12. Windows: activating the venv in PowerShell is blocked**
`.\.venv\Scripts\Activate.ps1` can be stopped by execution policy. Either
`Set-ExecutionPolicy -Scope Process RemoteSigned`, or skip activation and
call the interpreter directly: `.\.venv\Scripts\python.exe ...` (all repo
docs use the direct form).

---

Still stuck? Open an issue with the output of:

```bash
python -c "import sys, numpy, scentience_olfaction as so; print(sys.version); print('numpy', numpy.__version__); print('scentience-olfaction', so.__version__ if hasattr(so,'__version__') else 'dev')"
pip list
```
