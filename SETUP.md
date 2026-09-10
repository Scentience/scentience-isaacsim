# Installation and validation

## Standalone Python

Use Python 3.10 or newer. Linux, macOS and Windows can run the NumPy core.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[viz,envs]"
python examples/01_minimal.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Run commands from the
repository root. Optional extras: `gpu` (Warp), `torch` (batched device),
`train` (Stable-Baselines3), `yaml` (species registry), and `dev` (development
and validation tools). Use `test` for CPU CI without Torch/Warp.

For the optional Scentience Python SDK, use `python -m pip install -e ".[bridge]"`.
The bridge supports `scentience>=2.2.2,<3`; the offline example
`python examples/07_scentience_sdk.py --ovl` needs no API key, radio or model
download. See [SDK integration](docs/SDK_INTEGRATION.md) for live hardware use.

## Isaac Sim and Isaac Lab

Use the interpreter supplied by your Isaac Sim 6.x / Isaac Lab 3.x
installation. Install this repository in editable mode in that environment;
avoid replacing its Torch, Warp, NumPy, CUDA or Kit packages with a second
environment's versions.

```bash
python -m pip install -e .
python scripts/validate_install.py
```

Launch Kit before importing Isaac-dependent modules. Follow
[the integration guide](docs/ISAAC_INTEGRATION.md) for scene configuration and
[compatibility notes](docs/ISAAC_COMPATIBILITY.md) for live checks. The optional
Kit extension lives in `isaac_extension/scentience.isaac.olfaction`; add
`isaac_extension` to Extension Manager search paths after installing the core.

## Verify changes

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -m "not isaac and not slow"
python -m pytest -m slow
python -m build
```

The long plume-statistics tests are CPU workloads and may take several minutes.
Warp can execute parity tests on CPU when a CUDA GPU is unavailable. Actual
Isaac rendering and robotics simulation still require a supported Isaac host.

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing `gymnasium`, `matplotlib`, or `stable_baselines3` | Install the matching `envs`, `viz`, or `train` extra using the same Python interpreter. |
| `omni`/Kit import errors | Start the Isaac application before importing simulator modules; a plain Python process cannot replace Kit. |
| No odor at a probe | Warm up transport, verify downwind position and sensor height, and inspect `world.truth()` plus `plot_plume.py`. |
| CO₂ changes slowly or repeats | SCD30 has 20 s diffusion time constant; SCD4x has 60 s. Output is sample-and-hold, independently of your control rate. |
| CO₂ has a doubled ambient offset | A plume with `background_ppm.carbon_dioxide` requires absolute concentration mode; world/Isaac defaults select it automatically. |
| Policy sees weak stereo cues | Compare baseline spacing, wind speed, sampling period and sensor lag; a hardware-size baseline may yield sub-tick arrival differences. |
| Warp rejects an option | Use `transport_backend="numpy"` for the documented full reference model; unsupported physics is rejected instead of ignored. |
| Cache directory is not writable | Set `MPLCONFIGDIR` and `WARP_CACHE_PATH` to writable directories before running. |
| A run directory already exists | Choose a new `--out` directory; experiment commands avoid replacing previous results. |
