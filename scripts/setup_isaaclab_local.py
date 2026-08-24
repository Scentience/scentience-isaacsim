"""One-command environment for testing the Isaac Lab wrapper WITHOUT a GPU
or an Isaac Sim install.

Creates `.venv-isaaclab/` beside the repo root and installs the genuine
`isaaclab==2.3.2` wheel (no-deps, to skip its Isaac-Sim-only requirements)
plus the pure-Python dependencies its import graph actually needs -- each one
on this list was discovered empirically by following ModuleNotFoundErrors.

    python scripts/setup_isaaclab_local.py
    # then:
    .venv-isaaclab/Scripts/python scripts/check_isaaclab_binding.py .
    .venv-isaaclab/Scripts/python scripts/demo_isaaclab_sensor_local.py

Works on CPU-only machines; uses CUDA automatically when present.
"""
import os
import subprocess
import venv

# Discovered by following the wheel's import graph; see
# check_isaaclab_binding.py's docstring for the provenance of each entry.
DEPS = [
    "numpy<2",              # isaaclab 2.3.2 pins numpy<2
    "torch",
    "warp-lang>=1.5",
    "usd-core",             # real pxr; only PhysX-specific schemas are stubbed
    "toml", "packaging", "flatdict", "prettytable",
    "scipy", "trimesh", "pyyaml", "h5py",
    "matplotlib", "opencv-python-headless", "hidapi",
    "gymnasium",
]


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_dir = os.path.join(repo, ".venv-isaaclab")
    py = os.path.join(env_dir, "Scripts" if os.name == "nt" else "bin",
                      "python" + (".exe" if os.name == "nt" else ""))

    if not os.path.exists(py):
        print("[setup] creating", env_dir)
        venv.EnvBuilder(with_pip=True).create(env_dir)
    else:
        print("[setup] reusing", env_dir)

    def pip(*args):
        cmd = [py, "-m", "pip", "install", "--quiet", *args]
        print("[setup]", " ".join(cmd[3:]))
        return subprocess.call(cmd)

    rc = pip("--upgrade", "pip")
    rc |= pip(*DEPS)
    rc |= pip("--no-deps", "isaaclab==2.3.2")
    if rc:
        print("[setup] FAILED -- see pip output above")
        return 1

    print("[setup] done. Try:")
    rel = os.path.relpath(py, repo)
    print(f"  {rel} scripts/check_isaaclab_binding.py .")
    print(f"  {rel} scripts/demo_isaaclab_sensor_local.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
