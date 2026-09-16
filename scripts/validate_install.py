"""Live dependency and class-binding checks for Isaac Sim 6 / Isaac Lab 3.

Run with Isaac Lab's Python: ./isaaclab.sh -p /repo/scripts/validate_install.py --headless
This launches the real application. It does not create a scene; follow with
verify_in_isaac.py to exercise views, sensors, real robot assets and resets.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_checks():
    checks = []

    def check(name, fn):
        try:
            detail = fn()
            print(f"[PASS] {name}: {detail}")
            checks.append(True)
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            checks.append(False)

    def versions():
        found = {}
        for name, major in (("isaacsim", "6"), ("isaaclab", "3")):
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                module = __import__(name)
                version = getattr(module, "__version__", "unknown")
            if version.split(".")[0] != major:
                raise ValueError(f"expected {name} {major}.x; detected {version}")
            found[name] = version
        return found

    def binding():
        from isaaclab.sensors import SensorBase, ImuCfg
        from scentience_isaaclab.olfactory_sensor import OlfactorySensor, OlfactorySensorCfg
        if "env_mask" not in inspect.signature(SensorBase._update_buffers_impl).parameters:
            raise ValueError("requires Lab 3 env_mask API")
        cfg = OlfactorySensorCfg(prim_path="/World/envs/env_.*/Robot/base")
        cfg.validate()
        assert issubclass(OlfactorySensor, SensorBase) and cfg.class_type is OlfactorySensor
        assert cfg.offset.rot == ImuCfg.OffsetCfg().rot == (0., 0., 0., 1.)
        assert cfg.copy().to_dict() == cfg.to_dict()
        return "real SensorBase/configclass; xyzw offset; config round trip"

    def devices():
        import torch
        import warp as wp
        return {"torch": torch.__version__, "warp": wp.__version__,
                "torch_cuda": torch.cuda.is_available(), "warp_cuda_count": wp.get_cuda_device_count()}

    check("target versions", versions)
    check("sensor and standard Imu bindings", binding)
    check("tensor backends (CPU is allowed for diagnostic runs)", devices)
    print("Binding checks only; live scene verification still requires verify_in_isaac.py.")
    return int(not all(checks))


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    try:
        return run_checks()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
