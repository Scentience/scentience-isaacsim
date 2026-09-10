"""Inspect official Isaac Lab 3 source without importing or executing Isaac.

Usage: python scripts/check_isaaclab_contract.py /path/to/IsaacLab
Or pass one or more wheels containing isaaclab and isaaclab_physx source.
A passing result proves source shape only, never live PhysX or robot binding.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import zipfile


class Sources:
    def __init__(self, paths):
        self.files = {}
        for path in map(Path, paths):
            if path.is_dir():
                for file in path.rglob("*.py"):
                    self.files[file.as_posix()] = file.read_text(encoding="utf-8")
            elif zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        if name.endswith(".py"):
                            self.files[f"{path}!/{name}"] = archive.read(name).decode("utf-8")
            else:
                raise ValueError(f"not a source directory or wheel: {path}")

    def read(self, suffix):
        matches = [(p, text) for p, text in self.files.items() if p.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected one {suffix}, found {len(matches)}; provide Lab 3 + PhysX source")
        return matches[0][1]

    def cls(self, suffix, name):
        for node in ast.walk(ast.parse(self.read(suffix))):
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        raise ValueError(f"missing class {name} in {suffix}")


def method(cls, name):
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"missing {cls.name}.{name}")


def check_sources(sources):
    results = []

    def check(name, fn):
        try:
            fn()
            results.append((name, True, ""))
        except Exception as exc:
            results.append((name, False, str(exc)))

    def sensor():
        cls = sources.cls("isaaclab/sensors/sensor_base.py", "SensorBase")
        args = [a.arg for a in method(cls, "_update_buffers_impl").args.args]
        if args[1:] != ["env_mask"]:
            raise ValueError(f"requires Lab 3 env_mask API; got {args}")
        for name in ("_initialize_impl", "_update_outdated_buffers", "_resolve_rigid_body_ancestor_expr", "update"):
            method(cls, name)
        args = [a.arg for a in method(cls, "reset").args.args]
        if args != ["self", "env_ids", "env_mask"]:
            raise ValueError(f"reset API drift: {args}")
        text = sources.read("isaaclab/sensors/sensor_base.py")
        for member in ("_timestamp_last_update", "_timestamp", "_num_envs", "get_clone_plan"):
            if member not in text:
                raise ValueError(f"missing {member}")

    def imu():
        cls = sources.cls("isaaclab/sensors/imu/base_imu_data.py", "BaseImuData")
        for name in ("ang_vel_b", "lin_acc_b"):
            method(cls, name)
        cls = sources.cls("isaaclab/sensors/imu/imu_cfg.py", "OffsetCfg")
        rot = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id == "rot")
        if ast.literal_eval(rot.value) != (0., 0., 0., 1.):
            raise ValueError("ImuCfg identity is not xyzw")

    def views():
        method(sources.cls("isaaclab/assets/rigid_object/base_rigid_object.py", "BaseRigidObject"), "root_view")
        method(sources.cls("isaaclab_physx/physics/physx_manager.py", "PhysxManager"), "get_physics_sim_view")
        text = sources.read("isaaclab_physx/sensors/imu/imu.py")
        for name in ("create_rigid_body_view", "get_transforms", "get_gravity"):
            if name not in text:
                raise ValueError(f"PhysX Imu no longer uses {name}")

    def config():
        cls = sources.cls("isaaclab/sensors/sensor_base_cfg.py", "SensorBaseCfg")
        names = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
        if not {"prim_path", "update_period", "debug_vis"}.issubset(names):
            raise ValueError(f"SensorBaseCfg fields changed: {names}")
        method(sources.cls("isaaclab/managers/observation_manager.py", "ObservationManager"), "reset")
        text = sources.read("isaaclab/managers/manager_term_cfg.py")
        if "history_length" not in text or "flatten_history_dim" not in text:
            raise ValueError("observation history config unavailable")

    check("SensorBase masks, lazy capture, reset and clone initialization", sensor)
    check("lightweight Imu data and xyzw mount convention", imu)
    check("RigidObject root_view and PhysX sensor view access", views)
    check("sensor config and observation manager history", config)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args()
    try:
        sources = Sources(args.sources)
        results = check_sources(sources)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1
    print("SOURCE INSPECTION ONLY — Isaac was not executed")
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    try:
        digest = hashlib.sha256(sources.read("isaaclab/sensors/sensor_base.py").encode()).hexdigest()
        print(f"SensorBase SHA256: {digest}")
    except ValueError:
        pass
    return int(any(not ok for _, ok, _ in results))


if __name__ == "__main__":
    raise SystemExit(main())
