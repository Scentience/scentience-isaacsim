"""
Episode recorder: structured logs for training data, debugging, and the
figures that will back publications.

Format: compressed NPZ (arrays) + JSON sidecar (metadata) -- both readable
everywhere with zero optional dependencies. Channel and species ORDER is
explicit in the metadata, always; a log whose column meaning depends on code
version is a corrupted log that does not know it yet. Parquet export can land
later without changing callers (roadmap).
"""

from __future__ import annotations

import json
import time
import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


class EpisodeRecorder:
    def __init__(self, out_dir: str | Path, channel_names, species_names,
                 extra_meta: dict | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        reserved = {"channel_names", "species_names", "format_version", "n_steps", "fields", "written_unix"}
        if reserved & (extra_meta or {}).keys():
            raise ValueError("extra_meta cannot override recorder schema fields")
        self.meta = {"channel_names": list(channel_names),
                     "species_names": list(species_names),
                     "format_version": 1,
                     **(extra_meta or {})}
        self._rows: dict[str, list] = {}
        self._episode = 0
        self._shapes: dict[str, tuple] = {}

    def log(self, **fields) -> None:
        if not fields:
            raise ValueError("A recorded row must contain at least one field")
        if self._rows and fields.keys() != self._rows.keys():
            raise ValueError("Every row in an episode must contain the same fields")
        arrays = {k: np.array(v, dtype=np.float64, copy=True) for k, v in fields.items()}
        if any(k in self._shapes and v.shape != self._shapes[k] for k, v in arrays.items()):
            raise ValueError("Field shapes must remain constant within an episode")
        for k, v in arrays.items():
            self._shapes[k] = v.shape
            self._rows.setdefault(k, []).append(v)

    def end_episode(self, **episode_meta) -> Path:
        if {"channel_names", "species_names", "format_version", "n_steps", "fields", "written_unix"} & episode_meta.keys():
            raise ValueError("episode metadata cannot override recorder schema fields")
        arrays = {k: np.stack(v) for k, v in self._rows.items() if v}
        meta = {**self.meta, **episode_meta,
                "n_steps": int(next(iter(arrays.values())).shape[0]) if arrays else 0,
                "fields": sorted(arrays),
                "written_unix": time.time()}
        payload = json.dumps(meta, indent=2, allow_nan=False)
        # Reserve a new filename, including after a process restart. Never
        # silently overwrite another experiment's episode_00000.
        while True:
            stem = f"episode_{self._episode:05d}"
            path = self.out_dir / f"{stem}.npz"
            self._episode += 1
            if path.with_suffix(".json").exists():
                continue
            try:
                handle = path.open("xb")
                break
            except FileExistsError:
                continue
        try:
            with handle:
                np.savez_compressed(handle, **arrays)
            path.with_suffix(".json").write_text(payload, encoding="utf-8")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self._rows.clear()
        self._shapes.clear()
        return path

    @staticmethod
    def load(npz_path: str | Path) -> tuple[dict, dict]:
        npz_path = Path(npz_path)
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = dict(data)
        meta = json.loads(npz_path.with_suffix(".json").read_text())
        return arrays, meta


def run_metadata(config: dict, seed: int, **extra) -> dict:
    """Record software versions and a stable configuration fingerprint."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    versions = {"python": platform.python_version()}
    for package in ("scentience-olfaction", "numpy", "torch", "warp-lang", "gymnasium",
                    "stable-baselines3"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return {**extra, "seed": seed, "config": config, "versions": versions,
            "config_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
