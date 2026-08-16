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
from pathlib import Path

import numpy as np


class EpisodeRecorder:
    def __init__(self, out_dir: str | Path, channel_names, species_names,
                 extra_meta: dict | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta = {"channel_names": list(channel_names),
                     "species_names": list(species_names),
                     "format_version": 1,
                     **(extra_meta or {})}
        self._rows: dict[str, list] = {}
        self._episode = 0

    def log(self, **fields) -> None:
        for k, v in fields.items():
            self._rows.setdefault(k, []).append(np.asarray(v, dtype=np.float64))

    def end_episode(self, **episode_meta) -> Path:
        arrays = {k: np.stack(v) for k, v in self._rows.items() if v}
        stem = f"episode_{self._episode:05d}"
        np.savez_compressed(self.out_dir / f"{stem}.npz", **arrays)
        meta = {**self.meta, **episode_meta,
                "n_steps": int(next(iter(arrays.values())).shape[0]) if arrays else 0,
                "fields": sorted(arrays),
                "written_unix": time.time()}
        (self.out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        self._rows.clear()
        self._episode += 1
        return self.out_dir / f"{stem}.npz"

    @staticmethod
    def load(npz_path: str | Path) -> tuple[dict, dict]:
        npz_path = Path(npz_path)
        arrays = dict(np.load(npz_path))
        meta = json.loads(npz_path.with_suffix(".json").read_text())
        return arrays, meta
