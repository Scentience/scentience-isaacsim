"""Strict JSON configuration for reproducible experiments, without optional parsers."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from .emitters.emitters import BoxEmitter, LineEmitter, PointEmitter
from .plume.filament import FilamentPlumeConfig


def dataclass_from_dict(cls, values: dict):
    """Reject misspelled keys instead of silently changing an experiment."""
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} requires an object")
    allowed = {f.name for f in fields(cls) if f.init and not f.name.startswith("_")}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def plume_config(values: dict) -> FilamentPlumeConfig:
    values = dict(values)
    if values.get("emitters") is not None:
        emitters = []
        for entry in values["emitters"]:
            entry = dict(entry)
            kind = entry.pop("type", "point")
            classes = {"point": PointEmitter, "line": LineEmitter, "box": BoxEmitter}
            if kind not in classes:
                raise ValueError(f"Unknown emitter type {kind!r}; choose {tuple(classes)}")
            emitters.append(dataclass_from_dict(classes[kind], entry))
        values["emitters"] = emitters
    return dataclass_from_dict(FilamentPlumeConfig, values)


def load_world(path: str | Path):
    """Load plume and device configuration. Unknown keys are errors."""
    from .api import OlfactionWorld
    from .plume.filament import FilamentPlume

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"plume", "sensor_profile", "device_profile", "device_config", "seed",
               "stereo_baseline_m"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError(f"World configuration requires keys from {sorted(allowed)}")
    plume = plume_config(raw.pop("plume", {}))
    return OlfactionWorld(FilamentPlume(plume, seed=raw.get("seed", 0)), **raw)


def load_navigation(path: str | Path):
    """Load a PlumeNavConfig; Gymnasium is imported only on this path."""
    from .envs.plume_nav import PlumeNavConfig

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Navigation configuration requires an object")
    if "plume" in raw:
        raw["plume"] = plume_config(raw["plume"])
    return dataclass_from_dict(PlumeNavConfig, raw)


def config_dict(config) -> dict:
    """Serializable public dataclass fields, including explicit emitter types.

    Unbounded times are omitted, allowing standards-compliant JSON rather than
    JavaScript's nonstandard Infinity token. The constructor restores defaults.
    """
    import math

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if not k.startswith("_") and not (isinstance(v, float) and math.isinf(v))}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    out = clean(asdict(config))
    plume = getattr(config, "plume", config)
    target = out.get("plume", out)
    if getattr(plume, "emitters", None) is not None:
        for entry, emitter in zip(target["emitters"], plume.emitters):
            entry["type"] = "box" if isinstance(emitter, BoxEmitter) else (
                "line" if isinstance(emitter, LineEmitter) else "point")
    return out
