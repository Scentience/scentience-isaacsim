"""
Chemical species registry.

One place where species identity, physical properties, and naming live.
Everything downstream (emitters, transport, sensors) refers to species by
name and looks properties up here, so adding a species is a data change,
not a code change.

Design rules:
  * Extensible from JSON (always) and YAML (if PyYAML is present) without
    touching code.
  * Molecular identifiers (SMILES/InChI) are OPTIONAL metadata. The transport
    engine must never require them -- they exist so a later release can attach
    molecular embeddings (the COLIP/OVLM direction) without a breaking change.
  * Property values carry provenance in `sources` notes; defaults below are
    standard tabulated values (CRC-handbook-level facts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True)
class Species:
    name: str
    molar_mass_g_mol: float
    specific_gravity: float = 1.0
    """Density relative to air at 20 C. Used only if a buoyancy model is
    explicitly enabled -- v0.1 ships with buoyancy OFF (see CHEMICAL_MODEL.md)."""
    diffusivity_m2_s: float = 1.0e-5
    """Molecular diffusivity in air. Sets the floor of filament growth."""
    decay_rate_per_s: float = 0.0
    """First-order loss (photolysis, deposition, reaction). 0 = conserved."""
    aliases: tuple[str, ...] = ()
    smiles: str | None = None
    inchi: str | None = None
    notes: str = ""


# Standard tabulated values (molar masses exact; SG = M/28.96; diffusivities
# are textbook values at ~25 C, good to ~10%).
_DEFAULTS = [
    Species("ethanol", 46.07, 1.59, 1.18e-5, aliases=("C2H5OH", "EtOH"), smiles="CCO"),
    Species("methane", 16.04, 0.554, 2.2e-5, aliases=("CH4",), smiles="C"),
    Species("ammonia", 17.03, 0.588, 2.3e-5, aliases=("NH3",), smiles="N"),
    Species("carbon_monoxide", 28.01, 0.967, 2.0e-5, aliases=("CO",)),
    Species("carbon_dioxide", 44.01, 1.52, 1.6e-5, aliases=("CO2",)),
    Species("hydrogen", 2.016, 0.0696, 7.6e-5, aliases=("H2",)),
    Species("hydrogen_sulfide", 34.08, 1.19, 1.8e-5, aliases=("H2S",)),
    Species("nitrogen_dioxide", 46.01, 1.59, 1.4e-5, aliases=("NO2",)),
    Species("acetone", 58.08, 2.0, 1.06e-5, aliases=("C3H6O",), smiles="CC(C)=O"),
    Species("isopropanol", 60.10, 2.08, 1.0e-5, aliases=("IPA", "propan-2-ol")),
    Species("generic_voc", 100.0, 3.45, 8.0e-6, aliases=("VOC",),
            notes="placeholder for unspeciated VOC channels"),
]


class SpeciesRegistry:
    def __init__(self, species: list[Species] | None = None):
        self._by_name: dict[str, Species] = {}
        for s in (species if species is not None else _DEFAULTS):
            self.add(s)

    def add(self, s: Species) -> None:
        for key in (s.name, *s.aliases):
            k = key.lower()
            if k in self._by_name and self._by_name[k].name != s.name:
                raise ValueError(
                    f"name collision: {key!r} already maps to "
                    f"{self._by_name[k].name!r}; refusing to shadow it silently")
            self._by_name[k] = s

    def get(self, name: str) -> Species:
        try:
            return self._by_name[name.lower()]
        except KeyError:
            known = sorted({s.name for s in self._by_name.values()})
            raise KeyError(f"unknown species {name!r}; known: {known}") from None

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._by_name

    def names(self) -> list[str]:
        return sorted({s.name for s in self._by_name.values()})

    # ------------------------------------------------------------- file I/O
    @classmethod
    def from_file(cls, path: str | Path) -> "SpeciesRegistry":
        """Load from JSON always; YAML if PyYAML is installed. The file
        EXTENDS the defaults (override by reusing a name is an error --
        replace the registry wholesale if you mean to redefine a species)."""
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # optional dependency, deliberately
            except ImportError as e:
                raise ImportError(
                    "YAML species files need PyYAML (pip install pyyaml); "
                    "or use JSON, which needs nothing") from e
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        reg = cls()
        for entry in raw.get("species", []):
            entry["aliases"] = tuple(entry.get("aliases", ()))
            reg.add(Species(**entry))
        return reg

    def to_json(self, path: str | Path | None = None) -> str:
        payload = {"species": [asdict(s) for s in
                               sorted({s.name: s for s in self._by_name.values()}.values(),
                                      key=lambda s: s.name)]}
        s = json.dumps(payload, indent=2)
        if path:
            Path(path).write_text(s)
        return s


DEFAULT_REGISTRY = SpeciesRegistry()
