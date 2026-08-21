import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from scentience_olfaction.chemistry.registry import SpeciesRegistry, Species, DEFAULT_REGISTRY
from scentience_olfaction.emitters.emitters import PointEmitter, BoxEmitter


def test_registry_alias_and_collision():
    assert DEFAULT_REGISTRY.get("EtOH").name == "ethanol"
    assert DEFAULT_REGISTRY.get("CO2").name == "carbon_dioxide"
    r = SpeciesRegistry()
    with pytest.raises(ValueError):   # alias shadowing must be loud
        r.add(Species("fake", 10.0, aliases=("CO2",)))
    with pytest.raises(KeyError):
        r.get("unobtainium")


def test_registry_json_roundtrip(tmp_path):
    p = tmp_path / "sp.json"
    DEFAULT_REGISTRY.to_json(p)
    r2 = SpeciesRegistry.from_file(p)
    assert r2.get("ethanol").molar_mass_g_mol == 46.07


def test_emitter_rate_exact_longrun():
    e = PointEmitter((0, 0, 0), release_rate_hz=2.5)
    rng = np.random.default_rng(0)
    total = sum(e.n_release(t * 0.1, 0.1, rng) for t in range(1000))  # 100 s
    assert total == 250  # fractional accumulator has no long-run bias


def test_emitter_pulse_and_window():
    e = PointEmitter((0, 0, 0), release_rate_hz=10, t_start=1.0, t_stop=2.0)
    rng = np.random.default_rng(0)
    assert e.n_release(0.5, 0.1, rng) == 0
    assert sum(e.n_release(1.0 + i * 0.1, 0.1, rng) for i in range(10)) == 10
    assert e.n_release(2.5, 0.1, rng) == 0
    p = PointEmitter((0, 0, 0), pulse_on_s=1.0, pulse_off_s=1.0)
    assert p.active(0.5) and not p.active(1.5) and p.active(2.5)


def test_box_emitter_positions_inside():
    b = BoxEmitter((1, 2, 3), size=(0.5, 0.5, 0.5))
    pts = b.sample_positions(200, np.random.default_rng(1))
    assert np.all(pts >= [1, 2, 3]) and np.all(pts <= [1.5, 2.5, 3.5])
