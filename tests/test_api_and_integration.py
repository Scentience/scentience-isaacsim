import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scentience_olfaction import (OlfactionWorld, FilamentPlume,
                                  FilamentPlumeConfig, Species, SpeciesRegistry)
from scentience_olfaction.airflow.fields import GridAirflow


def test_world_five_line_contract():
    w = OlfactionWorld.simple(seed=3)
    for _ in range(50):
        w.step(0.05)
    r = w.read((5, 0, 1), dt=0.05)
    assert set(r) == set(w.channel_names) and len(r) == 11
    t = w.truth((5, 0, 1))
    assert set(t) == {"ethanol"}
    assert w.wind_at((5, 0, 1)).shape == (3,)


def test_world_reset_determinism():
    w1 = OlfactionWorld.simple(seed=9)
    w2 = OlfactionWorld.simple(seed=9)
    for _ in range(100):
        w1.step(0.05); w2.step(0.05)
    assert w1.truth((4, 0, 1)) == w2.truth((4, 0, 1))
    w1.reset(seed=9)
    assert w1.plume.n_alive == 0


def test_species_decay_loses_mass():
    reg = SpeciesRegistry()
    reg.add(Species("fastdecay", 30.0, decay_rate_per_s=0.5))
    cfg = FilamentPlumeConfig(species="fastdecay", max_age_s=1e9,
                              domain_min=(-99, -99, -99), domain_max=(999, 99, 99))
    p = FilamentPlume(cfg, seed=0, registry=reg)
    p.step(0.1)                      # release a couple of filaments
    m0 = p.total_moles().sum()
    assert m0 > 0
    # stop releasing: push emitter window shut, then advect 4 s = 2 half-lives*
    p.emitters[0].t_stop = 0.0
    for _ in range(40):
        p.step(0.1)
    m1 = p.total_moles().sum()
    expect = m0 * np.exp(-0.5 * 4.0)
    assert abs(m1 - expect) / expect < 0.05, (m1, expect)


def test_plume_follows_grid_airflow():
    """Integration: a grid field pointing +y must carry the plume +y, not the
    legacy cfg.wind_mean direction."""
    dims = (40, 40, 8)
    u = np.zeros((*dims, 3)); u[..., 1] = 1.0        # wind blows +y
    af = GridAirflow(origin=np.array([-5.0, -5.0, 0.0]), cell_size=0.5, u=u)
    cfg = FilamentPlumeConfig(source_pos=(0, 0, 1), wind_mean=(1, 0, 0),
                              turbulence_intensity=0.1, meander_std_rad=0.0,
                              domain_min=(-5, -5, 0), domain_max=(15, 15, 4))
    p = FilamentPlume(cfg, seed=1, airflow=af)
    for _ in range(400):
        p.step(0.02)
    pos = p.pos[p.alive]
    assert pos[:, 1].mean() > 3.0 * abs(pos[:, 0].mean()), \
        "filaments should travel +y with the grid field"
