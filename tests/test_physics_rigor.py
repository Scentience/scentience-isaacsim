"""
Plume physics vs closed-form predictions. These are the tests that make the
transport credible: each asserts an analytic property of the model, not a
regression snapshot.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from scentience_olfaction.plume.filament import (FilamentPlume,
                                                 FilamentPlumeConfig, GAUSS3D)


def test_single_filament_peak_matches_closed_form():
    """Ballistic filament (no turbulence): peak concentration at a probe on
    the trajectory must equal 20 ppm * (sigma0/sigma(t_transit))^3 exactly --
    validates release normalisation, growth, and sampling in one shot."""
    cfg = FilamentPlumeConfig(release_rate_hz=100.0, turbulence_intensity=0,
                              sigma_u_floor=0, meander_std_rad=0, gamma=2e-3,
                              sigma0=0.05, ppm_center_initial=20.0, max_age_s=1e9)
    p = FilamentPlume(cfg, seed=0)
    p.step(0.01)
    p.emitters[0].t_stop = 0.0
    assert p.n_alive == 1
    probe = np.array([[3.0, 0.0, 1.0]])
    peak = 0.0
    for _ in range(2000):
        p.step(0.01)
        peak = max(peak, p.sample(probe)[0])
    sig = math.sqrt(cfg.sigma0 ** 2 + cfg.gamma * 3.0)
    c_pred = 20.0 * (cfg.sigma0 / sig) ** 3
    assert abs(peak / c_pred - 1.0) < 0.03

    # growth law exact while we have a lone filament
    s_meas = p.sigma[p.alive][0]
    s_pred = math.sqrt(cfg.sigma0 ** 2 + cfg.gamma * p.age[p.alive][0])
    assert abs(s_meas - s_pred) < 1e-9


@pytest.mark.slow
def test_dispersion_dt_invariant():
    """Plume width at fixed downwind distance must not depend on the
    timestep -- the end-to-end expression of the exact OU update. A dt-scaled
    turbulence kick fails this at ratio ~2."""
    def width_at(dt):
        c = FilamentPlumeConfig(turbulence_intensity=0.3, lagrangian_timescale=1.0,
                                meander_std_rad=0, release_rate_hz=60,
                                max_age_s=30, max_filaments=6000)
        q = FilamentPlume(c, seed=11)
        for _ in range(int(40 / dt)):
            q.step(dt)
        pos = q.pos[q.alive]
        m = (pos[:, 0] > 7) & (pos[:, 0] < 9)
        return pos[m, 1].std()
    r = width_at(0.005) / width_at(0.02)
    assert 0.8 < r < 1.25, f"width ratio {r:.3f}: dispersion is dt-dependent"


@pytest.mark.slow
def test_mass_flux_conservation():
    """Time-averaged flux of gas through a downwind plane must equal the
    source emission rate (within the ~3% truncated by the 3-sigma cutoff and
    sampling error). The strongest single validation of the concentration
    normalisation chain."""
    cfg = FilamentPlumeConfig(release_rate_hz=40, turbulence_intensity=0.25,
                              meander_std_rad=0, max_age_s=60, max_filaments=8000,
                              domain_min=(-5, -30, -20), domain_max=(90, 30, 25))
    p = FilamentPlume(cfg, seed=2)
    for _ in range(3000):
        p.step(0.01)
    ys = np.linspace(-6, 6, 41)
    zs = np.linspace(-5, 7, 41)
    dy, dz = ys[1] - ys[0], zs[1] - zs[0]
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    plane = np.stack([np.full(Y.size, 8.0), Y.ravel(), Z.ravel()], 1)
    acc = np.zeros(plane.shape[0])
    nsamp = 300
    for i in range(nsamp * 5):
        p.step(0.01)
        if i % 5 == 0:
            acc += p.sample(plane)
    cbar = acc / nsamp
    n_air = cfg.n_air_mol_per_m3()
    flux = (cbar / 1e6 * n_air * 1.0 * dy * dz).sum()
    q_src = p.emitters[0].mass_flux_mol_s(n_air)
    assert 0.80 < flux / q_src < 1.15, f"flux/source = {flux / q_src:.3f}"


def test_chunked_sampling_equals_brute_force():
    cfg = FilamentPlumeConfig(release_rate_hz=60, max_filaments=3000)
    p = FilamentPlume(cfg, seed=3)
    for _ in range(800):
        p.step(0.01)
    pts = np.random.default_rng(1).uniform([-1, -5, 0], [15, 5, 3], (37, 3))
    full = p.sample_species(pts).sum(1)
    idx = np.flatnonzero(p.alive)
    fp, fs, fm = p.pos[idx], p.sigma[idx], p.moles[idx]
    n_air = cfg.n_air_mol_per_m3()
    cpk = 1e6 * fm / (n_air * GAUSS3D * fs ** 3)
    d2 = ((pts[:, None, :] - fp[None, :, :]) ** 2).sum(2)
    w = d2 < (3.0 * fs[None, :]) ** 2
    brute = np.where(w, cpk[None, :] * np.exp(-0.5 * d2 / fs[None, :] ** 2), 0).sum(1)
    assert np.allclose(full, brute, atol=1e-10)
