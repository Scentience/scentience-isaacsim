"""Transport contracts: analytic mass budgets, boundaries, lifecycle, backend scope.

Warp runs on CPU in ordinary CI; the same cases also run on CUDA when available.
Deterministic parity disables turbulence; statistical parity stays in its existing test.
"""
import math

import numpy as np
import pytest

from scentience_olfaction.airflow.fields import GridAirflow
from scentience_olfaction.chemistry.registry import Species, SpeciesRegistry
from scentience_olfaction.emitters.emitters import BoxEmitter, LineEmitter, PointEmitter
from scentience_olfaction.geometry.occupancy import OccupancyGrid
from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig


def config(**kw):
    return FilamentPlumeConfig(**{
        "release_rate_hz": 20., "wind_mean": (0., 0., 0.),
        "turbulence_intensity": 0., "sigma_u_floor": 0., "meander_std_rad": 0.,
        "gamma": 0., "max_filaments": 100, **kw})


@pytest.fixture(params=["numpy", "warp_cpu", "warp_cuda"])
def backend(request):
    if request.param == "numpy":
        return lambda cfg, **kw: FilamentPlume(cfg, **kw)
    wp = pytest.importorskip("warp")
    from scentience_olfaction.transport.filament_warp import WarpFilamentPlume
    device = request.param.removeprefix("warp_")
    if device == "cuda" and not wp.get_cuda_device_count():
        pytest.skip("CUDA unavailable")
    return lambda cfg, **kw: WarpFilamentPlume(cfg, device=device, **kw)


def sample(p, point=(0., 0., 1.)):
    if isinstance(p, FilamentPlume):
        return p.sample([point])
    p.set_probes(np.tile(point, (p.n_envs, 1)))
    return p.sample()


def assert_balanced(p):
    d = p.diagnostics()
    assert np.allclose(d["balance_error_mol"], 0., atol=1e-14)
    assert np.allclose(d["released_mol"] + d["pool_rejected_mol"], d["emitted_mol"])
    return d


@pytest.mark.parametrize("kw", [
    {"gamma": -1}, {"sigma0": 0}, {"temperature_k": 0}, {"pressure_atm": np.inf},
    {"lagrangian_timescale": 0}, {"meander_timescale": -1}, {"max_step_s": 0},
    {"max_filaments": 0}, {"max_filaments": 2.5}, {"max_filaments": True},
    {"release_rate_hz": -1}, {"turbulence_intensity": np.nan}, {"max_age_s": -1},
    {"wind_mean": (1, 2)}, {"domain_min": (70, 0, 0)}, {"buoyancy_model": "typo"},
    {"background_ppm": {"CO2": -1}}, {"molar_rate_mol_s": -1},
    {"molar_rate_mol_s": 1, "release_rate_hz": 0},
])
def test_invalid_config(kw):
    with pytest.raises(ValueError):
        config(**kw)


@pytest.mark.parametrize("dt", [-.01, np.nan, np.inf, -np.inf])
def test_invalid_dt_is_atomic(backend, dt):
    p = backend(config())
    with pytest.raises(ValueError):
        p.step(dt)
    assert p.t == 0 and np.asarray(p.n_alive).sum() == 0
    assert not np.any(p.diagnostics()["emitted_mol"])


def test_zero_step_is_noop(backend):
    p, q = backend(config(), seed=17), backend(config(), seed=17)
    p.step(0)
    p.step(.05); q.step(.05)
    assert np.array_equal(sample(p), sample(q))
    assert p.t == q.t


@pytest.mark.parametrize("ctor,kw", [
    (PointEmitter, {"sigma0": 0}), (PointEmitter, {"release_rate_hz": -1}),
    (PointEmitter, {"molar_rate_mol_s": np.nan}),
    (PointEmitter, {"pulse_on_s": 0}), (PointEmitter, {"pulse_off_s": -1}),
    (PointEmitter, {"t_start": 2, "t_stop": 1}),
    (LineEmitter, {"end": (np.nan, 0, 0)}), (BoxEmitter, {"size": (-1, 1, 1)}),
])
def test_invalid_emitter(ctor, kw):
    with pytest.raises(ValueError):
        ctor((0, 0, 1), **kw)


def test_window_and_pulses_integrate_across_step_boundaries():
    rng = np.random.default_rng(3)
    e = PointEmitter((0, 0, 1), release_rate_hz=20, t_start=.25, t_stop=.75)
    assert e.n_release(0, 1, rng) == 10
    pulse = dict(release_rate_hz=100, t_start=.15, t_stop=2.85,
                 pulse_on_s=.2, pulse_off_s=.3)
    a, b = (PointEmitter((0, 0, 1), **pulse) for _ in range(2))
    coarse = a.n_release(0, 3, rng)
    fine = sum(b.n_release(i * .01, .01, rng) for i in range(300))
    assert coarse == fine == 120
    assert PointEmitter((0, 0, 1), release_rate_hz=1).n_release(0, 0, rng) == 0


def test_fractional_rate_does_not_lose_packet_at_integer_boundary(backend):
    p = backend(config(release_rate_hz=1))
    for _ in range(20):
        p.step(.05)
    assert np.asarray(p.n_alive).sum() == 1


def test_emitters_are_owned_per_plume_and_per_entry():
    e = PointEmitter((0, 0, 1), release_rate_hz=2.5)
    cfg = config(emitters=[e, e])
    a, b = FilamentPlume(cfg), FilamentPlume(cfg)
    assert a.emitters[0] is not a.emitters[1] and a.emitters[0] is not e
    for p in (a, b, a, b):
        p.step(.1)
    assert a.n_alive == b.n_alive == 0
    a.reset(seed=5)
    assert b.emitters[0]._accum == pytest.approx(.5)
    assert e._accum == 0
    a.emitters[0].position = (3, 0, 1)
    assert b.emitters[0].position == (0, 0, 1)


@pytest.mark.parametrize("rate,sigma,temp,pressure", [(20, .1, 293.15, 1), (40, .2, 310, .8)])
def test_physical_source_rate_independent_of_packet_size_and_air(backend, rate, sigma, temp, pressure):
    p = backend(config(release_rate_hz=rate, sigma0=sigma, temperature_k=temp,
                       pressure_atm=pressure, molar_rate_mol_s=2e-6))
    p.step(.5)
    d = assert_balanced(p)
    assert np.allclose(d["retained_mol"], 1e-6, rtol=1e-12, atol=0)
    assert np.allclose(d["emitted_filaments"], rate * .5)


def test_background_survives_empty_pool_reset_and_adds_to_source(backend):
    p = backend(config(species="CO2", background_ppm={"CO2": 420}, release_rate_hz=20))
    assert p.species_names == ["carbon_dioxide"]
    assert np.allclose(sample(p), 420)
    p.step(.05)
    assert np.allclose(sample(p), 440, rtol=1e-6)
    d = assert_balanced(p)
    assert np.all(d["retained_mol"] > 0)
    p.reset(seed=3)
    assert np.allclose(sample(p), 420)
    assert not np.any(p.total_moles())


def test_background_only_species_and_alias_aggregation():
    p = FilamentPlume(config(emitters=[], background_ppm={"CO2": 420, "ethanol": 2}))
    assert p.species_names == ["carbon_dioxide", "ethanol"]
    assert np.array_equal(p.sample_species([[0, 0, 1], [100, 0, 1]]), [[420, 2], [420, 2]])
    assert not p.total_moles().any()
    with pytest.raises(ValueError, match="duplicate background"):
        FilamentPlume(config(background_ppm={"CO2": 420, "carbon_dioxide": 420}))
    e = [PointEmitter((0, 0, 1), species=s) for s in ("CO2", "carbon_dioxide")]
    p = FilamentPlume(config(emitters=e))
    p.step(.05)
    assert p.species_names == ["carbon_dioxide"]
    assert p.sample_species([[0, 0, 1]])[0, 0] == pytest.approx(40)


def test_large_step_matches_explicit_substeps(backend):
    cfg = config(wind_mean=(1, .2, 0), turbulence_intensity=.3,
                 meander_std_rad=.2, gamma=.001)
    a, b = backend(cfg, seed=17), backend(cfg, seed=17)
    a.step(.2)
    for _ in range(4):
        b.step(.05)
    assert np.array_equal(sample(a), sample(b))
    assert np.array_equal(a.diagnostics()["emitted_filaments"], b.diagnostics()["emitted_filaments"])
    # Verify actual state, not just a potentially empty probe.
    pa = a.pos if isinstance(a, FilamentPlume) else a.pos.numpy()
    pb = b.pos if isinstance(b, FilamentPlume) else b.pos.numpy()
    assert np.array_equal(pa, pb)


def test_seeded_reset_replays_full_state(backend):
    cfg = config(turbulence_intensity=.3, wind_mean=(1, 0, 0), meander_std_rad=.2)
    p = backend(cfg, seed=4)
    p.step(.23)
    p.reset(seed=9)
    q = backend(cfg, seed=9)
    for _ in range(7):
        p.step(.03); q.step(.03)
    assert p.t == q.t
    assert np.array_equal(sample(p), sample(q))
    pa = p.pos if isinstance(p, FilamentPlume) else p.pos.numpy()
    qa = q.pos if isinstance(q, FilamentPlume) else q.pos.numpy()
    assert np.array_equal(pa, qa)
    for key, value in p.diagnostics().items():
        assert np.array_equal(value, q.diagnostics()[key]), key


def test_pool_rejects_without_replacing_live_mass(backend):
    p = backend(config(max_filaments=2, release_rate_hz=100, wind_mean=(1, 0, 0),
                       molar_rate_mol_s=1e-6))
    p.step(.05)  # 5 arrivals into 2 slots, exercises n_new > capacity
    p.step(.05)
    d = assert_balanced(p)
    assert np.all(d["emitted_filaments"] == 10)
    assert np.all(d["released_filaments"] == 2)
    assert np.all(d["pool_rejected_filaments"] == 8)
    age = p.age[p.alive] if isinstance(p, FilamentPlume) else p.age.numpy()[p.alive.numpy() == 1]
    assert np.allclose(age, .1)  # Warp used to reset these ages by overwriting.
    assert np.isfinite(sample(p)).all()


@pytest.mark.parametrize("kw,key", [
    ({"max_age_s": .01}, "age_mol"),
    ({"wind_mean": (10, 0, 0), "domain_max": (.1, 20, 6)}, "domain_mol"),
])
def test_culling_mass_is_accounted(backend, kw, key):
    p = backend(config(**kw))
    p.step(.05)
    d = assert_balanced(p)
    assert np.all(d[key] > 0) and not d["retained_mol"].any()


def test_cpu_decay_and_outlet_mass_accounting():
    registry = SpeciesRegistry([Species("tracer", 30, decay_rate_per_s=.5)])
    p = FilamentPlume(config(species="tracer", molar_rate_mol_s=2e-6), registry=registry)
    p.step(.05)
    p.emitters[0].t_stop = 0
    p.step(.2)
    d = assert_balanced(p)
    initial = 2e-6 / 20
    assert d["retained_mol"][0] == pytest.approx(initial * math.exp(-.5 * .25))
    assert d["decayed_mol"][0] == pytest.approx(initial - d["retained_mol"][0])
    occ = OccupancyGrid.from_boxes((-1, -1, 0), (1, 1, 2), .25, [])
    p = FilamentPlume(config(source_pos=(.7, 0, 1), wind_mean=(2, 0, 0)), occupancy=occ)
    p.step(.05)
    assert assert_balanced(p)["outlet_mol"][0] > 0


def test_slip_dilution_uses_each_emitters_initial_radius():
    emitters = [PointEmitter((0, 0, 1), species="CO2", sigma0=s) for s in (.05, .2)]
    p = FilamentPlume(config(emitters=emitters, buoyancy_model="slip"))
    p.step(.05)
    assert p.pos[p.alive][0, 2] == pytest.approx(p.pos[p.alive][1, 2])


def test_grid_clamps_continuously_at_boundary_and_supports_singleton_axes():
    u = np.zeros((3, 1, 1, 3)); u[:, 0, 0, 0] = [0, 1, 2]
    g = GridAirflow(np.zeros(3), 1., u)
    pts = [[-.5, .5, .5], [.5, .5, .5], [2.5, .5, .5], [3., .5, .5]]
    assert np.allclose(g.velocity(pts)[:, 0], [0, 0, 2, 2])
    assert g.velocity([[2.5 - 1e-8, .5, .5]])[0, 0] == pytest.approx(2)
    with pytest.raises(ValueError):
        g.velocity([[np.nan, 0, 0]])


def test_warp_rejects_unsupported_physics():
    pytest.importorskip("warp")
    from scentience_olfaction.transport.filament_warp import WarpFilamentPlume
    for kw in ({"emitters": []}, {"buoyancy_model": "slip"}, {"background_ppm": {"CO2": 420}}):
        with pytest.raises(NotImplementedError):
            WarpFilamentPlume(config(**kw), device="cpu")
    registry = SpeciesRegistry([Species("tracer", 30, decay_rate_per_s=.5)])
    with pytest.raises(NotImplementedError, match="decay"):
        WarpFilamentPlume(config(species="tracer"), registry=registry, device="cpu")
    with pytest.raises(NotImplementedError):
        WarpFilamentPlume(config(), occupancy=object(), device="cpu")


def test_warp_partial_reset_clears_phase_and_preserves_other_environment():
    pytest.importorskip("warp")
    from scentience_olfaction.transport.filament_warp import WarpFilamentPlume
    cfg = config(release_rate_hz=10, wind_mean=(1, 0, 0), turbulence_intensity=.2, meander_std_rad=.2)
    a, b = (WarpFilamentPlume(cfg, n_envs=2, device="cpu", seed=7) for _ in range(2))
    a.step(.15); b.step(.15)
    a.reset([0])
    assert a.env_time_s[0] == 0
    a.step(.05); b.step(.05)
    assert a.n_alive[0] == 0  # leftover fractional release was cleared
    assert np.array_equal(a.pos.numpy()[1], b.pos.numpy()[1])
    assert np.array_equal(a.alive.numpy()[1], b.alive.numpy()[1])
    assert np.array_equal(a.meander[1], b.meander[1])
    for key, value in a.diagnostics().items():
        if key != "species_names":
            assert np.array_equal(value[1], b.diagnostics()[key][1]), key
    with pytest.raises(ValueError):
        a.reset([-1])
    with pytest.raises(ValueError):
        a.reset([0], seed=3)


def test_deterministic_cpu_warp_sampling_parity():
    pytest.importorskip("warp")
    from scentience_olfaction.transport.filament_warp import WarpFilamentPlume
    cfg = config(wind_mean=(1, 0, 0), gamma=.002, background_ppm={"ethanol": 2})
    a, b = FilamentPlume(cfg), WarpFilamentPlume(cfg, device="cpu")
    for _ in range(20):
        a.step(.02); b.step(.02)
    for x in np.linspace(-.2, 1, 15):
        assert np.allclose(sample(a, (x, 0, 1)), sample(b, (x, 0, 1)), rtol=2e-5, atol=1e-6)
    assert np.allclose(a.total_moles(), b.total_moles())
    b.set_probes(np.array([[.2, 0, 1]]))
    assert np.allclose(b.sample(), b.sample_torch().numpy().ravel())
