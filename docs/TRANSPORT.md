# Chemical plume transport

The NumPy solver is the scientific reference. The Warp solver implements its
uniform-wind, single-species point-source subset. The default Gaussian
normalization, Farrell Model 2 growth, stationary per-filament OU velocities,
shared bearing meander, and three-sigma sampling cutoff define the default model.

## Configuration and units

```python
from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig

cfg = FilamentPlumeConfig(
    species="carbon_dioxide",
    molar_rate_mol_s=1e-6,
    release_rate_hz=40.0,
    background_ppm={"carbon_dioxide": 420.0},  # chosen scenario baseline, not a calibration
    max_step_s=0.05,
)
plume = FilamentPlume(cfg, seed=7)
plume.step(0.1)
ppm = plume.sample_species([[2.0, 0.0, 1.0]])
budget = plume.diagnostics()
```

`molar_rate_mol_s=None` preserves the legacy source: a packet contains
`(ppm_center_initial / 1e6) * n_air * (2*pi)**1.5 * sigma0**3` moles.
Setting a molar rate Q overrides `ppm_center_initial`; each packet contains
`Q / release_rate_hz` moles. The release frequency then controls packet
resolution without changing the nominal molar source rate. A positive Q needs
a positive release frequency. Changing packet resolution still changes local
intermittency and peaks; it is not a claim of identical concentration records.

Explicit `PointEmitter`, `LineEmitter`, and `BoxEmitter` objects accept the same
`molar_rate_mol_s` option. Set it on each emitter when using `cfg.emitters`;
combining explicit emitters with a config-level molar rate is rejected. The
existing `mass_flux_mol_s(n_air)` method returns the nominal rate while the
source is ON, before duty-cycle averaging and stochastic modulation.

`background_ppm` defaults to an empty mapping, preserving zero ambient
concentration. Values are nonnegative ppm by volume. The CPU solver includes
background-only species, including when `emitters=[]`, and adds background at
every query even with an empty pool or after reset. Names and aliases are
canonicalized through the registry: `CO2` becomes `carbon_dioxide`. Duplicate
background aliases are rejected instead of silently summing two baselines.
Emitter aliases for the same species contribute to one concentration column.

Background is a prescribed, spatially uniform external reservoir. It does not
decay, advect, get line-of-sight gated, or enter the filament mass ledger. For
CO2 experiments, configure the scenario's measured ambient concentration and
interpret source concentration as excess above that background. Subtract the
known background before applying an excess-plume intermittency statistic.

Construct a new plume to change its configuration. For moving CPU sources,
mutate `plume.emitters[i].position`; changing an original emitter passed to a
constructor does not change the plume. Each plume owns a fresh copy of each
emitter, including separate accumulators and modulation state. Source windows
and strengths may be changed on those owned CPU emitters between steps; changing
species identities or the emitter list requires reconstruction. Configure
airflow and the registry before construction. An explicitly supplied airflow
object is used directly; use a separate object for each world unless shared
time evolution is intentional.

## Time integration and lifecycle

`step(dt)` rejects negative or nonfinite dt before changing state. Zero is a
no-op. A step is split into equal internal steps no larger than `max_step_s`
(default 0.05 s), `lagrangian_timescale / 10`, and, when meander is enabled,
`meander_timescale / 10`. Existing steps within these limits keep the original
update sequence and random draw order.

The OU **velocity** update is exact, and `sigma**2 += gamma * dt` is unchanged.
Position still uses the existing explicit velocity-times-dt update. Substepping
controls its error; it does not make integrated OU displacement exact or impose
a geometry-dependent CFL condition. Choose a smaller `max_step_s` for sharp
CFD gradients and vary it in convergence checks. Wall collision uses swept
voxel traversal and does not require a displacement smaller than a voxel.

Emitter release counts integrate the actual overlap of `[t, t+dt)` with start,
stop, and periodic ON windows. Fractional packets accumulate across OFF periods;
they are not discarded. A tolerance based on floating-point spacing prevents
delaying packets at integer boundaries. Modulated rates remain an OU-driven
approximation evaluated per active internal step. All newly released packets
are still placed at the start of an internal step and advanced through that
step: birth timing and finite-duration emission are resolved only to internal dt.

`FilamentPlume.reset(seed=...)` reproduces a fresh run with the same configuration
and seed. `WarpFilamentPlume.reset(env_ids=None, seed=...)` offers that same
full-reset replay contract. Resets clear pools, time, release fractions, and
diagnostics. Source and probe placement/configuration remain in effect.
Unseeded resets continue the random streams rather than promising replay.

Warp also supports `reset(env_ids=[...])`: selected environments have their
pool, local time, cursor, release phase, meander angle, and ledger cleared;
other environments' trajectories and random streams are unchanged. Seeded
partial resets are explicitly rejected; use a full reset for seeded replay.
`t` is the batch clock; `env_time_s` exposes time since each environment's reset.

## Walls and outlets

`OccupancyGrid.collide_and_slide(p_old, p_new)` accepts matching `(N, 3)` arrays
and returns `(positions, outlet_mask)`. Swept traversal checks every crossed
voxel, stops on the free side of the first wall, and sweeps the remaining
tangential motion. It checks all touched neighbours at simultaneous edge or
corner crossings. Conservative corner contacts can stop multiple motion axes;
sliding along a flat wall preserves tangential displacement. Contact positions
have a floating-point-scale offset into free space to keep repeated pushes stable.

The first outlet or domain exit along the swept path is reported for culling,
even if the requested endpoint is free. A wall before an outlet blocks access;
sliding into an outlet still culls the filament. Starting inside a solid voxel
raises `ValueError`; starting in an outlet reports immediate culling. Inputs
are not modified. Moves within one free voxel take a fast path; longer moves
use batched traversal with work proportional to crossed cells until first contact.

## Pool and mass diagnostics

Both backends retain living filaments and reject new packets when the pool is
full. Rejected releases are counted and are not queued for later emission.
Release precedes advection/culling, preserving the reference's update order;
a slot expiring this step becomes available on the next internal step. CPU
emitters are served in list order, so saturation can starve later sources.
Choose enough capacity for the release rate and residence time and inspect the
ledger; conservation bookkeeping does not make a saturated simulation faithful.

`diagnostics()` returns detached NumPy arrays and `species_names`. CPU arrays
have shape `(S,)`; Warp arrays have shape `(n_envs, 1)`. Quantities are cumulative
since reset except `retained_mol` and `pending_mol`:

| Key | Meaning |
|---|---|
| `emitted_mol`, `emitted_filaments` | Whole packets requested by sources, including rejected packets |
| `released_mol`, `released_filaments` | Packets admitted into the pool |
| `pool_rejected_mol`, `pool_rejected_filaments` | Source packets lost because the pool was full |
| `retained_mol` | Moles in live filaments, also returned by `total_moles()` |
| `decayed_mol` | First-order loss from released filaments |
| `outlet_mol` | Remaining mass removed by occupancy outlets |
| `domain_mol` | Remaining mass removed by crossing configured domain limits |
| `age_mol` | Remaining mass removed at the maximum age |
| `pending_mol` | Current fractional packets held by emitter accumulators |
| `balance_error_mol` | Residual of the balance below (roundoff expected) |

```text
emitted_mol = retained_mol + pool_rejected_mol + decayed_mol
              + outlet_mol + domain_mol + age_mol
released_mol = emitted_mol - pool_rejected_mol
```

For a steady physical source before resets, requested mass plus pending mass
tracks Q times active time. When changing an owned CPU source's strength, first
finish or reset its fractional packet: pending mass is evaluated using its
current packet strength. Decay is accounted before culling; cull categories
are disjoint, with outlet, domain, then age precedence. Warp has no decay or
occupancy, so those corresponding ledger fields are zero.

The ledger tracks packet moles, not the spatial integral of the sampled field.
The unrenormalized three-sigma spherical cutoff retains about 97.07% of a
Gaussian's mass in unbounded free space. Line-of-sight masking and domain
boundaries can remove more sampled mass. These existing approximations remain;
renormalizing them would change the scientific model and concentration peaks.

## CPU and Warp feature contract

| Feature | NumPy | Warp |
|---|---|---|
| Legacy point source, molar-rate option | Yes | Yes, one source per environment |
| Uniform wind, OU velocity, shared meander, growth | Yes | Yes, independent environments |
| Background ppm | Multiple species, including ambient-only | Must match its single source species |
| Explicit emitter objects, line/box sources, pulsing/modulation | Yes | `NotImplementedError` |
| Swept occupancy collision, sliding, outlets | Yes | `NotImplementedError` |
| Custom airflow, buoyancy slip | Yes | `NotImplementedError` |
| Registry aliases | Canonicalized | Canonicalized |
| Per-species decay | Yes | Nonzero decay rejected |
| Pool rejection and mass ledger | Yes | Yes |
| Seeded replay | Full plume | Full batch |
| Sampling | `(M, S)` via `sample_species(points)` | One probe per environment; `sample()` is `(n,)`, `sample_torch()` is `(n,1)` |

Warp uses float32 state and concentration arithmetic; NumPy uses float64.
Random generators differ, so stochastic parity is statistical. The Warp ring
cursor searches only dead slots and retains live gas when the pool is full.
`source` and `set_probes()` remain available for per-environment positions. Torch results
are views of reused output buffers; clone them when retaining historical samples.
Diagnostics and reset take host snapshots; they are not CUDA-graph-safe operations.

## Model limitations

| Aspect | Limit |
|---|---|
| Geometry | Collision is against occupied voxels. Features absent from the grid, including subvoxel walls missed during rasterization, cannot block transport. Edge/corner contacts are conservative. |
| Concentration masking | `line_of_sight` rejects solid endpoints, including identical points inside a solid. Interior point marching remains approximate and can miss corner intersections. This sampling approximation is separate from swept filament collision; prescribed background remains spatially uniform. |
| Dispersion | OU velocity updates are exact; position integration and packet birth timing remain resolved to the internal timestep. |
| Growth and buoyancy | `gamma` controls growth; registry molecular diffusivity is not applied as a floor. Optional slip is phenomenological and disabled by default. |
| Mass interpretation | The ledger conserves packet bookkeeping. Gaussian truncation, wall masking, and source packets rejected by a full pool limit field fidelity. |
| Ambient gas | Background is a prescribed uniform reservoir, not a transported or reacting field. |
| Mean airflow | Potential flow does not reproduce wakes or recirculation; use an appropriate imported CFD field when needed. |

Intermittency gates are regression checks for a specified scenario and threshold,
not general certification against turbulence theory. `tail_exponent` reports an
empirical CCDF slope. Celani et al.'s -3/2 probability-density exponent corresponds
to -1/2 for an ideal untruncated CCDF; finite cutoffs affect empirical fits. See
the [primary paper, equations (5) and (6)](https://arxiv.org/html/1411.3507#S3).
The numerical regression-gate thresholds are unchanged.

## Validation

`tests/test_occupancy.py` covers swept wall crossings in both directions, edge
and vertex contacts, successive sliding collisions, outlet ordering, translated
grids, repeated wall pushes, and a fast-plume mass-budget regression. Random
first-contact results are checked against independent analytic segment/box
intersections. Line-of-sight tests cover solid endpoints, degenerate segments,
batch queries, and suppression of nearby plume contributions inside solids.
`tests/test_transport_contract.py` covers source rates, background,
mass accounting, lifecycle, and backend contracts, including CUDA where available.

The CI `backends` job runs `pytest -q -m "not isaac"`, including the slow plume
gate, dispersion, and mass-flux tests. The `core` jobs run the nonslow NumPy
suite on Python 3.10 and 3.12. Local CPU testing uses Warp 1.17 with
`WARP_CACHE_PATH=/tmp/scentience-warp-cache MPLCONFIGDIR=/tmp/scentience-mpl`;
CUDA coverage requires a CUDA-enabled build and device.

Release validation on 2026-09-09: the full non-Isaac suite completed with
**328 passed, 14 CUDA skips** in 83.95 s. The final focused contract run,
including the line-of-sight endpoint checks, completed with **131 passed**.
These results validate the CPU backends; Isaac execution and CUDA execution
are not covered by these runs.
