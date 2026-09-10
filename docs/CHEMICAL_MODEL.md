# Chemical transport model

Filament (Lagrangian puff) model after Farrell, Murlis, Long, Li and Cardé,
*Environmental Fluid Mechanics* 2:143–169 (2002). The original implementation
and its scientific defaults are retained; no GADEN source is included.

## Equations

Farrell Model 2 growth and ideal-gas ppm conversion:

```text
sigma(t) = sqrt(sigma0² + gamma * t)
C_i(x) = 10⁶ N_i exp(-|x-x_i|² / (2 sigma_i²)) / (n_air (2 pi)^(3/2) sigma_i³)
n_air = pressure_atm / (R * temperature_k)
```

Untruncated Gaussian concentration conserves each filament's moles. The
sampling implementation uses a finite spherical cutoff and line-of-sight
masking; their integral can be smaller. See [transport accounting](TRANSPORT.md)
for the distinction between packet mass and sampled-field mass.

Each filament has a stationary Ornstein–Uhlenbeck turbulent velocity:

```text
a = exp(-dt / T_L)
u_next = a * u + sigma_u * sqrt(1-a²) * normal(0,1)
sigma_u = max(turbulence_intensity * |wind_mean|, sigma_u_floor)
```

The velocity update is exact; displacement uses explicit advection with bounded
substeps. A shared OU bearing process represents large-scale meander. First-order
species decay multiplies moles by `exp(-decay_rate * dt)`. The configured `gamma`
controls growth; registry molecular diffusivity is metadata and is not added
as a second diffusion term.

## Scope and validation

The model resolves intermittency economically for navigation research. It is
not a CFD solver or a reacting-gas/thermal model. Mean airflow may be supplied
from external CFD. Obstacle collision and concentration visibility use an
occupancy approximation; walls do not automatically produce wakes or a
mass-conserving reflected concentration field. Optional buoyancy slip remains a
phenomenological, disabled-by-default approximation.

The supplied long-record regression scenario includes a meander ablation and
checks intermittency, blank-duration variability and peak-to-mean ratio. These
empirical bounds do not certify arbitrary environments. The empirical CCDF tail
slope is distinct from the -3/2 probability-density scaling discussed by
[Celani et al., PRX 4, 041015](https://doi.org/10.1103/PhysRevX.4.041015).
Use the [research workflow](RESEARCH_WORKFLOW.md) to record thresholds and assess
convergence. Current APIs, backend scope and numerical limits are in
[TRANSPORT.md](TRANSPORT.md).
