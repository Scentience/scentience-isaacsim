# Chemical transport model

Filament (Lagrangian puff) model after Farrell, Murlis, Long, Li & Carde,
Env. Fluid Mech. 2:143-169 (2002). Implemented from the published equations;
no GADEN (LGPL-3.0) source is used or transcribed.

## Equations

Growth (Farrell "Model 2"):    sigma(t) = sqrt(sigma0^2 + gamma t)
Concentration (mass-conserving Gaussian, 3-sigma cutoff, LoS-gated):

    C_i(x) = 1e6 * N_i / (n_air (2 pi)^{3/2} sigma_i^3) * exp(-d_i^2 / 2 sigma_i^2)
    N_i fixed at release:  N = (ppm_c/1e6) n_air (2 pi)^{3/2} sigma0^3
    n_air = P / (R T)

Small-scale turbulence -- per-filament OU velocity, EXACT discrete update:

    u' <- e^{-dt/T_L} u' + sigma_u sqrt(1 - e^{-2 dt/T_L}) xi
    sigma_u = I |U|,  floor sigma_u_floor at ~zero wind

Large-scale meander -- ONE shared OU process on wind bearing (T ~ 10-20 s).
Measured consequence (600 s @ 100 Hz, probe 8 m downwind): blank-duration
CV 2.31 with meander, 0.96 without. CV < 1 = sub-exponential blanks = an
environment easier than reality. The gate fails it.

Decay: per-species first-order, moles *= exp(-lambda dt).
Walls: axis-decomposed slide (exact for axis-aligned geometry, approximate at
corners); OUTLET cells and domain exit cull filaments; concentration is
line-of-sight gated through the occupancy grid.

## Deliberate deviations from common practice

* OU with sqrt(1-a^2), never a dt-scaled kick: a dt-scaled kick makes
  turbulent diffusivity proportional to the timestep (halve dt -> plume
  narrows by sqrt 2). Guarded by test_ou_stationary_variance.
* Trilinear wind sampling, never nearest-cell.
* OU state seeded from the stationary distribution at release.

## What is deliberately NOT modelled (v0.1)

* Buoyancy: OFF by default. The optional "slip" model is a dilution-scaled
  phenomenological term, clearly labelled; a defensible dense-gas model
  (gravity currents) is roadmap work, not a silent constant.
* Wakes/recirculation: potential flow has none; import RANS when they matter.
* GPU path scope: single-species point sources, no occupancy, no decay
  (documented in ISAAC_COMPATIBILITY.md; CPU reference has all features).
* Reactive chemistry, aerosol physics, thermals.

## Validation targets (the realism gate)

blank/whiff CV > 1; tail exponent bracketing -3/2 (Celani, Villermaux &
Vergassola, PRX 4:041015 (2014)); intermittency 0.02-0.95; peak-to-mean > 3
(~14 near source, Farrell 2002). Enforced in CI by tests/test_plume_gate.py.
