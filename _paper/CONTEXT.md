# Paper context (NOT a draft -- outline material only)

Per Kordel: do not write the paper yet. This file accumulates everything
needed to construct the outline quickly later, mirroring the COLIP-2 flow
(foundation/outline here -> rewritten in Kordel's voice).

## Working titles (candidates)
* "Olfaction for Isaac Sim: Chemical Plume Transport and Calibrated Virtual
  Olfactory Sensors for Robot Learning"
* "First Olfactory Sensor Models for GPU-Parallel Robotics Simulation"

## Candidate contributions (each maps to code + a measured number)
1. First olfactory/chemical sensing package for Isaac Sim (negative-result
   search documented: no prior gas sensor in Isaac Sim/Lab as of Aug 2026).
2. Filament transport with two-scale turbulence; meander ablation shows
   blank-CV 2.31 vs 0.96 (gate PASS vs FAIL) -- reproduce:
   `pytest -m slow tests/test_plume_gate.py`
3. A CI-enforced plume realism gate (targets from Celani 2014 / Farrell
   2002); tail exponent -1.68 (q75-q99 fit; range-dependence documented in
   validation/plume_stats.py docstring).
4. Whiff retention through sensor dynamics: 19% (tau_fall 12 s) vs 97%
   (46 ms, Dennler-class constants) on the identical plume -- reproduce:
   `python scripts/validate_physics.py`. THE headline number.
5. Evidence-provenance system; claim_check() as executable scientific
   integrity (ties to standardization position paper arXiv:2506.00398).
6. OIO reference implementation on 4 platform presets; heading-error
   reduction ~91% (quadruped example, seed 0); observability asymmetry
   (crosswind corrected, downwind provably not) asserted in tests.
7. Hardware-shaped everything: BLE-schema frames consumable by the
   `scentience` client ecosystem; Gymnasium env whose observation is device
   channels, never ground truth.

## Measured numbers table (regenerate before writing)
| quantity | value | command |
|---|---|---|
| blank CV with/without meander | 2.31 / 0.96 | pytest -m slow tests/test_plume_gate.py |
| blank tail slope (q75-q99) | -1.68 | scripts/validate_physics.py |
| whiff retention slow/fast | 19% / 97% | scripts/validate_physics.py |
| cast&surge vs random final dist | 5.8 m vs 12.1 m (6 seeds) | examples/04_gym_baseline.py |
| OIO heading reduction (quadruped) | ~91% | examples/03_olfactory_inertial_odometry.py |
| mass conservation ratio | 1.0000 | tests: test_registry_emitters / core checks |
| OU sigma_u dt-invariance | 0.308/0.300/0.294 @ dt 5/10/50 ms | tests/test_warp_parity.py |

## Related-work skeleton
GADEN (Monroy 2017, LGPL, ROS; cited not used) . Farrell 2002 . pompy (MIT)
. Celani/Villermaux/Vergassola PRX 2014 . Singh et al. NMI 2023
(PlumeTrackNets) . Loisy & Eloy OTTO 2023 . Dennler 2024 Sci. Adv. .
Scentience line: standardization (2506.00398), OIO (2506.04539),
chronoamperometry (2506.04540), diffusion GNN (2506.00455), Chasing Ghosts
(2602.19577), scentience-plume-envs.

## Honesty section material
Isaac integration unvalidated until validate_install.py passes; GPU-path
scoping; buoyancy off; coefficients DIGITIZED not MEASURED; gate thresholds
partly author-chosen (blank-CV > 1 defensible from Celani; peak-to-mean > 3
convention).
