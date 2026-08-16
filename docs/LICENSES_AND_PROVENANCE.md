# License and source audit (v0.1)

## Runtime dependencies
| package | license | role |
|---|---|---|
| numpy | BSD-3-Clause | core (only mandatory dep) |
| warp-lang | Apache-2.0 | optional GPU path (bundled with Isaac Sim) |
| gymnasium | MIT | optional envs |
| torch | BSD-3-Clause | optional Isaac-scale device model |
| scentience (PyPI) | Scentience's own | optional bridge target; not imported by core |

All permissive. Nothing copyleft is imported, vendored, or transcribed.

## Equations and data
* Farrell et al. 2002 (filament model), Celani et al. 2014 (statistics),
  Akenine-Moller 2001 (tri-box SAT), Ng et al. 1999 (reward shaping), RAE
  TN-106 (PID CFs), datasheets (MiCS-6814, SCD4x, Alphasense B4): published
  equations, algorithms, and measured facts -- not copyrightable; cited.
* Dennler et al. 2024 (Sci. Adv.): measured time constants only, credited;
  their heater-modulation METHOD is not implemented.
* France et al. (Chasing Ghosts, OIO, chronoamperometry papers): concepts
  reimplemented with citation; author is Scentience's founder.

## Explicit flag
GADEN (github.com/MAPIRlab/gaden) is **LGPL-3.0**. It is cited as related
work and used only as a behavioural cross-check. No GADEN code was read into
this implementation; the plume is implemented from Farrell's paper. A future
GADEN head-to-head (roadmap v0.4) reads its OUTPUT data only, which carries
no license obligation. No workaround needed.
