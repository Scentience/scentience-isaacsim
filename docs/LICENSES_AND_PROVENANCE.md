# Licenses and provenance

Two separate questions, answered separately:

1. **Licensing** -- what you may do with this code, what we ask in return, and
   what we do not warrant.
2. **Provenance** -- where every equation and every constant came from, and
   which of them are strong enough to support a published claim.

---

## 1. Licensing

### The license

This project is licensed under the **Apache License, Version 2.0**. The full
text is in `LICENSE` at the repository root; it governs. Nothing in this
document adds to, subtracts from, or modifies those terms.

Apache-2.0 is deliberate. This package is meant to become shared
infrastructure for machine olfaction, and infrastructure cannot be copyleft or
non-commercial without excluding the commercial robotics stacks where it has to
live. Permissive licensing is a strategic choice, not an oversight.

### Citation

**If you use this software in work you publish, cite it.**

This is a scholarly obligation and a condition of the community we want, and we
ask for it in the strongest terms the license permits. It is not an additional
legal restriction on use -- Apache-2.0 grants you the right to use this code
commercially, privately, and without citing anyone, and we are not taking that
back.

Two mechanisms make the request more than decorative:

- **`CITATION.cff`** at the repository root is machine-readable. GitHub renders
  a "Cite this repository" button from it, and Zenodo, `cffconvert`, and most
  reference managers can consume it directly.
- **`NOTICE`** at the repository root carries the attribution and citation
  request. Apache-2.0 **section 4(c)** requires that anyone redistributing this
  work -- modified or not -- carry the `NOTICE` contents forward into their
  derivative work. So the citation request does propagate downstream as a
  license condition, even though the act of citing is not itself compelled.

If you use it, cite it:

```bibtex
@software{france_scentience_olfaction,
  author  = {France, Kordel K. and {Scentience, Inc.}},
  title   = {scentience-olfaction: olfactory sensing and chemical plume
             simulation for Isaac Sim},
  year    = {2026},
  version = {0.1.0},
  license = {Apache-2.0},
  url     = {https://github.com/scentience/scentience-isaacsim}
}
```

Please also cite the underlying science you rely on, not just this package.
The plume transport is Farrell et al. (2002); the fast-sensor time constants
are Dennler et al. (2024); the OIO reference implementation is France et al.
(arXiv:2506.04539). Full references are listed at the end of
[`CHEMICAL_MODEL.md`](CHEMICAL_MODEL.md) and in `NOTICE`.

### Research status and warranty

**This is research software. Treat its output as a hypothesis, not a
measurement.**

Specifically, as of v0.1:

- The Isaac Sim / Isaac Lab integration has **never been executed in a live
  install** (see `ISAAC_COMPATIBILITY.md`). The supported paths are standalone
  Python and Gymnasium.
- Sensor sensitivity coefficients are **DIGITIZED or SYNTHESIZED**, not
  measured on Scentience hardware. The MiCS-6814 datasheet publishes log-log
  graphs, not tabulated coefficients. `claim_check()` will refuse to let you
  state an absolute-ppm claim on this evidence, and you should let it.
- The plume model is validated against **published turbulence statistics**, not
  against a wind tunnel. It reproduces the right *statistical character*; it is
  not a CFD solution of your room.
- Nothing here is calibrated, qualified, or certified for safety, medical,
  environmental-compliance, or life-critical use. **Do not use simulator output
  to decide whether a real atmosphere is safe to breathe.**

Apache-2.0 sections 7 and 8 govern warranty and liability: the software is
provided **"AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**, and no
contributor is liable for any damages arising from its use. Scentience, Inc.
accepts no liability for errors in this software or for any decision made on
the basis of its output. By using this repository you acknowledge that.

Read sections 7 and 8 in `LICENSE` for the operative text. This paragraph is a
plain-language summary, not a substitute.

---

## 2. Provenance of the physics

### GADEN: cited, never transcribed

[GADEN](https://github.com/MAPIRlab/gaden) (Monroy et al., *Sensors* 2017) is
the reference filament simulator in robot olfaction. It is **LGPL-3.0**.
This package is Apache-2.0 and must stay cleanly separated from it.

The separation is procedural, not merely asserted:

- `plume/filament.py` is implemented from the **published equations** in
  Farrell et al. (2002), *Environmental Fluid Mechanics* 2:143-169.
- **No GADEN source was read, copied, adapted, translated, or consulted while
  writing it.** GADEN is used only as a *behavioural* cross-check -- "does a
  plume of roughly this character emerge?" -- never as an implementation
  reference.
- Where this implementation and GADEN differ, the difference is documented and
  justified against the literature in `CHEMICAL_MODEL.md`. Two such deviations
  are deliberate corrections (exact OU integration; shared large-scale meander).

If you contribute, this constraint is binding: see `CONTRIBUTING.md`. No
LGPL/GPL code may enter this codebase.

### Third-party dependencies

| Dependency | License | Required? |
|---|---|---|
| NumPy | BSD-3-Clause | yes -- the only hard dependency |
| Warp (`warp-lang`) | Apache-2.0 | optional, `[gpu]` |
| PyTorch | BSD-3-Clause | optional, `[torch]` |
| Gymnasium | MIT | optional, `[envs]` |
| SciPy | BSD-3-Clause | optional, `[dev]` |
| `scentience` client | see that package | optional, `[bridge]` |
| Isaac Sim / Isaac Lab | NVIDIA license | optional, not bundled |

All permissive. Nothing copyleft is required to run the core.

Isaac Sim and Isaac Lab are **not vendored and not redistributed** here. The
`isaac_extension/` directory contains only our own Kit extension scaffold.

### Evidence levels on every constant

Every physical constant carries an `Evidence` level, enforced at runtime by
`scentience_olfaction/provenance.py`:

| Level | Meaning |
|---|---|
| `MEASURED` | measured on real hardware, with n and uncertainty |
| `DATASHEET` | tabulated by the part manufacturer |
| `DIGITIZED` | read off a published graph, or inverted from a driver fit |
| `SYNTHESIZED` | transferred from a related species/die by stated analogy |
| `ASSUMED` | a modelling choice, defensible but unmeasured |

`claim_check(claim, level, depends_on=...)` refuses any claim the underlying
evidence cannot support, and names the offending coefficients. Run
`python scripts/provenance_demo.py` to see it decline two real claims.

As of v0.1 the shipped MiCS-6814 coefficients are `DIGITIZED`/`SYNTHESIZED`,
and `plume.gamma` is `ASSUMED`. That is why absolute-ppm accuracy is not
claimed anywhere in this repository. It is stated at runtime, not buried here.

---

## Questions

Licensing questions: open an issue at
<https://github.com/scentience/scentience-isaacsim/issues>. For anything with
legal weight, talk to your own counsel -- this document is written by
engineers, for engineers, and is not legal advice.
