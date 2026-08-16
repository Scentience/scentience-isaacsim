"""
Amperometric electrochemical (EC) gas sensor model.

Structurally different from MOX in the two ways that matter:

  * LINEAR in concentration.  Current is stoichiometric with the
    diffusion-limited flux through the membrane:  I = (n F A D / delta) * C.
    So cross-sensitivity IS a linear mixing matrix here (the thing that is
    wrong for MOX is right for EC).
  * The transient of a potential-step measurement follows the Cottrell
    equation  I(t) = n F A sqrt(D) C / sqrt(pi t)  -- the t^{-1/2} tail is why
    classical chronoamperometry needs tens of seconds per reading, and why
    France & Daescu (arXiv:2506.04540, IEEE BioSensors 2025) accelerate it by
    inferring the plateau from the early transient.  `cottrell_current()`
    reproduces the physics so that acceleration strategy can be prototyped
    in simulation.

The default channel constants follow the Chasing Ghosts stack (France et al.,
arXiv:2602.19577): a two-electrode cell with a room-temperature ionic liquid
transducer, electrode area 2.25 cm^2, read by chronoamperometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

F_FARADAY = 96485.33212  # C/mol


@dataclass
class ECChannelConfig:
    name: str = "ec"
    # Linear sensitivity per species [nA/ppm]. This is the mixing-matrix row.
    sensitivity_na_per_ppm: dict[str, float] = field(default_factory=dict)
    tau_s: float = 25.0                # first-order lag (membrane diffusion)
    zero_current_na: float = -10.0     # baseline at 20 C
    zero_tempco_na_per_k: float = -2.0 # zero drifts with temperature
    span_tempco_per_k: float = 0.008   # sensitivity ~+0.8 %/K around 20 C
    noise_na: float = 1.5              # white current noise (1 sigma)
    drift_na_per_sqrt_s: float = 0.02  # baseline random walk
    # Cottrell parameters (chronoamperometric mode)
    n_electrons: int = 2
    area_cm2: float = 2.25             # Chasing Ghosts ItalSens cell
    diffusivity_cm2_s: float = 1.0e-5


class ECChannel:
    def __init__(self, cfg: ECChannelConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        self._y_na = 0.0
        self._drift = 0.0

    def step(self, conc_ppm: dict[str, float], dt: float, temp_c: float = 20.0) -> dict:
        c = self.cfg
        span = 1.0 + c.span_tempco_per_k * (temp_c - 20.0)
        target = span * sum(c.sensitivity_na_per_ppm.get(g, 0.0) * v
                            for g, v in conc_ppm.items())
        alpha = 1.0 - math.exp(-dt / max(c.tau_s, 1e-6))
        self._y_na += alpha * (target - self._y_na)
        self._drift += c.drift_na_per_sqrt_s * math.sqrt(dt) * self.rng.standard_normal()
        zero = c.zero_current_na + c.zero_tempco_na_per_k * (temp_c - 20.0)
        i_na = self._y_na + zero + self._drift + c.noise_na * self.rng.standard_normal()
        return {"current_na": i_na, "signal_na": self._y_na}

    def cottrell_current(self, conc_mol_cm3: float, t_s: np.ndarray) -> np.ndarray:
        """Ideal Cottrell transient I(t) [A] for a potential step at t=0.
        I = n F A sqrt(D) C / sqrt(pi t). Diverges at t->0 as physics says it
        should; callers window it (real front ends saturate)."""
        c = self.cfg
        t = np.maximum(np.asarray(t_s, np.float64), 1e-6)
        return (c.n_electrons * F_FARADAY * c.area_cm2 *
                math.sqrt(c.diffusivity_cm2_s) * conc_mol_cm3 / np.sqrt(math.pi * t))


# Default profiles. Sensitivities are ILLUSTRATIVE (order-of-magnitude from
# Alphasense B4-class datasheets: hundreds of nA/ppm for the target gas,
# percent-level cross terms except H2 on CO cells) -- provenance registered by
# the device model.
EC_CO = ECChannelConfig(
    name="ec_co",
    sensitivity_na_per_ppm={"carbon_monoxide": 500.0, "hydrogen": 200.0,
                            "hydrogen_sulfide": 5.0, "ethanol": 2.0},
    tau_s=25.0)

EC_H2S = ECChannelConfig(
    name="ec_h2s",
    sensitivity_na_per_ppm={"hydrogen_sulfide": 1500.0, "carbon_monoxide": 3.0,
                            "nitrogen_dioxide": -100.0},
    tau_s=20.0)
