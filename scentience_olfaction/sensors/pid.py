"""
Photoionization detector (PID) model.

Physics: VUV photons (9.8 / 10.6 / 11.7 eV lamps) ionise any species whose
ionisation energy is below the lamp energy; collected current is linear in
concentration and non-selective. Readings are reported as isobutylene
equivalents; per-species correction factors (CF) convert:

    reading_isobutylene_equiv = sum_i  C_i / CF_i        (CF=inf -> invisible)

CF values below are from the published RAE Systems TN-106 tables (tabulated
measurement data = facts; no license restriction). A species whose IE exceeds
the lamp energy has CF=inf here, which is the honest encoding of "the lamp
cannot see it" -- H2, CO, CO2, CH4 for all standard lamps.

Known incumbent bug, deliberately not inherited: GADEN ships 10.47 as the
ethanol "correction factor" at 11.7 eV; 10.47 eV is ethanol's ionisation
ENERGY in the same table (the 10.6 eV CF for ethanol is ~3.1). We cite the
table itself and test that ethanol at 10.6 eV uses 3.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

INF = math.inf

# RAE TN-106 correction factors (10.6 eV lamp is the standard field lamp).
# Species not listed for a lamp: not detectable (CF=inf).
CORRECTION_FACTORS = {
    9.8: {"isobutylene": 1.0, "toluene": 0.54, "benzene": 0.55, "acetone": 1.2,
          "ethanol": 10.0, "isopropanol": 500.0},
    10.6: {"isobutylene": 1.0, "toluene": 0.50, "benzene": 0.53, "acetone": 1.1,
           "ethanol": 3.1, "isopropanol": 6.0, "ammonia": 9.7,
           "hydrogen_sulfide": 3.3},
    11.7: {"isobutylene": 1.0, "toluene": 0.51, "benzene": 0.60, "acetone": 1.4,
           "isopropanol": 2.7, "ammonia": 5.7, "methanol": 2.5},
}
# Never PID-visible at any standard lamp energy (IE too high):
PID_BLIND = {"hydrogen", "carbon_monoxide", "carbon_dioxide", "methane"}


@dataclass
class PIDConfig:
    lamp_ev: float = 10.6
    tau_s: float = 3.0                # inlet + electronics; PIDs are fast
    noise_ppm: float = 0.02          # isobutylene-equivalent
    humidity_quench_at_90rh: float = 0.30
    """Fractional signal loss at 90 %RH (typ. 20-40 % for 10.6 eV lamps)."""


class PIDChannel:
    def __init__(self, cfg: PIDConfig, rng: np.random.Generator):
        if cfg.lamp_ev not in CORRECTION_FACTORS:
            raise ValueError(f"lamp_ev must be one of {sorted(CORRECTION_FACTORS)}")
        self.cfg = cfg
        self.rng = rng
        self._cf = CORRECTION_FACTORS[cfg.lamp_ev]
        self.reset()

    def reset(self) -> None:
        self._y = 0.0

    def correction_factor(self, species: str) -> float:
        if species in PID_BLIND:
            return INF
        return self._cf.get(species, INF)

    def step(self, conc_ppm: dict[str, float], dt: float, rh_pct: float = 50.0) -> dict:
        target = 0.0
        for g, c in conc_ppm.items():
            cf = self.correction_factor(g)
            if math.isfinite(cf) and cf > 0:
                target += c / cf
        # humidity quench, linear in RH above 0
        target *= 1.0 - self.cfg.humidity_quench_at_90rh * (rh_pct / 90.0)
        alpha = 1.0 - math.exp(-dt / max(self.cfg.tau_s, 1e-6))
        self._y += alpha * (target - self._y)
        return {"ppm_isobutylene_equiv":
                self._y + self.cfg.noise_ppm * self.rng.standard_normal()}
