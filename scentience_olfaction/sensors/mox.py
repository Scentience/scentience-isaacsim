"""
MOX (metal-oxide) gas sensor model, MiCS-6814 class.

Full signal chain, in the order the physics actually happens:

    C_g(t)  species concentrations [ppm]
      -> power-law steady state, superposed across species in RESISTANCE space
      -> humidity / temperature modulation (multiplicative in log space)
      -> asymmetric first-order lag (tau_rise != tau_fall)
      -> transport delay from inlet dead volume
      -> multiplicative baseline drift (random walk) + 1/f + white noise
      -> voltage divider
      -> ADC quantisation
      -> counts

The divider + quantisation stages matter more than they look: resolution in
R_s degrades as (R_s + R_L)^2, so a RED die sitting at 1.5 MOhm in clean air
is nearly unresolvable on a 12-bit ADC.  Quantising concentration instead of
voltage hides this entirely and is the standard way sim data fails to transfer.

Steady-state coefficients (A, beta) are digitised from open-source driver
constants, NOT from the SGX datasheet, which publishes log-log graphs only.
They are labelled ILLUSTRATIVE and must be replaced by per-unit calibration
before any quantitative claim.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def absolute_humidity(temp_c: float, rh_pct: float) -> float:
    """g/m^3 from degrees C and % RH (Magnus). MOX responds to AH, not RH."""
    es = 6.112 * math.exp(17.62 * temp_c / (243.12 + temp_c))
    return 216.7 * (rh_pct / 100.0) * es / (273.15 + temp_c)


@dataclass
class MoxChannelConfig:
    """One MOX die (MiCS-6814 has three: RED, NH3, OX)."""

    name: str

    # --- steady state:  Rs/R0 = A * C^(-beta), superposed over species -------
    # ILLUSTRATIVE -- replace with per-unit calibration.
    sensitivity: dict[str, tuple[float, float]] = field(default_factory=dict)

    # --- unit-to-unit variation ---------------------------------------------
    r0_nominal: float = 4.0e5  # [ohm]
    r0_range: tuple[float, float] = (1.0e5, 1.5e6)  # datasheet spread: 15x
    rs_r0_clean_air: float = 1.0

    # --- dynamics -----------------------------------------------------------
    # Defaults are the SLOW regime (packaged sensor, still air).  A fast
    # heater + low dead volume gets to ~0.09 s (Dennler et al. 2024, Sci Adv).
    tau_rise_s: float = 3.0
    tau_fall_s: float = 12.0
    dead_volume_delay_s: float = 0.0
    # tau shortens with forced convection over the die; a robot at 1 m/s does
    # not have the same time constant as the same part sitting on a bench.
    tau_flow_exponent: float = 0.5
    tau_flow_ref_mps: float = 0.1

    # --- environment coupling ------------------------------------------------
    humidity_coeff: float = -0.010  # d(ln Rs)/d(AH)   [m^3/g]
    humidity_sensitivity_coeff: float = 0.0  # AH * ln(C) cross-term
    activation_energy_ev: float = 0.0

    # --- noise / drift --------------------------------------------------------
    drift_sigma_per_sqrt_s: float = 2.0e-4  # random walk on ln R0
    white_noise_frac: float = 2.0e-3
    flicker_noise_frac: float = 3.0e-3

    # --- readout --------------------------------------------------------------
    r_load: float = 1.0e5  # [ohm]
    v_supply: float = 3.3
    v_ref: float = 3.3
    adc_bits: int = 12


class MoxChannel:
    def __init__(self, cfg: MoxChannelConfig, rng: np.random.Generator, randomize: bool = True):
        self.cfg = cfg
        self.rng = rng
        self._randomize = randomize
        self.reset()

    def reset(self) -> None:
        c = self.cfg
        if self._randomize:
            # R0 log-uniform over the datasheet spread. Sampling a single
            # nominal value produces unrealistically consistent virtual units
            # and is the fastest way to train a policy that cannot transfer.
            lo, hi = c.r0_range
            self.r0 = float(np.exp(self.rng.uniform(math.log(lo), math.log(hi))))
        else:
            self.r0 = c.r0_nominal
        self.ln_drift = 0.0
        self._flicker = np.zeros(4)
        self._y = c.rs_r0_clean_air  # filtered Rs/R0
        self._delay: deque[float] = deque()
        self._delay_t = 0.0
        self.t = 0.0

    # ------------------------------------------------------------------ model
    def _steady_state(self, conc_ppm: dict[str, float], temp_c: float, rh_pct: float) -> float:
        """Rs/R0 target. Superposition is over resistance DECREMENT, not conc."""
        c = self.cfg
        base = c.rs_r0_clean_air
        decrement = 0.0
        for gas, (A, beta) in c.sensitivity.items():
            cg = conc_ppm.get(gas, 0.0)
            if cg <= 1e-9:
                continue
            r = A * cg ** (-beta)
            if beta > 0:  # reducing gas: lowers Rs
                decrement += max(base - min(r, base), 0.0)
            else:  # oxidizing gas (NO2): raises Rs
                decrement -= max(r - base, 0.0)
        rs_r0 = max(base - decrement, 1e-3)

        ah = absolute_humidity(temp_c, rh_pct)
        ln_rs = math.log(rs_r0) + c.humidity_coeff * ah
        if c.humidity_sensitivity_coeff:
            ctot = sum(conc_ppm.get(g, 0.0) for g in c.sensitivity)
            if ctot > 1e-9:
                ln_rs += c.humidity_sensitivity_coeff * ah * math.log(ctot)
        if c.activation_energy_ev:
            kT = 8.617333e-5 * (temp_c + 273.15)
            ln_rs += c.activation_energy_ev / kT - c.activation_energy_ev / (8.617333e-5 * 293.15)
        return math.exp(ln_rs)

    def step(
        self,
        conc_ppm: dict[str, float],
        dt: float,
        temp_c: float = 20.0,
        rh_pct: float = 50.0,
        flow_mps: float = 0.0,
        heater_level: float = 1.0,
    ) -> dict[str, float]:
        c = self.cfg
        target = self._steady_state(conc_ppm, temp_c, rh_pct)

        # --- asymmetric first-order lag, flow- and heater-corrected ----------
        tau = c.tau_rise_s if target < self._y else c.tau_fall_s
        if flow_mps > 0.0 and c.tau_flow_exponent:
            tau *= (max(flow_mps, 1e-3) / c.tau_flow_ref_mps) ** (-c.tau_flow_exponent)
        tau /= max(heater_level, 1e-3)  # hotter plate = faster surface kinetics
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))  # exact, stable for any dt
        self._y += alpha * (target - self._y)

        # --- transport delay (inlet + housing dead volume) -------------------
        y = self._y
        if c.dead_volume_delay_s > 0.0:
            self._delay.append(self._y)
            n = max(1, int(round(c.dead_volume_delay_s / dt)))
            while len(self._delay) > n:
                self._delay.popleft()
            y = self._delay[0]

        # --- baseline drift: random walk on ln R0 ----------------------------
        self.ln_drift += c.drift_sigma_per_sqrt_s * math.sqrt(dt) * self.rng.standard_normal()
        r0_t = self.r0 * math.exp(self.ln_drift)

        # --- noise: white + 1/f as a bank of AR(1) poles ---------------------
        noise = c.white_noise_frac * self.rng.standard_normal()
        if c.flicker_noise_frac:
            for i, tau_i in enumerate((0.1, 1.0, 10.0, 100.0)):
                a = math.exp(-dt / tau_i)
                self._flicker[i] = a * self._flicker[i] + math.sqrt(
                    max(1.0 - a * a, 0.0)
                ) * self.rng.standard_normal()
                noise += c.flicker_noise_frac * self._flicker[i] / 2.0

        rs = max(y * r0_t * (1.0 + noise), 1.0)

        # --- readout: divider then quantise the VOLTAGE ----------------------
        v = c.v_supply * c.r_load / (rs + c.r_load)
        q = c.v_ref / (2**c.adc_bits)
        counts = int(min(max(math.floor(v / q), 0), 2**c.adc_bits - 1))
        v_q = counts * q

        self.t += dt
        # Reconstructed Rs, i.e. what firmware can actually recover.
        rs_hat = c.r_load * (c.v_supply / max(v_q, q * 0.5) - 1.0)
        return {
            "rs_true": rs,
            "rs_measured": rs_hat,
            "ratio_measured": rs_hat / r0_t,
            "counts": counts,
            "volts": v_q,
            "r0_current": r0_t,
        }


# --------------------------------------------------------------------------
# MiCS-6814 profile.  ILLUSTRATIVE COEFFICIENTS -- calibrate before use.
# Derived algebraically from open-source driver inverse fits (C = a*r^b), which
# were themselves digitised from the datasheet's log-log graphs.
# --------------------------------------------------------------------------
MICS6814_RED = MoxChannelConfig(
    name="red",
    sensitivity={
        "carbon_monoxide": (3.37, 0.847),
        "hydrogen": (0.91, 0.599),
        "ethanol": (1.31, 0.645),
        "methane": (4.33, 0.227),
        "hydrogen_sulfide": (2.0, 0.6),
    },
    r0_nominal=4.0e5,
    r0_range=(1.0e5, 1.5e6),
    rs_r0_clean_air=1.0,
)

MICS6814_NH3 = MoxChannelConfig(
    name="nh3",
    sensitivity={"ammonia": (0.94, 0.465), "ethanol": (1.0, 0.35)},
    r0_nominal=3.0e5,
    r0_range=(1.0e4, 1.5e6),
    rs_r0_clean_air=1.0,
)

MICS6814_OX = MoxChannelConfig(
    name="ox",
    sensitivity={"nitrogen_dioxide": (6.35, -1.01)},  # oxidizing: negative beta
    r0_nominal=5.0e3,
    r0_range=(8.0e2, 2.0e4),
    rs_r0_clean_air=1.0,
    r_load=1.0e4,
)

# Fast configuration -- Dennler et al. 2024 (Sci. Adv. 10, eadp1764):
# 150-400 C heater square wave at 20 Hz, 1 kHz / 24-bit readout, low dead
# volume => onset 87 +/- 20 ms, recovery 106 +/- 24 ms.  This is the regime
# where a sensor can actually RESOLVE plume whiffs rather than integrate them.
FAST_OVERRIDES = dict(
    tau_rise_s=0.038,  # t90 ~ 87 ms  =>  tau = t90 / ln(10)
    tau_fall_s=0.046,  # t90 ~ 106 ms
    dead_volume_delay_s=0.005,
    adc_bits=24,
    white_noise_frac=5.0e-4,
)
