"""
Scentience V1 device model -- vectorised over parallel environments.

Emits the Scentience BLE/Sockets channel schema, so a consumer cannot tell
simulated output from hardware output without inspecting the transport. That
is deliberate: it is what makes a policy trained here deployable without an
adapter, and it is what makes the simulator a credible reference for the
device API.

Torch throughout, on the sim device. No host round trip.

ALL sensitivity coefficients are ILLUSTRATIVE (see provenance.py). They are
algebraic inversions of open-source driver fits that were themselves digitised
from the MiCS-6814 datasheet's log-log graphs -- the datasheet publishes no
tabulated (A, beta). Replace via calibration/fit.py before any absolute claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover - only hit in a numpy-only install
    # The batched device below genuinely needs torch, but CHANNELS, PROFILES and
    # register_coefficients() are pure metadata. ARCHITECTURE.md invariant 4
    # ("core imports with no Isaac, no GPU, no torch") and the README's
    # provenance claim both require those to stay readable without it, so defer
    # the failure to the first actual torch use instead of import time.
    class _TorchUnavailable:
        def __getattr__(self, name):
            raise ImportError(
                "scentience_olfaction.sensors.scentience_v1 needs torch for the "
                "batched device (attribute 'torch.%s'): pip install torch. "
                "The NumPy device is scentience_olfaction.sensors.device_np." % name
            )

    torch = _TorchUnavailable()

from ..provenance import Evidence, ProvenanceRegistry, coeff, synthesized_from

# Channel order IS the Scentience BLE schema. Do not reorder.
CHANNELS = (
    "mics1_red", "mics1_nh3", "mics1_ox",
    "mics2_red", "mics2_nh3", "mics2_ox",
    "co2_ppm", "temperature_c", "relative_humidity", "ec1", "ec2",
)

MOX_CHANNELS = (0, 1, 2, 3, 4, 5)

PROFILES = {
    # tau_rise, tau_fall [s]
    "packaged_slow": (3.0, 12.0),
    # Dennler et al. 2024, Sci. Adv. 10:eadp1764 -- 150-400 C at 20 Hz, 1 kHz
    # readout, onset 87+/-20 ms, recovery 106+/-24 ms. tau = t90 / ln(10).
    "fast_modulated": (0.038, 0.046),
}


def register_coefficients(reg: ProvenanceRegistry) -> ProvenanceRegistry:
    """Declare every constant this model uses, with its evidence level."""
    reg.register("mics6814.A.ethanol", coeff(
        1.31, Evidence.DIGITIZED,
        "algebraic inversion of open-source driver fit C = 1.52 * r^-1.55",
        notes="MiCS-6814 rev 8 publishes log-log graphs only; no tabulated A/beta"))
    reg.register("mics6814.beta.ethanol", coeff(0.645, Evidence.DIGITIZED, "as above"))
    reg.register("mics6814.A.carbon_monoxide", coeff(
        3.37, Evidence.DIGITIZED, "inverted from C = 4.2 * r^-1.18"))
    reg.register("mics6814.beta.carbon_monoxide", coeff(0.847, Evidence.DIGITIZED, "as above"))
    reg.register("mics6814.A.ammonia", coeff(
        0.94, Evidence.DIGITIZED, "inverted from C = 0.877 * r^-2.15"))
    reg.register("mics6814.beta.ammonia", coeff(0.465, Evidence.DIGITIZED, "as above"))
    reg.register("mics6814.A.hydrogen_sulfide", coeff(
        2.0, Evidence.SYNTHESIZED,
        synthesized_from("ethanol", "hydrogen_sulfide", "same RED die, comparable reducing beta"),
        notes="H2S appears on the datasheet RED curve but no range is tabulated"))
    reg.register("mics6814.R0.red", coeff(
        4.0e5, Evidence.DATASHEET, "SGX MiCS-6814 rev 8, RED die", units="ohm",
        notes="100k-1.5M across units; sampled log-uniform per episode"))
    reg.register("device.tau.packaged_slow", coeff(
        12.0, Evidence.ASSUMED, "typical packaged MOX in still air", units="s",
        notes="NOT measured on Scentience hardware; measure via step exposure"))
    reg.register("device.tau.fast_modulated", coeff(
        0.046, Evidence.MEASURED, "Dennler et al. 2024 Sci. Adv. 10:eadp1764",
        units="s", conditions="150-400 C 20 Hz square wave, 1 kHz readout, low dead volume",
        n_units=8, uncertainty=0.010,
        notes="measured on THAT hardware, not on a Scentience OPU"))
    reg.register("scd4x.tau63", coeff(
        60.0, Evidence.DATASHEET, "Sensirion SCD4x datasheet v1.5", units="s",
        notes="photoacoustic, NOT NDIR; t90 ~138 s, cannot resolve a plume whiff"))
    return reg


@dataclass
class DeviceConfig:
    sensor_profile: str = "packaged_slow"
    r0_range: tuple[float, float] = (1.0e5, 1.5e6)
    rs_r0_clean_air: float = 1.0
    drift_sigma_per_sqrt_s: float = 2.0e-4
    white_noise_frac: float = 2.0e-3
    ambient_temp_c: float = 20.0
    ambient_rh: float = 50.0
    humidity_coeff: float = -0.010


class ScentienceV1Device:
    """Vectorised (n_envs,) device. One virtual unit per environment."""

    def __init__(self, cfg: DeviceConfig, n_envs: int, device, randomize: bool = True):
        self.cfg = cfg
        self.n = n_envs
        self.dev = device
        self.provenance = register_coefficients(ProvenanceRegistry())
        self.tau_rise, self.tau_fall = PROFILES[cfg.sensor_profile]
        self.A = 1.31
        self.beta = 0.645
        self.reset(None, randomize)

    # --------------------------------------------------------------- state
    def reset(self, env_ids=None, randomize: bool = True):
        n, dev = self.n, self.dev
        idx = slice(None) if env_ids is None else torch.as_tensor(list(env_ids), device=dev)
        if env_ids is None:
            self.r0 = torch.empty(n, len(MOX_CHANNELS), device=dev)
            self.y = torch.full((n, len(MOX_CHANNELS)), self.cfg.rs_r0_clean_air, device=dev)
            self.ln_drift = torch.zeros(n, len(MOX_CHANNELS), device=dev)
            self.A_e = torch.full((n, len(MOX_CHANNELS)), self.A, device=dev)
            self.beta_e = torch.full((n, len(MOX_CHANNELS)), self.beta, device=dev)
        else:
            self.y[idx] = self.cfg.rs_r0_clean_air
            self.ln_drift[idx] = 0.0

        if randomize:
            # Per-episode unit-to-unit variation. Without this every episode
            # trains against the same virtual device and the policy overfits
            # to a specific R0 that no real unit has.
            lo, hi = (math.log(v) for v in self.cfg.r0_range)
            shape = (n if env_ids is None else len(list(env_ids)), len(MOX_CHANNELS))
            self.r0[idx] = torch.exp(torch.rand(shape, device=dev) * (hi - lo) + lo)
            self.A_e[idx] = self.A * (1.0 + 0.15 * torch.randn(shape, device=dev))
            self.beta_e[idx] = self.beta * (1.0 + 0.08 * torch.randn(shape, device=dev))
        else:
            self.r0[idx] = 4.0e5

    # ---------------------------------------------------------------- step
    def step(self, conc_ppm: torch.Tensor, dt: float) -> torch.Tensor:
        """
        conc_ppm : (n, S) ground-truth concentration
        returns  : (n, C) channels in the Scentience BLE schema order
        """
        c = conc_ppm[:, :1].clamp_min(1e-9).expand(-1, len(MOX_CHANNELS))
        base = self.cfg.rs_r0_clean_air

        # Power law, superposed as a RESISTANCE decrement (not linear in conc).
        r = self.A_e * c.pow(-self.beta_e)
        target = (base - (base - torch.minimum(r, torch.full_like(r, base)))).clamp_min(1e-3)

        # Absolute humidity acts in log space.
        t, rh = self.cfg.ambient_temp_c, self.cfg.ambient_rh
        es = 6.112 * math.exp(17.62 * t / (243.12 + t))
        ah = 216.7 * (rh / 100.0) * es / (273.15 + t)
        target = target * math.exp(self.cfg.humidity_coeff * ah)

        # Asymmetric first-order lag. Gas arriving LOWERS Rs, hence target < y.
        tau = torch.where(target < self.y, self.tau_rise, self.tau_fall)
        alpha = 1.0 - torch.exp(-dt / tau)  # exact; stable for any dt
        self.y = self.y + alpha * (target - self.y)

        self.ln_drift = self.ln_drift + self.cfg.drift_sigma_per_sqrt_s * math.sqrt(dt) * \
            torch.randn_like(self.ln_drift)
        r0_t = self.r0 * torch.exp(self.ln_drift)
        rs = (self.y * r0_t * (1.0 + self.cfg.white_noise_frac *
                               torch.randn_like(self.y))).clamp_min(1.0)

        out = torch.zeros(self.n, len(CHANNELS), device=self.dev)
        out[:, :len(MOX_CHANNELS)] = rs / r0_t  # ratio, drift-invariant feature
        out[:, CHANNELS.index("temperature_c")] = t
        out[:, CHANNELS.index("relative_humidity")] = rh
        # co2_ppm, ec1, ec2 remain zero until those channel models land.
        return out


def build_device(device_profile: str, sensor_profile: str, n_envs: int,
                 device, randomize: bool = True) -> ScentienceV1Device:
    if device_profile != "scentience_v1":
        raise ValueError(
            f"unknown device profile {device_profile!r}; only 'scentience_v1' exists. "
            "Add profiles in configs/, not by editing this file.")
    if sensor_profile not in PROFILES:
        raise ValueError(
            f"unknown sensor profile {sensor_profile!r}; choose from {tuple(PROFILES)}. "
            "This choice changes whiff retention by roughly 5x -- it is not cosmetic.")
    return ScentienceV1Device(
        DeviceConfig(sensor_profile=sensor_profile), n_envs, device, randomize)
