"""Batched Scentience V1 model using the same channel calibrations as NumPy.

Schema compatibility does not establish hardware fidelity. Default empirical
coefficients remain illustrative; see docs/SCENTIENCE_MODELS.md for evidence,
precision, kinetics and randomization limits. Torch is optional at import.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral
import warnings

import numpy as np

from .device_np import CHANNELS, DeviceConfig
from .mox import absolute_humidity, finite_value

try:
    import torch
except ImportError:  # pragma: no cover - only hit in a numpy-only install
    # The batched device below genuinely needs torch, but CHANNELS, PROFILES and
    # register_coefficients() are pure metadata and must remain readable without
    # the optional backend. Defer
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

MOX_CHANNELS = (0, 1, 2, 3, 4, 5)

PROFILES = {
    # tau_rise, tau_fall [s]
    "packaged_slow": (3.0, 12.0),
    # Dennler et al. 2024, Sci. Adv. 10:eadp1764 -- 150-400 C at 20 Hz, 1 kHz
    # readout, onset 87+/-20 ms, recovery 106+/-24 ms. tau = t90 / ln(10).
    "fast_modulated": (0.038, 0.046),
}


def register_coefficients(reg: ProvenanceRegistry) -> ProvenanceRegistry:
    """Legacy source references; these do not certify user-supplied calibration."""
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
        0.046, Evidence.ASSUMED, "response-time transfer from Dennler et al. 2024 Sci. Adv. 10:eadp1764",
        units="s", conditions="150-400 C 20 Hz square wave, 1 kHz readout, low dead volume",
        n_units=8, uncertainty=0.010,
        notes="source experiment measured on different hardware; transfer to Scentience is assumed"))
    reg.register("scd4x.tau63", coeff(
        60.0, Evidence.DATASHEET, "Sensirion SCD4x datasheet v1.5", units="s",
        notes="photoacoustic NDIR; t90 ~138 s, filters short plume whiffs"))
    reg.register("device.stereo_baseline_m", coeff(
        0.04, Evidence.ASSUMED, "typical dev-board spacing of the two MiCS-6814 dies",
        units="m",
        notes="NOT measured on the Scentience dev kit; measure the centre-to-centre "
              "die spacing on your board and set OlfactionWorld(stereo_baseline_m=...) "
              "/ PlumeNavConfig.stereo_baseline_m before quantitative stereo claims"))
    return reg


class ScentienceV1Device:
    """Batched full-channel model with independent environment clocks/RNGs.

    Arithmetic is torch-native. Input validation, per-environment generators
    and optional delay-history bookkeeping use Python and may synchronize the
    device. This is not a claim of a fused, synchronization-free GPU kernel.
    """

    def __init__(self, cfg: DeviceConfig, n_envs: int, device, randomize: bool = True,
                 *, species_names: tuple[str, ...] = ("ethanol",), seed: int | None = 0):
        if not isinstance(cfg, DeviceConfig):
            raise ValueError("cfg must be a shared DeviceConfig")
        self.cfg = deepcopy(cfg)
        self.cfg.__post_init__()
        if isinstance(n_envs, bool) or not isinstance(n_envs, Integral) or n_envs <= 0:
            raise ValueError("n_envs must be a positive integer")
        if not isinstance(species_names, tuple) or not species_names or any(
            not isinstance(s, str) or not s for s in species_names
        ) or len(set(species_names)) != len(species_names):
            raise ValueError("species_names must be a nonempty tuple of unique nonempty strings")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        self.n, self.species_names = int(n_envs), species_names
        self.dev = torch.device(device)
        self.dtype = getattr(torch, cfg.dtype)
        if self.dev.type == "mps" and self.dtype == torch.float64:
            raise ValueError("MPS does not support float64; explicitly select float32 or CPU/CUDA")
        self.mox_configs = self.cfg.resolved_mox()
        if self.dtype == torch.float32 and any(c.adc_bits >= 24 for c in self.mox_configs):
            warnings.warn("float32 cannot promise exact 24-bit-or-higher ADC reconstruction; "
                          "select dtype='float64' on CPU/CUDA for precision studies",
                          UserWarning, stacklevel=2)
        self.provenance = register_coefficients(ProvenanceRegistry())
        self._p = {name: self._tensor([getattr(c, name) for c in self.mox_configs])
                   for name in (
                       "r0_nominal", "rs_r0_clean_air", "tau_rise_s", "tau_fall_s",
                       "tau_flow_exponent", "tau_flow_ref_mps", "humidity_coeff",
                       "humidity_sensitivity_coeff", "activation_energy_ev",
                       "drift_sigma_per_sqrt_s", "white_noise_frac", "flicker_noise_frac",
                       "r_load", "v_supply", "v_ref", "polarity")}
        self._a = self._tensor([[c.sensitivity.get(g, (1.0, 0.0))[0]
                                for g in species_names] for c in self.mox_configs])
        self._beta = self._tensor([[c.sensitivity.get(g, (1.0, 0.0))[1]
                                   for g in species_names] for c in self.mox_configs])
        self._ec_sensitivity = self._tensor([
            [c.sensitivity_na_per_ppm.get(g, 0.0) for g in species_names]
            for c in self.cfg.ec_channels])
        self._ep = {name: self._tensor([getattr(c, name) for c in self.cfg.ec_channels])
                    for name in ("tau_s", "zero_current_na", "zero_tempco_na_per_k",
                                 "span_tempco_per_k", "noise_na", "drift_na_per_sqrt_s")}
        self._levels = self._tensor([2 ** c.adc_bits for c in self.mox_configs])
        self._max_counts = torch.tensor([2 ** c.adc_bits - 1 for c in self.mox_configs],
                                        device=self.dev, dtype=torch.int64)
        self._co2_index = species_names.index("carbon_dioxide") if "carbon_dioxide" in species_names else None
        self._max_delay = max(c.dead_volume_delay_s for c in self.mox_configs)
        self._delays = self._tensor([c.dead_volume_delay_s for c in self.mox_configs])
        root = np.random.SeedSequence(seed)
        self._rng = []
        for i in range(self.n):
            env_seed = int(np.random.SeedSequence(root.entropy, spawn_key=(i,)).generate_state(
                1, dtype=np.uint64)[0]) % (2**63 - 1)
            # No CPU-RNG fallback hidden behind a requested accelerator.
            self._rng.append(torch.Generator(device=self.dev).manual_seed(env_seed))
        self.y = self._tensor(np.zeros((self.n, 6)))
        self.r0 = torch.zeros_like(self.y)
        self.ln_drift = torch.zeros_like(self.y)
        self._flicker = self._tensor(np.zeros((self.n, 6, 4)))
        self._ec_y = self._tensor(np.zeros((self.n, 2)))
        self._ec_drift = torch.zeros_like(self._ec_y)
        self._co2_y = self._tensor(np.zeros(self.n))
        self._co2_held = torch.zeros_like(self._co2_y)
        self._asc_offset = torch.zeros_like(self._co2_y)
        self._window_min = torch.zeros_like(self._co2_y)
        # Python double clocks avoid float32 cadence drift on long episodes.
        self._elapsed = [0.0] * self.n
        self._sample_elapsed = [0.0] * self.n
        self._asc_elapsed = [0.0] * self.n
        self._history = [deque() for _ in range(self.n)]
        self.last_mox_readings = {key: torch.full_like(self.y, float("nan")) for key in (
            "rs_true", "rs_measured", "ratio_measured", "ratio_baseline", "volts", "r0_current")}
        self.last_mox_readings["counts"] = torch.full(
            (self.n, 6), -1, device=self.dev, dtype=torch.int64)
        self.reset(randomize=randomize)

    def _tensor(self, value):
        return torch.as_tensor(value, device=self.dev, dtype=self.dtype)

    def _ids(self, env_ids):
        if env_ids is None:
            values = list(range(self.n))
        elif isinstance(env_ids, slice):
            values = list(range(self.n))[env_ids]
        elif isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            if env_ids.dtype == torch.bool:
                if env_ids.numel() != self.n:
                    raise ValueError("boolean env_ids must have n_envs elements")
                values = env_ids.nonzero(as_tuple=True)[0].tolist()
            elif env_ids.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                values = env_ids.tolist()
            else:
                raise ValueError("env_ids must contain integers, not floating-point indices")
        else:
            try:
                values = list(env_ids)
            except TypeError as exc:
                raise ValueError("env_ids must be an iterable of integer indices") from exc
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in values):
            raise ValueError("env_ids must contain integer indices")
        if len(set(values)) != len(values) or any(i < 0 or i >= self.n for i in values):
            raise ValueError("env_ids must be unique and in [0, n_envs)")
        return values, torch.tensor(values, device=self.dev, dtype=torch.long)

    def _normal(self, ids, shape):
        return torch.stack([torch.randn(shape, generator=self._rng[i],
                                        device=self.dev, dtype=self.dtype) for i in ids])

    def reset(self, env_ids=None, randomize: bool = False):
        """Reset selected dynamics; False restores nominal R0, True resamples.

        RNG streams continue rather than rewind. Reconstruct with the same
        seed and repeat the call sequence to replay an experiment.
        """
        if not isinstance(randomize, bool):
            raise ValueError("randomize must be boolean")
        ids, idx = self._ids(env_ids)
        if not ids:
            return
        self.y[idx] = self._p["rs_r0_clean_air"]
        self.ln_drift[idx] = 0
        self._flicker[idx] = 0
        self._ec_y[idx] = 0
        self._ec_drift[idx] = 0
        self._co2_y[idx] = self.cfg.co2.ambient_baseline_ppm
        self._co2_held[idx] = self.cfg.co2.ambient_baseline_ppm
        self._asc_offset[idx] = 0
        self._window_min[idx] = math.inf
        self.r0[idx] = self._p["r0_nominal"]
        if randomize:
            lo = self._tensor([math.log(c.r0_range[0]) for c in self.mox_configs])
            hi = self._tensor([math.log(c.r0_range[1]) for c in self.mox_configs])
            u = torch.stack([torch.rand((6,), generator=self._rng[i],
                                         device=self.dev, dtype=self.dtype) for i in ids])
            self.r0[idx] = torch.exp(lo + u * (hi - lo))
        for i in ids:
            self._elapsed[i] = self._sample_elapsed[i] = self._asc_elapsed[i] = 0.0
            self._history[i].clear()
        for key, value in self.last_mox_readings.items():
            value[idx] = -1 if key == "counts" else float("nan")

    def _validate_input(self, conc, n, name):
        if not isinstance(conc, torch.Tensor) or conc.shape != (n, len(self.species_names)):
            raise ValueError(f"{name} must be a tensor of shape ({n}, {len(self.species_names)})")
        if conc.device != self.y.device or conc.dtype != self.dtype:
            raise ValueError(f"{name} must use device={self.y.device}, dtype={self.dtype}; "
                             "convert explicitly before stepping")
        if not bool(torch.isfinite(conc).all()) or bool((conc < 0).any()):
            raise ValueError(f"{name} must contain finite, nonnegative ppm values")

    def _delayed(self, ids, dt, initial, target, tau, current):
        if not self._max_delay:
            return current
        result = current.clone()
        for row, i in enumerate(ids):
            start, end = self._elapsed[i], self._elapsed[i] + dt
            history = self._history[i]
            history.append((start, end, initial[row].clone(), target[row].clone(), tau[row].clone()))
            oldest_query = end - self._max_delay
            while len(history) > 1 and history[0][1] <= oldest_query:
                history.popleft()
            # Compute query in Python double before converting to model dtype.
            query = [end - c.dead_volume_delay_s for c in self.mox_configs]
            prehistory = torch.tensor([q <= 0 for q in query], device=self.dev)
            result[row] = torch.where(prehistory, self._p["rs_r0_clean_air"], result[row])
            for left, right, y0, tgt, time_constant in history:
                mask = torch.tensor([left < q <= right for q in query], device=self.dev)
                local_dt = self._tensor([max(q - left, 0) for q in query])
                value = y0 - torch.expm1(-local_dt / time_constant) * (tgt - y0)
                result[row] = torch.where(mask, value, result[row])
        return result

    def step(self, conc_ppm, dt: float, conc_ppm_2=None, env_ids=None):
        """Return (K, 11) channels for the K selected environments, in order.

        Concentrations are (K, S), in species_names order. CO2 follows
        cfg.co2.concentration_mode: excess (legacy) or absolute ppm. Inputs
        are held constant during the positive scalar dt.
        Right air feeds only the right MOX triplet. Other environments do not
        advance. Unknown configured species have zero calibrated sensitivity.
        """
        finite_value("dt", dt, positive=True)
        ids, idx = self._ids(env_ids)
        self._validate_input(conc_ppm, len(ids), "conc_ppm")
        if conc_ppm_2 is not None:
            self._validate_input(conc_ppm_2, len(ids), "conc_ppm_2")
        if not ids:
            return torch.empty((0, len(CHANNELS)), device=self.dev, dtype=self.dtype)
        with torch.no_grad():
            return self._step(conc_ppm, float(dt), conc_ppm_2, ids, idx)

    def _step(self, conc_ppm, dt, conc_ppm_2, ids, idx):
        p, cfg = self._p, self.cfg
        right = conc_ppm if conc_ppm_2 is None else conc_ppm_2
        concentrations = torch.cat((conc_ppm[:, None, :].expand(-1, 3, -1),
                                     right[:, None, :].expand(-1, 3, -1)), dim=1)
        base = p["rs_r0_clean_air"]
        ratios = self._a * concentrations.clamp_min(1e-9).pow(-self._beta)
        delta = torch.where(self._beta > 0, (base[:, None] - ratios).clamp_min(0),
                            -(ratios - base[:, None]).clamp_min(0))
        active = (self._beta != 0) & (concentrations > 1e-9)
        delta = torch.where(active, delta, 0.0)
        target = (base - delta.sum(dim=-1)).clamp_min(1e-3)
        ah = absolute_humidity(cfg.ambient_temp_c, cfg.ambient_rh)
        ln_target = target.log() + p["humidity_coeff"] * ah
        total = torch.where(self._beta != 0, concentrations, 0.0).sum(dim=-1)
        ln_target += p["humidity_sensitivity_coeff"] * ah * torch.where(
            total > 1e-9, total.clamp_min(1e-9).log(), 0.0)
        ln_target += p["activation_energy_ev"] / (8.617333e-5 * (cfg.ambient_temp_c + 273.15)) \
            - p["activation_energy_ev"] / (8.617333e-5 * 293.15)
        target = ln_target.exp()
        if not bool(torch.isfinite(target).all()) or bool((target <= 0).any()):
            raise ValueError("MOX target is outside the finite range of configured arithmetic")
        initial = self.y[idx]
        tau = torch.where(p["polarity"] * (target - initial) > 0,
                          p["tau_rise_s"], p["tau_fall_s"])
        if cfg.flow_mps > 0:
            tau = tau * (max(cfg.flow_mps, 1e-3) / p["tau_flow_ref_mps"]).pow(-p["tau_flow_exponent"])
        tau = (tau / max(cfg.heater_level, 1e-3)).clamp_min(1e-6)
        self.y[idx] = initial - torch.expm1(-dt / tau) * (target - initial)
        delayed = self._delayed(ids, dt, initial, target, tau, self.y[idx])
        if any(c.drift_sigma_per_sqrt_s for c in self.mox_configs):
            self.ln_drift[idx] += p["drift_sigma_per_sqrt_s"] * math.sqrt(dt) * self._normal(ids, (6,))
        r0_current = self.r0[idx] * self.ln_drift[idx].exp()
        noise = torch.zeros_like(initial)
        if any(c.white_noise_frac for c in self.mox_configs):
            noise += p["white_noise_frac"] * self._normal(ids, (6,))
        if any(c.flicker_noise_frac for c in self.mox_configs):
            a = self._tensor([math.exp(-dt / t) for t in (0.1, 1., 10., 100.)])
            self._flicker[idx] = a * self._flicker[idx] + (1 - a*a).clamp_min(0).sqrt() * self._normal(ids, (6, 4))
            noise += p["flicker_noise_frac"] * self._flicker[idx].sum(dim=-1) / 2
        rs = (delayed * r0_current * (1 + noise)).clamp_min(1)
        volts = p["v_supply"] * p["r_load"] / (rs + p["r_load"])
        q = p["v_ref"] / self._levels
        counts = torch.minimum((volts / q).floor().clamp_min(0).to(torch.int64), self._max_counts)
        measured_volts = counts * q
        rs_hat = p["r_load"] * (p["v_supply"] / torch.maximum(measured_volts, q * 0.5) - 1)
        diagnostics = dict(rs_true=rs, rs_measured=rs_hat, ratio_measured=rs_hat / r0_current,
                           ratio_baseline=rs_hat / self.r0[idx], counts=counts.to(torch.int64),
                           volts=measured_volts, r0_current=r0_current)
        for key, value in diagnostics.items():
            self.last_mox_readings[key][idx] = value
        out = torch.empty((len(ids), len(CHANNELS)), device=self.dev, dtype=self.dtype)
        out[:, :6] = diagnostics[cfg.ratio_feature]
        out[:, 7], out[:, 8] = cfg.ambient_temp_c, cfg.ambient_rh
        ep = self._ep
        span = 1 + ep["span_tempco_per_k"] * (cfg.ambient_temp_c - 20)
        ec_target = (conc_ppm @ self._ec_sensitivity.T) * span
        self._ec_y[idx] += -torch.expm1(-dt / ep["tau_s"]) * (ec_target - self._ec_y[idx])
        if any(c.drift_na_per_sqrt_s for c in cfg.ec_channels):
            self._ec_drift[idx] += ep["drift_na_per_sqrt_s"] * math.sqrt(dt) * self._normal(ids, (2,))
        out[:, 9:] = self._ec_y[idx] + ep["zero_current_na"] + ep["zero_tempco_na_per_k"] * (cfg.ambient_temp_c - 20) + self._ec_drift[idx]
        if any(c.noise_na for c in cfg.ec_channels):
            out[:, 9:] += ep["noise_na"] * self._normal(ids, (2,))
        out[:, 6] = self._step_co2(conc_ppm, dt, ids, idx)
        for i in ids:
            self._elapsed[i] += dt
        return out

    def _step_co2(self, conc_ppm, dt, ids, idx):
        cc = self.cfg.co2
        tolerance = 1e-12 * min(cc.sample_interval_s, cc.asc_window_s)
        for row, i in enumerate(ids):
            missing = cc.ambient_baseline_ppm if cc.concentration_mode == "absolute" else 0.0
            concentration = conc_ppm[row, self._co2_index] if self._co2_index is not None else missing
            true_ppm = concentration if cc.concentration_mode == "absolute" else cc.ambient_baseline_ppm + concentration
            remaining = dt
            while remaining > 0:
                segment = min(remaining, cc.sample_interval_s - self._sample_elapsed[i])
                if cc.asc_enabled:
                    segment = min(segment, cc.asc_window_s - self._asc_elapsed[i])
                self._co2_y[i] += -math.expm1(-segment / cc.tau63_s) * (true_ppm - self._co2_y[i])
                self._sample_elapsed[i] += segment
                remaining = max(0.0, remaining - segment)
                if cc.asc_enabled:
                    self._window_min[i] = torch.minimum(self._window_min[i], self._co2_y[i])
                    self._asc_elapsed[i] += segment
                    if self._asc_elapsed[i] >= cc.asc_window_s - tolerance:
                        self._asc_offset[i] += cc.asc_gain * (cc.asc_target_ppm - self._window_min[i])
                        self._window_min[i] = math.inf
                        self._asc_elapsed[i] = 0.0
                if self._sample_elapsed[i] >= cc.sample_interval_s - tolerance:
                    self._sample_elapsed[i] = 0.0
                    reading = self._co2_y[i] + self._asc_offset[i]
                    if cc.accuracy_bias_fraction:
                        # Accuracy is a bounded systematic bias, not Gaussian noise.
                        y = self._co2_y[i]
                        envelope = cc.accuracy_base_ppm + cc.accuracy_frac * y
                        for upper, base, fraction in reversed(cc.accuracy_bands):
                            envelope = torch.where(y <= upper, base + fraction * y, envelope)
                        lo, hi = cc.accuracy_range_ppm
                        reading += cc.accuracy_bias_fraction * torch.where(
                            (y >= lo) & (y <= hi), envelope, 0.0)
                    if cc.repeatability_ppm:
                        reading += cc.repeatability_ppm * torch.randn(
                            (), generator=self._rng[i], device=self.dev, dtype=self.dtype)
                    if cc.resolution_ppm:
                        reading = torch.round(reading / cc.resolution_ppm) * cc.resolution_ppm
                    if cc.output_range_ppm is not None:
                        reading = reading.clamp(*cc.output_range_ppm)
                    self._co2_held[i] = reading
        return self._co2_held[idx]


def build_device(device_profile: str, sensor_profile: str, n_envs: int,
                 device, randomize: bool = True, *,
                 species_names: tuple[str, ...] = ("ethanol",), seed: int | None = 0,
                 config: DeviceConfig | Mapping | None = None) -> ScentienceV1Device:
    if device_profile != "scentience_v1":
        raise ValueError(f"unknown device profile {device_profile!r}; only 'scentience_v1' exists")
    if isinstance(config, Mapping):
        config = DeviceConfig.from_dict({"sensor_profile": sensor_profile, **config})
    cfg = DeviceConfig(sensor_profile=sensor_profile) if config is None else config
    if not isinstance(cfg, DeviceConfig):
        raise ValueError("config must be a DeviceConfig; use DeviceConfig.from_dict for JSON")
    if cfg.sensor_profile != sensor_profile:
        raise ValueError("sensor_profile conflicts with config.sensor_profile")
    return ScentienceV1Device(cfg, n_envs, device, randomize,
                             species_names=species_names, seed=seed)
