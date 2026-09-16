"""Adapters around existing transport/device models; no Isaac imports.

NumPy is the general reference path. Warp is restricted to its implemented
single-source, single-passive-species transport; unsupported physics fails.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch

from scentience_olfaction.chemistry.registry import DEFAULT_REGISTRY
from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig

from ._torch import indices, load_callable


def canonical_species(names):
    result = tuple(DEFAULT_REGISTRY.get(name).name for name in names)
    if not result or len(set(result)) != len(result):
        raise ValueError("species must be nonempty and unique after alias resolution")
    return result


def plume_config(cfg, species):
    result = deepcopy(cfg) if cfg is not None else FilamentPlumeConfig(species=species[0])
    result.species = DEFAULT_REGISTRY.get(result.species).name
    if result.emitters is not None:
        for emitter in result.emitters:
            emitter.species = DEFAULT_REGISTRY.get(emitter.species).name
    emitted = {e.species for e in result.build_emitters()}
    if hasattr(result, "canonical_background"):
        result.background_ppm = result.canonical_background(DEFAULT_REGISTRY)
        emitted.update(result.background_ppm)
    if not emitted.issubset(species):
        raise ValueError(f"plume emits {sorted(emitted - set(species))} absent from sensor.species")
    return result


class NumpyPlumeBatch:
    def __init__(self, cfg, species, n_envs, device, seed):
        self.species_names, self.device = species, device
        self.plumes = [FilamentPlume(deepcopy(cfg), seed=seed + i) for i in range(n_envs)]
        self._columns = [species.index(name) for name in self.plumes[0].species_names]

    def step(self, dt):
        for plume in self.plumes:
            plume.step(dt)

    def sample(self, probes_local, env_ids):
        """Return independent (K, P, S) samples and (K, 3) wind."""
        points, ids = probes_local.detach().cpu().numpy(), env_ids.cpu().tolist()
        conc = np.zeros((*points.shape[:2], len(self.species_names)), dtype=np.float32)
        wind = np.zeros((len(ids), 3), dtype=np.float32)
        for row, env_id in enumerate(ids):
            plume = self.plumes[env_id]
            conc[row][:, self._columns] = plume.sample_species(points[row])
            wind[row] = plume.airflow.velocity(points[row].mean(axis=0)[None])[0]
        return (torch.as_tensor(conc, device=self.device),
                torch.as_tensor(wind, device=self.device))

    def reset(self, env_ids):
        for env_id in env_ids.cpu().tolist():
            self.plumes[env_id].reset()


class WarpPlumeBatch:
    def __init__(self, cfg, species, n_envs, device, seed):
        if len(species) != 1 or cfg.emitters is not None or cfg.buoyancy_model != "none":
            raise ValueError("Warp supports one passive species/source and no buoyancy; use numpy")
        if DEFAULT_REGISTRY.get(species[0]).decay_rate_per_s != 0:
            raise ValueError("Warp does not implement species decay; use numpy")
        from scentience_olfaction.transport.filament_warp import WarpFilamentPlume

        self.plume = WarpFilamentPlume(cfg, n_envs=n_envs, device=str(device), seed=seed)
        self._probes = torch.zeros(n_envs, 3, device=device)

    def step(self, dt):
        self.plume.step(dt)

    def sample(self, probes_local, env_ids):
        samples = []
        # The core Warp API always samples the full batch and aliases its output.
        # Scatter selected probes into a full buffer; clone before the next probe.
        for probe in probes_local.unbind(dim=1):
            self._probes[env_ids] = probe
            self.plume.set_probes_torch(self._probes)
            samples.append(self.plume.sample_torch()[env_ids].clone())
        return torch.stack(samples, dim=1), self.plume.wind_torch()[env_ids].clone()

    def reset(self, env_ids):
        if env_ids.numel():
            ids = None if env_ids.numel() == self._probes.shape[0] else env_ids.cpu().tolist()
            self.plume.reset(ids)
            self._probes[env_ids] = 0


class NumpyDeviceBatch:
    """Explicit CPU reference adapter; configurable single-unit manufacturer factory.

    Factory kwargs: device_profile, sensor_profile, seed, config.
    A unit exposes channel_names, step(dict, dt, conc_ppm_2=dict)->dict, reset().
    """

    def __init__(self, *, device_profile, sensor_profile, n_envs, device, randomize,
                 species_names, seed=0, config=None, factory=None):
        self.n, self.device, self.species_names = n_envs, device, species_names
        self.seed, self._episodes = seed, [0] * n_envs
        self._kwargs = dict(device_profile=device_profile, sensor_profile=sensor_profile,
                            config=config)
        self._factory = load_callable(factory) if factory is not None else None
        self.units = [self._make(i, randomize) for i in range(n_envs)]
        self.channel_names = tuple(self.units[0].channel_names)
        if any(tuple(u.channel_names) != self.channel_names for u in self.units):
            raise ValueError("all device units must expose identical channel_names")

    @staticmethod
    def _default_factory(*, device_profile, sensor_profile, seed, randomize, config):
        from scentience_olfaction.sensors.factory import create_sensor

        return create_sensor(device_profile, sensor_profile, seed=seed, config=config,
                             randomize=randomize)

    def _make(self, i, randomize):
        kwargs = dict(**deepcopy(self._kwargs), seed=self.seed + i + self.n * self._episodes[i])
        if self._factory is None:
            return self._default_factory(**kwargs, randomize=randomize)
        return self._factory(**kwargs)

    def step(self, conc, dt, conc_ppm_2=None, env_ids=None):
        ids = indices(self.n, self.device, env_ids)
        left = conc.detach().cpu().numpy()
        right = left if conc_ppm_2 is None else conc_ppm_2.detach().cpu().numpy()
        times = torch.as_tensor(dt).expand(len(ids)).cpu().numpy()
        if left.shape != (len(ids), len(self.species_names)) or right.shape != left.shape:
            raise ValueError("concentration must have selected-row shape (K, S)")
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("dt must be finite and nonnegative")
        rows = []
        for row, i in enumerate(ids.cpu().tolist()):
            reading = self.units[i].step(dict(zip(self.species_names, left[row])),
                                         float(times[row]),
                                         conc_ppm_2=dict(zip(self.species_names, right[row])))
            rows.append([reading[name] for name in self.channel_names])
        return torch.tensor(rows, dtype=conc.dtype, device=self.device).reshape(
            len(ids), len(self.channel_names))

    def reset(self, env_ids=None, randomize=True):
        for i in indices(self.n, self.device, env_ids).cpu().tolist():
            if randomize:
                self._episodes[i] += 1
                unit = self._make(i, True)
                if tuple(unit.channel_names) != self.channel_names:
                    raise ValueError("device channel schema changed on reset")
                self.units[i] = unit
            else:
                if self._factory is None and self._kwargs["device_profile"] == "scentience_v1":
                    unit = self.units[i].device
                    calibration = [channel.r0 for channel in unit.mox]
                    self.units[i].reset(randomize=False)
                    # The scalar core restores nominal R0 on False. At the
                    # integration boundary False means preserve this unit.
                    for channel, r0 in zip(unit.mox, calibration):
                        channel.r0 = r0
                else:
                    self.units[i].reset()


class DeviceInputAdapter:
    """Selected-row scalar-dt dispatch; concentration basis resolved at creation.

    Models see zero-order-held input over the actual interval since capture.
    Explicit background is absolute input. No concentration is subtracted here.
    """

    def __init__(self, model, channel_names, species, background):
        self.model, self.channel_names = model, channel_names
        self.dtype = getattr(model, "dtype", torch.float32)
        # Resolve actual constructed configs, including manufacturer defaults.
        unit = model.units[0] if isinstance(model, NumpyDeviceBatch) else model
        inner = getattr(unit, "device", unit)
        cc = getattr(getattr(inner, "cfg", None), "co2", None)
        if cc is None:
            cc = getattr(getattr(unit, "channel", None), "cfg", None)
        bg = background.get("carbon_dioxide")
        if bg is not None and "carbon_dioxide" in species and "co2_ppm" in channel_names:
            if cc is None or not hasattr(cc, "concentration_mode"):
                raise ValueError("CO2 background requires a device with an explicit CO2 config")
            if cc.concentration_mode != "absolute":
                raise ValueError("CO2 background requires concentration_mode='absolute'")

    def step(self, conc, dt, conc_ppm_2=None, env_ids=None):
        left = conc.clone()
        right = left.clone() if conc_ppm_2 is None else conc_ppm_2.clone()
        # Cast to the model's declared arithmetic dtype at the boundary.
        dtype = getattr(self.model, "dtype", conc.dtype)
        left, right = left.to(dtype=dtype), right.to(dtype=dtype)
        elapsed = torch.as_tensor(dt, device=conc.device).expand(len(conc))
        if not bool(torch.isfinite(elapsed).all()) or bool((elapsed <= 0).any()):
            raise ValueError("selected device capture intervals must be finite and positive")
        ids = env_ids if env_ids is not None else torch.arange(len(conc), device=conc.device)
        out = torch.empty(len(conc), len(self.channel_names), device=conc.device, dtype=dtype)
        # Most captures have one group. Reset/lazy reads may split the batch.
        for value in elapsed.unique():
            selected = elapsed == value
            reading = self.model.step(left[selected], float(value),
                                      conc_ppm_2=right[selected], env_ids=ids[selected])
            if reading.shape != (int(selected.sum()), len(self.channel_names)):
                raise ValueError("device step returned an invalid selected-row channel shape")
            out[selected] = reading
        return out

    def reset(self, env_ids=None, randomize=True):
        calibration = None
        if not randomize and isinstance(getattr(self.model, "r0", None), torch.Tensor):
            selected = indices(self.model.r0.shape[0], self.model.r0.device, env_ids)
            calibration = self.model.r0[selected].clone()
        self.model.reset(env_ids, randomize=randomize)
        if calibration is not None:
            self.model.r0[selected] = calibration


def resolve_device_config(cfg, background):
    """Mirror OlfactionWorld's absolute-CO2 reservoir rule without mutating cfg."""
    config = deepcopy(cfg.device_config)
    bg = background.get("carbon_dioxide")
    if bg is not None and cfg.device_profile != "emstat_pico":
        scentience = cfg.device_profile == "scentience_v1"
        if config is None:
            co2 = {"concentration_mode": "absolute", "ambient_baseline_ppm": bg}
            config = {"co2": co2} if scentience else co2
        elif isinstance(config, dict):
            co2 = config.setdefault("co2", {}) if scentience else config
            if co2.get("concentration_mode", "absolute") != "absolute":
                raise ValueError("background CO2 requires concentration_mode='absolute'")
            co2.setdefault("concentration_mode", "absolute")
            co2.setdefault("ambient_baseline_ppm", bg)
        else:
            co2 = config.co2 if scentience else config
            if co2.concentration_mode != "absolute":
                raise ValueError("background CO2 requires concentration_mode='absolute'")
            co2 = replace(co2, ambient_baseline_ppm=bg)
            config = replace(config, co2=co2) if scentience else co2
    if cfg.device_backend == "torch" and cfg.device_profile == "scentience_v1" and isinstance(config, dict):
        from scentience_olfaction.sensors.device_np import DeviceConfig

        config = DeviceConfig.from_dict({"sensor_profile": cfg.sensor_profile, **config})
    return config


def build_backends(cfg, n, device):
    species = canonical_species(cfg.species)
    pcfg = plume_config(cfg.plume, species)
    transports = {"numpy": NumpyPlumeBatch, "warp": WarpPlumeBatch}
    if cfg.transport_backend not in transports:
        raise ValueError("transport_backend must be 'numpy' or 'warp'")
    kwargs = dict(device_profile=cfg.device_profile, sensor_profile=cfg.sensor_profile,
                  n_envs=n, device=device, randomize=cfg.randomize_per_episode,
                  species_names=species, seed=cfg.seed,
                  config=resolve_device_config(cfg, getattr(pcfg, "background_ppm", {})))
    if cfg.device_backend == "numpy":
        model = NumpyDeviceBatch(**kwargs, factory=cfg.numpy_device_factory)
    elif cfg.device_backend == "torch":
        factory = cfg.torch_device_factory or "scentience_olfaction.sensors.scentience_v1:build_device"
        model = load_callable(factory)(**kwargs)
    else:
        raise ValueError("device_backend must be 'numpy' or 'torch'")
    names = getattr(model, "channel_names", None)
    if names is None and cfg.device_profile == "scentience_v1" and cfg.torch_device_factory is None:
        from scentience_olfaction.sensors.scentience_v1 import CHANNELS

        names = CHANNELS
    names = tuple(names or ())
    if not names or len(set(names)) != len(names):
        raise ValueError("device.channel_names must be nonempty and unique")
    if cfg.channel_names is not None and tuple(cfg.channel_names) != names:
        raise ValueError(f"configured channel_names do not match device schema {names}")
    model = DeviceInputAdapter(model, names, species, getattr(pcfg, "background_ppm", {}))
    return transports[cfg.transport_backend](pcfg, species, n, device, cfg.seed), model, species
