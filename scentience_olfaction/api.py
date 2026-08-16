"""
The five-line API.

    from scentience_olfaction import OlfactionWorld

    world = OlfactionWorld.simple()          # ethanol source, 1 m/s wind
    world.step(0.05)
    reading = world.read((5.0, 0.0, 1.0))    # what the DEVICE reports
    truth   = world.truth((5.0, 0.0, 1.0))   # ground-truth ppm, for debugging

Everything else in the package is reachable from here, but nothing else is
required.  Complexity is opt-in: pass emitters/occupancy/airflow to the
constructor when you need them, ignore them when you don't.
"""

from __future__ import annotations

import numpy as np

from .plume.filament import FilamentPlume, FilamentPlumeConfig
from .sensors.device_np import CHANNELS, DeviceState, ScentienceV1


class OlfactionWorld:
    """A plume plus one or more virtual Scentience devices."""

    def __init__(self, plume: FilamentPlume,
                 sensor_profile: str = "packaged_slow", seed: int | None = 0):
        self.plume = plume
        self.sensor_profile = sensor_profile
        self._devices: dict[str, ScentienceV1] = {}
        self._seed = seed or 0
        self.channel_names = CHANNELS

    # ----------------------------------------------------------- constructors
    @classmethod
    def simple(cls, source=(0.0, 0.0, 1.0), species: str = "ethanol",
               wind=(1.0, 0.0, 0.0), strength_ppm: float = 20.0,
               sensor_profile: str = "packaged_slow",
               seed: int = 0) -> "OlfactionWorld":
        """One source, one wind, sane defaults. Start here."""
        cfg = FilamentPlumeConfig(source_pos=tuple(source), species=species,
                                  wind_mean=tuple(wind),
                                  ppm_center_initial=strength_ppm)
        return cls(FilamentPlume(cfg, seed=seed), sensor_profile, seed)

    # ---------------------------------------------------------------- devices
    def _device(self, name: str) -> ScentienceV1:
        if name not in self._devices:
            self._devices[name] = ScentienceV1(
                self.sensor_profile, seed=self._seed + len(self._devices) + 1)
        return self._devices[name]

    # ------------------------------------------------------------------- API
    def step(self, dt: float) -> None:
        self.plume.step(dt)

    def read(self, position, dt: float | None = None, name: str = "nose",
             state: DeviceState | None = None) -> dict[str, float]:
        """Simulated device reading at a world position -- what the robot sees.
        `dt` defaults to the last plume step size assumption of 0.05 s; pass
        your control period for correct sensor dynamics."""
        conc = self.truth(position)
        return self._device(name).step(conc, dt if dt is not None else 0.05, state)

    def truth(self, position) -> dict[str, float]:
        """Ground-truth ppm by species. For debugging, labels, and reward
        shaping. Do not feed to a policy you intend to deploy."""
        c = self.plume.sample_species(np.atleast_2d(np.asarray(position, float)))[0]
        return dict(zip(self.plume.species_names, c.tolist()))

    def wind_at(self, position) -> np.ndarray:
        return self.plume.airflow.velocity(np.atleast_2d(np.asarray(position, float)))[0]

    def reset(self, seed: int | None = None) -> None:
        self.plume.reset(seed)
        for d in self._devices.values():
            d.reset()
