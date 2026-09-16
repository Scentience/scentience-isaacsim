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
import hashlib
from copy import deepcopy
from dataclasses import replace

from .plume.filament import FilamentPlume, FilamentPlumeConfig
from .sensors.device_np import DeviceState
from .sensors.factory import create_sensor


class OlfactionWorld:
    """A plume plus independently seeded devices using one sensor configuration."""

    def __init__(self, plume: FilamentPlume,
                 sensor_profile: str = "packaged_slow", seed: int | None = 0,
                 stereo_baseline_m: float = 0.04, *,
                 device_profile: str = "scentience_v1", device_config=None):
        self.plume = plume
        self.sensor_profile = sensor_profile
        self.device_profile = device_profile
        self.device_config = deepcopy(device_config)
        self._devices = {}
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        self._seed = seed if seed is not None else int(np.random.SeedSequence().entropy)
        self._episode = 0
        if not np.isfinite(stereo_baseline_m) or stereo_baseline_m < 0:
            raise ValueError("stereo_baseline_m must be finite and nonnegative")
        self.stereo_baseline_m = stereo_baseline_m
        """Centre-to-centre spacing of the two MiCS-6814 dies. Evidence:
        ASSUMED (typical dev-board spacing) -- measure YOUR kit and set it.
        Used only when `read(..., heading=...)` is given; see provenance."""
        self._last_dt = 0.05
        self._resolve_background()
        self.channel_names = self._device("nose").channel_names

    def _resolve_background(self):
        """An explicit atmospheric CO2 reservoir makes truth absolute ppm."""
        background = self.plume.cfg.canonical_background(self.plume.registry).get("carbon_dioxide")
        if background is None or self.device_profile == "emstat_pico":
            return
        cfg = self.device_config
        if cfg is None:
            co2 = {"concentration_mode": "absolute", "ambient_baseline_ppm": background}
            self.device_config = {"co2": co2} if self.device_profile == "scentience_v1" else co2
        elif isinstance(cfg, dict):
            if self.device_profile == "scentience_v1":
                co2 = cfg.setdefault("co2", {})
            else:
                co2 = cfg
            if co2.get("concentration_mode", "absolute") != "absolute":
                raise ValueError("background CO2 requires device concentration_mode='absolute'")
            co2.setdefault("concentration_mode", "absolute")
            co2.setdefault("ambient_baseline_ppm", background)
        else:
            co2 = cfg.co2 if self.device_profile == "scentience_v1" else cfg
            if co2.concentration_mode != "absolute":
                raise ValueError("background CO2 requires device concentration_mode='absolute'")
            if self.device_profile == "scentience_v1":
                self.device_config = replace(cfg, co2=replace(co2, ambient_baseline_ppm=background))
            else:
                self.device_config = replace(co2, ambient_baseline_ppm=background)

    # ----------------------------------------------------------- constructors
    @classmethod
    def simple(cls, source=(0.0, 0.0, 1.0), species: str = "ethanol",
               wind=(1.0, 0.0, 0.0), strength_ppm: float = 20.0,
               sensor_profile: str = "packaged_slow",
               seed: int = 0, **world_options) -> "OlfactionWorld":
        """One source, one wind, sane defaults. Start here."""
        cfg = FilamentPlumeConfig(source_pos=tuple(source), species=species,
                                  wind_mean=tuple(wind),
                                  ppm_center_initial=strength_ppm)
        return cls(FilamentPlume(cfg, seed=seed), sensor_profile, seed, **world_options)

    # ---------------------------------------------------------------- devices
    def _device(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("device name must be a nonempty string")
        if name not in self._devices:
            name_seed = int.from_bytes(hashlib.blake2s(name.encode(), digest_size=8).digest(), "little")
            self._devices[name] = create_sensor(
                self.device_profile, sensor_profile=self.sensor_profile,
                seed=self._seed + name_seed + self._episode, config=self.device_config)
        return self._devices[name]

    # ------------------------------------------------------------------- API
    def step(self, dt: float) -> None:
        self.plume.step(dt)
        if dt > 0:
            self._last_dt = dt

    def read(self, position, dt: float | None = None, name: str = "nose",
             state: DeviceState | None = None,
             heading: float | None = None,
             orientation_wxyz=None) -> dict[str, float]:
        """Simulated device reading at a world position -- what the robot sees.
        `dt` defaults to the last nonzero world.step period (initially 0.05 s).
        Each read advances that named device; call once per sensor tick.
        Pass orientation_wxyz for stereo sensing on a pitched/rolled robot.

        Stereo olfaction: pass `heading` (rad, world frame, XY plane) and the
        two MiCS dies sample separated points -- `chem_left_*` is the LEFT die,
        `chem_right_*` the RIGHT die, `stereo_baseline_m` apart, perpendicular to
        the heading. Without `heading`, both dies sample `position` (mono),
        which is the pre-stereo behaviour, unchanged."""
        dt = dt if dt is not None else self._last_dt
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("sensor dt must be finite and positive")
        p = _position(position)
        if heading is not None and (not np.isfinite(heading) or orientation_wxyz is not None):
            raise ValueError("provide one finite heading or an orientation_wxyz quaternion")
        left_n = None
        if orientation_wxyz is not None:
            q = np.asarray(orientation_wxyz, float)
            if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
                raise ValueError("orientation_wxyz must be a finite unit quaternion")
            axis = np.array([0.0, 1.0, 0.0])
            left_n = axis + 2 * np.cross(q[1:], np.cross(q[1:], axis) + q[0] * axis)
        elif heading is not None:
            left_n = np.array([-np.sin(heading), np.cos(heading), 0.0])
        dev = self._device(name)
        if left_n is None or self.stereo_baseline_m <= 0.0:
            return dev.step(self.truth(p), dt, state)
        half = 0.5 * self.stereo_baseline_m
        conc_l = self.truth(p + half * left_n)
        conc_r = self.truth(p - half * left_n)
        return dev.step(conc_l, dt, state, conc_ppm_2=conc_r)

    def truth(self, position) -> dict[str, float]:
        """Ground-truth ppm by species. For debugging, labels, and reward
        shaping. Do not feed to a policy you intend to deploy."""
        c = self.plume.sample_species(_position(position)[None])[0]
        return dict(zip(self.plume.species_names, c.tolist()))

    def wind_at(self, position) -> np.ndarray:
        return self.plume.airflow.velocity(_position(position)[None])[0]

    def read_ble(self, position, dt: float | None = None, name: str = "nose", *,
                 uid: str = "SIM001", timestamp: str | None = None,
                 pressure_hpa: float = 1010.0, side: str = "left",
                 include_sim_metadata: bool = True, state: DeviceState | None = None,
                 heading: float | None = None, orientation_wxyz=None) -> dict:
        """Advance one Scentience sensor tick and return an SDK-shaped reading.

        Default timestamps use plume time since 1970-01-01 UTC, preserving
        deterministic subsecond timing. Supply a timestamp to align an external
        clock. Use bridge.ble_frame on an existing reading to avoid a new tick.
        """
        from datetime import datetime, timedelta, timezone
        from .bridge.ble_schema import ble_frame, _validate_frame_options

        if self.device_profile != "scentience_v1":
            raise ValueError("read_ble requires device_profile='scentience_v1'")
        _validate_frame_options(uid, timestamp, pressure_hpa, side, include_sim_metadata)
        if timestamp is None:
            stamp = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self.plume.t)
            timestamp = stamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
        reading = self.read(position, dt=dt, name=name, state=state,
                            heading=heading, orientation_wxyz=orientation_wxyz)
        frame = ble_frame(reading, uid=uid, timestamp=timestamp, pressure_hpa=pressure_hpa,
                          device_config=self._device(name).configuration, side=side,
                          include_sim_metadata=include_sim_metadata)
        if include_sim_metadata:
            frame["_sim_time_s"] = float(self.plume.t)
        return frame

    def sensor_diagnostics(self, name: str = "nose") -> dict:
        """Snapshot of the most recent instrument diagnostics, without a tick."""
        return self._device(name).last_reading

    def configuration(self) -> dict:
        """Resolved JSON-compatible scenario and calibration for experiment logs."""
        from .config import config_dict

        return {"plume": config_dict(self.plume.cfg), "seed": self._seed,
                "device_profile": self.device_profile, "sensor_profile": self.sensor_profile,
                "stereo_baseline_m": self.stereo_baseline_m,
                "device_config": config_dict(self._device("nose").configuration)}

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            if not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a nonnegative integer")
            self._seed, self._episode = seed, 0
        else:
            self._episode += 1
        self.plume.reset(seed)
        self._devices.clear()
        self._last_dt = 0.05


def _position(position):
    p = np.asarray(position, float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("position must be a finite XYZ vector in metres")
    return p
