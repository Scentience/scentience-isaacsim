"""
Scentience V1 device -- single-instance NumPy implementation.

This is the device model for standalone Python, the Gymnasium environment,
and single-robot use: one virtual unit, stepped scalar-wise, no torch, no
Isaac.  The vectorised torch twin (sensors/scentience_v1.py) is for Isaac Lab
RL at scale; tests/test_device_parity.py holds the two to the same step
response when torch is available.

Channel schema == the Scentience BLE/Sockets API ordering, so downstream code
cannot tell simulation from hardware without inspecting the transport.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .electrochemical import EC_CO, EC_H2S, ECChannel
from .mox import (FAST_OVERRIDES, MICS6814_NH3, MICS6814_OX, MICS6814_RED,
                  MoxChannel, MoxChannelConfig)
from .scd4x import SCD4xChannel, SCD4xConfig

CHANNELS = (
    "mics1_red", "mics1_nh3", "mics1_ox",
    "mics2_red", "mics2_nh3", "mics2_ox",
    "co2_ppm", "temperature_c", "relative_humidity", "ec1", "ec2",
)

SENSOR_PROFILES = ("packaged_slow", "fast_modulated")


@dataclass
class DeviceState:
    temp_c: float = 20.0
    rh_pct: float = 50.0
    flow_mps: float = 0.3
    heater_level: float = 1.0


class ScentienceV1:
    """
    One virtual Reconnaisscent-class unit: 2x MiCS-6814 (RED/NH3/OX each),
    an SCD4x CO2 channel, and 2 electrochemical cells, per the stack described
    in France et al. (arXiv:2602.19577) and the Scentience product docs.

    `sensor_profile` selects the MOX dynamic regime:
      packaged_slow  tau_fall ~12 s -- retains ~19% of plume whiff events
      fast_modulated tau_fall ~46 ms (Dennler-class) -- retains ~97%
    The choice changes the POMDP the robot is solving. State it in results.
    """

    def __init__(self, sensor_profile: str = "packaged_slow",
                 seed: int | None = 0, randomize_unit: bool = True):
        if sensor_profile not in SENSOR_PROFILES:
            raise ValueError(f"sensor_profile must be one of {SENSOR_PROFILES}")
        self.sensor_profile = sensor_profile
        self.rng = np.random.default_rng(seed)

        def mox(cfg: MoxChannelConfig) -> MoxChannel:
            if sensor_profile == "fast_modulated":
                cfg = replace(cfg, **FAST_OVERRIDES)
            return MoxChannel(cfg, self.rng, randomize=randomize_unit)

        self.mox = [mox(MICS6814_RED), mox(MICS6814_NH3), mox(MICS6814_OX),
                    mox(MICS6814_RED), mox(MICS6814_NH3), mox(MICS6814_OX)]
        self.co2 = SCD4xChannel(SCD4xConfig(), self.rng)
        self.ec = [ECChannel(EC_CO, self.rng), ECChannel(EC_H2S, self.rng)]

    def reset(self) -> None:
        for m in self.mox:
            m.reset()
        self.co2.reset()
        for e in self.ec:
            e.reset()

    def step(self, conc_ppm: dict[str, float], dt: float,
             state: DeviceState | None = None) -> dict[str, float]:
        st = state or DeviceState()
        out: dict[str, float] = {}
        for name, ch in zip(CHANNELS[:6], self.mox):
            r = ch.step(conc_ppm, dt, temp_c=st.temp_c, rh_pct=st.rh_pct,
                        flow_mps=st.flow_mps, heater_level=st.heater_level)
            out[name] = r["ratio_measured"]
        out["co2_ppm"] = self.co2.step(conc_ppm.get("carbon_dioxide", 0.0), dt)["co2_ppm"]
        out["temperature_c"] = st.temp_c
        out["relative_humidity"] = st.rh_pct
        out["ec1"] = self.ec[0].step(conc_ppm, dt, temp_c=st.temp_c)["current_na"]
        out["ec2"] = self.ec[1].step(conc_ppm, dt, temp_c=st.temp_c)["current_na"]
        return out

    def observation_vector(self, reading: dict[str, float]) -> np.ndarray:
        return np.array([reading[c] for c in CHANNELS], dtype=np.float64)
