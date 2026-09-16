"""Torch snapshots of the standard Isaac Lab 3 Imu, without synthesizing pose.

Lab 3 provides sensor-frame angular velocity and proper acceleration. Gravity
removal requires an explicit, synchronized orientation and world gravity from
the caller; neither is observable from the lightweight Imu data alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._torch import as_torch, quaternion, quat_apply, quat_inverse_apply


@dataclass(frozen=True)
class ImuSample:
    angular_velocity: torch.Tensor  # (N,3), rad/s
    specific_force: torch.Tensor | None  # (N,3); absent for native no-gravity input without gravity_w
    timestamp: torch.Tensor         # (N,), simulation episode seconds at capture
    valid: torch.Tensor             # (N,), timestamp > 0 (not a warmup quality guarantee)
    linear_acceleration: torch.Tensor | None = None  # gravity removed, if requested
    frame: str = "imu"
    orientation_w: torch.Tensor | None = None  # native Sim orientation converted to xyzw; not actor data


def read_imu(sensor, *, timestamp=None, rotation_target_from_imu=None,
             frame="imu", gravity_w=None, orientation_w=None) -> ImuSample:
    """Read Lab 3 ProxyArray (.torch), Warp, or torch Imu data as a snapshot.

    ``orientation_w`` is the xyzw IMU-to-world attitude at the *capture* time.
    ``gravity_w`` is actual simulation gravity, e.g. (0,0,-9.81), NOT a bias.
    Supplying both computes a = specific_force + R_world_to_imu * gravity.
    ``rotation_target_from_imu`` only rotates axes about the same origin; it
    cannot transport acceleration to a displaced robot origin without lever
    arm dynamics. No orientation is inferred from accelerometer measurements.

    Standard ImuData has no public timestamp. When one is not provided, this
    explicitly uses SensorBase._timestamp_last_update after the lazy data read.
    This dependency is checked by check_isaaclab_contract.py; wall-clock time
    is never substituted. Timestamps restart on per-environment reset.
    """
    data = sensor.data
    omega = as_torch(data.ang_vel_b).clone()
    force = as_torch(data.lin_acc_b).clone()
    if omega.ndim != 2 or omega.shape[-1] != 3 or force.shape != omega.shape:
        raise ValueError("Imu vectors must have shape (N,3)")
    if timestamp is None:
        timestamp = getattr(sensor, "_timestamp_last_update", None)
        if timestamp is None:
            raise ValueError("provide capture timestamp; sensor has no Lab 3 capture clock")
    stamp = as_torch(timestamp).to(device=omega.device).clone()
    if stamp.shape != omega.shape[:1] or not bool(torch.isfinite(stamp).all()):
        raise ValueError("timestamp must be finite with shape (N,)")
    if bool((stamp < 0).any()):
        raise ValueError("timestamp must be nonnegative episode seconds")
    acceleration = None
    if (gravity_w is None) != (orientation_w is None):
        raise ValueError("gravity removal requires both gravity_w and capture orientation_w")
    if gravity_w is not None:
        gravity = torch.as_tensor(gravity_w, dtype=force.dtype, device=force.device)
        if gravity.shape not in ((3,), force.shape) or not bool(torch.isfinite(gravity).all()):
            raise ValueError("gravity_w must be finite (3,) or (N,3)")
        q_w = quaternion(orientation_w, device=force.device)
        if q_w.shape not in ((4,), (len(force), 4)):
            raise ValueError("orientation_w must be (4,) or (N,4)")
        acceleration = force + quat_inverse_apply(q_w, gravity)
    if rotation_target_from_imu is not None:
        q = quaternion(rotation_target_from_imu, device=force.device)
        if q.shape not in ((4,), (len(force), 4)):
            raise ValueError("rotation_target_from_imu must be (4,) or (N,4)")
        if frame == "imu":
            raise ValueError("name the target frame when rotating IMU axes")
        omega, force = quat_apply(q, omega), quat_apply(q, force)
        if acceleration is not None:
            acceleration = quat_apply(q, acceleration)
    elif frame != "imu":
        raise ValueError("a named target frame requires rotation_target_from_imu")
    return ImuSample(omega, force, stamp, stamp > 0, acceleration, frame)


def read_sim_imu_frame(frame_data, *, read_gravity: bool, is_valid: bool,
                       gravity_w=None, device="cpu") -> ImuSample:
    """Convert one native Sim 6 experimental IMUSensor.get_data() dictionary.

    No simulator APIs are invoked. Input vectors are local-frame SI values and
    orientation is native Sim wxyz, converted to xyzw here. Caller must pass the
    exact read_gravity flag used for acquisition and raw reading validity: the
    documented get_data dict has NO validity field and can hold an old sample
    on invalid reads. Neither validity nor gravity mode can be guessed from it.

    Expects time, physics_step, linear_acceleration, angular_velocity, orientation.
    If gravity is supplied it is actual world gravity in m/s², not a sensor bias.
    When read_gravity=False, linear_acceleration is already gravity-free; do not
    add gravity again. orientation_w is simulation state, not an inertial-only
    attitude estimate. Supports the new API, not deprecated get_current_frame.
    """
    if type(read_gravity) is not bool or type(is_valid) is not bool:
        raise ValueError("read_gravity and is_valid must be explicit booleans")
    required = {"time", "physics_step", "linear_acceleration", "angular_velocity", "orientation"}
    if not required.issubset(frame_data):
        raise ValueError(f"native Sim 6 frame missing {sorted(required - frame_data.keys())}")

    def tensor(name, shape):
        value = torch.as_tensor(frame_data[name], device=device, dtype=torch.float32).clone()
        if value.shape != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite with shape {shape}")
        return value

    acceleration = tensor("linear_acceleration", (3,)).unsqueeze(0)
    omega = tensor("angular_velocity", (3,)).unsqueeze(0)
    q = quaternion(tensor("orientation", (4,))[[1, 2, 3, 0]]).unsqueeze(0)
    # Keep simulation time in float64 rather than downcasting a long-running clock.
    stamp = torch.as_tensor(frame_data["time"], device=device, dtype=torch.float64).clone()
    step = tensor("physics_step", ())
    if stamp.ndim or not bool(torch.isfinite(stamp)) or stamp < 0 or step < 0 or step != step.floor():
        raise ValueError("native time/physics_step must be nonnegative simulation clocks")
    stamp = stamp.reshape(1)
    gravity = None
    if gravity_w is not None:
        gravity = torch.as_tensor(gravity_w, device=device, dtype=torch.float32)
        if gravity.shape != (3,) or not bool(torch.isfinite(gravity).all()):
            raise ValueError("gravity_w must be a finite xyz vector in m/s²")
        gravity = quat_inverse_apply(q, gravity)
    if read_gravity:
        force = acceleration
        linear = None if gravity is None else acceleration + gravity
    else:
        linear = acceleration
        force = None if gravity is None else acceleration - gravity
    return ImuSample(omega, force, stamp, torch.tensor([is_valid], device=device),
                     linear, orientation_w=q)
