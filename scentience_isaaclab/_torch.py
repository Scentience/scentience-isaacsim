"""Small, Isaac-independent tensor boundary helpers. Quaternions are xyzw."""

from __future__ import annotations

import importlib

import torch


def as_torch(value) -> torch.Tensor:
    """Accept torch, Lab 3 ProxyArray, or Warp without copying tensor storage."""
    if isinstance(value, torch.Tensor):
        return value
    proxy = getattr(value, "torch", None)
    if proxy is not None:
        return proxy() if callable(proxy) else proxy
    import warp as wp

    return wp.to_torch(value)


def indices(n, device, env_ids=None, env_mask=None):
    """Lab 3 masks take priority; reject ambiguous/invalid index inputs."""
    if env_mask is not None:
        mask = as_torch(env_mask)
        if mask.dtype != torch.bool or mask.shape != (n,):
            raise ValueError(f"env_mask must be bool with shape ({n},)")
        return mask.to(device).nonzero(as_tuple=True)[0]
    if env_ids is None:
        return torch.arange(n, device=device)
    if isinstance(env_ids, slice):
        return torch.arange(n, device=device)[env_ids]
    ids = torch.as_tensor(env_ids, device=device)
    if ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("env_ids must be a one-dimensional integer sequence")
    if bool(((ids < 0) | (ids >= n)).any()) or ids.unique().numel() != ids.numel():
        raise ValueError("env_ids must be unique and in range")
    return ids.long()


def quaternion(value, *, device=None):
    q = torch.as_tensor(value, dtype=torch.float32, device=device)
    if q.shape[-1:] != (4,) or not bool(torch.isfinite(q).all()):
        raise ValueError("rotation must be a finite xyzw quaternion")
    norm = q.norm(dim=-1, keepdim=True)
    if bool((norm < 1e-8).any()):
        raise ValueError("rotation quaternion cannot be zero")
    return q / norm


def quat_apply(q, v):
    xyz, v = torch.broadcast_tensors(q[..., :3], v)
    uv = 2 * torch.linalg.cross(xyz, v)
    return v + q[..., 3:] * uv + torch.linalg.cross(xyz, uv)


def quat_inverse_apply(q, v):
    return quat_apply(torch.cat((-q[..., :3], q[..., 3:]), dim=-1), v)


def quat_mul(a, b):
    av, bv = torch.broadcast_tensors(a[..., :3], b[..., :3])
    return torch.cat((a[..., 3:] * bv + b[..., 3:] * av + torch.linalg.cross(av, bv),
                      a[..., 3:] * b[..., 3:] - (av * bv).sum(-1, keepdim=True)), -1)


def load_callable(target):
    if callable(target):
        return target
    if not isinstance(target, str) or ":" not in target:
        raise ValueError("factory must be callable or 'package.module:function'")
    module, name = target.split(":", 1)
    obj = getattr(importlib.import_module(module), name)
    if not callable(obj):
        raise TypeError(f"{target!r} is not callable")
    return obj
