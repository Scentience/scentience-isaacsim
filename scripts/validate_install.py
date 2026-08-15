"""
Run this INSIDE Isaac Sim to validate the integration.

    ./python.sh /path/to/scripts/validate_install.py

Until this passes, docs/ISAAC_COMPATIBILITY.md must continue to say the Isaac
integration is unvalidated. Paste the output into that file with a date.
"""
import sys

CHECKS = []
def check(name):
    def deco(fn):
        CHECKS.append((name, fn)); return fn
    return deco


@check("isaacsim + isaaclab import, versions reported")
def _c1():
    import isaacsim, isaaclab
    return f"isaacsim={getattr(isaacsim,'__version__','?')} isaaclab={getattr(isaaclab,'__version__','?')}"


@check("SensorBase API shape matches what we subclass (2.3.x vs 3.0)")
def _c2():
    import inspect
    from isaaclab.sensors import SensorBase
    sig = inspect.signature(SensorBase._update_buffers_impl)
    p = list(sig.parameters)[1]
    if p == "env_ids":
        return "Isaac Lab 2.x API (env_ids) -- matches this code"
    raise AssertionError(
        f"_update_buffers_impl takes {p!r}, not 'env_ids'. This is the 3.0 "
        "env_mask API; port olfactory_sensor.py before proceeding.")


@check("warp available and reports a CUDA device")
def _c3():
    import warp as wp
    wp.init()
    n = wp.get_cuda_device_count()
    if n == 0:
        raise AssertionError("no CUDA device; the GPU transport path will not run")
    return f"{n} CUDA device(s)"


@check("our sensor cfg constructs and binds class_type")
def _c4():
    from scentience_isaaclab.olfactory_sensor import OlfactorySensorCfg, OlfactorySensor
    cfg = OlfactorySensorCfg(prim_path="/World/envs/env_.*/Robot/base")
    assert cfg.class_type is OlfactorySensor, "class_type not bound"
    return f"{len(cfg.channel_names)} channels, profile={cfg.sensor_profile}"


@check("warp/numpy physics parity")
def _c5():
    import subprocess, pathlib
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).parent.parent /
                        "tests" / "test_warp_parity.py")], capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(r.stdout[-800:] + r.stderr[-800:])
    return "parity + OU dt-invariance pass"


if __name__ == "__main__":
    fails = 0
    for name, fn in CHECKS:
        try:
            print(f"[ OK ] {name}\n       {fn()}")
        except Exception as e:
            fails += 1
            print(f"[FAIL] {name}\n       {type(e).__name__}: {e}")
    print(f"\n{len(CHECKS)-fails}/{len(CHECKS)} checks passed")
    if fails:
        print("Isaac integration is NOT validated. Do not update ISAAC_COMPATIBILITY.md.")
    sys.exit(1 if fails else 0)
