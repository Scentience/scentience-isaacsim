"""Execute scentience_isaaclab against GENUINE isaaclab 2.3.x code.

Usage (in a venv with `pip install --no-deps isaaclab==2.3.2` plus these
pure-Python deps -- each one was discovered by running this harness and
following the ModuleNotFoundErrors: torch, numpy<2, usd-core, toml,
packaging, warp-lang, flatdict, prettytable, scipy, trimesh, pyyaml, h5py,
matplotlib, opencv-python-headless, hidapi, gymnasium):
    python scripts/check_isaaclab_binding.py <repo-root>

Tier between 'static source inspection' and 'live Isaac run': the Omniverse kit
runtime (carb / omni.kit / omni.timeline / isaacsim.core.simulation_manager)
is stubbed with inert placeholders, but everything from the isaaclab wheel --
SensorBase, SensorBaseCfg, configclass machinery, math utils -- is the real
shipped code, executed for real.

What this proves, when green:
  - `import isaaclab`, `isaaclab.sensors`, `isaaclab.utils` execute
  - our OlfactorySensor class DEFINES against the real SensorBase
    (bad overrides / metaclass clashes fail at class-creation time)
  - the real @configclass machinery processes OlfactorySensorCfg
  - cfg constructs; cfg.class_type binds to OlfactorySensor
    (validate_install.py check 4, on real isaaclab instead of a live install)
  - the 2.x env_ids signature our subclass implements matches the base
What it deliberately does NOT prove: PhysX views, prim binding, timeline
callbacks, rendering -- anything needing the kit runtime. Those stay
'unvalidated in a live install'.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab_kit_stubs import add_isaaclab_paths, install_stubs  # noqa: E402

add_isaaclab_paths()
install_stubs()

checks = []


def check(name, fn):
    try:
        detail = fn()
        checks.append((name, True))
        print("  [OK ] %s%s" % (name, (" -- " + detail) if detail else ""))
    except Exception as e:
        checks.append((name, False))
        print("  [FAIL] %s\n         %s: %s" % (name, type(e).__name__, e))


def c1():
    import isaaclab
    return "isaaclab %s (real wheel, kit runtime stubbed)" % isaaclab.__version__


def c2():
    import inspect
    from isaaclab.sensors import SensorBase
    p = list(inspect.signature(SensorBase._update_buffers_impl).parameters)[1]
    assert p == "env_ids", "3.0 env_mask API detected: %r" % p
    return "SensorBase._update_buffers_impl(%s) -- 2.x API" % p


def c3():
    # class DEFINITION executes against the real SensorBase
    from scentience_isaaclab.olfactory_sensor import OlfactorySensor, OlfactorySensorCfg
    from isaaclab.sensors import SensorBase, SensorBaseCfg
    assert issubclass(OlfactorySensor, SensorBase)
    assert issubclass(OlfactorySensorCfg, SensorBaseCfg)
    return "OlfactorySensor subclasses real SensorBase; Cfg subclasses real SensorBaseCfg"


def c4():
    # validate_install.py check 4, verbatim
    from scentience_isaaclab.olfactory_sensor import OlfactorySensor, OlfactorySensorCfg
    cfg = OlfactorySensorCfg(prim_path="/World/envs/env_.*/Robot/base")
    assert cfg.class_type is OlfactorySensor, "class_type not bound"
    return "%d channels, profile=%s" % (len(cfg.channel_names), cfg.sensor_profile)


def c5():
    # our override signature matches the base's 2.x contract
    import inspect
    from scentience_isaaclab.olfactory_sensor import OlfactorySensor
    p = list(inspect.signature(OlfactorySensor._update_buffers_impl).parameters)[1]
    assert p == "env_ids", p
    return "override takes env_ids"


def c6():
    # real configclass round-trip: to_dict/replace machinery
    from scentience_isaaclab.olfactory_sensor import OlfactorySensorCfg
    cfg = OlfactorySensorCfg(prim_path="/World/x")
    d = cfg.to_dict()
    assert "update_period" in d and "prim_path" in d
    cfg2 = cfg.replace(update_period=0.5)
    assert cfg2.update_period == 0.5 and cfg.prim_path == cfg2.prim_path
    return "real @configclass to_dict/replace work on our Cfg"


def c7():
    from isaaclab.utils import math as math_utils
    import torch
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # identity, wxyz
    v = torch.tensor([[1.0, 2.0, 3.0]])
    out = math_utils.quat_apply(q, v)
    assert torch.allclose(out, v), out
    return "real quat_apply(identity) is identity (wxyz confirmed)"




def c8():
    import scentience_isaaclab.mdp as mdp
    assert callable(mdp.gas_channels) and callable(mdp.wind_body)
    return "mdp observation terms import under real isaaclab"


def c9():
    # The DirectRLEnv task cfg must construct under the real configclass
    # machinery (this is where missing/incompatible fields explode).
    from scentience_isaaclab.tasks.plume_nav.plume_nav_env_cfg import PlumeNavEnvCfg
    cfg = PlumeNavEnvCfg()
    d = cfg.to_dict()
    assert "sim" in d and "scene" in d, sorted(d)[:8]
    return "PlumeNavEnvCfg constructs; %d top-level fields" % len(d)


def c10():
    # Importing tasks must register the gym id, and its entry points must
    # resolve to real modules (a typo here fails only at gym.make time).
    import importlib
    import gymnasium as gym
    import scentience_isaaclab.tasks  # noqa: F401  (registration side effect)
    spec = gym.spec("Isaac-PlumeNav-Scentience-v0")
    for target in (spec.entry_point, spec.kwargs["env_cfg_entry_point"]):
        mod, _, attr = target.partition(":")
        obj = getattr(importlib.import_module(mod), attr.split(".")[0])
        assert obj is not None
    return "gym id registered; entry points resolve"


check("import isaaclab (genuine 2.3.2)", c1)
check("SensorBase API is 2.x (validate_install check 2)", c2)
check("our classes define against real SensorBase", c3)
check("cfg constructs, class_type binds (validate_install check 4)", c4)
check("our override signature matches base", c5)
check("real configclass machinery round-trips our Cfg", c6)
check("real math_utils.quat_apply sanity", c7)
check("mdp observation terms import", c8)
check("DirectRLEnv task cfg constructs (real configclass)", c9)
check("gym task id registers and entry points resolve", c10)

fails = sum(1 for _, ok in checks if not ok)
print("\n%d/%d passed" % (len(checks) - fails, len(checks)))
sys.exit(1 if fails else 0)
