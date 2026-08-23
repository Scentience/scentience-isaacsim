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
import types

REPO = sys.argv[1]
sys.path.insert(0, REPO)

# The PyPI `isaaclab` wheel is a launcher shell whose outer __init__ bootstraps
# the Isaac Sim kernel on import. The actual library is vendored at
# <site-packages>/isaaclab/source/isaaclab/. Put that inner root FIRST on
# sys.path so `import isaaclab` resolves to the genuine library package and the
# launcher shell is never executed.
site = [p for p in sys.path if p.endswith("site-packages")][0]
source_root = os.path.join(site, "isaaclab", "source")
inner = os.path.join(source_root, "isaaclab")
assert os.path.isdir(os.path.join(inner, "isaaclab", "sensors")), inner
# The wheel vendors sibling packages (isaaclab_contrib, isaaclab_assets, ...)
# next to the core; isaaclab.scene imports isaaclab_contrib at module scope,
# so every source/<pkg> root has to be importable.
for d in sorted(os.listdir(source_root)):
    p = os.path.join(source_root, d)
    if os.path.isdir(p):
        sys.path.insert(0, p)


class _StubModule(types.ModuleType):
    """Explicit attrs win; anything else auto-mints a permissive MagicMock."""

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        from unittest.mock import MagicMock
        v = MagicMock(name=self.__name__ + "." + k)
        v.__name__ = k
        setattr(self, k, v)
        return v


def stub(name, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        mod_name = ".".join(parts[:i])
        if mod_name not in sys.modules:
            m = _StubModule(mod_name)
            m.__path__ = []  # mark as package so submodule imports resolve
            sys.modules[mod_name] = m
            if i > 1:
                setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], m)
    for k, v in attrs.items():
        setattr(sys.modules[name], k, v)
    return sys.modules[name]


# ---- kit runtime stubs (inert; nothing here is isaaclab code) ----
class _Settings:
    def get(self, *a, **k):
        return None

    def set(self, *a, **k):
        pass

    get_as_bool = get_as_int = get_as_float = get_as_string = get


stub("carb",
     log_info=lambda *a, **k: None,
     log_warn=lambda *a, **k: None,
     log_error=lambda *a, **k: None,
     settings=types.SimpleNamespace(get_settings=lambda: _Settings(),
                                    ISettings=_Settings))


class _Sub:
    def unsubscribe(self):
        pass


class _EventStream:
    def create_subscription_to_pop(self, *a, **k):
        return _Sub()

    def create_subscription_to_pop_by_type(self, *a, **k):
        return _Sub()


class _App:
    def get_message_bus_event_stream(self):
        return _EventStream()

    def get_update_event_stream(self):
        return _EventStream()

    def print_and_log(self, *a):
        pass


stub("omni")
stub("omni.kit")
stub("omni.kit.app", get_app=lambda: _App(), get_app_interface=lambda: _App())


class _Timeline:
    class TimelineEventType:
        PLAY = 0
        STOP = 1
        PAUSE = 2

    def get_timeline_event_stream(self):
        return _EventStream()

    def is_playing(self):
        return False

    def is_stopped(self):
        return True


stub("omni.timeline",
     get_timeline_interface=lambda: _Timeline(),
     TimelineEventType=_Timeline.TimelineEventType)
stub("omni.log", info=lambda *a, **k: None, warn=lambda *a, **k: None,
     error=lambda *a, **k: None)
stub("omni.usd", get_context=lambda: None)
stub("omni.client",
     Result=types.SimpleNamespace(OK=0),
     stat=lambda *a, **k: (0, None),
     read_file=lambda *a, **k: (0, None, b""))
stub("omni.kit.commands", execute=lambda *a, **k: (False, None))
stub("omni.physx", get_physx_interface=lambda: None,
     get_physx_simulation_interface=lambda: None)
stub("isaacsim.core.utils")
stub("isaacsim.core.utils.stage", get_current_stage=lambda: None)


class SimulationManager:
    @staticmethod
    def get_physics_sim_view():
        return None

    @staticmethod
    def register_callback(*a, **k):
        return 0

    @staticmethod
    def deregister_callback(*a, **k):
        pass


stub("isaacsim", SimulationApp=type("SimulationApp",(object,),{}))
stub("isaacsim.core")
stub("isaacsim.core.simulation_manager", SimulationManager=SimulationManager)
# isaacsim.core.version: isaaclab/__init__ may print the Isaac Sim version
stub("isaacsim.core.version", get_version=lambda: ("5.1.0",) * 8)


# ---- auto-stub: any OTHER kit-runtime module resolves to a permissive mock.
# Applies ONLY to the kit namespaces (carb/omni/isaacsim/pxr fallbacks); the
# isaaclab wheel itself always loads its real code.
import importlib.abc
import importlib.machinery
from unittest.mock import MagicMock

KIT_ROOTS = ("carb", "omni", "isaacsim")


class _KitAutoStub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in KIT_ROOTS and name not in sys.modules:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = MagicMock(name="kitstub:" + spec.name)
        m.__spec__ = spec
        m.__name__ = spec.name
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _KitAutoStub())

# pxr is REAL (usd-core wheel), but Isaac ships extra schema modules usd-core
# lacks (PhysxSchema, Semantics). Graft namespaces that mint REAL classes on
# attribute access -- isaaclab's configclass machinery reads `.__name__` on
# these, which a MagicMock does not provide.
import pxr


class _MintMeta(type):
    """Metaclass: attribute access on a minted class mints a nested class, so
    chains like PhysxSchema.Tokens.boundingCube resolve at import time. Every
    minted object is a real class -- it has __name__, is callable, and behaves
    under `in`/`issubclass` checks the way configclass machinery expects."""

    def __getattr__(cls, k):
        if k.startswith("__"):
            raise AttributeError(k)
        sub = _MintMeta(k, (), {"__module__": cls.__module__})
        setattr(cls, k, sub)
        return sub


class _SchemaNS(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        cls = _MintMeta(k, (), {"__module__": self.__name__})
        setattr(self, k, cls)
        return cls


for _schema in ("PhysxSchema", "Semantics", "UsdGeom", "UsdPhysics", "Usd",
                "UsdShade", "UsdUtils", "Sdf", "Gf", "Vt", "Tf"):
    if not hasattr(pxr, _schema):
        ns = _SchemaNS("pxr." + _schema)
        setattr(pxr, _schema, ns)
        sys.modules["pxr." + _schema] = ns


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
