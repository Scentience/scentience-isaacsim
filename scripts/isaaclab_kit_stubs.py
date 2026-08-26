"""Kit-runtime stubs: run GENUINE isaaclab 2.3.x code without Isaac Sim.

Shared by scripts/check_isaaclab_binding.py (validation harness) and
scripts/demo_isaaclab_sensor_local.py (runtime demo). Everything from the
isaaclab wheel is real, executed code; only the Omniverse kit runtime
(carb / omni.* / isaacsim.*) is replaced with inert placeholders, plus the
PhysX-only pxr schema modules usd-core lacks.

Call `add_isaaclab_paths()` then `install_stubs()` BEFORE importing isaaclab.
"""
import os
import sys
import types


def add_isaaclab_paths() -> str:
    """Make `import isaaclab` resolve to the wheel's vendored library.

    The PyPI `isaaclab` wheel is a launcher shell whose outer __init__
    bootstraps the Isaac Sim kernel on import; the actual library lives at
    <site-packages>/isaaclab/source/isaaclab/, with sibling packages
    (isaaclab_contrib, isaaclab_assets, ...) beside it that isaaclab.scene
    imports at module scope.
    """
    site = [p for p in sys.path if p.endswith("site-packages")][0]
    source_root = os.path.join(site, "isaaclab", "source")
    inner = os.path.join(source_root, "isaaclab")
    assert os.path.isdir(os.path.join(inner, "isaaclab", "sensors")), (
        "isaaclab wheel not found -- pip install --no-deps isaaclab==2.3.2 "
        "plus the deps listed in check_isaaclab_binding.py")
    for d in sorted(os.listdir(source_root)):
        p = os.path.join(source_root, d)
        if os.path.isdir(p):
            sys.path.insert(0, p)
    return inner


def install_stubs() -> None:
    """Install the kit-runtime stub modules into sys.modules."""
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
    class _UsdContext:
        def get_stage(self):
            return None

        def get_stage_event_stream(self):
            return _EventStream()

    stub("omni.usd", get_context=lambda: _UsdContext())
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
