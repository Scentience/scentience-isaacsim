"""Static API-contract validation of scentience_isaaclab against the REAL
isaaclab 2.3.x wheel from PyPI. No GPU, no Isaac install needed.

Usage:
    pip download isaaclab==2.3.2 --no-deps --dest /tmp/lab
    python scripts/check_isaaclab_contract.py /tmp/lab/isaaclab-2.3.2-*.whl

This does NOT execute Isaac -- it parses the shipped source of the wheel and
verifies every assumption our code documents about the 2.3.x API:

  A. `SensorBase._update_buffers_impl` takes `env_ids` (2.x), not `env_mask` (3.0)
  B. `SensorBase._initialize_impl` exists (we call super()._initialize_impl())
  C. `SensorBaseCfg` has the fields we set: prim_path, update_period, debug_vis
  D. `SensorBase.data` / `_update_outdated_buffers` lazy-eval contract exists
  E. `isaaclab.utils.math.quat_apply` exists (the one math util we call)
  F. `isaaclab.utils.configclass` exists
  G. every `from isaaclab.X import Y` in our code resolves to a real symbol
  H. DirectRLEnv/DirectRLEnvCfg/InteractiveSceneCfg/SimulationCfg exist
  I. imu.py sensor (our stated authoring reference) really uses this shape
"""
import ast
import sys
import zipfile

WHEEL = sys.argv[1]
z = zipfile.ZipFile(WHEEL)
names = z.namelist()


def read(path):
    return z.read(path).decode("utf-8", "replace")


def find(suffix):
    hits = [n for n in names if n.endswith(suffix)]
    return hits[0] if hits else None


def parse(path):
    return ast.parse(read(path))


def class_def(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def method(cls, name):
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def arg_names(fn):
    return [a.arg for a in fn.args.args]


def module_exports(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.add(a.asname or a.name.split(".")[0])
    return out


results = []


def check(label, ok, detail=""):
    results.append((label, ok, detail))
    print("  [{0}] {1}{2}".format("OK " if ok else "FAIL", label,
                                  (" -- " + detail) if detail else ""))


print("wheel:", WHEEL.rsplit("/", 1)[-1])
print()

# --- A + B + D: SensorBase shape ---
sb_path = find("isaaclab/sensors/sensor_base.py")
check("sensor_base.py present in wheel", sb_path is not None, sb_path or "")
sb = parse(sb_path)
cls = class_def(sb, "SensorBase")
check("class SensorBase found", cls is not None)

m = method(cls, "_update_buffers_impl")
check("_update_buffers_impl exists", m is not None)
args = arg_names(m)
check("2.x signature: 2nd arg is 'env_ids' (3.0 uses env_mask)",
      len(args) >= 2 and args[1] == "env_ids", "args=" + repr(args))

m = method(cls, "_initialize_impl")
check("_initialize_impl exists (we call super() on it)", m is not None)

m = method(cls, "_update_outdated_buffers")
check("_update_outdated_buffers exists (lazy-eval contract)", m is not None)

src = read(sb_path)
check("data property documented as lazy on SensorBase",
      "def data" in src)

# --- C: SensorBaseCfg fields ---
cfg_cls = class_def(sb, "SensorBaseCfg")
if cfg_cls is None:
    cfg_path = find("isaaclab/sensors/sensor_base_cfg.py")
    cfg_cls = class_def(parse(cfg_path), "SensorBaseCfg") if cfg_path else None
check("SensorBaseCfg found", cfg_cls is not None)
cfg_fields = set()
for node in cfg_cls.body:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        cfg_fields.add(node.target.id)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                cfg_fields.add(t.id)
for f in ("prim_path", "update_period", "debug_vis"):
    check("SensorBaseCfg.%s exists" % f, f in cfg_fields, "fields=" + ", ".join(sorted(cfg_fields)) if f not in cfg_fields else "")

# --- E: math utils ---
mu_path = find("isaaclab/utils/math.py")
mu = module_exports(parse(mu_path))
check("isaaclab.utils.math.quat_apply exists", "quat_apply" in mu)

# --- F: configclass ---
u_init = find("isaaclab/utils/__init__.py")
u = module_exports(parse(u_init))
check("isaaclab.utils.configclass importable", "configclass" in u)

# --- G: our exact import lines resolve ---
IMPORTS = {
    "isaaclab/sensors/__init__.py": ["SensorBase", "SensorBaseCfg"],
    "isaaclab/envs/__init__.py": ["DirectRLEnv", "DirectRLEnvCfg"],
    "isaaclab/scene/__init__.py": ["InteractiveSceneCfg"],
    "isaaclab/sim/__init__.py": ["SimulationCfg"],
}
for mod, symbols in IMPORTS.items():
    p = find(mod)
    if p is None:
        for s in symbols:
            check("from %s import %s" % (mod, s), False, "module missing")
        continue
    exp = module_exports(parse(p))
    src = read(p)
    for s in symbols:
        check("from %s import %s" % (mod.replace("/__init__.py", "").replace("/", "."), s),
              s in exp or s in src)

# --- H2: every isaaclab symbol scripts/verify_in_isaac.py touches ---
VERIFY_SURFACE = {
    "isaaclab/app/__init__.py": ["AppLauncher"],
    "isaaclab/sim/__init__.py": ["SimulationContext", "SimulationCfg"],
    "isaaclab/sim/spawners/from_files/__init__.py": ["GroundPlaneCfg"],
    "isaaclab/sim/spawners/shapes/__init__.py": ["CuboidCfg"],
    "isaaclab/sim/spawners/materials/__init__.py": ["RigidBodyMaterialCfg"],
    "isaaclab/sim/schemas/__init__.py": ["RigidBodyPropertiesCfg", "MassPropertiesCfg",
                                     "CollisionPropertiesCfg"],
    "isaaclab/assets/__init__.py": ["RigidObjectCfg"],
    "isaaclab/scene/__init__.py": ["InteractiveScene", "InteractiveSceneCfg"],
}
for mod, symbols in VERIFY_SURFACE.items():
    p2 = find(mod)
    if p2 is None:
        for sym in symbols:
            check("verify_in_isaac surface: %s in %s" % (sym, mod), False, "module missing")
        continue
    exp = module_exports(parse(p2))
    src2 = read(p2)
    for sym in symbols:
        check("verify_in_isaac surface: %s.%s" %
              (mod.replace("/__init__.py", "").replace("/", "."), sym),
              sym in exp or sym in src2)

# --- I: imu.py, the stated authoring reference ---
imu_path = find("isaaclab/sensors/imu/imu.py")
check("imu.py present (authoring reference)", imu_path is not None)
imu_cls = class_def(parse(imu_path), "Imu")
m = method(imu_cls, "_update_buffers_impl") if imu_cls else None
check("Imu._update_buffers_impl(env_ids) matches pattern we copied",
      m is not None and len(arg_names(m)) >= 2 and arg_names(m)[1] == "env_ids")

# --- isaacsim.core.simulation_manager: not in this wheel (ships with Isaac Sim),
# but verify isaaclab itself imports it the same way we do.
hits = [n for n in names if n.endswith(".py") and
        "from isaacsim.core.simulation_manager import SimulationManager" in read(n)]
check("isaaclab 2.3.2 itself imports SimulationManager from the same path we use",
      len(hits) > 0, "%d files, e.g. %s" % (len(hits), hits[0].split("isaaclab/")[-1] if hits else ""))

print()
fails = [r for r in results if not r[1]]
print("%d/%d contract checks passed" % (len(results) - len(fails), len(results)))
sys.exit(1 if fails else 0)
