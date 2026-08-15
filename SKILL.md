---
name: scentience-isaac-olfaction
description: Build, extend, test, and package the Scentience olfactory simulation stack for NVIDIA Isaac Sim and Isaac Lab — chemical emitters, GPU filament plume transport, virtual Scentience sensor models, plume-realism validation, RL environments, OmniGraph/ROS 2 interfaces, navigation benchmarks, and sim-to-real calibration. Use when implementing or modifying the Scentience Isaac olfaction extension or related chemical-perception simulation code. Do not use for unrelated Isaac Sim work.
---

# Scentience Isaac Olfaction

## Mission

Add chemical perception and virtual Scentience olfactory sensors to NVIDIA Isaac
Sim / Isaac Lab, at a fidelity sufficient that **a policy trained in it transfers
to Scentience hardware**.

That last clause is the whole specification. A simulator that produces a
plausible-looking plume and a smooth concentration reading is easy and worthless:
it trains gradient ascent, which fails on the first real whiff. Every design
decision below is downstream of transfer.

```
Isaac Sim stage  ──►  occupancy grid + wind field   (offline, once per scene)
                              │
Isaac Lab physics step ──►  filament plume, Warp/GPU  ──► C(x,t) per species
                              │
                       virtual Scentience sensor
              (power law · cross-sens · asym. lag · drift · ADC)
                              │
        ┌─────────────────────┼──────────────────┬──────────────┐
    Isaac Lab obs         Python API        OmniGraph/ROS 2   dataset log
        │
    RL policy (recurrent — a scalar without history is not enough)
```

---

## 0. Non-negotiables

These are the things that, if you get them wrong, make everything else moot.

1. **Ground truth and observation are different objects.** Ground truth is for
   labels, reward shaping, evaluation, and debugging. The policy sees only
   simulated sensor output. Never wire `C(x,t)` into an observation term.

2. **The plume must pass the realism gate (§5) before any policy is trained.**
   Not "should" — a plume that fails the gate produces a policy that cannot
   transfer, and you will not discover this until hardware. Run the gate in CI.

3. **The sensor time constant is the most important parameter in the system.**
   If `tau_fall` ≫ whiff duration, the sensor integrates rather than resolves,
   and the observable is a low-pass, asymmetrically distorted shadow of the
   field. Measured on this stack (§5.3): a packaged MOX at `tau_fall = 12 s`
   retains **19%** of ground-truth whiff events; a Dennler-class fast sensor at
   `tau_fall = 46 ms` retains **97%**. Same plume. Choose the profile that
   matches the hardware you will deploy on, and state which one you used.

4. **GPU-resident from the start.** Not NumPy-first-optimise-later. See §2.

5. **Isaac Lab `SensorBase` is the primary integration; the Kit extension is an
   optional wrapper.** See §6.

6. **Do not invent NVIDIA APIs, and do not invent Scentience calibration
   coefficients.** Shipped MOX/EC constants are illustrative and must be
   labelled as such until fitted against measured exposure data (§9).

---

## 1. Target versions — verified, do not re-derive

Isaac Sim is open source. `git clone --depth 1 --filter=blob:none --sparse
https://github.com/isaac-sim/IsaacSim.git` and read the reference
implementations directly rather than guessing from docs.

| | Recommended | Notes |
|---|---|---|
| Isaac Sim | **5.1** | 6.0.1 is current; 5.1 is the last with full OSS sensor sources |
| Isaac Lab | **2.3.x** | `main` carries 3.0.0 untagged |
| Warp | bundled `omni.warp.core` ≥1.13 | no separate install |

Breaking changes to route around:

- `isaacsim.sensors.physx` — **removed in 6.0**. Never build on it.
- `isaacsim.sensors.physics` — deprecated in 6.0 in favour of
  `isaacsim.sensors.experimental.physics`.
- Isaac Lab **2.x → 3.0** changes `SensorBase._update_buffers_impl(env_ids:
  Sequence[int])` to `(env_mask: wp.array)`, and `data.field` (torch) to
  `data.field.torch` (ProxyArray). Mechanical, but pin your version and say so.
- `IsaacSimExtensionTemplate` does not exist. The real ones are
  `isaac-sim/IsaacLabExtensionTemplate` (use this) and
  `isaac-sim/isaacsim-app-template`.

**Reference implementations to copy from, in priority order:**

1. `isaaclab/sensors/imu/imu.py` — the exact `SensorBase` subclass pattern,
   including the PhysX tensor-view pose read and the xyzw→wxyz `.roll(1, -1)`.
2. `isaacsim.sensors.physics/python/impl/effort_sensor.py` — the only
   first-party non-RTX sensor that is pure Python; shows physics-step
   subscription and rate decoupling.
3. `isaacsim.robot.schema/sensor_schema/SensorSchema.usda` — how NVIDIA
   authors a **codeless** USD sensor schema (`skipCodeGeneration = true`).

Record what you actually validated in `docs/ISAAC_COMPATIBILITY.md`. Never claim
an Isaac integration test passed if Isaac was unavailable.

---

## 2. Transport: Warp-first, filaments only

### 2.1 Why not NumPy-first

The obvious plan — write it in NumPy, port to GPU when profiling says to — is
wrong here specifically, for three reasons:

- Isaac Lab runs `N` environments on GPU. A NumPy transport forces a
  device→host→device round trip **every step, per env**. That cost does not
  show up in a single-env profile and does not go away with vectorisation; it
  is architectural.
- Warp ships inside Isaac Sim. There is no dependency to add, and
  `wp.from_torch` / `wp.to_torch` are zero-copy on CUDA, so sensor output lands
  in the policy's tensor without a copy.
- Warp kernels are Python and JIT-compiled. The iteration cost that normally
  justifies "prototype in NumPy" is not there.

Keep a NumPy reference implementation for tests and CI (must run without a GPU),
and assert the two agree. Do not make it the runtime path.

### 2.2 Fidelity tiers — three, not four

| Tier | Model | Use |
|---|---|---|
| 0 | Gaussian plume, closed form | smoke tests, curriculum stage 0, unit tests only |
| 1 | **Filament (Farrell)** — the default | everything real |
| 2 | Filament on a precomputed CFD wind field | fixed scenes, published results |

**There is no voxel/advection-diffusion tier.** Codex-style plans list a grid
solver above the puff model as "higher fidelity". It is not. Semi-Lagrangian
advection interpolates at the back-traced point, and that interpolation
numerically diffuses away exactly the filamentous structure that olfactory
navigation depends on. On a 10 cm grid you get a Gaussian plume with extra
steps and 100× the cost. If you want a grid, use it for the **wind field**
(steady, smooth, genuinely grid-friendly), never for the concentration field.

### 2.3 The filament model

Farrell et al. 2002, *Env. Fluid Mech.* 2:143–169. Implement from the published
equations. **Do not transcribe GADEN source — it is LGPL-3.0**, and a Python
port of it is a derivative work. Cross-check numeric behaviour against GADEN;
that is fine.

State per filament is exactly: position, sigma, turbulent velocity, age.

**Growth** (Farrell "Model 2"):

```
sigma(t) = sqrt(sigma_0^2 + gamma * t)     <=>     sigma^2 += gamma*dt
```

**Concentration**, mass-conserving Gaussian, superposed with a 3-sigma cutoff:

```
C_i(x) = N_fil / (n_air * (2 pi)^{3/2} sigma_i^3) * exp(-d_i^2 / (2 sigma_i^2))
C(x)   = sum_i C_i(x)   for   d_i < 3 sigma_i   and   line-of-sight(x, p_i)
```

`N_fil` is fixed at birth so mass is conserved as sigma grows. The
line-of-sight test against the occupancy grid is the cheap stand-in for "gas
does not diffuse through walls" — without it, filaments leak through geometry.

**Turbulence — TWO scales, both required.**

*Small scale*, per filament, Ornstein-Uhlenbeck, **exact** discrete update:

```
a = exp(-dt / T_L)
u' <- a*u' + sigma_u * sqrt(1 - a^2) * xi        xi ~ N(0, I)
```

Use this form, not Euler-Maruyama: it is unconditionally stable for any `dt`,
and it is correct. A memoryless Gaussian kick scaled by `dt` (rather than
`sqrt(dt)`) makes effective turbulent diffusivity proportional to your
timestep — halve `dt`, the plume narrows by `sqrt(2)`. That is a bug shape, and
it is present in at least one widely-used simulator. Seed `u'` from the
stationary distribution at release; starting at zero creates a spurious laminar
segment near the source.

*Large scale*, **one** OU process on wind bearing shared by every filament,
`T_meander ≈ 10–20 s` ≫ `T_L`:

```
theta <- a_m*theta + sigma_theta * sqrt(1 - a_m^2) * xi     (advance once per step)
```

This is Farrell's "large meander" term and it is **not optional**. Measured
ablation on this stack (§5.3): with meander, blank-duration CV = **2.31**
(heavy-tailed, passes); without it, CV = **0.96** — sub-exponential blanks, a
plume with no long absences, an environment materially easier than reality.
Per-filament turbulence alone does not produce it, because independent kicks
average out across the filament population.

**Implementation notes.** Fixed-capacity pool with an `alive` mask — never
dynamic allocation; the ping-pong compaction idiom is CPU-shaped. Cull by age
and domain exit. For sensor queries with a handful of robots and a few thousand
filaments, brute-force reduction per sensor beats building an acceleration
structure; reach for `wp.HashGrid` only when you need dense fields for
visualisation. Put `n_envs` in the outer kernel dimension; envs can share one
wind field and occupancy grid with per-env source position as the
domain-randomisation axis. Wrap the per-step kernel sequence in
`wp.ScopedCapture()` — launch overhead dominates when several small kernels run
per RL step.

### 2.4 Wind fields

Ranked by effort. Do not start at the bottom.

1. **Uniform mean + OU turbulence + bearing meander.** Costs nothing, and with
   §2.3 it already passes the realism gate. Start here.
2. **Potential flow** — solve `∇²φ = 0` on the occupancy grid, `u = ∇φ`. Smooth,
   divergence-free, goes *around* furniture. No wakes or recirculation. Good
   fallback for procedurally generated scenes.
3. **Steady RANS from OpenFOAM** (`simpleFoam`, k-ε), one run per scene, minutes.
   Store as `(Nx,Ny,Nz,3)` float16 or a NanoVDB volume.
4. **Recommended hybrid:** RANS mean field **+** per-filament OU perturbation
   scaled by the local turbulent kinetic energy from the same solve,
   `sigma_u = sqrt(2k/3)`. Obstacle-correct mean flow with physically-scaled,
   spatially-varying turbulence, from a *steady* solve.

Sample with **trilinear** interpolation (`wp.volume_sample(..., wp.Volume.LINEAR)`
on a NanoVDB grid). Nearest-cell lookup produces grid-aligned artefacts in
filament trajectories. If you use a time sequence of wind snapshots, interpolate
between them; stepping discontinuously injects a velocity jump every snapshot
interval that shows up directly in the statistics.

### 2.5 What about PhysX Flow?

Flow is a real GPU advecting-scalar solver and it is the wrong tool here.
Readback is via the `FlowNanoVdbReadback` OmniGraph node, documented as having
*"some latency due to async readback"*; it is tied to the **render** loop, not
physics, which destroys rate decoupling; and there is no documented path from
its `uint[]` NanoVDB buffers into a `wp.Volume`. Use it offline to *generate*
`.nvdb` frames if you like. Do not put it in the RL loop. PhysX particle/SPH
fluids model liquids, not dilute gas advection, and are far too expensive for
vectorised envs.

---

## 3. Occupancy grid from the Isaac stage

The plume needs to know where the walls are. Rasterise once, at scene load:

1. Collect collision meshes from the USD stage.
2. Triangle→voxel via a separating-axis triangle-box test → `OBSTACLE`.
3. Flood-fill from a known-empty point → `FREE`. Everything unreached stays
   non-free. (This is why a seed point is required; do not try to infer it.)
4. Domain boundary cells → `OUTLET` (filaments entering are culled).
5. Cache to disk keyed by a hash of the stage, so it is not recomputed per run.

Cell states: `FREE, OBSTACLE, OUTLET, OUT_OF_BOUNDS`. Filament–obstacle response
is **slide**, not stop or reflect: on hit, revert, compute the axis-aligned
pseudo-normal from the cell transition, reject the remaining displacement off
it, recurse. Cap sub-steps (≤4) inside the kernel and clamp `dt` so a filament
never moves more than ~2 cells — unbounded sub-stepping causes warp divergence.

---

## 4. Sensor model

### 4.1 Chain

```
C_g(t) [ppm]
  → steady-state response, superposed in RESISTANCE space
  → temperature / absolute-humidity modulation (in log space)
  → heater-state dependence
  → asymmetric first-order lag, flow- and heater-corrected
  → transport delay (inlet + housing dead volume)
  → baseline random walk + 1/f + white noise
  → voltage divider
  → ADC quantisation
  → counts
```

### 4.2 The parts that are usually wrong

**Superposition is not linear in concentration.** A `W @ concentration_vector`
cross-sensitivity matrix is correct for electrochemical cells and wrong for MOX.
MOX is a power law; contributions superpose as resistance decrements:

```
Rs/R0 = max( S_air - sum_g [ S_air - min(A_g * C_g^(-beta_g), S_air) ] , eps )
```

with `beta > 0` for reducing gases and `beta < 0` for oxidizing (NO₂ raises
`Rs`). Keep the linear-`W` form for the EC and PID channels, where it is right.

**Quantise the voltage, not the concentration.** MOX is read through a divider:

```
V = Vcc * R_L / (Rs + R_L)      counts = floor(V / q),  q = Vref / 2^N
```

so `∂Rs/∂V ∝ (Rs + R_L)²` — resolution in `Rs` collapses at high `Rs`. A RED die
at 1.5 MΩ in clean air on a 12-bit ADC is nearly unresolvable. Quantising
concentration hides this entirely and is a reliable way to build a sim whose
data does not transfer.

**Humidity acts on absolute humidity, and on sensitivity, not just baseline:**

```
ln Rs = ln A - beta*ln C + Ea/(kT) + gamma*AH + delta*AH*ln C
AH [g/m^3] = 216.7 * (RH/100) * 6.112*exp(17.62T/(243.12+T)) / (273.15+T)
```

The `delta` cross-term matters — humidity changes the slope, and in the field
humidity covaries with the plume (evaporative sources are humid), so it is a
confound *correlated with the label*.

**Time constants are not intrinsic.** The 3–16 s values in the robotics
literature are package + chamber constants. A fast heater with low dead volume
reaches ~90 ms (Dennler et al. 2024, *Sci. Adv.* 10:eadp1764 — 150→400 °C square
wave at 20 Hz, 1 kHz / 24-bit readout, onset 87 ± 20 ms, recovery 106 ± 24 ms).
`tau` also shortens with forced convection over the die, so a robot at 1 m/s and
the same part on a bench are not the same sensor. Model both:

```
tau_eff = tau_0 * (flow / flow_ref)^(-p) / heater_level
alpha   = 1 - exp(-dt / tau_eff)         # exact; stable for any dt
y      += alpha * (target - y)
tau     = tau_rise if target < y else tau_fall     # for MOX, gas arriving LOWERS Rs
```

Ship at least two profiles — `packaged_slow` and `fast_modulated` — and make the
choice explicit in every result.

### 4.3 Domain randomisation is a first-class feature, not a nice-to-have

MiCS-6814 `R0` spans **100 kΩ – 1.5 MΩ** unit-to-unit (datasheet, RED die) —
a 15× spread — and drifts 25–40 % CV over a year against 0.4–1.2 % short-term
repeatability. Sampling a single nominal `R0` produces virtual units far more
consistent than any two real ones. Randomise **per episode**, log-uniform over
the datasheet range:

`R0, A_g, beta_g, tau_rise, tau_fall, drift sigma, humidity coeff, R_L, dead volume`

And feed the policy **drift-invariant features only** — ratios, derivatives,
EMA at several timescales, area-under-curve, baseline-tracked deflection. Never
absolute `Rs`. Firmware runs a slow-EMA baseline tracker; put it in the sim so
the policy sees what firmware will actually hand it.

### 4.4 Scentience V1 profile

Load composition from `configs/scentience_v1.yaml`; do not hard-code. Base stack
per the published Scentience work: 2× MiCS-6814 (RED/NH3/OX each), SCD-4x CO₂
plus T/RH context, 2× electrochemical.

Channel-specific notes:
- **SCD-4x is photoacoustic, not NDIR.** `tau63 = 60 s` → `t90 ≈ 138 s`. It
  cannot see a plume whiff; model it as environmental context, not an olfactory
  channel. Model its ASC too: a rolling 7-day minimum forced to 400 ppm will
  drag calibration down in a continuously-occupied space. That is a real
  behaviour worth exposing.
- **EC channels** are linear in C (`I = nFAD/δ · C`), with Cottrell
  `I(t) = nFA√D·C/√(πt)` for the chronoamperometric transient. `t90` 15–60 s.
  Cross-sensitivity is large and one-sided (CO-B4: H₂ < 50 % of reading) — use
  the mixing matrix.
- **MOX heater program** is a list of `{duration_s, level}`. Heater state changes
  *sensor response*, never room concentration. Selectivity coefficients may vary
  per heater phase; the per-cycle log-normalised, background-subtracted feature
  from Dennler et al. is inherently drift-robust and worth implementing as the
  default feature extractor for modulated mode.

### 4.5 Learned transfer model

Optional plug-in: `[C_g, T, RH, heater state, recent history] → channels`. Small
MLP/GRU/TCN. PyTorch stays an optional dependency. Persist and **validate at
load**: dataset hash, species order, channel order, normalisation, expected
sample rate, version. Fail loudly on mismatch rather than silently producing
garbage in a different unit.

---

## 5. Plume realism gate — run this in CI

This is the section codex-style plans omit entirely, and it is the difference
between a simulator you can trust and one you cannot.

### 5.1 Metrics

Log a virtual probe at a fixed point at ≥100 Hz for ≥10 minutes, then compute:

- **intermittency** — fraction of time above a detection threshold
- **whiff / blank duration** distributions: median, CV, log-log CCDF slope
- **peak-to-mean** ratio
- conditional mean given detection

Threshold every series at the **same absolute concentration**, or for sensor
output at **3σ of that sensor's own clean-air noise floor after baseline
tracking** — which is how a detection threshold is set on hardware.

### 5.2 Targets

| Metric | Target | Source |
|---|---|---|
| blank-duration CV | **> 1** (heavy-tailed; exponential is CV = 1) | Celani, Villermaux & Vergassola, PRX 4:041015 |
| whiff/blank CCDF slope | roughly −1 to −1.5 over 1–2 decades, exponential cutoff at the large-eddy time | ibid. |
| intermittency | 0.02–0.95; ~0.5 at ~2 m off-axis | Farrell et al. 2002 |
| peak-to-mean | ≳ 3, ~14 near source | Farrell et al. 2002 |

**A blank-duration CV below 1 is a failure, not a warning.** It means the plume
has no large-scale meander, and the environment is easier than reality in
precisely the dimension the policy is supposed to learn.

### 5.3 Measured reference values on this stack

600 s @ 100 Hz, probe 8 m downwind on axis, 1 m/s mean wind, `I = 0.30`,
`T_L = 1.5 s`, `sigma_theta = 0.22 rad`, `T_meander = 15 s`:

| Configuration | intermittency | peak/mean | whiffs | blank CV | gate |
|---|---|---|---|---|---|
| filament, small + large scale | 0.368 | 18.9 | 338 | **2.31** | PASS |
| filament, meander ablated | 0.543 | 16.5 | 449 | **0.96** | FAIL |
| Gaussian plume + slow meander | 1.000 | 3.1 | 1 | — | FAIL |
| → MOX `tau_fall` 12 s (packaged) | 0.616 | 5.8 | **65** | 2.10 | PASS |
| → MOX `tau_fall` 46 ms (fast) | 0.430 | 4.9 | **329** | 2.47 | PASS |

Read the last two rows carefully. Both pass the plume gate, because the plume is
the same. But the slow sensor retains **19 %** of ground-truth whiff events and
stretches median whiff duration from 0.56 s to 1.16 s; the fast sensor retains
**97 %**. The policy trained on each is solving a different POMDP. This is the
single most consequential number in the system and it belongs in every result
you publish.

One more finding worth keeping: an unscaled 0.1–2 ppm ethanol plume is *below*
the MiCS-6814's useful range (datasheet: 10–500 ppm ethanol), and 1/f drift
swamps it entirely — the "sensor" reads noise. Check that your source strength
puts the plume inside the part's dynamic range before concluding anything about
navigation performance.

### 5.4 CI wiring

`pytest` job, CPU-only, fixed seed, ~60 s of simulated time (shorter records
need a looser whiff-count floor). Assert the gate passes and that the summary
statistics match a stored reference within tolerance. Regressions in plume
physics are otherwise invisible until a training run mysteriously gets easier.

---

## 6. Isaac integration — Isaac Lab first

### 6.1 The primary path needs no Kit extension and no USD schema

Isaac Lab's own docs bless "custom sensors implemented in Python that do not
require creating any USD prim or schema". For RL, subclass `SensorBase`:

```python
@configclass
class OlfactorySensorCfg(SensorBaseCfg):
    class_type: type = OlfactorySensor
    offset: OffsetCfg = OffsetCfg()
    species: tuple[str, ...] = ("ethanol",)
    profile: str = "scentience_v1"
    sensor_profile: str = "packaged_slow"   # or "fast_modulated"

class OlfactorySensor(SensorBase):
    @property
    def data(self):
        self._update_outdated_buffers()      # REQUIRED — lazy eval contract
        return self._data

    def _initialize_impl(self):
        super()._initialize_impl()           # REQUIRED
        self._view = SimulationManager.get_physics_sim_view() \
            .create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
        # allocate plume state, sensor state, wp arrays

    def _update_buffers_impl(self, env_ids):
        pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], -1)
        quat_w = quat_w.roll(1, dims=-1)                       # xyzw -> wxyz
        p = pos_w + math_utils.quat_apply(quat_w, self._offset_pos_b[env_ids])
        wp.launch(sample_plume, dim=p.shape[0], inputs=[...])  # GPU, batched
        self._data.channels[env_ids] = self._sensor.step(...)
```

Registration is: subclass `SensorBaseCfg`, set `class_type`, add the cfg as a
field on `InteractiveSceneCfg`. There is no registry, decorator, or entry point.
`cfg.update_period` gives fixed-rate decoupling from physics for free, in
simulated seconds, vectorised across envs — do not hand-roll an accumulator.

Use `DirectRLEnv`, not the manager-based workflow: the plume is bespoke non-USD
state you want to step yourself in `_pre_physics_step`.

Package as an **Isaac Lab external extension** (`IsaacLabExtensionTemplate`) —
a pip-installable Python package, no `extension.toml`, no `IExt`.

### 6.2 The Kit extension is the second path, for GUI / ROS deployment

Only when you need scene authoring, the viewport, or a non-RL robot. Then:

- `omni.ext.IExt` with `on_startup`/`on_shutdown`; one chemical service per
  stage; **no transport equations in `extension.py`**.
- Drive it from `SimulationManager.register_callback(fn,
  event=SimulationEvent.PHYSICS_POST_STEP)` (6.0) or
  `IsaacEvents.POST_PHYSICS_STEP` (5.x). Hold sensors in a `weakref.WeakSet` —
  strong refs in a global callback registry are the classic crash-on-stage-close
  bug, and NVIDIA's own new sensor manager uses a WeakSet for exactly this.
- Never drive a sensor from a render/app update callback if you need rate
  guarantees.
- Stage representation: namespaced custom attributes
  (`scentience:sensor:profile`, `scentience:chemical:species`, …) are enough and
  persist in USD automatically. If you want typed and discoverable, use a
  **codeless applied API schema** (`skipCodeGeneration = true`,
  `apiSchemaType = "singleApply"`) — no C++ build, generated with `usdGenSchema`.
  Skip compiled schemas entirely.
- Separate rates with accumulators: physics 120 Hz / transport 20 Hz / sensor
  10 Hz / ROS 10 Hz / viz 5 Hz. Test nested and rotated prims, moving sensors,
  moving emitters, and that pause does not advance chemical time.

### 6.3 OmniGraph and ROS 2

One Python OG node, `Scentience Read Olfaction`. Model it on
`OgnIsaacReadEffortSensor.py`: per-instance state via `BaseResetNode`,
`release_instance` cleanup, `db.outputs.execOut =
og.ExecutionAttributeState.ENABLED`. Output a **stable array plus documented
channel ordering**, not dozens of fixed pins.

For ROS 2: **do not start with a custom message.** A custom `.msg` requires the
user to source a ROS 2 workspace containing it *before* launching Isaac Sim —
real friction, for no benefit at first. Ship `std_msgs/Float32MultiArray` (plus
a latched `channel_names` on a separate topic) and add
`scentience_olfaction_msgs/Olfaction` later, when someone asks. Ground-truth
publishing is a separate topic, off by default. Nothing ROS-specific may block
standalone Python use.

---

## 7. Repository layout

Core packages are ordinary Python with **no mandatory Isaac dependency**; Isaac
is an adapter layer.

```
scentience-isaac-olfaction/
├── scentience_olfaction/          # importable without Isaac
│   ├── chemistry/{species,mixture,registry}.py
│   ├── emitters/{base,point,surface,volume}.py
│   ├── transport/{base,gaussian,filament_np,filament_warp,occupancy}.py
│   ├── airflow/{base,uniform,potential,grid}.py
│   ├── sensors/{base,dynamics,noise,mox,ec,scd4x,pid,scentience_v1,learned}.py
│   ├── validation/{plume_stats,sensor_stats,references}.py   # the gate
│   ├── calibration/{dataset,fit,metrics}.py
│   └── logging/{recorder,schema}.py
├── scentience_isaaclab/           # SensorBase + DirectRLEnv + cfgs  (primary)
├── isaac_extension/               # Kit extension + OGN nodes       (secondary)
├── configs/                       # chemicals, device profiles, plume presets
├── examples/  tests/{unit,integration,regression}  docs/  scripts/
```

---

## 8. Reuse what already exists

Before writing anything, read and, where possible, subsume:

- **`scentience/scentience-plume-envs`** — existing Gymnasium envs (`env_a`
  autoregressive plume generation, `env_b` OIO navigation). Keep the Gymnasium
  API surface; swap the Gaussian + Dryden field for the filament model behind it
  so existing training scripts keep working.
- **`KordelFranceTech/ChasingGhosts`** — Expected SARSA(λ), MiCS-6814 + Cottrell
  EC sensor models, olfactory-inertial odometry with dual-timescale EMA bout
  detection. The sensor models port directly. Its reported failure mode — IMU
  payload bias from the added sensor mass, non-linear and battery-dependent —
  should become a modelled perturbation, not a footnote.
- **Scentience BLE/Sockets API schema** — 14 chemical channels
  (CO2, NH3, NO, NO2, CO, C2H5OH, H2, CH4, C3H8, C4H10, H2S, HCHO, SO2, VOC)
  plus env and battery fields. **Make the simulator emit this exact schema**, so
  a consumer cannot tell sim from hardware without looking at the transport.
  Note the published schema does not state units — fix that in the simulator and
  push it back to the docs.
- **GADEN VGR dataset** — ready-made dispersion simulations in 3-D models of
  real houses. A free scene bank; converting its grids to `.nvdb` is cheap.

---

## 9. Sim-to-real calibration

Dataset schema: `timestamp, species, known_concentration, mixture, T, RH,
heater_state, channels[], device_id, session_id`.

1. Split by **session and device**, never randomly — random splits leak temporal
   correlation and will report a transfer accuracy you do not have.
2. Fit static (A, beta per gas per channel) then temporal (`tau_rise`,
   `tau_fall`, dead volume).
3. Evaluate on **held-out devices**, not just held-out sessions. Cross-device is
   the number that predicts transfer.
4. Metrics per channel: RMSE/MAE, correlation, temporal lag error, rise/recovery
   curve error.
5. Replay held-out exposure sequences through the simulator and compare.
6. Report the **sim2real gap explicitly** — no established benchmark exists for
   machine olfaction, which is an argument for publishing one, not for skipping it.

Until fitted, every coefficient is labelled `ILLUSTRATIVE`. Shipped MiCS-6814
`(A, beta)` values are algebraic inversions of open-source driver constants that
were themselves digitised from datasheet log-log graphs — the datasheet
publishes no tabulated coefficients. Say so in the docstring, not just the docs.

---

## 10. Phases — vertical slice first

Codex-style plans put Isaac at phase 3 and RL at phase 6+. Invert that: the
product is *a policy that transfers*, so get one end-to-end path working, then
deepen.

**Phase 0 — vertical slice (the only phase that matters at first).**
NumPy filament plume + MOX channel + realism gate + one probe script. No Isaac.
*Exit:* the gate passes, and the ablation table in §5.3 reproduces.

**Phase 1 — GPU + Isaac Lab.** Warp filament kernel (asserted equal to the NumPy
reference), `SensorBase` subclass, `DirectRLEnv`, obs = baseline-tracked
deflection + derivative + multi-timescale EMA + local wind + robot state.
*Exit:* a recurrent policy trains on N parallel envs and beats a
chemotaxis+casting baseline.

**Phase 2 — geometry.** Occupancy grid from the USD stage, line-of-sight, slide
response, potential-flow or RANS wind.
*Exit:* plume respects walls; source localisation works in a real Isaac scene.

**Phase 3 — device fidelity.** Full Scentience V1 profile, all 14 channels,
heater modulation, domain randomisation, BLE-schema-compatible output.
*Exit:* sim output is schema-identical to hardware.

**Phase 4 — interfaces.** Kit extension, OG node, ROS 2, visualisation
(2-D slice, capped filament rendering, source/sensor markers — throttled
separately, never mandatory).

**Phase 5 — benchmarks and calibration.** Episode recorder (Parquet, explicit
species/channel order in metadata), multimodal comparison harness
(vision / olfaction / both), calibration workflow against real exposure data.

---

## 11. Scientific integrity

Never state or imply that: a Gaussian plume reproduces indoor turbulence; a
filament model is CFD-equivalent; concentrations are chemically validated when
they are not; a virtual MiCS/EC response is a hardware-accurate digital twin
before calibration; olfaction replaces GPS, vision, or LiDAR.

Prefer "approximate", "robotics-grade", "phenomenological", "calibrated
against". Every published result states the Isaac Sim / Isaac Lab version, the
plume config, the **sensor profile** (`packaged_slow` vs `fast_modulated`), and
the realism-gate output. The framework's job is to make fidelity *measurable*,
not to make it look high.

Treat imported config, NPZ, VDB, CFD, and model files as untrusted: no pickle by
default, validate dimensions and units, bound allocations, no code execution in
config.

---

## 12. Definition of done, v1

- [ ] Realism gate implemented, running in CI, and **passing** with the §5.3
      ablation reproducible.
- [ ] NumPy and Warp transports agree to tolerance; NumPy path runs GPU-free.
- [ ] Filament plume with both turbulence scales; blank CV > 1.
- [ ] Occupancy grid from a real USD stage; filaments do not cross walls.
- [ ] `SensorBase` subclass working under Isaac Lab, vectorised over envs.
- [ ] Sensor chain complete through ADC quantisation.
- [ ] Both `packaged_slow` and `fast_modulated` profiles, with the whiff-retention
      number measured and documented for each.
- [ ] Per-episode domain randomisation over `R0`, `(A, beta)`, `tau`, drift.
- [ ] Ground truth strictly separated from observation.
- [ ] Output conforms to the Scentience BLE/Sockets channel schema, **with units**.
- [ ] Recurrent policy beats chemotaxis+casting on source localisation.
- [ ] Python-only path works with no Isaac installed.
- [ ] OmniGraph read node; ROS 2 publishing documented and runnable.
- [ ] Episode logging with explicit channel/species ordering in metadata.
- [ ] Calibration workflow documented; **no invented coefficients**; every
      illustrative constant labelled in code, not only in docs.
- [ ] `docs/ISAAC_COMPATIBILITY.md` records what was actually validated, and
      against which versions.

---

## 13. Completion report format

1. **Implemented** — files/features.
2. **Validated** — commands run, and the realism-gate numbers.
3. **Isaac compatibility** — exact versions and APIs exercised, or an explicit
   statement that Isaac was unavailable.
4. **Known limitations** — physical-model limitations first.
5. **Next highest-value step** — one recommendation.
