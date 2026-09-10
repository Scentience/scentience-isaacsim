# Ecosystem alignment and contribution scope

Assessment date: 2026-09-09. This is an integration design and release plan, not
an assertion of NVIDIA acceptance or third-party performance parity.

## Packages and conventions reviewed

| Primary source | Relevant convention | Application here |
|---|---|---|
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | Separate sensor/data/config classes, task registration, MDP terms, optional learning runners | Sensor adapter, IMU bridge, task registration helper and framework-neutral core |
| [Isaac Lab project template](https://github.com/isaac-sim/IsaacLabExtensionTemplate) | External projects; template generation has moved into Isaac Lab and the old template is archived | Keep an independently installable extension; do not copy the obsolete template wholesale |
| [Pegasus Simulator](https://github.com/PegasusSimulator/PegasusSimulator) | Vehicle/controller/sensor separation, standalone examples, documentation and attribution | Attach chemical sensing to a vehicle link without replacing its flight dynamics or autopilot |
| [Isaac Lab learning-library comparison](https://isaac-sim.github.io/IsaacLab/develop/source/overview/reinforcement-learning/rl_frameworks.html) | RSL-RL, skrl, RL-Games and SB3 adapters | Preserve standard observations/reset semantics and explicit runner config entry points |

The existing import layout is retained to avoid breaking users:

```text
scentience_olfaction/   NumPy transport, sensors, environments, metrics and recording
scentience_isaaclab/    SensorBase adapter, IMU conversion and MDP integration
isaac_extension/       Optional Kit UI extension
configs/               Small editable experiment configurations
examples/              Standalone examples
scripts/               Validation, training, benchmarking and visual reports
tests/                 Functional, physical and backend contract tests
docs/                  Integration, scientific assumptions and research workflows
```

This mirrors the separation of responsibilities in mature Isaac projects while
keeping a NumPy-only installation useful. A physical robot task owns actuators,
contacts, locomotion/flight control and rewards; the olfactory package supplies
measurements. Sensor mounting on a rigid link applies to humanoids, quadrupeds,
wheeled bases, manipulators and drones. Actual USD link paths, gravity settings
and offsets must be checked for each asset. Rotor wash, moving-body flow and
wake coupling require a suitable external airflow field; a mount alone does
not model those effects.

## Learning research

Recurrent deep-RL agents have been used for turbulent plume tracking because
odor encounters and wind history provide information unavailable in one sample.
See [Singh et al., Nature Machine Intelligence 5, 58–70](https://www.nature.com/articles/s42256-022-00599-w).
The supplied observation-history wrapper supports a feedforward baseline;
recurrent PPO remains a useful comparison in an existing Isaac runner. PPO is
the common Isaac starting point. SAC/TD3 provide off-policy comparisons in the
standalone training command. These are integration choices, not a universal
ranking of algorithms.

## A concrete NVIDIA showcase

Use one scene with two sensor streams: sampled air concentration for visualization
and the instrument response actually available to the robot. Show a controlled
wind change, plume loss/reacquisition, stereo sensing and the difference between
fast MOX and slow CO₂ context. Pair the robot video with concentration and sensor
traces, then report held-out success and declaration statistics. Include one
humanoid, quadruped and aerial mount check using existing NVIDIA assets; preserve
their controllers. Publish actual hardware/runtime versions, scene seeds and
calibration identifiers with the results.

The repository supplies the visualization, baseline, logging and mounting
components. A new live Isaac recording, robot-specific training result, and
measured calibration dataset still need to be produced on the supported host.

## Candidate upstream contribution

A reviewable first contribution is the robot-independent sensor contract, IMU
observation example and a minimal mount demonstration. Keep source-search
research, detailed Scentience calibration and benchmark assets in the external
package initially. This reduces the maintenance/API surface the Isaac Lab team
would need to own. Discuss scope with maintainers before moving model code.

The existing sensor license restricts research use; the package's old Apache
metadata did not describe those terms. The license files remain unchanged.
Isaac Lab's [licensing guidance](https://isaac-sim.github.io/IsaacLab/main/source/refs/license.html)
uses permissive project licensing, so redistribution of the current sensor
models is an unresolved prerequisite for an upstream merge. No relicensing,
upstream issue, email, PR or publication is performed by these changes.
