# Roadmap (bi-weekly cadence)

v0.1 (this release) -- timestamp release: filament transport (CPU full /
GPU fast path), MiCS-6814 + EC + SCD4x + PID + V1 device, occupancy +
potential flow, realism gate in CI, Gymnasium env + cast-and-surge baseline,
OIO reference implementation, BLE-schema bridge, provenance system.
Isaac Lab sensor written but UNVALIDATED in a live install.

v0.2 -- Isaac validation pass (validate_install.py on real 5.1 install; paste
output into ISAAC_COMPATIBILITY.md), GPU multi-species + occupancy in Warp,
DirectRLEnv example task, .nvdb plume export for offline fields.

v0.3 -- calibration workflow against real Scentience exposure logs
(calibration/fit.py), measured (A, beta, tau) for at least ethanol on >= 5
units, sensor-stats validation report.

v0.4 -- GADEN head-to-head on the VGR scenes (statistics comparison),
RANS wind import path, recorded-vs-simulated whiff statistics.

v0.5 -- COLIP-2 / olfaction-vision-language integration (deliberately held
out of v0.1; multimodality lands as its own release), Sockets-API server if
the client packages add the protocol (docs/UPSTREAM_REQUESTS.md).

Later: learned sensor transfer models, RL training recipes + trained
baselines, ROS 2 publishing, dense-gas buoyancy model, arXiv paper.
