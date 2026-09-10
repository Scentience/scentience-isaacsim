# Dependency and provenance notes

## Repository terms

The authoritative terms are [LICENSE](../LICENSE),
[LICENSE_MODEL](../scentience_olfaction/sensors/LICENSE_MODEL),
[NOTICE](../NOTICE) and [ACKNOWLEDGEMENTS](../ACKNOWLEDGEMENTS). None was changed
in this update. The sensor model terms limit use and derivatives to research.
Earlier Apache-2.0 package/citation metadata was inaccurate and has been removed;
that correction does not change or replace any license grant.

New sensor modifications are described in [the changelog](../CHANGELOG.md),
[SCENTIENCE_MODELS.md](SCENTIENCE_MODELS.md) and [SENSOR_PROFILES.md](SENSOR_PROFILES.md).
Scentience Sensor Research Model is licensed under the Scentience Sensor
Research Model License Agreement.

## Selected software dependencies

| Dependency | License | Use |
|---|---|---|
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | BSD-3-Clause | Only mandatory dependency |
| [Warp](https://github.com/NVIDIA/warp/blob/main/LICENSE.md) | Apache-2.0 | Optional transport backend |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | BSD-style | Optional tensor device model |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE) | MIT | Optional environment API |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE) | MIT | New optional PPO/SAC/TD3 training command |
| [Matplotlib](https://matplotlib.org/stable/project/license.html) | PSF-based permissive license | Optional static plots |
| [PyYAML](https://github.com/yaml/pyyaml/blob/main/LICENSE) | MIT | Optional species file parser |
| pytest, pytest-cov, Ruff, build, SciPy | MIT, BSD or Apache/MIT | Development and tests |

The new training extra uses Stable-Baselines3 without vendor SDKs or proprietary
calibration services. The old `bridge` extra that pulled the separately licensed
Scentience client has been removed; the pure numeric BLE schema adapter remains.
No GADEN code or new copyleft runtime has been introduced. Other Isaac learning
runners are documented integrations, not mandatory dependencies.

This is a direct-dependency inventory, not an assertion that every asset, binary
or transitive component in a full Isaac installation has identical terms.
Isaac runtime/asset licensing remains governed by NVIDIA's distributions;
no robot meshes, firmware or proprietary SDKs are redistributed here.

## Equations, calibration and evidence

Filament transport follows the cited Farrell model; turbulent-plume statistics
are discussed against Celani et al. Instrument profiles identify manufacturer
sources and simulation assumptions. Published response times from another
sensor setup do not establish accuracy on Scentience hardware. The evidence
registry distinguishes assumed, synthesized, digitized, datasheet and measured
values; use scoped `claim_check()` for claims depending on registered values.
The registry is not complete coverage of every configurable coefficient.

Keep calibration data and their permitted use with your experiment record.
A calibration ID identifies a dataset or experiment; it is not a certification
of accuracy. No manufacturer firmware or SDK was copied into the new models.
