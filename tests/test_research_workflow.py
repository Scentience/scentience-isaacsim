"""Configuration, replay, recorder integrity, and the public learning contract."""

import json

import numpy as np
import pytest

from scentience_olfaction import FilamentPlume, FilamentPlumeConfig
from scentience_olfaction.config import config_dict, plume_config
from scentience_olfaction.recording.recorder import EpisodeRecorder, run_metadata


def test_config_roundtrip_and_typo_rejection():
    cfg = plume_config({"emitters": [{"type": "line", "position": [0, 0, 1], "end": [0, 2, 1]}]})
    restored = plume_config(json.loads(json.dumps(config_dict(cfg), allow_nan=False)))
    assert restored.emitters[0].end == [0, 2, 1]
    with pytest.raises(ValueError, match="Unknown"):
        plume_config({"turbulance_intensity": 0.5})
    meta = run_metadata(config_dict(cfg), 7)
    assert meta["config_sha256"] == run_metadata(config_dict(restored), 7)["config_sha256"]


def test_recorder_snapshots_schema_and_restart(tmp_path):
    rec = EpisodeRecorder(tmp_path, ["a"], ["ethanol"])
    obs = np.array([1.0])
    rec.log(obs=obs, t=0.0)
    obs[:] = 9
    with pytest.raises(ValueError, match="same fields"):
        rec.log(obs=obs)
    with pytest.raises(ValueError, match="shapes"):
        rec.log(obs=[1, 2], t=1)
    first = rec.end_episode()
    arrays, meta = rec.load(first)
    assert arrays["obs"].tolist() == [[1.0]] and meta["n_steps"] == 1
    restarted = EpisodeRecorder(tmp_path, ["a"], ["ethanol"])
    restarted.log(obs=[2], t=0)
    assert restarted.end_episode() != first


def test_slice_is_read_only_and_in_ppm():
    from scentience_olfaction.visualization import concentration_slice
    plume = FilamentPlume(FilamentPlumeConfig(max_filaments=20), seed=1)
    plume.step(0.1)
    time, pos = plume.t, plume.pos.copy()
    x, y, c = concentration_slice(plume, resolution=(8, 6))
    assert c.shape == (len(y), len(x)) and np.isfinite(c).all() and (c >= 0).all()
    assert plume.t == time and np.array_equal(plume.pos, pos)


def test_gym_contract_history_reset_and_discounted_potential():
    gym = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env
    from scentience_olfaction.envs.plume_nav import PlumeNavConfig, PlumeNavEnv
    from scentience_olfaction.envs.wrappers import ObservationHistory
    cfg = PlumeNavConfig(warmup_s=0, timeout_s=1, shaping_gamma=0.99)
    env = PlumeNavEnv(cfg)
    check_env(env, skip_render_check=True)
    env.reset(seed=7)
    d0 = env._dist_to_source()
    _, reward, _, _, info = env.step([0.5, 0])
    assert reward == pytest.approx(cfg.shaping_scale * (d0 - 0.99 * info["dist"]))
    wrapped = ObservationHistory(env, 3)
    first, _ = wrapped.reset(seed=4)
    wrapped.step([1, 0])
    again, _ = wrapped.reset(seed=4)
    np.testing.assert_array_equal(first, again)
    assert first.shape == (33,)
    assert gym.make("Scentience-PlumeNav-v0", cfg=cfg).observation_space.shape == (11,)


def test_terminal_potential_and_action_validation():
    pytest.importorskip("gymnasium")
    from scentience_olfaction.envs.plume_nav import PlumeNavConfig, PlumeNavEnv
    env = PlumeNavEnv(PlumeNavConfig(warmup_s=0, shaping_gamma=0.99))
    env.reset(seed=0)
    with pytest.raises(ValueError, match="finite"):
        env.step([np.nan, 0])
    env._pos[:] = [0, 0]
    previous = env._prev_dist
    _, reward, term, trunc, info = env.step([0, 0])
    assert term and not trunc and info["is_success"]
    assert reward == pytest.approx(10 + env.cfg.shaping_scale * previous)
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0, 0])


def test_world_clock_named_replay_and_background_alias():
    from scentience_olfaction import OlfactionWorld
    cfg = FilamentPlumeConfig(emitters=[], background_ppm={"CO2": 420})
    world = OlfactionWorld(FilamentPlume(cfg), device_profile="scd41",
                           device_config={"repeatability_ppm": 0}, seed=7)
    world.step(5.0)
    assert world.read((0, 0, 1))["co2_ppm"] == 420
    first = world.read((0, 0, 1), name="other")
    world.reset(seed=7)
    world.step(5.0)
    assert world.read((0, 0, 1), name="other") == first
    assert world.sensor_diagnostics("other")["co2_ppm"] == 420


def test_censored_events_and_stats_validation():
    from scentience_olfaction.validation.plume_stats import whiff_blank_durations, summarize
    whiffs, blanks = whiff_blank_durations(np.array([1, 1, 0, 0]), 0.1, 0.5)
    assert not len(whiffs) and not len(blanks)
    whiffs, blanks = whiff_blank_durations(np.array([0, 1, 1, 0]), 0.1, 0.5)
    assert whiffs.tolist() == [0.2] and not len(blanks)
    with pytest.raises(ValueError, match="nonempty"):
        summarize(np.array([]), 0.1, 0.5)
