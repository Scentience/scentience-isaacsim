"""The verification plot script must keep producing its two figures --
they are release collateral (see docs/RELEASE_VALIDATION.md, fifth pass)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib.util

import pytest

pytest.importorskip("matplotlib")


def _load():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                        "plot_verification.py")
    spec = importlib.util.spec_from_file_location("plot_verification", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plot_script_writes_both_figures(tmp_path):
    mod = _load()
    paths = mod.main(["--outdir", str(tmp_path), "--seconds", "10"])
    assert len(paths) == 2
    for p in paths:
        assert os.path.exists(p), p
        assert os.path.getsize(p) > 20_000, f"{p} suspiciously small"
    names = {os.path.basename(p) for p in paths}
    assert names == {"sensor_bandwidth.png", "stereo_lateralisation.png"}
