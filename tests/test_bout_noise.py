import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scentience_olfaction.oio.oio import BoutDetector, BoutDetectorConfig


def test_no_bouts_in_pure_noise():
    """The 3-sigma threshold must actually deliver a low false-positive rate
    on gaussian noise -- otherwise every OIO result is built on phantom
    detections."""
    det = BoutDetector(BoutDetectorConfig(noise_sigma=0.01))
    rng = np.random.default_rng(0)
    for _ in range(20000):                        # 200 s at 100 Hz
        det.step(0.01 * rng.standard_normal(), 0.01)
    assert det.n_bouts <= 3, f"{det.n_bouts} false bouts in pure noise"


def test_bouts_detected_at_modest_snr():
    det = BoutDetector(BoutDetectorConfig(noise_sigma=0.01))
    rng = np.random.default_rng(1)
    hits = 0
    for i in range(20000):
        x = 0.01 * rng.standard_normal()
        if (i // 100) % 20 == 0:                  # 1 s pulse every 20 s
            x += 0.06                             # 6-sigma whiff
        r = det.step(x, 0.01)
        hits += r["onset"]
    assert 8 <= det.n_bouts <= 12, f"expected ~10 bouts, got {det.n_bouts}"
