from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CAREY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAREY_DIR))

from bpm_analysis import BpmEstimate, choose_bpm, estimate_bpm  # noqa: E402


class BpmAnalysisTests(unittest.TestCase):
    def test_filename_bpm_wins(self) -> None:
        decision = choose_bpm(
            filename_bpm=128,
            lm_bpm=100,
            local_estimate=BpmEstimate(bpm=66.2, confidence=10.0, candidates=()),
        )

        self.assertEqual(decision.bpm, 128)
        self.assertEqual(decision.source, "filename")

    def test_local_bpm_overrides_unrelated_lm_guess(self) -> None:
        decision = choose_bpm(
            lm_bpm=100,
            local_estimate=BpmEstimate(bpm=66.2, confidence=10.0, candidates=()),
        )

        self.assertEqual(decision.bpm, 66)
        self.assertEqual(decision.source, "local_overrode_lm")

    def test_lm_bpm_survives_when_in_same_tempo_family(self) -> None:
        decision = choose_bpm(
            lm_bpm=133,
            local_estimate=BpmEstimate(bpm=66.2, confidence=10.0, candidates=()),
        )

        self.assertEqual(decision.bpm, 133)
        self.assertEqual(decision.source, "lm_agrees_with_local")

    def test_low_confidence_local_bpm_does_not_override_lm(self) -> None:
        decision = choose_bpm(
            lm_bpm=100,
            local_estimate=BpmEstimate(bpm=66.2, confidence=1.05, candidates=()),
        )

        self.assertEqual(decision.bpm, 100)
        self.assertEqual(decision.source, "lm_local_low_confidence")

    def test_estimator_locks_simple_click_track(self) -> None:
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "click_90.wav"
            sample_rate = 22050
            duration = 20.0
            bpm = 90
            audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
            beat_period = 60 / bpm
            click = np.hanning(256).astype(np.float32) * 0.8
            for beat in np.arange(0, duration, beat_period):
                start = int(beat * sample_rate)
                end = min(len(audio), start + len(click))
                audio[start:end] += click[: end - start]
            sf.write(str(path), audio, sample_rate, subtype="PCM_16")

            estimate = estimate_bpm(path)

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertLessEqual(abs(estimate.bpm - bpm), 3.0)


if __name__ == "__main__":
    unittest.main()
