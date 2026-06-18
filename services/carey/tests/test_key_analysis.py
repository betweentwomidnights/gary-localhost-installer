from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CAREY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAREY_DIR))

from key_analysis import KeyEstimate, choose_key, estimate_key, normalize_keyscale  # noqa: E402


class KeyAnalysisTests(unittest.TestCase):
    def test_normalize_keyscale_accepts_common_spellings(self) -> None:
        self.assertEqual(normalize_keyscale("D minor"), "D minor")
        self.assertEqual(normalize_keyscale("d min"), "D minor")
        self.assertEqual(normalize_keyscale("Bb major"), "A# major")
        self.assertEqual(normalize_keyscale("B♭ major"), "A# major")
        self.assertEqual(normalize_keyscale("N/A"), "")

    def test_local_key_overrides_unrelated_lm_guess_when_confident(self) -> None:
        decision = choose_key(
            lm_keyscale="G minor",
            local_estimate=KeyEstimate(
                keyscale="D minor",
                confidence=0.2,
                candidates=(),
            ),
        )

        self.assertEqual(decision.keyscale, "D minor")
        self.assertEqual(decision.source, "local_overrode_lm")

    def test_low_confidence_local_key_does_not_override_lm(self) -> None:
        decision = choose_key(
            lm_keyscale="G minor",
            local_estimate=KeyEstimate(
                keyscale="D minor",
                confidence=0.12,
                candidates=(),
            ),
        )

        self.assertEqual(decision.keyscale, "G minor")
        self.assertEqual(decision.source, "lm_local_low_confidence")

    def test_estimator_locks_simple_d_minor_audio(self) -> None:
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "d_minor.wav"
            sample_rate = 22050
            duration = 8.0
            t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
            notes = [146.832, 174.614, 220.0]  # D3, F3, A3
            audio = sum(np.sin(2 * np.pi * note * t) for note in notes)
            audio += 0.5 * sum(np.sin(2 * np.pi * note * 2 * t) for note in notes)
            audio = (audio / np.max(np.abs(audio))).astype(np.float32)
            sf.write(str(path), audio, sample_rate, subtype="PCM_16")

            estimate = estimate_key(path)

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertEqual(estimate.keyscale, "D minor")


if __name__ == "__main__":
    unittest.main()
