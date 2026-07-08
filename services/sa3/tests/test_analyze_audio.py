from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import analyze_audio
import bpm_analysis
import key_analysis
from bpm_analysis import BpmEstimate
from key_analysis import KeyEstimate


class BuildSuggestionTests(unittest.TestCase):
    def test_bare_format_bpm_and_key(self):
        self.assertEqual(analyze_audio.build_suggestion(145, "C minor"), "145 bpm, C minor")

    def test_bpm_only(self):
        self.assertEqual(analyze_audio.build_suggestion(120, ""), "120 bpm")

    def test_key_only(self):
        self.assertEqual(analyze_audio.build_suggestion(None, "A major"), "A major")

    def test_empty_when_nothing_estimated(self):
        self.assertEqual(analyze_audio.build_suggestion(None, ""), "")


class AnalyzeTests(unittest.TestCase):
    def test_uses_local_estimates_in_bare_format(self):
        bpm_est = BpmEstimate(bpm=145.4, confidence=1.83, candidates=())
        key_est = KeyEstimate(keyscale="C minor", confidence=0.21, candidates=())
        with (
            patch.object(bpm_analysis, "estimate_bpm", return_value=bpm_est),
            patch.object(key_analysis, "estimate_key", return_value=key_est),
        ):
            result = analyze_audio.analyze(Path("dummy.wav"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["bpm"], 145)
        self.assertEqual(result["keyscale"], "C minor")
        self.assertEqual(result["suggestion"], "145 bpm, C minor")
        self.assertEqual(result["bpm_confidence"], 1.83)
        self.assertEqual(result["key_confidence"], 0.21)

    def test_low_confidence_key_still_suggested(self):
        # SA3 has no LM fallback; a low-confidence key must not be silently dropped.
        bpm_est = BpmEstimate(bpm=90.0, confidence=1.1, candidates=())
        key_est = KeyEstimate(keyscale="F# major", confidence=0.02, candidates=())
        with (
            patch.object(bpm_analysis, "estimate_bpm", return_value=bpm_est),
            patch.object(key_analysis, "estimate_key", return_value=key_est),
        ):
            result = analyze_audio.analyze(Path("dummy.wav"))

        self.assertEqual(result["keyscale"], "F# major")
        self.assertEqual(result["suggestion"], "90 bpm, F# major")

    def test_silent_audio_yields_empty_suggestion(self):
        with (
            patch.object(bpm_analysis, "estimate_bpm", return_value=None),
            patch.object(key_analysis, "estimate_key", return_value=None),
        ):
            result = analyze_audio.analyze(Path("dummy.wav"))

        self.assertTrue(result["ok"])
        self.assertIsNone(result["bpm"])
        self.assertEqual(result["keyscale"], "")
        self.assertEqual(result["suggestion"], "")


class EnsureScipyTests(unittest.TestCase):
    def test_noop_when_present(self):
        with (
            patch.object(analyze_audio.importlib.util, "find_spec", return_value=object()),
            patch.object(analyze_audio.subprocess, "check_call") as check_call,
        ):
            analyze_audio.ensure_scipy()
        check_call.assert_not_called()

    def test_installs_when_missing(self):
        # Missing on first probe, present after install.
        with (
            patch.object(analyze_audio.importlib.util, "find_spec", side_effect=[None, object()]),
            patch.object(analyze_audio.subprocess, "check_call") as check_call,
        ):
            analyze_audio.ensure_scipy()
        check_call.assert_called_once()
        install_cmd = check_call.call_args.args[0]
        self.assertIn("pip", install_cmd)
        self.assertIn(analyze_audio.SCIPY_REQUIREMENT, install_cmd)

    def test_points_to_rebuild_env_when_install_fails(self):
        import subprocess

        with (
            patch.object(analyze_audio.importlib.util, "find_spec", return_value=None),
            patch.object(
                analyze_audio.subprocess,
                "check_call",
                side_effect=subprocess.CalledProcessError(1, "pip"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild env"):
                analyze_audio.ensure_scipy()

    def test_points_to_rebuild_env_when_still_missing_after_install(self):
        with (
            patch.object(analyze_audio.importlib.util, "find_spec", side_effect=[None, None]),
            patch.object(analyze_audio.subprocess, "check_call"),
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild env"):
                analyze_audio.ensure_scipy()


if __name__ == "__main__":
    unittest.main()
