from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CAREY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAREY_DIR))

import sa3_autolabel  # noqa: E402


class BuildReuseArgsTests(unittest.TestCase):
    """The reused train_lora_job helpers index MODEL_MAP with args.model, and
    that map is keyed by full model names rather than the CLI aliases."""

    def cli(self, model: str) -> SimpleNamespace:
        return SimpleNamespace(
            job_id="job", name=None, run_dir=".", log_path=".", cancel_path=".",
            status_path=".", current_job_path=".", dataset_dir=".",
            carey_url="http://127.0.0.1:8003", caption_lm_model="acestep-5Hz-lm-1.7B",
            model=model, caption_timeout=1.0, caption_startup_timeout=1.0,
            model_load_timeout=1.0, carey_stop_timeout=1.0,
            caption_window_seconds=1.0, analysis_duration=1.0,
            style="", trigger="", job_dir=".",
        )

    def test_the_default_alias_resolves_to_a_real_model_key(self) -> None:
        from train_lora_job import MODEL_MAP

        args = sa3_autolabel.build_reuse_args(self.cli("base"))

        self.assertEqual(args.model, "acestep-v15-base")
        # The bug this covers surfaced as a bare KeyError('base') in the UI.
        self.assertIn(args.model, MODEL_MAP)

    def test_a_full_model_name_is_left_alone(self) -> None:
        args = sa3_autolabel.build_reuse_args(self.cli("acestep-v15-xl-base"))

        self.assertEqual(args.model, "acestep-v15-xl-base")

    def test_an_unknown_model_says_so_instead_of_raising_a_key_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown ACE-Step training model"):
            sa3_autolabel.build_reuse_args(self.cli("nonsense"))


class Sa3AudioDiscoveryTests(unittest.TestCase):
    def test_matches_ui_audio_extensions_including_aiff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = {
                "a.wav",
                "b.flac",
                "c.mp3",
                "d.ogg",
                "e.opus",
                "f.m4a",
                "g.aiff",
                "h.aif",
            }
            for name in expected | {"ignore.txt"}:
                (root / name).touch()

            discovered = {path.name for path in sa3_autolabel.discover_sa3_audio(root)}

        self.assertEqual(discovered, expected)


class GenreValidationTests(unittest.TestCase):
    def test_normalizes_genre_lists(self) -> None:
        genre = sa3_autolabel.usable_genre(
            {"genres": ["ambient", " drone ", ""]},
            Path("song.wav"),
        )
        self.assertEqual(genre, "ambient, drone")

    def test_rejects_missing_or_placeholder_genres(self) -> None:
        for value in (None, "", "N/A", "unknown"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "no usable genre"):
                    sa3_autolabel.usable_genre({"genre": value}, Path("song.wav"))

    def test_retries_bad_full_track_genre_with_excerpt(self) -> None:
        args = SimpleNamespace(
            caption_window_seconds=0.0,
            caption_fallback_window_seconds=120.0,
        )
        with patch.object(
            sa3_autolabel,
            "request_music_analysis",
            side_effect=[{"genre": ""}, {"genre": "dark ambient"}],
        ) as request:
            result, genre = sa3_autolabel.request_valid_genre_analysis(
                args,
                object(),
                Path("song.wav"),
            )

        self.assertEqual(result, {"genre": "dark ambient"})
        self.assertEqual(genre, "dark ambient")
        self.assertEqual(
            [call.kwargs["caption_window_seconds"] for call in request.call_args_list],
            [0.0, 120.0],
        )


class TerminalStatusTests(unittest.TestCase):
    @staticmethod
    def cli(root: Path) -> list[str]:
        run_dir = root / "run"
        return [
            "--dataset-dir",
            str(root),
            "--run-dir",
            str(run_dir),
            "--log-path",
            str(run_dir / "autolabel.log"),
            "--status-path",
            str(run_dir / "status.json"),
            "--cancel-path",
            str(run_dir / "cancel.requested"),
            "--current-job-path",
            str(run_dir / "current_job.json"),
        ]

    def test_failure_is_recorded_after_caption_service_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "song.wav"
            audio.touch()
            events: list[tuple[str, dict]] = []

            def update_status(_args, **updates):
                events.append(("status", updates))

            def stop_server(args, _server):
                events.append(("stop", {}))
                update_status(
                    args,
                    status="running",
                    phase="stopping-caption-service",
                )

            with (
                patch.object(sa3_autolabel, "update_status", side_effect=update_status),
                patch.object(sa3_autolabel, "ensure_carey_stopped"),
                patch.object(sa3_autolabel, "start_caption_server", return_value=object()),
                patch.object(sa3_autolabel, "wait_for_carey"),
                patch.object(sa3_autolabel, "ensure_carey_model_loaded"),
                patch.object(
                    sa3_autolabel,
                    "request_valid_genre_analysis",
                    side_effect=RuntimeError("captioner exploded"),
                ),
                patch.object(sa3_autolabel, "stop_caption_server", side_effect=stop_server),
            ):
                result = sa3_autolabel.main(self.cli(root))

        self.assertEqual(result, 1)
        self.assertEqual(events[-1][0], "status")
        self.assertEqual(events[-1][1]["status"], "failed")
        self.assertEqual(events[-1][1]["error"], "captioner exploded")
        self.assertLess(
            next(index for index, event in enumerate(events) if event[0] == "stop"),
            len(events) - 1,
        )

    def test_success_is_recorded_after_caption_service_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "song.wav"
            audio.touch()
            events: list[tuple[str, dict]] = []

            def update_status(_args, **updates):
                events.append(("status", updates))

            def stop_server(args, _server):
                events.append(("stop", {}))
                update_status(
                    args,
                    status="running",
                    phase="stopping-caption-service",
                )

            bpm = SimpleNamespace(bpm=96, source="local")
            key = SimpleNamespace(keyscale="C minor", source="local")
            with (
                patch.object(sa3_autolabel, "update_status", side_effect=update_status),
                patch.object(sa3_autolabel, "ensure_carey_stopped"),
                patch.object(sa3_autolabel, "start_caption_server", return_value=object()),
                patch.object(sa3_autolabel, "wait_for_carey"),
                patch.object(sa3_autolabel, "ensure_carey_model_loaded"),
                patch.object(
                    sa3_autolabel,
                    "request_valid_genre_analysis",
                    return_value=({"genre": "trip-hop"}, "trip-hop"),
                ),
                patch.object(sa3_autolabel, "decide_sidecar_bpm", return_value=bpm),
                patch.object(sa3_autolabel, "decide_sidecar_key", return_value=key),
                patch.object(sa3_autolabel, "stop_caption_server", side_effect=stop_server),
            ):
                result = sa3_autolabel.main(self.cli(root))

            sidecar = audio.with_suffix(".txt").read_text(encoding="utf-8").strip()

        self.assertEqual(result, 0)
        self.assertEqual(sidecar, "trip-hop, 96 bpm, C minor")
        self.assertEqual(events[-1][0], "status")
        self.assertEqual(events[-1][1]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
