from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

CAREY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAREY_DIR))

import train_lora_job  # noqa: E402


def make_args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model="base",
        checkpoint_dir=root / "checkpoints",
        max_duration=240.0,
        rank=64,
        alpha=128,
        module_profile="balanced",
        learning_rate=3e-4,
        batch_size=1,
        gradient_accumulation=1,
        epochs=150,
        save_every=25,
        save_best=True,
        save_best_after=25,
        cfg_ratio=0.15,
        timestep_mu=-0.4,
        instrumental=True,
        loss_weighting="min_snr",
        snr_gamma=5.0,
    )


class TrainLoraJobTests(unittest.TestCase):
    def test_registration_records_base_and_xl_model_families(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "adapter"
            checkpoint.mkdir()
            registry_path = root / "lora_registry.json"

            for model, expected_family in (("base", "standard"), ("xl-base", "xl")):
                args = make_args(root)
                args.model = model
                args.name = f"test-{expected_family}"
                args.dataset_dir = root
                args.adapter_type = "dora"
                args.lora_catalog_path = None
                args.lora_registry_path = registry_path
                args.captions_json_path = None

                train_lora_job.register_trained_lora(args, checkpoint)

            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry["test-standard"]["model_family"], "standard")
            self.assertEqual(registry["test-xl"]["model_family"], "xl")

    def test_run_step_surfaces_trainer_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            failure = "[FAIL] VRAM preflight needs 512 MiB more free GPU memory"
            proc = Mock(pid=1234)
            proc.poll.return_value = 1

            def start_trainer(*_args, **_kwargs):
                (output_dir / ".training-failure.txt").write_text(failure, encoding="utf-8")
                return proc

            with (
                patch.object(train_lora_job, "check_cancel"),
                patch.object(train_lora_job, "update_status"),
                patch.object(train_lora_job.subprocess, "Popen", side_effect=start_trainer),
            ):
                with self.assertRaisesRegex(RuntimeError, "VRAM preflight"):
                    train_lora_job.run_step(
                        argparse.Namespace(),
                        ["trainer"],
                        "training",
                        "Training ACE-Step LoRA",
                        cwd=output_dir,
                    )

    def test_command_builders_match_existing_carey_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = make_args(root)
            preprocess = train_lora_job.build_preprocess_command(
                args,
                root / "dataset.json",
                root / "tensors",
                root / "output",
            )
            train = train_lora_job.build_train_command(
                args,
                root / "tensors",
                root / "output",
            )

            self.assertIn("--preprocess", preprocess)
            self.assertIn("--dataset-json", preprocess)
            self.assertIn("--gradient-checkpointing", train)
            self.assertIn("--offload-encoder", train)
            self.assertIn("--vram-preflight", train)
            self.assertIn("--use-dora", train)
            self.assertEqual(train[train.index("--shift") + 1], "1.0")
            self.assertEqual(train[train.index("--gradient-accumulation") + 1], "1")
            self.assertEqual(train[train.index("--max-duration") + 1], "240.0")
            self.assertEqual(train[train.index("--module-profile") + 1], "balanced")
            self.assertEqual(train[train.index("--timestep-mu") + 1], "-0.4")
            self.assertIn("--save-best", train)
            self.assertEqual(train[train.index("--save-best-after") + 1], "25")
            self.assertEqual(train[train.index("--loss-weighting") + 1], "min_snr")
            self.assertEqual(train[train.index("--snr-gamma") + 1], "5.0")

    def test_normalize_analysis_result_handles_query_result_shapes(self) -> None:
        nested = json.dumps(
            [
                {
                    "status_message": "Full Hardware Analysis Success",
                    "prompt": "warm ambient",
                    "bpm": 90,
                }
            ]
        )
        result = train_lora_job.normalize_analysis_result(nested)
        self.assertEqual(result["prompt"], "warm ambient")
        self.assertEqual(result["bpm"], 90)

    def test_analysis_is_instrumental_uses_explicit_value_or_lyrics(self) -> None:
        self.assertTrue(
            train_lora_job.analysis_is_instrumental({"lyrics": "[Instrumental]"})
        )
        self.assertFalse(
            train_lora_job.analysis_is_instrumental(
                {"is_instrumental": "false", "lyrics": ""}
            )
        )
        self.assertTrue(
            train_lora_job.analysis_is_instrumental(
                {"lyrics": "sung lyrics"},
                default=True,
            )
        )
        self.assertTrue(
            train_lora_job.analysis_is_instrumental(
                {"lyrics": "[Intro]\n[Synth melody and beat starts]\n[Outro]"}
            )
        )
        self.assertFalse(
            train_lora_job.analysis_is_instrumental(
                {"lyrics": "[Verse]\nactual sung words"}
            )
        )

    def test_caption_lm_model_choices_include_quality_tiers(self) -> None:
        parser = train_lora_job.build_parser()
        action = next(
            item for item in parser._actions if item.dest == "caption_lm_model"
        )
        self.assertEqual(tuple(action.choices), train_lora_job.CAPTION_LM_MODELS)
        self.assertEqual(action.default, "acestep-5Hz-lm-0.6B")

    def test_caption_window_defaults_to_full_track(self) -> None:
        parser = train_lora_job.build_parser()
        action = next(
            item for item in parser._actions if item.dest == "caption_window_seconds"
        )
        self.assertEqual(action.default, 0.0)

    def test_genre_ratio_defaults_to_sidestep_reference_value(self) -> None:
        parser = train_lora_job.build_parser()
        action = next(item for item in parser._actions if item.dest == "genre_ratio")
        self.assertEqual(action.default, 20)

    def test_adapter_defaults_to_dora_recipe(self) -> None:
        parser = train_lora_job.build_parser()
        action = next(item for item in parser._actions if item.dest == "adapter_type")
        self.assertEqual(action.default, "dora")

    def test_min_snr_defaults_to_settled_recipe(self) -> None:
        parser = train_lora_job.build_parser()
        loss_action = next(
            item for item in parser._actions if item.dest == "loss_weighting"
        )
        gamma_action = next(item for item in parser._actions if item.dest == "snr_gamma")

        self.assertEqual(loss_action.default, "min_snr")
        self.assertEqual(gamma_action.default, 5.0)

    def test_training_defaults_to_balanced_profile_and_best_checkpoint(self) -> None:
        parser = train_lora_job.build_parser()
        profile = next(item for item in parser._actions if item.dest == "module_profile")
        save_best = next(item for item in parser._actions if item.dest == "save_best")
        save_after = next(
            item for item in parser._actions if item.dest == "save_best_after"
        )

        self.assertEqual(profile.default, "balanced")
        self.assertTrue(save_best.default)
        self.assertEqual(save_after.default, 25)

    def test_dataset_type_controls_timestep_mu_unless_overridden(self) -> None:
        self.assertEqual(
            train_lora_job.resolve_timestep_mu(
                argparse.Namespace(timestep_mu=None, instrumental=True)
            ),
            -0.4,
        )
        self.assertEqual(
            train_lora_job.resolve_timestep_mu(
                argparse.Namespace(timestep_mu=None, instrumental=False)
            ),
            0.0,
        )
        self.assertEqual(
            train_lora_job.resolve_timestep_mu(
                argparse.Namespace(timestep_mu=-0.2, instrumental=False)
            ),
            -0.2,
        )

    def test_auto_timesignature_is_omitted_unless_requested(self) -> None:
        result = {"timesignature": "3"}

        self.assertEqual(
            train_lora_job.auto_timesignature(
                argparse.Namespace(include_auto_timesignature=False),
                result,
            ),
            "",
        )
        self.assertEqual(
            train_lora_job.auto_timesignature(
                argparse.Namespace(include_auto_timesignature=True),
                result,
            ),
            "3",
        )

    def test_music_analysis_requests_selected_lm_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "clip.wav"
            audio.write_bytes(b"RIFF")
            args = argparse.Namespace(
                carey_url="http://127.0.0.1:8011",
                analysis_duration=8.0,
                caption_timeout=30.0,
                caption_lm_model="acestep-5Hz-lm-1.7B",
                cancel_path=root / "cancel.requested",
            )
            accepted = Mock()
            accepted.raise_for_status.return_value = None
            accepted.json.return_value = {"data": {"task_id": "task-1"}}
            completed = Mock()
            completed.raise_for_status.return_value = None
            completed.json.return_value = {
                "data": [
                    {
                        "status": 1,
                        "result": {
                            "status_message": "Full Hardware Analysis Success",
                            "lm_model": "acestep-5Hz-lm-1.7B",
                            "prompt": "chiptune",
                        },
                    }
                ]
            }
            client = Mock()
            client.post.side_effect = [accepted, completed]

            result = train_lora_job.request_music_analysis(args, client, audio)

            request_data = client.post.call_args_list[0].kwargs["data"]
            self.assertEqual(
                request_data["lm_model_path"],
                "acestep-5Hz-lm-1.7B",
            )
            self.assertEqual(request_data["audio_duration"], "8.0")
            self.assertEqual(result["lm_model"], "acestep-5Hz-lm-1.7B")

    def test_music_analysis_auto_duration_uses_source_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "clip.wav"
            audio.write_bytes(b"RIFF")
            args = argparse.Namespace(analysis_duration=0.0)

            original = train_lora_job.audio_duration_seconds
            try:
                train_lora_job.audio_duration_seconds = Mock(return_value=147.25)
                self.assertEqual(
                    train_lora_job.resolve_music_analysis_duration(args, audio),
                    147.25,
                )
            finally:
                train_lora_job.audio_duration_seconds = original

    def test_prepare_caption_audio_crops_long_files_to_caption_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "long.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(b"\x00\x00" * 8000 * 5)

            args = argparse.Namespace(caption_window_seconds=2.0)
            prepared = train_lora_job.prepare_caption_audio(args, audio)
            try:
                self.assertNotEqual(prepared.path, audio)
                self.assertEqual(prepared.cleanup_path, prepared.path)
                self.assertAlmostEqual(prepared.duration, 2.0, places=2)
                self.assertAlmostEqual(prepared.offset, 1.5, places=1)
                self.assertAlmostEqual(
                    train_lora_job.audio_duration_seconds(prepared.path),
                    2.0,
                    places=2,
                )
            finally:
                if prepared.cleanup_path:
                    prepared.cleanup_path.unlink(missing_ok=True)

    def test_caption_server_uses_private_lm_api_profile(self) -> None:
        args = argparse.Namespace(
            carey_url="http://127.0.0.1:8013",
            model="base",
            caption_lm_model="acestep-5Hz-lm-1.7B",
        )

        command = train_lora_job.build_caption_server_command(args)
        self.assertIn("api_server.py", command[1])
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "8013")
        self.assertIn("--no-init", command)
        self.assertEqual(
            command[command.index("--lm-model-path") + 1],
            "acestep-5Hz-lm-1.7B",
        )

        env = train_lora_job.build_caption_server_env(
            args,
            {
                "PYTHONPATH": "existing",
                "ACESTEP_NO_INIT": "true",
                "ACESTEP_CONFIG_PATH2": "acestep-v15-xl-base",
            },
        )
        self.assertEqual(env["ACESTEP_CONFIG_PATH"], "acestep-v15-base")
        self.assertEqual(env["ACESTEP_INIT_LLM"], "true")
        self.assertEqual(env["ACESTEP_LM_MODEL_PATH"], "acestep-5Hz-lm-1.7B")
        self.assertEqual(env["ACESTEP_LM_BACKEND"], "pt")
        self.assertEqual(env["ACESTEP_LM_OFFLOAD_TO_CPU"], "true")
        self.assertEqual(env["ACESTEP_NO_INIT"], "true")
        self.assertEqual(env["ACESTEP_OFFLOAD_TO_CPU"], "true")
        self.assertEqual(env["ACESTEP_OFFLOAD_DIT_TO_CPU"], "true")
        self.assertEqual(env["ACESTEP_UNDERSTAND_MAX_NEW_TOKENS"], "1024")
        self.assertEqual(env["ACESTEP_UNDERSTAND_TEMPERATURE"], "0.3")
        self.assertEqual(env["ACESTEP_USE_FLASH_ATTENTION"], "false")
        self.assertEqual(env["ACESTEP_COMPILE_MODEL"], "false")
        self.assertNotIn("ACESTEP_CONFIG_PATH2", env)
        self.assertTrue(env["PYTHONPATH"].startswith(str(train_lora_job.SERVICE_DIR)))

        small_args = argparse.Namespace(
            carey_url="http://127.0.0.1:8013",
            model="base",
            caption_lm_model="acestep-5Hz-lm-0.6B",
        )
        small_env = train_lora_job.build_caption_server_env(small_args, {})
        self.assertEqual(small_env["ACESTEP_OFFLOAD_DIT_TO_CPU"], "false")

    def test_captioning_existing_sidecars_logs_skip_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            run_dir = root / "run"
            dataset.mkdir()
            run_dir.mkdir()
            audio = dataset / "clip.wav"
            audio.write_bytes(b"RIFF")
            audio.with_suffix(".txt").write_text(
                "caption: existing\nlyrics: [Instrumental]\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                dataset_dir=dataset,
                overwrite_captions=False,
                status_path=run_dir / "status.json",
                current_job_path=run_dir / "current.json",
                job_id="skip-caption",
                name="skip-caption",
                run_dir=run_dir,
                log_path=run_dir / "job.log",
                cancel_path=run_dir / "cancel.requested",
                caption_lm_model="acestep-5Hz-lm-1.7B",
            )

            captioned = train_lora_job.caption_with_understand_music(args)

            self.assertEqual(captioned, 0)
            self.assertIn("already have sidecars", args._caption_skip_message)
            status = json.loads(args.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["phase"], "captioning-skipped")
            self.assertEqual(status["captionedCount"], 0)
            self.assertIn("overwrite captions", status["message"])

    def test_caption_quality_gate_rejects_raw_audio_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "bad.wav"
            with self.assertRaisesRegex(RuntimeError, "audio-code tokens"):
                train_lora_job.validate_caption_analysis_result(
                    audio,
                    {
                        "prompt": "A guitar groove <|audio_code_123|>",
                        "genre": "rock",
                        "lyrics": "[Instrumental]",
                    },
                )

    def test_caption_quality_gate_rejects_embedded_metadata_in_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "bad.wav"
            with self.assertRaisesRegex(RuntimeError, "embedded metadata"):
                train_lora_job.validate_caption_analysis_result(
                    audio,
                    {
                        "prompt": "A synth song duration: 90",
                        "genre": "electronic",
                        "lyrics": "[Instrumental]",
                    },
                )

    def test_caption_quality_gate_rejects_repeated_punctuation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "bad.wav"
            with self.assertRaisesRegex(RuntimeError, "repeated"):
                train_lora_job.validate_caption_analysis_result(
                    audio,
                    {
                        "prompt": "!" * 200,
                        "genre": "rock",
                        "lyrics": "[Instrumental]",
                    },
                )

    def test_caption_quality_gate_allows_legacy_lyrics_only_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "legacy.wav"
            train_lora_job.validate_caption_analysis_result(
                audio,
                {"lyrics": "human written lyrics"},
                require_caption=False,
            )

    def test_request_valid_music_analysis_retries_excerpt_after_bad_full_track(self) -> None:
        args = argparse.Namespace(
            caption_window_seconds=0.0,
            caption_fallback_window_seconds=120.0,
        )
        audio = Path("song.wav")
        bad = {"prompt": "!" * 200, "genre": "rock", "lyrics": "[Instrumental]"}
        good = {
            "prompt": "An energetic instrumental rock track with bright synth melodies.",
            "genre": "rock",
            "lyrics": "[Instrumental]",
        }
        original = train_lora_job.request_music_analysis
        mocked_request = Mock(side_effect=[bad, good])
        try:
            train_lora_job.request_music_analysis = mocked_request
            result = train_lora_job.request_valid_music_analysis(args, Mock(), audio)
        finally:
            train_lora_job.request_music_analysis = original

        self.assertEqual(result, good)
        windows = [
            call.kwargs["caption_window_seconds"]
            for call in mocked_request.call_args_list
        ]
        self.assertEqual(windows, [0.0, 120.0])

    def test_ensure_carey_model_loaded_calls_load_for_no_init_backend(self) -> None:
        args = argparse.Namespace(
            carey_url="http://127.0.0.1:8011",
            model="base",
            model_load_timeout=900.0,
        )
        health = Mock()
        health.json.return_value = {
            "data": {"initialized": False, "current_model": None}
        }
        health.raise_for_status.return_value = None
        loaded = Mock()
        loaded.json.return_value = {"code": 200, "data": {"status": "loaded"}}
        loaded.raise_for_status.return_value = None
        client = Mock()
        client.get.return_value = health
        client.post.return_value = loaded

        train_lora_job.ensure_carey_model_loaded(args, client)

        client.post.assert_called_once_with(
            "http://127.0.0.1:8011/v1/load",
            params={"config_path": "acestep-v15-base"},
            timeout=900.0,
        )

    def test_dry_run_prepares_dataset_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "audio"
            dataset.mkdir()
            audio = dataset / "demo.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(b"\x00\x00" * 800)

            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "status.json").write_text(
                json.dumps({"status": "failed", "error": "stale failure"}),
                encoding="utf-8",
            )
            argv = [
                "--job-id",
                "demo-1",
                "--name",
                "Demo",
                "--dataset-dir",
                str(dataset),
                "--checkpoint-dir",
                str(root / "checkpoints"),
                "--run-dir",
                str(run_dir),
                "--status-path",
                str(run_dir / "status.json"),
                "--current-job-path",
                str(root / "current.json"),
                "--log-path",
                str(root / "demo.log"),
                "--dry-run",
            ]
            self.assertEqual(train_lora_job.main(argv), 0)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            plan = json.loads((run_dir / "training_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(status["phase"], "prepared")
            self.assertEqual(status["sampleCount"], 1)
            self.assertIsNone(status["error"])
            self.assertFalse(plan["fisher"])
            self.assertEqual(plan["optimizer"], "adamw")
            self.assertEqual(plan["quantization"], "disabled")
            self.assertEqual(
                Path(plan["tensorsDir"]).parent,
                Path(plan["outputDir"]),
            )


if __name__ == "__main__":
    unittest.main()
