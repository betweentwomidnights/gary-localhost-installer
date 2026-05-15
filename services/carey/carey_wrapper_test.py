import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import carey_wrapper


class CareyWrapperModelSelectionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._original_current_model = carey_wrapper._current_model

    def tearDown(self):
        carey_wrapper._current_model = self._original_current_model

    def test_cover_form_data_uses_turbo_generation_defaults_and_cover_nofsq(self):
        job = carey_wrapper.Job(task_id="job-1", task_type="cover", bpm=120, duration=42.0)
        req = SimpleNamespace(
            audio_data="",
            bpm=120,
            caption="orchestral version",
            lyrics="[Instrumental]",
            language="en",
            key_scale="",
            cover_noise_strength=0.2,
            audio_cover_strength=0.3,
            guidance_scale=7.5,
            inference_steps=64,
            use_src_as_ref=False,
            no_fsq=False,
            time_signature="4",
            batch_size=1,
            audio_format="wav",
        )

        data = carey_wrapper._build_form_data(job, req, "ignored.wav")

        self.assertEqual(data["guidance_scale"], str(carey_wrapper.COVER_GUIDANCE_SCALE))
        self.assertEqual(data["inference_steps"], str(carey_wrapper.COVER_INFERENCE_STEPS))
        self.assertEqual(data["task_type"], "cover-nofsq")

    def test_cover_form_data_no_fsq_flag_is_backcompat_noop(self):
        job = carey_wrapper.Job(task_id="job-1b", task_type="cover", bpm=120, duration=42.0)
        for no_fsq in (False, True):
            with self.subTest(no_fsq=no_fsq):
                req = SimpleNamespace(
                    audio_data="",
                    bpm=120,
                    caption="orchestral version",
                    lyrics="[Instrumental]",
                    language="en",
                    key_scale="",
                    cover_noise_strength=0.2,
                    audio_cover_strength=0.3,
                    guidance_scale=1.0,
                    inference_steps=8,
                    use_src_as_ref=False,
                    no_fsq=no_fsq,
                    time_signature="4",
                    batch_size=1,
                    audio_format="wav",
                )

                data = carey_wrapper._build_form_data(job, req, "ignored.wav")

                self.assertEqual(data["task_type"], "cover-nofsq")

    def test_lego_form_data_uses_simple_fallback_caption_pool_when_empty(self):
        job = carey_wrapper.Job(task_id="job-1c", task_type="lego", bpm=120, duration=8.0)
        req = SimpleNamespace(
            bpm=120,
            caption="  ",
            lyrics="",
            language="en",
            key_scale="",
            guidance_scale=7.0,
            inference_steps=50,
            track_name="brass",
            time_signature="4",
            batch_size=1,
            audio_format="wav",
        )

        with patch.object(carey_wrapper.random, "choice", return_value="jazzy trumpet solo") as choice_mock:
            data = carey_wrapper._build_form_data(job, req, "ignored.wav")

        choice_mock.assert_called_once_with(carey_wrapper.TRACK_CAPTION_POOLS["brass"])
        self.assertEqual(data["caption"], "jazzy trumpet solo")

    def test_lego_form_data_preserves_explicit_caption(self):
        job = carey_wrapper.Job(task_id="job-1d", task_type="lego", bpm=120, duration=8.0)
        req = SimpleNamespace(
            bpm=120,
            caption="bright flute hook",
            lyrics="",
            language="en",
            key_scale="",
            guidance_scale=7.0,
            inference_steps=50,
            track_name="woodwinds",
            time_signature="4",
            batch_size=1,
            audio_format="wav",
        )

        with patch.object(carey_wrapper.random, "choice") as choice_mock:
            data = carey_wrapper._build_form_data(job, req, "ignored.wav")

        choice_mock.assert_not_called()
        self.assertEqual(data["caption"], "bright flute hook")

    def test_lego_caption_pools_are_short_and_complete(self):
        self.assertEqual(set(carey_wrapper.TRACK_CAPTION_POOLS), carey_wrapper.ALLOWED_TRACKS)
        for track_name, captions in carey_wrapper.TRACK_CAPTION_POOLS.items():
            with self.subTest(track_name=track_name):
                self.assertGreaterEqual(len(captions), 4)
                self.assertEqual(carey_wrapper.TRACK_CAPTIONS[track_name], captions[0])
                for caption in captions:
                    self.assertLessEqual(len(caption.split()), 5)

    def test_complete_form_data_preserves_requested_generation_values(self):
        job = carey_wrapper.Job(task_id="job-2", task_type="complete", bpm=120, target_duration=64.0)
        req = SimpleNamespace(
            audio_data="",
            bpm=120,
            audio_duration=64.0,
            caption="anthemic synthwave",
            lyrics="",
            language="en",
            key_scale="C minor",
            guidance_scale=6.5,
            inference_steps=32,
            use_src_as_ref=True,
            time_signature="4",
            batch_size=1,
            audio_format="wav",
        )

        data = carey_wrapper._build_form_data(job, req, "ignored.wav")

        self.assertEqual(data["guidance_scale"], "6.5")
        self.assertEqual(data["inference_steps"], "32")
        self.assertEqual(data["key_scale"], "C minor")

    async def test_ensure_required_model_swaps_only_when_needed(self):
        cover_job = carey_wrapper.Job(task_id="job-3", task_type="cover", bpm=120)
        carey_wrapper._current_model = carey_wrapper.ACESTEP_BASE_CONFIG
        client = object()
        req = SimpleNamespace(model="xl-turbo", track_name="")

        with patch.object(carey_wrapper, "_unload_model", AsyncMock()) as unload_mock, patch.object(
            carey_wrapper,
            "_load_model",
            AsyncMock(return_value=carey_wrapper.ACESTEP_TURBO_CONFIG),
        ) as load_mock:
            await carey_wrapper._ensure_required_model(client=client, job=cover_job, req=req)

        unload_mock.assert_awaited_once()
        load_mock.assert_awaited_once_with(client, carey_wrapper.ACESTEP_TURBO_CONFIG)
        self.assertEqual(carey_wrapper._current_model, carey_wrapper.ACESTEP_TURBO_CONFIG)
        self.assertEqual(cover_job.status, carey_wrapper.JobStatus.LOADING)

    async def test_ensure_required_model_skips_swap_when_model_matches(self):
        lego_job = carey_wrapper.Job(task_id="job-4", task_type="lego", bpm=120)
        carey_wrapper._current_model = carey_wrapper.ACESTEP_BASE_CONFIG
        client = object()
        req = SimpleNamespace(model="base", track_name="drums")

        with patch.object(carey_wrapper, "_unload_model", AsyncMock()) as unload_mock, patch.object(
            carey_wrapper,
            "_load_model",
            AsyncMock(),
        ) as load_mock:
            await carey_wrapper._ensure_required_model(client=client, job=lego_job, req=req)

        unload_mock.assert_not_called()
        load_mock.assert_not_called()
        self.assertEqual(carey_wrapper._current_model, carey_wrapper.ACESTEP_BASE_CONFIG)

    def test_lego_tracks_always_use_acestep_base_backend(self):
        for track_name in ("vocals", "backing_vocals", "drums"):
            with self.subTest(track_name=track_name):
                self.assertEqual(
                    carey_wrapper._backend_key_for("lego", requested_model="sft", track_name=track_name),
                    "regular",
                )
                self.assertEqual(
                    carey_wrapper._required_model_for_task("lego", requested_model="sft", track_name=track_name),
                    carey_wrapper.ACESTEP_LEGO_CONFIG,
                )
                self.assertEqual(carey_wrapper.ACESTEP_LEGO_CONFIG, "acestep-v15-base")

    def test_lego_lora_request_is_ignored(self):
        req = SimpleNamespace(lora="unknown-adapter", model="sft", track_name="vocals")

        carey_wrapper._validate_lora_request("lego", req)

    async def test_ensure_required_model_swaps_vocal_lego_from_sft_to_base(self):
        lego_job = carey_wrapper.Job(task_id="job-4b", task_type="lego", bpm=120)
        carey_wrapper._current_model = carey_wrapper.ACESTEP_SFT_CONFIG
        client = object()
        req = SimpleNamespace(model="sft", track_name="vocals")

        with patch.object(carey_wrapper, "_unload_model", AsyncMock()) as unload_mock, patch.object(
            carey_wrapper,
            "_load_model",
            AsyncMock(return_value=carey_wrapper.ACESTEP_LEGO_CONFIG),
        ) as load_mock:
            await carey_wrapper._ensure_required_model(client=client, job=lego_job, req=req)

        unload_mock.assert_awaited_once()
        load_mock.assert_awaited_once_with(client, carey_wrapper.ACESTEP_LEGO_CONFIG)
        self.assertEqual(carey_wrapper._current_model, carey_wrapper.ACESTEP_LEGO_CONFIG)

    def test_sync_checkpoint_overrides_copies_python_files_into_runtime_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overrides_dir = root / "checkpoint_overrides" / "acestep-v15-turbo"
            checkpoints_dir = root / "checkpoints" / "acestep-v15-turbo"
            overrides_dir.mkdir(parents=True)
            checkpoints_dir.mkdir(parents=True)

            src = overrides_dir / "modeling_acestep_v15_turbo.py"
            dst = checkpoints_dir / "modeling_acestep_v15_turbo.py"
            src.write_text("patched = True\n", encoding="utf-8")
            dst.write_text("patched = False\n", encoding="utf-8")

            synced = carey_wrapper._sync_checkpoint_overrides(root)

            self.assertEqual(synced, [dst])
            self.assertEqual(dst.read_text(encoding="utf-8"), "patched = True\n")


class CareyWrapperProgressResolutionTest(unittest.TestCase):
    def test_resolve_generation_progress_reads_nested_acestep_progress(self):
        result = {
            "progress_text": "Generating music (batch size: 1)...",
            "result": '[{"progress": 0.63, "stage": "Generating music (batch size: 1)..."}]',
        }

        progress, label = carey_wrapper._resolve_generation_progress(result)

        self.assertEqual(progress, 63)
        self.assertEqual(label, "Generating music (batch size: 1)...")

    def test_resolve_generation_progress_maps_decode_chunks_into_decode_band(self):
        result = {
            "progress_text": "Decoding audio chunks: 29/58",
            "result": '[{"progress": 0.8, "stage": "Decoding audio..."}]',
        }

        progress, label = carey_wrapper._resolve_generation_progress(result)

        self.assertEqual(progress, 87)
        self.assertEqual(label, "Decoding audio... (29/58)")

    def test_resolve_generation_progress_uses_finalize_floor(self):
        result = {
            "progress_text": "Preparing audio data...",
            "result": '[{"progress": 0.8, "stage": "Preparing audio data..."}]',
        }

        progress, label = carey_wrapper._resolve_generation_progress(result)

        self.assertEqual(progress, 96)
        self.assertEqual(label, "Preparing audio data...")


if __name__ == "__main__":
    unittest.main()
