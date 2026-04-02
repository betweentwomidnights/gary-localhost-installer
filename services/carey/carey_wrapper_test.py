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

    def test_cover_form_data_uses_turbo_generation_defaults(self):
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
            time_signature="4",
            batch_size=1,
            audio_format="wav",
        )

        data = carey_wrapper._build_form_data(job, req, "ignored.wav")

        self.assertEqual(data["guidance_scale"], str(carey_wrapper.COVER_GUIDANCE_SCALE))
        self.assertEqual(data["inference_steps"], str(carey_wrapper.COVER_INFERENCE_STEPS))

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

        with patch.object(carey_wrapper, "_unload_model", AsyncMock()) as unload_mock, patch.object(
            carey_wrapper,
            "_load_model",
            AsyncMock(return_value=carey_wrapper.ACESTEP_TURBO_CONFIG),
        ) as load_mock:
            await carey_wrapper._ensure_required_model(client=client, job=cover_job)

        unload_mock.assert_awaited_once()
        load_mock.assert_awaited_once_with(client, carey_wrapper.ACESTEP_TURBO_CONFIG)
        self.assertEqual(carey_wrapper._current_model, carey_wrapper.ACESTEP_TURBO_CONFIG)
        self.assertEqual(cover_job.status, carey_wrapper.JobStatus.LOADING)

    async def test_ensure_required_model_skips_swap_when_model_matches(self):
        lego_job = carey_wrapper.Job(task_id="job-4", task_type="lego", bpm=120)
        carey_wrapper._current_model = carey_wrapper.ACESTEP_BASE_CONFIG
        client = object()

        with patch.object(carey_wrapper, "_unload_model", AsyncMock()) as unload_mock, patch.object(
            carey_wrapper,
            "_load_model",
            AsyncMock(),
        ) as load_mock:
            await carey_wrapper._ensure_required_model(client=client, job=lego_job)

        unload_mock.assert_not_called()
        load_mock.assert_not_called()
        self.assertEqual(carey_wrapper._current_model, carey_wrapper.ACESTEP_BASE_CONFIG)

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
