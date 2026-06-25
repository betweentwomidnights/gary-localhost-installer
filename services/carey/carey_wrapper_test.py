import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import carey_wrapper


class CareyWrapperLoraFamilyTest(unittest.TestCase):
    def setUp(self):
        self._original_registry = dict(carey_wrapper.LORA_REGISTRY)
        self._original_caption_pools = dict(carey_wrapper._caption_pools)

    def tearDown(self):
        carey_wrapper.LORA_REGISTRY.clear()
        carey_wrapper.LORA_REGISTRY.update(self._original_registry)
        carey_wrapper._caption_pools.clear()
        carey_wrapper._caption_pools.update(self._original_caption_pools)

    def test_registry_and_caption_pools_follow_active_model_family(self):
        registry = {
            "legacy-standard": {
                "path": "C:/loras/legacy-standard",
                "backends": ["base", "turbo"],
            },
            "standard-style": {
                "path": "C:/loras/standard-style",
                "model_family": "standard",
                "backends": ["base", "turbo"],
            },
            "xl-style": {
                "path": "C:/loras/xl-style",
                "model_family": "xl",
                "backends": ["base", "turbo"],
            },
        }
        captions = {
            "default": ["default caption"],
            "legacy-standard": ["legacy caption"],
            "standard-style": ["standard caption"],
            "xl-style": ["xl caption"],
            "orphaned-pool": ["should not be exposed"],
        }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry_path = root / "lora_registry.json"
            captions_path = root / "captions.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            captions_path.write_text(json.dumps(captions), encoding="utf-8")

            for active_family, expected_loras, expected_pools in (
                (
                    "standard",
                    {"legacy-standard", "standard-style"},
                    {"default", "legacy-standard", "standard-style"},
                ),
                ("xl", {"xl-style"}, {"default", "xl-style"}),
            ):
                with self.subTest(active_family=active_family), patch.object(
                    carey_wrapper, "LORA_REGISTRY_PATH", registry_path
                ), patch.object(
                    carey_wrapper, "CAPTIONS_PATH", captions_path
                ), patch.object(
                    carey_wrapper,
                    "_primary_runtime_family",
                    return_value=active_family,
                ):
                    carey_wrapper._load_lora_registry()
                    carey_wrapper._load_captions()

                    self.assertEqual(set(carey_wrapper.LORA_REGISTRY), expected_loras)
                    self.assertEqual(set(carey_wrapper._caption_pools), expected_pools)


class CareyWrapperModelSelectionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._original_current_model = carey_wrapper._current_model
        self._original_manage_model_lifecycle = carey_wrapper.MANAGE_MODEL_LIFECYCLE
        self._original_unload_model_after_job = carey_wrapper.UNLOAD_MODEL_AFTER_JOB

    def tearDown(self):
        carey_wrapper._current_model = self._original_current_model
        carey_wrapper.MANAGE_MODEL_LIFECYCLE = self._original_manage_model_lifecycle
        carey_wrapper.UNLOAD_MODEL_AFTER_JOB = self._original_unload_model_after_job

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

    def test_generation_request_models_expose_seed(self):
        requests = (
            carey_wrapper.LegoRequest(audio_data="audio", track_name="drums", bpm=120),
            carey_wrapper.CompleteRequest(audio_data="audio", bpm=120, audio_duration=16.0),
            carey_wrapper.CoverRequest(audio_data="audio", bpm=120, caption="dub remix"),
        )

        for request in requests:
            with self.subTest(request_type=type(request).__name__):
                self.assertEqual(request.seed, -1)
                fixed_request = request.model_copy(update={"seed": 42})
                self.assertEqual(fixed_request.seed, 42)

    def test_generation_form_data_forwards_fixed_and_random_seed(self):
        cases = (
            (
                carey_wrapper.Job(task_id="seed-lego", task_type="lego", bpm=120, duration=8.0),
                carey_wrapper.LegoRequest(
                    audio_data="audio", track_name="drums", bpm=120, caption="drum break"
                ),
            ),
            (
                carey_wrapper.Job(task_id="seed-complete", task_type="complete", bpm=120, duration=8.0),
                carey_wrapper.CompleteRequest(
                    audio_data="audio", bpm=120, audio_duration=16.0, caption="synthwave"
                ),
            ),
            (
                carey_wrapper.Job(task_id="seed-cover", task_type="cover", bpm=120, duration=8.0),
                carey_wrapper.CoverRequest(
                    audio_data="audio", bpm=120, caption="dub remix"
                ),
            ),
        )

        for job, request in cases:
            with self.subTest(task_type=job.task_type, seed="fixed"):
                fixed_data = carey_wrapper._build_form_data(
                    job, request.model_copy(update={"seed": 42}), "ignored.wav"
                )
                self.assertEqual(fixed_data["seed"], "42")
                self.assertEqual(fixed_data["use_random_seed"], "false")

            with self.subTest(task_type=job.task_type, seed="random"):
                random_data = carey_wrapper._build_form_data(job, request, "ignored.wav")
                self.assertNotIn("seed", random_data)
                self.assertEqual(random_data["use_random_seed"], "true")

    def test_extract_form_data_does_not_expose_seed_controls(self):
        job = carey_wrapper.Job(task_id="seed-extract", task_type="extract", bpm=120, duration=8.0)
        request = carey_wrapper.ExtractRequest(
            audio_data="audio", track_name="drums", bpm=120
        )

        data = carey_wrapper._build_form_data(job, request, "ignored.wav")

        self.assertNotIn("seed", data)
        self.assertNotIn("use_random_seed", data)

    def test_completed_status_returns_backend_seed(self):
        job = carey_wrapper.Job(
            task_id="seed-status",
            task_type="cover",
            bpm=120,
            status=carey_wrapper.JobStatus.COMPLETED,
            audio_b64="audio",
            seed_used="42,84",
        )

        response = carey_wrapper._build_status_response(job)
        payload = json.loads(response.body)

        self.assertEqual(payload["seed"], "42,84")

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

    def test_lego_tracks_always_use_active_base_backend(self):
        for track_name in ("vocals", "backing_vocals", "drums"):
            with self.subTest(track_name=track_name):
                self.assertEqual(
                    carey_wrapper._backend_key_for("lego", requested_model="sft", track_name=track_name),
                    "base",
                )
                self.assertEqual(
                    carey_wrapper._required_model_for_task("lego", requested_model="sft", track_name=track_name),
                    carey_wrapper.ACESTEP_LEGO_CONFIG,
                )

    def test_lego_lora_request_validates_base_backend_and_family(self):
        carey_wrapper.LORA_REGISTRY.clear()
        carey_wrapper.LORA_REGISTRY.update(
            {
                "standard-style": {
                    "path": "C:/loras/standard-style",
                    "model_family": "standard",
                    "backends": ["base"],
                },
                "turbo-only": {
                    "path": "C:/loras/turbo-only",
                    "model_family": "standard",
                    "backends": ["turbo"],
                },
                "xl-style": {
                    "path": "C:/loras/xl-style",
                    "model_family": "xl",
                    "backends": ["base"],
                },
            }
        )

        carey_wrapper._validate_lora_request(
            "lego",
            SimpleNamespace(lora="standard-style", model="sft", track_name="vocals"),
        )

        for lora_name in ("unknown-adapter", "turbo-only", "xl-style"):
            with self.subTest(lora_name=lora_name), self.assertRaises(carey_wrapper.HTTPException):
                carey_wrapper._validate_lora_request(
                    "lego",
                    SimpleNamespace(lora=lora_name, model="sft", track_name="vocals"),
                )

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

    async def test_teardown_unloads_full_model_after_completed_job(self):
        job = carey_wrapper.Job(task_id="job-5", task_type="complete", bpm=120)
        job.status = carey_wrapper.JobStatus.COMPLETED
        req = SimpleNamespace(model="turbo", track_name="")
        client = object()
        carey_wrapper.MANAGE_MODEL_LIFECYCLE = True
        carey_wrapper.UNLOAD_MODEL_AFTER_JOB = True
        carey_wrapper._current_model = carey_wrapper.ACESTEP_TURBO_CONFIG

        with patch.object(carey_wrapper, "_unload_model", AsyncMock()) as unload_model:
            await carey_wrapper._teardown_generation_resources(
                client,
                job,
                req,
                lora_config=None,
                lora_name="",
            )

        unload_model.assert_awaited_once_with(client)
        self.assertIsNone(carey_wrapper._current_model)
        self.assertEqual(job.status, carey_wrapper.JobStatus.COMPLETED)

    async def test_teardown_completed_job_survives_model_unload_failure(self):
        job = carey_wrapper.Job(task_id="job-6", task_type="complete", bpm=120)
        job.status = carey_wrapper.JobStatus.COMPLETED
        req = SimpleNamespace(model="turbo", track_name="")
        client = object()
        carey_wrapper.MANAGE_MODEL_LIFECYCLE = True
        carey_wrapper.UNLOAD_MODEL_AFTER_JOB = True
        carey_wrapper._current_model = carey_wrapper.ACESTEP_TURBO_CONFIG

        with patch.object(carey_wrapper, "_unload_model", AsyncMock(side_effect=RuntimeError("boom"))):
            await carey_wrapper._teardown_generation_resources(
                client,
                job,
                req,
                lora_config=None,
                lora_name="",
            )

        self.assertEqual(carey_wrapper._current_model, carey_wrapper.ACESTEP_TURBO_CONFIG)
        self.assertEqual(job.status, carey_wrapper.JobStatus.COMPLETED)

    async def test_teardown_unloads_lora_when_full_model_unload_disabled(self):
        job = carey_wrapper.Job(task_id="job-7", task_type="complete", bpm=120)
        job.status = carey_wrapper.JobStatus.COMPLETED
        req = SimpleNamespace(model="turbo", track_name="")
        client = object()
        carey_wrapper.MANAGE_MODEL_LIFECYCLE = True
        carey_wrapper.UNLOAD_MODEL_AFTER_JOB = False

        with patch.object(carey_wrapper, "_unload_lora", AsyncMock()) as unload_lora:
            await carey_wrapper._teardown_generation_resources(
                client,
                job,
                req,
                lora_config={"path": "adapter"},
                lora_name="adapter",
            )

        unload_lora.assert_awaited_once_with(client)

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
