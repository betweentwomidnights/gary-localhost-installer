import math
import tempfile
import types
import unittest
from pathlib import Path

import torch

from api import (
    apply_loudness_chain,
    cached_hf_snapshot_file,
    continuation_mask_bounds,
    continuation_splice_params,
    splice_continuation_source,
)


class _FixedDecoder(torch.nn.Module):
    def __init__(self, audio: torch.Tensor):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.audio = audio

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.audio.to(device=latents.device, dtype=latents.dtype).clone()


def _pipe_with_audio(audio: torch.Tensor):
    pretransform = _FixedDecoder(audio)
    return types.SimpleNamespace(
        model=types.SimpleNamespace(pretransform=pretransform)
    )


class ContinuationSpliceTests(unittest.TestCase):
    def test_finds_decoder_file_in_commit_pinned_snapshot_without_main_ref(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = (
                Path(temp_dir)
                / "models--thepatch--same-l-decoder-lora"
                / "snapshots"
                / "commit-sha"
                / "squeakfix_v3.safetensors"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"adapter")

            resolved = cached_hf_snapshot_file(
                "thepatch/same-l-decoder-lora",
                "squeakfix_v3.safetensors",
                temp_dir,
            )

            self.assertEqual(resolved, str(checkpoint.resolve()))

    def test_request_boolean_controls_accept_explicit_off(self):
        parsed = continuation_splice_params(
            {
                "splice_source": "off",
                "splice_xfade": "0.125",
                "splice_gain_match": 0,
                "mask_overlap": "0.2",
            }
        )

        self.assertFalse(parsed["splice_source"])
        self.assertEqual(parsed["splice_xfade"], 0.125)
        self.assertFalse(parsed["splice_gain_match"])
        self.assertEqual(parsed["mask_overlap"], 0.2)

    def test_mask_overlap_moves_regeneration_into_source(self):
        overlap, mask_start = continuation_mask_bounds(8.0, 0.2)

        self.assertAlmostEqual(overlap, 0.2)
        self.assertAlmostEqual(mask_start, 7.8)

    def test_mask_overlap_is_clamped_for_short_sources(self):
        overlap, mask_start = continuation_mask_bounds(0.1, 0.2)

        self.assertAlmostEqual(overlap, 0.05)
        self.assertAlmostEqual(mask_start, 0.05)

    def test_mask_overlap_never_consumes_a_source_shorter_than_guard(self):
        overlap, mask_start = continuation_mask_bounds(0.03, 0.2)

        self.assertEqual(overlap, 0.0)
        self.assertAlmostEqual(mask_start, 0.03)

    def test_restores_source_head_and_leaves_generated_tail_untouched(self):
        generated = torch.arange(12, dtype=torch.float32).view(1, 1, -1)
        source = torch.full((1, 8), -2.0)

        output, meta = splice_continuation_source(
            generated,
            (10, source),
            mask_start_seconds=0.8,
            sample_rate=10,
            xfade_seconds=0.2,
            gain_match=False,
        )

        torch.testing.assert_close(output[..., :6], source[..., :6].unsqueeze(0))
        torch.testing.assert_close(output[..., 8:], generated[..., 8:])
        self.assertTrue(meta["splice_applied"])
        self.assertEqual(meta["splice_end_seconds"], 0.8)
        self.assertEqual(meta["splice_xfade_applied"], 0.2)
        self.assertEqual(meta["splice_gain"], 1.0)

    def test_gain_match_is_clamped_and_mono_source_expands_to_stereo(self):
        generated = torch.ones(1, 2, 8)
        source = torch.full((1, 8), 0.01)

        output, meta = splice_continuation_source(
            generated,
            (8, source),
            mask_start_seconds=1.0,
            sample_rate=8,
            xfade_seconds=0,
            gain_match=True,
        )

        self.assertEqual(output.shape, generated.shape)
        torch.testing.assert_close(output, torch.full_like(generated, 0.04))
        self.assertEqual(meta["splice_gain"], 4.0)

    def test_gain_match_does_not_attenuate_source_for_a_silent_model_head(self):
        generated = torch.zeros(1, 1, 8)
        source = torch.ones(1, 8)

        output, meta = splice_continuation_source(
            generated,
            (8, source),
            mask_start_seconds=1.0,
            sample_rate=8,
            xfade_seconds=0,
            gain_match=True,
        )

        torch.testing.assert_close(output, torch.ones_like(generated))
        self.assertEqual(meta["splice_gain"], 1.0)

    def test_equal_power_fade_meets_generated_audio_at_seam(self):
        sample_rate = 1000
        splice_end = 800
        timeline = torch.arange(1200, dtype=torch.float32) / sample_rate
        generated = torch.sin(2 * math.pi * 7 * timeline).view(1, 1, -1)
        source = (generated[..., :splice_end] * 0.5).squeeze(0)

        output, meta = splice_continuation_source(
            generated,
            (sample_rate, source),
            mask_start_seconds=splice_end / sample_rate,
            sample_rate=sample_rate,
            xfade_seconds=0.03,
            gain_match=True,
        )

        self.assertEqual(meta["splice_xfade_applied"], 0.03)
        self.assertAlmostEqual(meta["splice_gain"], 2.0, places=5)
        self.assertAlmostEqual(output[0, 0, splice_end - 1].item(), generated[0, 0, splice_end - 1].item(), places=5)
        torch.testing.assert_close(output[..., splice_end:], generated[..., splice_end:])

    def test_one_sample_fade_uses_generated_sample_at_seam(self):
        generated = torch.arange(6, dtype=torch.float32).view(1, 1, -1)
        source = torch.full((1, 4), -5.0)

        output, _ = splice_continuation_source(
            generated,
            (10, source),
            mask_start_seconds=0.4,
            sample_rate=10,
            xfade_seconds=0.1,
            gain_match=False,
        )

        self.assertAlmostEqual(output[0, 0, 3].item(), generated[0, 0, 3].item())
        torch.testing.assert_close(output[..., 4:], generated[..., 4:])

    def test_splice_runs_before_peak_normalization(self):
        decoded = torch.cat(
            [torch.full((1, 1, 4), 0.5), torch.ones(1, 1, 4)], dim=-1
        )
        params = {
            "mode": "continue",
            "duration": 0.8,
            "target_samples": 8,
            "inpaint_audio": (10, torch.full((1, 4), 0.25)),
            "inpaint_mask_start_seconds": 0.4,
            "continue": {},
            "splice_source": True,
            "splice_xfade": 0.0,
            "splice_gain_match": True,
            "latent_rescale": 1.0,
            "latent_shift": 0.0,
            "latent_target_std": None,
            "latent_adapt_min": 0.9,
            "latent_adapt_max": 1.0,
            "peak_normalize_db": -6.020599913279624,
            "limiter_ceiling_db": None,
            "limiter_knee": 0.8,
        }

        output, loudness = apply_loudness_chain(
            _pipe_with_audio(decoded),
            torch.zeros(1, 1, 1),
            params,
            sample_rate=10,
            session_id="test",
        )

        torch.testing.assert_close(output[..., :4], torch.full((1, 1, 4), 0.25))
        torch.testing.assert_close(output[..., 4:], torch.full((1, 1, 4), 0.5))
        self.assertEqual(loudness["decoded_peak"], 1.0)
        self.assertEqual(loudness["peak_normalize_gain"], 0.5)
        self.assertTrue(params["continue"]["splice_applied"])
        self.assertEqual(params["continue"]["splice_gain"], 2.0)


if __name__ == "__main__":
    unittest.main()
