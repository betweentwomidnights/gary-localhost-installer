"""Unit tests for cover/lego-specific conditioning mask behavior."""

import unittest

import torch

from acestep.core.generation.handler.conditioning_masks import ConditioningMaskMixin


class _Host(ConditioningMaskMixin):
    """Minimal host exposing device/sample rate for conditioning-mask tests."""

    def __init__(self):
        self.device = "cpu"
        self.sample_rate = 48000


class ConditioningMaskMixinTests(unittest.TestCase):
    """Validate task-type-sensitive cover and lego conditioning branches."""

    def test_cover_nofsq_marks_full_track_as_non_cover(self):
        """Cover-nofsq should bypass the FSQ cover flag and keep raw VAE latents."""
        host = _Host()
        chunk_masks, spans, is_covers, src_latents = host._build_chunk_masks_and_src_latents(
            batch_size=1,
            max_latent_length=8,
            instructions=["Generate audio semantic tokens based on the given conditions:"],
            audio_code_hints=[None],
            target_wavs=torch.ones(1, 2, 9600),
            target_latents=torch.ones(1, 8, 4),
            repainting_start=None,
            repainting_end=None,
            silence_latent_tiled=torch.zeros(8, 4),
            task_type="cover-nofsq",
        )

        self.assertEqual(chunk_masks.shape, (1, 8))
        self.assertEqual(spans, [("full", 0, 8)])
        self.assertFalse(bool(is_covers[0].item()))
        self.assertTrue(torch.equal(src_latents, torch.ones(1, 8, 4)))

    def test_lego_repaint_preserves_source_latents_inside_generation_span(self):
        """Lego should keep source latents intact instead of silencing the repaint span."""
        host = _Host()
        target_latents = torch.ones(1, 16, 2)
        silence_latent = torch.zeros(16, 2)
        _, spans, _, src_latents = host._build_chunk_masks_and_src_latents(
            batch_size=1,
            max_latent_length=16,
            instructions=["Generate the DRUMS track based on the audio context:"],
            audio_code_hints=[None],
            target_wavs=torch.ones(1, 2, 19200),
            target_latents=target_latents,
            repainting_start=[0.0],
            repainting_end=[0.32],
            silence_latent_tiled=silence_latent,
            task_type="lego",
        )

        _, start_latent, end_latent = spans[0]
        self.assertTrue(torch.equal(src_latents[0, start_latent:end_latent], target_latents[0, start_latent:end_latent]))

    def test_non_lego_repaint_silences_generation_span(self):
        """Regular repaint-style tasks should still zero the generated span in src latents."""
        host = _Host()
        target_latents = torch.ones(1, 16, 2)
        silence_latent = torch.zeros(16, 2)
        _, spans, _, src_latents = host._build_chunk_masks_and_src_latents(
            batch_size=1,
            max_latent_length=16,
            instructions=["Repaint the mask area based on the given conditions:"],
            audio_code_hints=[None],
            target_wavs=torch.ones(1, 2, 19200),
            target_latents=target_latents,
            repainting_start=[0.0],
            repainting_end=[0.32],
            silence_latent_tiled=silence_latent,
            task_type="repaint",
        )

        _, start_latent, end_latent = spans[0]
        self.assertTrue(torch.equal(src_latents[0, start_latent:end_latent], silence_latent[start_latent:end_latent]))


if __name__ == "__main__":
    unittest.main()
