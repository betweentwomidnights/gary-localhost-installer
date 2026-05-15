"""Tests for generation padding helpers."""

import unittest

import torch

from acestep.core.generation.handler.padding_utils import (
    COVER_EDGE_PAD_SAMPLES,
    COVER_EDGE_PAD_SECONDS,
    PaddingMixin,
)


class _Host(PaddingMixin):
    """Minimal host exposing target wav creation for padding tests."""

    def create_target_wavs(self, duration):
        frames = int(duration * 48000)
        return torch.zeros(2, frames)


class PaddingMixinTests(unittest.TestCase):
    """Verify task-specific source-audio padding behavior."""

    def test_cover_task_left_pads_source_audio_for_edge_trim(self):
        host = _Host()
        source = torch.arange(16, dtype=torch.float32).repeat(2, 1)

        repainting_start, repainting_end, target_wavs = host.prepare_padding_info(
            actual_batch_size=1,
            processed_src_audio=source,
            audio_duration=None,
            repainting_start=None,
            repainting_end=None,
            is_repaint_task=False,
            is_lego_task=False,
            is_cover_task=True,
            can_use_repainting=False,
        )

        self.assertIsNone(repainting_start)
        self.assertIsNone(repainting_end)
        self.assertEqual(tuple(target_wavs.shape), (1, 2, COVER_EDGE_PAD_SAMPLES + source.shape[-1]))
        self.assertTrue(torch.equal(target_wavs[0, :, :COVER_EDGE_PAD_SAMPLES], torch.zeros(2, COVER_EDGE_PAD_SAMPLES)))
        self.assertTrue(torch.equal(target_wavs[0, :, COVER_EDGE_PAD_SAMPLES:], source))

    def test_cover_edge_pad_duration_matches_sample_count(self):
        self.assertEqual(COVER_EDGE_PAD_SAMPLES, 9600)
        self.assertAlmostEqual(COVER_EDGE_PAD_SECONDS, 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
