import unittest

import torch

from dataset_processing.pre_encode import (
    _per_clip_latent_rms,
    _valid_latent_length,
    encode_with_per_track_norm,
)


class _LinearEncoder:
    def encode_audio(self, audio, chunked=False):
        del chunked
        return audio * 0.5


class _Pretransform:
    def __init__(self):
        self.model = _LinearEncoder()


class PreEncodeLoudnessTests(unittest.TestCase):
    def test_valid_latent_length_rounds_up_and_clamps(self):
        self.assertEqual(_valid_latent_length(3, 8, 4), 2)
        self.assertEqual(_valid_latent_length(99, 8, 4), 4)

    def test_rms_ignores_padded_latent_region(self):
        latents = torch.tensor([[[1.0, 1.0, 50.0, 50.0]]])

        measured = _per_clip_latent_rms(latents, [2], 4)

        self.assertAlmostEqual(measured[0], 1.0)

    def test_iterative_normalization_hits_target(self):
        audio = torch.ones(2, 1, 4)
        audio[1, :, 2:] = 0

        latents, gains, pre_norm, achieved, iterations = encode_with_per_track_norm(
            _Pretransform(),
            audio,
            [4, 2],
            target_latent_rms=0.9,
            max_iters=4,
            tolerance=0.03,
            chunked=False,
        )

        self.assertEqual(latents.shape, audio.shape)
        self.assertEqual(iterations, 1)
        self.assertEqual(pre_norm, [0.5, 0.5])
        self.assertTrue(all(abs(value - 0.9) < 1e-4 for value in achieved))
        self.assertTrue(all(abs(value - 1.8) < 1e-4 for value in gains))


if __name__ == "__main__":
    unittest.main()
