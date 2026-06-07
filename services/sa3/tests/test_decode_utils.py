import unittest

import torch

from stable_audio_3.inference.decode_utils import align_latents_for_decode


class DecodeAlignmentTests(unittest.TestCase):
    def test_aligns_latents_to_decoder_dtype_and_device(self):
        pretransform = torch.nn.Linear(2, 2).to(dtype=torch.float16)
        latents = torch.ones(1, 2, dtype=torch.float32)
        casts = []

        aligned = align_latents_for_decode(
            latents,
            pretransform,
            on_cast=lambda *details: casts.append(details),
        )

        self.assertEqual(aligned.dtype, torch.float16)
        self.assertEqual(aligned.device, next(pretransform.parameters()).device)
        self.assertEqual(len(casts), 1)

    def test_keeps_already_aligned_latents(self):
        pretransform = torch.nn.Linear(2, 2).to(dtype=torch.float16)
        latents = torch.ones(1, 2, dtype=torch.float16)

        aligned = align_latents_for_decode(latents, pretransform)

        self.assertIs(aligned, latents)


if __name__ == "__main__":
    unittest.main()
