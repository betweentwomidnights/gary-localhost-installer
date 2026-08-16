"""Tests for MelodyFlow's seed resolution.

These cover the request-facing contract. The reproducibility it buys was
measured separately against the real model: with the same seed two euler
transforms correlate at 1.000000 but are not bit-identical, because the VAE
encoder's convolutions pick nondeterministic cuDNN algorithms. Midpoint, which
skips the regularization path, did come out bit-identical. So the seed pins
the noise, not the kernels.
"""

import random
import unittest
from unittest.mock import patch

from localhost_melodyflow import AudioProcessingError, resolve_seed


class ResolveSeedTests(unittest.TestCase):
    def test_explicit_seed_is_used_as_is(self):
        self.assertEqual(resolve_seed(1234), 1234)
        self.assertEqual(resolve_seed("1234"), 1234)
        self.assertEqual(resolve_seed(0), 0)

    def test_missing_or_blank_asks_for_a_random_seed(self):
        for value in (None, "", -1, "-1"):
            with patch.object(random, "randint", return_value=4242) as randint:
                self.assertEqual(resolve_seed(value), 4242)
            randint.assert_called_once_with(0, 99999)

    def test_random_seeds_stay_in_the_range_the_ui_can_display(self):
        for _ in range(200):
            seed = resolve_seed(None)
            self.assertGreaterEqual(seed, 0)
            self.assertLessEqual(seed, 99999)

    def test_a_junk_seed_is_a_request_error_not_a_500(self):
        with self.assertRaisesRegex(AudioProcessingError, "Invalid seed"):
            resolve_seed("banana")

    def test_resolved_seed_is_always_non_negative(self):
        # torch.manual_seed rejects negatives, so nothing below zero may escape.
        for value in (-1, -99, "-5"):
            self.assertGreaterEqual(resolve_seed(value), 0)


if __name__ == "__main__":
    unittest.main()
