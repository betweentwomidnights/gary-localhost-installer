"""Tests for ACE-Step Min-SNR loss weighting."""

import unittest

import torch

from acestep.training_v2.loss_weighting import flow_min_snr_weights


class FlowMinSnrWeightsTests(unittest.TestCase):
    def test_low_snr_samples_keep_full_weight(self):
        weights = flow_min_snr_weights(torch.tensor([0.5, 0.9]), gamma=5.0)

        torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_high_snr_samples_are_clamped(self):
        timestep = torch.tensor([0.1])
        weights = flow_min_snr_weights(timestep, gamma=5.0)

        expected = torch.tensor([5.0 / 81.0])
        torch.testing.assert_close(weights, expected)

    def test_extreme_timesteps_stay_finite(self):
        weights = flow_min_snr_weights(torch.tensor([0.0, 1.0]), gamma=5.0)

        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(((weights >= 0.0) & (weights <= 1.0)).all())

    def test_gamma_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            flow_min_snr_weights(torch.tensor([0.5]), gamma=0.0)


if __name__ == "__main__":
    unittest.main()
