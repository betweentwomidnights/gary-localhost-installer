"""Regression tests for numerical trainer integrity guards."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from acestep.training_v2.trainer_helpers import (
    ensure_finite_gradients,
    verify_saved_adapter,
)


class TrainerIntegrityTests(unittest.TestCase):
    def test_nonfinite_gradient_is_rejected_before_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = torch.tensor([1.0, float("nan")])

        with self.assertRaisesRegex(FloatingPointError, "non-finite gradients"):
            ensure_finite_gradients([parameter])

    def test_nonfinite_saved_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            save_file(
                {"adapter.weight": torch.tensor([1.0, float("inf")])},
                str(output_dir / "adapter_model.safetensors"),
            )
            (output_dir / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA"}), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
                verify_saved_adapter(str(output_dir))

    def test_finite_saved_adapter_passes_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            save_file(
                {"adapter.weight": torch.tensor([0.0, 1.0])},
                str(output_dir / "adapter_model.safetensors"),
            )
            (output_dir / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA"}), encoding="utf-8"
            )

            verify_saved_adapter(str(output_dir))


if __name__ == "__main__":
    unittest.main()
