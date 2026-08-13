"""Tests for conservative single-device training choices on Windows ROCm."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.training_v2 import optim
from acestep.training_v2.trainer_fixed import (
    is_windows_rocm_runtime,
    should_use_fabric,
)


def fake_torch(*, hip=None):
    return SimpleNamespace(version=SimpleNamespace(hip=hip))


class RocmTrainingSelectionTests(unittest.TestCase):
    def test_detects_windows_rocm_before_fabric_setup(self):
        self.assertTrue(
            is_windows_rocm_runtime(
                platform="win32",
                torch_module=fake_torch(hip="7.2"),
            )
        )

    def test_windows_rocm_uses_direct_pytorch_loop(self):
        self.assertFalse(
            should_use_fabric(
                fabric_available=True,
                platform="win32",
                torch_module=fake_torch(hip="7.2"),
            )
        )

    def test_windows_cuda_keeps_fabric(self):
        self.assertTrue(
            should_use_fabric(
                fabric_available=True,
                platform="win32",
                torch_module=fake_torch(hip=None),
            )
        )

    def test_linux_rocm_keeps_fabric(self):
        self.assertTrue(
            should_use_fabric(
                fabric_available=True,
                platform="linux",
                torch_module=fake_torch(hip="7.2"),
            )
        )

    def test_missing_fabric_uses_direct_loop_everywhere(self):
        self.assertFalse(
            should_use_fabric(
                fabric_available=False,
                platform="linux",
                torch_module=fake_torch(hip=None),
            )
        )

    def test_rocm_adamw_does_not_request_fused_optimizer(self):
        sentinel = object()
        with patch.object(optim.torch.version, "hip", "7.2"), patch.object(
            optim, "AdamW", return_value=sentinel
        ) as adamw:
            result = optim.build_optimizer([object()], device_type="cuda")

        self.assertIs(result, sentinel)
        self.assertNotIn("fused", adamw.call_args.kwargs)

    def test_cuda_adamw_keeps_fused_optimizer(self):
        sentinel = object()
        with patch.object(optim.torch.version, "hip", None), patch.object(
            optim, "AdamW", return_value=sentinel
        ) as adamw:
            result = optim.build_optimizer([object()], device_type="cuda")

        self.assertIs(result, sentinel)
        self.assertTrue(adamw.call_args.kwargs["fused"])


if __name__ == "__main__":
    unittest.main()
