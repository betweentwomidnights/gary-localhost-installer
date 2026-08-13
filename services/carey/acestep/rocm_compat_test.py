"""Tests for the Windows ROCm torch.distributed compatibility hook."""

import unittest
from types import SimpleNamespace

from acestep.rocm_compat import install_windows_rocm_distributed_fallback


def fake_torch(*, hip=None, distributed=None):
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip),
        distributed=distributed if distributed is not None else SimpleNamespace(),
    )


class WindowsRocmDistributedCompatibilityTests(unittest.TestCase):
    def test_installs_single_process_fallback_for_incomplete_rocm_build(self):
        torch_module = fake_torch(hip="7.2")

        installed = install_windows_rocm_distributed_fallback(
            torch_module,
            os_name="nt",
        )

        self.assertTrue(installed)
        self.assertFalse(torch_module.distributed.is_initialized())
        self.assertEqual(torch_module.distributed.get_world_size(), 1)
        sentinel = object()
        self.assertIs(torch_module.distributed.nn.all_reduce(sentinel), sentinel)

    def test_fallback_rejects_multi_process_all_reduce(self):
        distributed = SimpleNamespace(
            is_initialized=lambda: True,
            get_world_size=lambda: 2,
        )
        torch_module = fake_torch(hip="7.2", distributed=distributed)
        install_windows_rocm_distributed_fallback(torch_module, os_name="nt")

        with self.assertRaisesRegex(RuntimeError, "only supports one process"):
            distributed.nn.all_reduce(object())

    def test_install_is_idempotent(self):
        torch_module = fake_torch(hip="7.2")

        self.assertTrue(
            install_windows_rocm_distributed_fallback(torch_module, os_name="nt")
        )
        self.assertFalse(
            install_windows_rocm_distributed_fallback(torch_module, os_name="nt")
        )

    def test_cuda_build_is_unchanged(self):
        torch_module = fake_torch(hip=None)

        installed = install_windows_rocm_distributed_fallback(
            torch_module,
            os_name="nt",
        )

        self.assertFalse(installed)
        self.assertFalse(hasattr(torch_module.distributed, "nn"))

    def test_complete_rocm_distributed_build_is_unchanged(self):
        distributed = SimpleNamespace(group=object(), ReduceOp=object())
        torch_module = fake_torch(hip="7.2", distributed=distributed)

        installed = install_windows_rocm_distributed_fallback(
            torch_module,
            os_name="nt",
        )

        self.assertFalse(installed)
        self.assertFalse(hasattr(distributed, "nn"))


if __name__ == "__main__":
    unittest.main()
