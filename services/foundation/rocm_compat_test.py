"""Tests for Foundation-1's Windows ROCm distributed compatibility hook."""

import unittest
from types import SimpleNamespace

from rocm_compat import install_windows_rocm_distributed_fallback


def fake_torch(*, hip="7.2", distributed=None):
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip),
        distributed=distributed if distributed is not None else SimpleNamespace(),
    )


class WindowsRocmDistributedFallbackTests(unittest.TestCase):
    def test_installs_reduce_op_and_single_process_all_reduce(self):
        torch_module = fake_torch()

        installed = install_windows_rocm_distributed_fallback(
            torch_module, os_name="nt"
        )

        self.assertTrue(installed)
        self.assertEqual(torch_module.distributed.ReduceOp.AVG, "avg")
        sentinel = object()
        self.assertIs(torch_module.distributed.all_reduce(sentinel), sentinel)

    def test_refuses_multi_process_collectives(self):
        distributed = SimpleNamespace(
            is_initialized=lambda: True,
            get_world_size=lambda *args, **kwargs: 2,
        )
        torch_module = fake_torch(distributed=distributed)
        install_windows_rocm_distributed_fallback(torch_module, os_name="nt")

        with self.assertRaisesRegex(RuntimeError, "only supports one process"):
            distributed.all_reduce(object())

    def test_replaces_collective_when_reduce_op_is_missing(self):
        original_all_reduce = object()
        distributed = SimpleNamespace(all_reduce=original_all_reduce)
        torch_module = fake_torch(distributed=distributed)

        install_windows_rocm_distributed_fallback(torch_module, os_name="nt")

        self.assertIsNot(distributed.all_reduce, original_all_reduce)
        self.assertEqual(distributed.ReduceOp.AVG, "avg")

    def test_leaves_complete_distributed_module_untouched(self):
        reduce_op = SimpleNamespace(AVG=object())
        all_reduce = object()
        distributed = SimpleNamespace(ReduceOp=reduce_op, all_reduce=all_reduce)
        torch_module = fake_torch(distributed=distributed)

        installed = install_windows_rocm_distributed_fallback(
            torch_module, os_name="nt"
        )

        self.assertFalse(installed)
        self.assertIs(distributed.ReduceOp, reduce_op)
        self.assertIs(distributed.all_reduce, all_reduce)

    def test_does_not_patch_non_rocm_or_non_windows_runtime(self):
        no_hip = fake_torch(hip=None)
        linux_rocm = fake_torch()

        self.assertFalse(
            install_windows_rocm_distributed_fallback(no_hip, os_name="nt")
        )
        self.assertFalse(
            install_windows_rocm_distributed_fallback(linux_rocm, os_name="posix")
        )
        self.assertFalse(hasattr(no_hip.distributed, "ReduceOp"))
        self.assertFalse(hasattr(linux_rocm.distributed, "ReduceOp"))


if __name__ == "__main__":
    unittest.main()
