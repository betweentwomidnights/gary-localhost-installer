"""Unit tests for torch.distributed cleanup in ``LLMHandler``."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from acestep.llm_inference import LLMHandler, resolve_lm_backend
    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - dependency guard
    LLMHandler = None
    resolve_lm_backend = None
    _IMPORT_ERROR = exc


@unittest.skipIf(LLMHandler is None, f"llm_inference import unavailable: {_IMPORT_ERROR}")
class LlmDistributedCleanupTests(unittest.TestCase):
    """Verify process-group cleanup helper avoids double initialization issues."""

    def test_cleanup_destroys_initialized_process_group(self):
        """Cleanup should call destroy when torch.distributed is initialized."""
        handler = LLMHandler()
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.destroy_process_group") as destroy_mock:
            handler._cleanup_torch_distributed_state()
        destroy_mock.assert_called_once()

    def test_cleanup_is_noop_when_not_initialized(self):
        """Cleanup should not destroy when process group is not initialized."""
        handler = LLMHandler()
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=False
        ), patch("torch.distributed.destroy_process_group") as destroy_mock:
            handler._cleanup_torch_distributed_state()
        destroy_mock.assert_not_called()


@unittest.skipIf(resolve_lm_backend is None, f"llm_inference import unavailable: {_IMPORT_ERROR}")
class LlmBackendSelectionTests(unittest.TestCase):
    @staticmethod
    def fake_torch(*, hip=None):
        return SimpleNamespace(version=SimpleNamespace(hip=hip))

    def test_rocm_uses_pytorch_even_though_device_is_named_cuda(self):
        self.assertEqual(
            resolve_lm_backend("vllm", "cuda", self.fake_torch(hip="7.2")),
            "pt",
        )

    def test_nvidia_cuda_keeps_vllm(self):
        self.assertEqual(
            resolve_lm_backend("vllm", "cuda", self.fake_torch(hip=None)),
            "vllm",
        )

    def test_non_cuda_device_uses_pytorch(self):
        self.assertEqual(
            resolve_lm_backend("vllm", "cpu", self.fake_torch(hip=None)),
            "pt",
        )

    def test_explicit_pytorch_backend_is_unchanged(self):
        self.assertEqual(
            resolve_lm_backend("pt", "cuda", self.fake_torch(hip="7.2")),
            "pt",
        )


if __name__ == "__main__":
    unittest.main()
