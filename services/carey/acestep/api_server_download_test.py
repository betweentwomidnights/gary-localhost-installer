"""Tests for component-scoped API server downloads."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep import api_server


class ApiServerSelectiveDownloadTests(unittest.TestCase):
    def test_huggingface_unified_repo_download_is_component_scoped(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "huggingface_hub.snapshot_download"
        ) as snapshot_download:
            result = api_server._download_from_huggingface(
                api_server.DEFAULT_REPO_ID,
                tmp,
                "acestep-5Hz-lm-1.7B",
            )

        snapshot_download.assert_called_once_with(
            repo_id=api_server.DEFAULT_REPO_ID,
            local_dir=tmp,
            local_dir_use_symlinks=False,
            allow_patterns=["acestep-5Hz-lm-1.7B/**"],
        )
        self.assertEqual(result, str(Path(tmp) / "acestep-5Hz-lm-1.7B"))

    def test_huggingface_separate_repo_download_has_no_filter(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "huggingface_hub.snapshot_download"
        ) as snapshot_download:
            model_name = "acestep-v15-xl-base"
            result = api_server._download_from_huggingface(
                "ACE-Step/acestep-v15-xl-base",
                tmp,
                model_name,
            )

        snapshot_download.assert_called_once_with(
            repo_id="ACE-Step/acestep-v15-xl-base",
            local_dir=str(Path(tmp) / model_name),
            local_dir_use_symlinks=False,
            allow_patterns=None,
        )
        self.assertEqual(result, str(Path(tmp) / model_name))

    def test_modelscope_unified_repo_download_is_component_scoped(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "modelscope.snapshot_download"
        ) as snapshot_download:
            result = api_server._download_from_modelscope(
                api_server.DEFAULT_REPO_ID,
                tmp,
                "acestep-v15-turbo",
            )

        snapshot_download.assert_called_once_with(
            model_id=api_server.DEFAULT_REPO_ID,
            local_dir=tmp,
            allow_patterns=["acestep-v15-turbo/**"],
        )
        self.assertEqual(result, str(Path(tmp) / "acestep-v15-turbo"))

    def test_partial_model_directory_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_name = "acestep-v15-xl-base"
            model_dir = root / model_name
            model_dir.mkdir()
            (model_dir / "config.json").write_bytes(b"config")

            self.assertFalse(api_server._checkpoint_files_valid(model_name, tmp))


if __name__ == "__main__":
    unittest.main()
