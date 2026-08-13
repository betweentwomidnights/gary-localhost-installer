"""Regression tests for selective ACE-Step model downloads."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep import model_downloader


def write_component(root: Path, component: str) -> None:
    component_dir = root / component
    component_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in model_downloader.COMPONENT_REQUIRED_FILES[component]:
        path = component_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")


class SelectiveModelDownloadTests(unittest.TestCase):
    def test_shared_models_do_not_require_turbo_or_language_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_component(root, "vae")
            write_component(root, "Qwen3-Embedding-0.6B")

            self.assertTrue(model_downloader.check_shared_models_exist(root))
            self.assertFalse(model_downloader.check_main_model_exists(root))

    def test_shared_download_filters_unified_repo_to_missing_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_component(root, "vae")

            def complete_download(repo_id, local_dir, token, source, allow_patterns):
                self.assertEqual(repo_id, model_downloader.MAIN_MODEL_REPO)
                self.assertEqual(allow_patterns, ["Qwen3-Embedding-0.6B/**"])
                write_component(root, "Qwen3-Embedding-0.6B")
                return True, "downloaded"

            with patch.object(
                model_downloader,
                "_smart_download",
                side_effect=complete_download,
            ):
                success, _ = model_downloader.ensure_shared_models(root)

            self.assertTrue(success)
            self.assertFalse((root / "acestep-v15-turbo").exists())
            self.assertFalse((root / "acestep-5Hz-lm-1.7B").exists())

    def test_default_turbo_download_requests_only_turbo_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def complete_download(repo_id, local_dir, token, source, allow_patterns):
                self.assertEqual(allow_patterns, ["acestep-v15-turbo/**"])
                write_component(root, "acestep-v15-turbo")
                return True, "downloaded"

            with patch.object(
                model_downloader,
                "_smart_download",
                side_effect=complete_download,
            ), patch.object(model_downloader, "_sync_model_code_files", return_value=[]):
                success, _ = model_downloader.ensure_dit_model(
                    "acestep-v15-turbo",
                    root,
                )

            self.assertTrue(success)

    def test_default_language_model_download_requests_only_language_model_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def complete_download(repo_id, local_dir, token, source, allow_patterns):
                self.assertEqual(allow_patterns, ["acestep-5Hz-lm-1.7B/**"])
                write_component(root, "acestep-5Hz-lm-1.7B")
                return True, "downloaded"

            with patch.object(
                model_downloader,
                "_smart_download",
                side_effect=complete_download,
            ), patch.object(model_downloader, "checkpoint_files_valid", return_value=True):
                success, _ = model_downloader.ensure_lm_model(
                    "acestep-5Hz-lm-1.7B",
                    root,
                )

            self.assertTrue(success)

    def test_partial_xl_checkpoint_is_not_treated_as_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "acestep-v15-xl-base"
            model_dir.mkdir()
            (model_dir / "config.json").write_bytes(b"config")

            self.assertFalse(
                model_downloader.check_model_exists("acestep-v15-xl-base", root)
            )

    def test_partial_xl_checkpoint_is_refreshed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_name = "acestep-v15-xl-base"
            model_dir = root / model_name
            model_dir.mkdir()
            (model_dir / "config.json").write_bytes(b"config")

            def complete_download(
                repo_id,
                local_dir,
                token,
                source,
                allow_patterns=None,
            ):
                self.assertEqual(repo_id, model_downloader.SUBMODEL_REGISTRY[model_name])
                self.assertIsNone(allow_patterns)
                write_component(root, model_name)
                return True, "downloaded"

            with patch.object(
                model_downloader,
                "_smart_download",
                side_effect=complete_download,
            ), patch.object(model_downloader, "_sync_model_code_files", return_value=[]):
                success, _ = model_downloader.download_submodel(model_name, root)

            self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
