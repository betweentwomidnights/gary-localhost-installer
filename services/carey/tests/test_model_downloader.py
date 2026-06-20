import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from acestep import model_downloader


class ModelDownloaderValidationTests(unittest.TestCase):
    def test_known_bad_lm_weight_is_not_treated_as_installed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp)
            model_dir = checkpoints / "acestep-5Hz-lm-1.7B"
            model_dir.mkdir()
            weight = model_dir / "model.safetensors"
            weight.write_bytes(b"not the upstream 1.7B weights")

            self.assertFalse(
                model_downloader.check_model_exists(
                    "acestep-5Hz-lm-1.7B",
                    checkpoints,
                )
            )

            backups = model_downloader.quarantine_invalid_checkpoint_files(
                "acestep-5Hz-lm-1.7B",
                checkpoints,
            )

            self.assertFalse(weight.exists())
            self.assertEqual(len(backups), 1)
            self.assertTrue(backups[0].name.startswith("model.safetensors.bad-"))

    def test_multi_shard_lm_validation_checks_each_expected_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp)
            model_dir = checkpoints / "acestep-5Hz-lm-4B"
            model_dir.mkdir()
            shard_a = model_dir / "model-00001-of-00002.safetensors"
            shard_b = model_dir / "model-00002-of-00002.safetensors"
            shard_a.write_bytes(b"first shard")
            shard_b.write_bytes(b"wrong second shard")

            expected = {
                "acestep-5Hz-lm-4B": {
                    shard_a.name: model_downloader._file_hash(shard_a),
                    shard_b.name: model_downloader._file_hash(shard_a),
                },
            }

            with patch.object(model_downloader, "EXPECTED_MODEL_FILE_SHA256", expected):
                self.assertFalse(
                    model_downloader.check_model_exists(
                        "acestep-5Hz-lm-4B",
                        checkpoints,
                    )
                )

                backups = model_downloader.quarantine_invalid_checkpoint_files(
                    "acestep-5Hz-lm-4B",
                    checkpoints,
                )

            self.assertTrue(shard_a.exists())
            self.assertFalse(shard_b.exists())
            self.assertEqual(len(backups), 1)
            self.assertTrue(backups[0].name.startswith(f"{shard_b.name}.bad-"))


if __name__ == "__main__":
    unittest.main()
