from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import train_lora_job


class TrainingDependencyTests(unittest.TestCase):
    def test_no_repair_when_training_dependencies_are_present(self):
        args = SimpleNamespace()

        with (
            patch.object(train_lora_job, "missing_training_dependencies", return_value=[]),
            patch.object(train_lora_job, "run_step") as run_step,
        ):
            train_lora_job.ensure_training_dependencies(args)

        run_step.assert_not_called()

    def test_missing_dependencies_are_installed_before_training(self):
        args = SimpleNamespace()

        with (
            patch.object(
                train_lora_job,
                "missing_training_dependencies",
                side_effect=[["accelerate>=0.30"], []],
            ),
            patch.object(train_lora_job, "run_step") as run_step,
        ):
            train_lora_job.ensure_training_dependencies(args)

        run_step.assert_called_once_with(
            args,
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "accelerate>=0.30",
            ],
            "environment-setup",
            "Installing missing SA3 LoRA training dependencies",
        )

    def test_failed_repair_points_to_rebuild_env(self):
        args = SimpleNamespace()

        with (
            patch.object(
                train_lora_job,
                "missing_training_dependencies",
                return_value=["accelerate>=0.30"],
            ),
            patch.object(
                train_lora_job,
                "run_step",
                side_effect=RuntimeError("environment-setup failed with exit code 1"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild env"):
                train_lora_job.ensure_training_dependencies(args)

    def test_incomplete_repair_points_to_rebuild_env(self):
        args = SimpleNamespace()

        with (
            patch.object(
                train_lora_job,
                "missing_training_dependencies",
                side_effect=[["accelerate>=0.30"], ["accelerate>=0.30"]],
            ),
            patch.object(train_lora_job, "run_step"),
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild env"):
                train_lora_job.ensure_training_dependencies(args)


if __name__ == "__main__":
    unittest.main()
