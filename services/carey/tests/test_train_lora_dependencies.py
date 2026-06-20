from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


CAREY_DIR = Path(__file__).resolve().parents[1]
if str(CAREY_DIR) not in sys.path:
    sys.path.insert(0, str(CAREY_DIR))

import train_lora_job  # noqa: E402


class CareyJobDependencyTests(unittest.TestCase):
    def test_no_repair_when_dependencies_are_present(self) -> None:
        args = SimpleNamespace()

        with (
            patch.object(train_lora_job, "missing_core_runtime_modules", return_value=[]),
            patch.object(train_lora_job, "missing_job_dependencies", return_value=[]),
            patch.object(train_lora_job, "run_step") as run_step,
        ):
            train_lora_job.ensure_job_dependencies(args)

        run_step.assert_not_called()

    def test_missing_safe_dependencies_are_installed_before_launch(self) -> None:
        args = SimpleNamespace()

        with (
            patch.object(train_lora_job, "missing_core_runtime_modules", return_value=[]),
            patch.object(
                train_lora_job,
                "missing_job_dependencies",
                side_effect=[["peft==0.18.1"], []],
            ),
            patch.object(train_lora_job, "run_step") as run_step,
        ):
            train_lora_job.ensure_job_dependencies(args)

        run_step.assert_called_once_with(
            args,
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "peft==0.18.1",
            ],
            "environment-setup",
            "Installing missing Carey captioning/training dependencies",
        )

    def test_missing_torch_requires_full_environment_rebuild(self) -> None:
        with (
            patch.object(
                train_lora_job,
                "missing_core_runtime_modules",
                return_value=["torch"],
            ),
            patch.object(train_lora_job, "run_step") as run_step,
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild env"):
                train_lora_job.ensure_job_dependencies(SimpleNamespace())

        run_step.assert_not_called()

    def test_failed_repair_points_to_rebuild_env(self) -> None:
        with (
            patch.object(train_lora_job, "missing_core_runtime_modules", return_value=[]),
            patch.object(
                train_lora_job,
                "missing_job_dependencies",
                return_value=["peft==0.18.1"],
            ),
            patch.object(
                train_lora_job,
                "run_step",
                side_effect=RuntimeError("environment-setup failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild env"):
                train_lora_job.ensure_job_dependencies(SimpleNamespace())

    def test_incomplete_repair_points_to_rebuild_env(self) -> None:
        with (
            patch.object(train_lora_job, "missing_core_runtime_modules", return_value=[]),
            patch.object(
                train_lora_job,
                "missing_job_dependencies",
                side_effect=[["peft==0.18.1"], ["peft==0.18.1"]],
            ),
            patch.object(train_lora_job, "run_step"),
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild env"):
                train_lora_job.ensure_job_dependencies(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
