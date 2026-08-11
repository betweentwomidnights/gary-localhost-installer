from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import train_lora_job
from dataset_processing import prompt_templates


class TrainingConfigTests(unittest.TestCase):
    def build_config(self, run_dir: Path, *, include: str = "", exclude: str = "") -> dict:
        args = SimpleNamespace(
            run_dir=run_dir,
            job_id="scope-test",
            rank=16,
            alpha=16,
            adapter_type="dora",
            lora_include=include,
            lora_exclude=exclude,
            learning_rate=1e-4,
            base_precision="bf16",
        )
        path = train_lora_job.build_model_config(args, run_dir / "t5gemma")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_efficient_scope_is_persisted_in_lora_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = self.build_config(
                Path(temp),
                include="transformer.layers",
                exclude="to_local_embed",
            )

        lora = payload["training"]["lora_config"]
        self.assertEqual(lora["include"], ["transformer.layers"])
        self.assertEqual(lora["exclude"], ["to_local_embed"])

    def test_full_scope_leaves_filters_unset(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = self.build_config(Path(temp))

        lora = payload["training"]["lora_config"]
        self.assertNotIn("include", lora)
        self.assertNotIn("exclude", lora)

    def test_no_sidecar_trains_with_an_empty_prompt(self):
        prompt_templates.set_config(
            {
                "prompt_config": {
                    "use_tags": True,
                    "use_paths": False,
                    "use_fixed": False,
                    "balance": {"tags": 40},
                    "tag_keys": ["prompt"],
                    "trigger": "",
                    "trigger_pct": 0,
                }
            }
        )

        result = prompt_templates.get_custom_metadata({"relpath": "ignored.wav"}, None)

        self.assertEqual(result["prompt"], "")

    def test_trigger_word_labels_audio_without_a_sidecar(self):
        prompt_templates.set_config(
            {
                "prompt_config": {
                    "use_tags": True,
                    "use_paths": False,
                    "use_fixed": False,
                    "balance": {"tags": 40},
                    "tag_keys": ["prompt"],
                    "trigger": "amd-test",
                    "trigger_pct": 100,
                }
            }
        )

        result = prompt_templates.get_custom_metadata({"relpath": "ignored.wav"}, None)

        self.assertEqual(result["prompt"], "amd-test")


if __name__ == "__main__":
    unittest.main()
