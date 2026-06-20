"""Tests for Gary prompt-variant preprocessing helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from acestep.training_v2 import preprocess


class PromptVariantTests(unittest.TestCase):
    def test_prompt_variants_add_genre_when_ratio_positive(self):
        meta = {"caption": "bright guitar song", "genre": "math rock"}

        self.assertEqual(
            preprocess._prompt_variants_for_sample(meta, 20),
            ["caption", "genre"],
        )

    def test_prompt_variants_skip_duplicate_or_forced_prompts(self):
        self.assertEqual(
            preprocess._prompt_variants_for_sample(
                {"caption": "rock", "genre": "rock"},
                20,
            ),
            ["caption"],
        )
        self.assertEqual(
            preprocess._prompt_variants_for_sample(
                {
                    "caption": "detailed caption",
                    "genre": "ambient",
                    "prompt_override": "caption",
                },
                20,
            ),
            ["caption"],
        )
        self.assertEqual(
            preprocess._prompt_variants_for_sample(
                {
                    "caption": "detailed caption",
                    "genre": "ambient",
                    "prompt_override": "genre",
                },
                0,
            ),
            ["genre"],
        )

    def test_variant_manifest_keeps_one_training_row_per_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp)
            song = out_path / "song.wav"
            plain = out_path / "plain.wav"
            (out_path / "song.pt").write_bytes(b"")
            (out_path / "song.genre.pt").write_bytes(b"")
            (out_path / "plain.pt").write_bytes(b"")

            count = preprocess._write_variant_manifest(
                out_path=out_path,
                audio_files=[song, plain],
                sample_meta={
                    "song.wav": {"caption": "detailed song", "genre": "electro rock"},
                    "plain.wav": {"caption": "plain song", "genre": ""},
                },
                ds_meta={"genre_ratio": 20, "tag_position": "prepend"},
            )

            manifest = json.loads((out_path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(count, 2)
            self.assertEqual(manifest["samples"], ["song.pt", "plain.pt"])
            self.assertEqual(
                manifest["sample_groups"],
                [
                    {"path": "song.pt", "genre_path": "song.genre.pt"},
                    {"path": "plain.pt"},
                ],
            )
            self.assertEqual(
                manifest["metadata"]["prompt_variant_strategy"],
                "epoch_rotating_track_swap",
            )
            self.assertEqual(manifest["metadata"]["samples_per_epoch"], 2)
            self.assertEqual(manifest["metadata"]["num_tensor_files"], 3)


if __name__ == "__main__":
    unittest.main()
