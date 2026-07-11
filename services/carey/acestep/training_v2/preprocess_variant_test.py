"""Tests for Gary prompt-variant preprocessing helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from acestep.training_v2 import preprocess, preprocess_discovery


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
            song_caption = preprocess._variant_final_path(out_path, song, "caption")
            song_genre = preprocess._variant_final_path(out_path, song, "genre")
            plain_caption = preprocess._variant_final_path(out_path, plain, "caption")
            song_caption.write_bytes(b"")
            song_genre.write_bytes(b"")
            plain_caption.write_bytes(b"")

            count = preprocess._write_variant_manifest(
                out_path=out_path,
                audio_files=[song, plain],
                sample_meta={
                    preprocess_discovery.audio_metadata_key(song): {
                        "caption": "detailed song",
                        "genre": "electro rock",
                    },
                    preprocess_discovery.audio_metadata_key(plain): {
                        "caption": "plain song",
                        "genre": "",
                    },
                },
                ds_meta={"genre_ratio": 20, "tag_position": "prepend"},
            )

            manifest = json.loads((out_path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(count, 2)
            self.assertEqual(
                manifest["samples"], [song_caption.name, plain_caption.name]
            )
            self.assertEqual(
                manifest["sample_groups"],
                [
                    {"path": song_caption.name, "genre_path": song_genre.name},
                    {"path": plain_caption.name},
                ],
            )
            self.assertEqual(
                manifest["metadata"]["prompt_variant_strategy"],
                "epoch_rotating_track_swap",
            )
            self.assertEqual(manifest["metadata"]["samples_per_epoch"], 2)
            self.assertEqual(manifest["metadata"]["num_tensor_files"], 3)

    def test_duplicate_basenames_keep_distinct_metadata_and_tensor_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drums = root / "drums" / "song.wav"
            vocals = root / "vocals" / "song.wav"
            drums.parent.mkdir()
            vocals.parent.mkdir()
            drums.write_bytes(b"audio")
            vocals.write_bytes(b"audio")
            dataset_json = root / "dataset.json"
            dataset_json.write_text(
                json.dumps(
                    {
                        "samples": [
                            {"audio_path": str(drums), "caption": "drum metadata"},
                            {"audio_path": str(vocals), "caption": "vocal metadata"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            metadata = preprocess_discovery.load_sample_metadata(
                str(dataset_json), [drums, vocals]
            )
            self.assertEqual(
                metadata[preprocess_discovery.audio_metadata_key(drums)]["caption"],
                "drum metadata",
            )
            self.assertEqual(
                metadata[preprocess_discovery.audio_metadata_key(vocals)]["caption"],
                "vocal metadata",
            )
            self.assertNotEqual(
                preprocess._variant_final_path(root, drums, "caption"),
                preprocess._variant_final_path(root, vocals, "caption"),
            )


if __name__ == "__main__":
    unittest.main()
