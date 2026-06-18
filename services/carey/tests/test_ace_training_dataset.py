from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

CAREY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAREY_DIR))

from ace_training_dataset import (  # noqa: E402
    build_dataset_json,
    parse_key_value_sidecar,
    write_canonical_sidecar,
)


class AceTrainingDatasetTests(unittest.TestCase):
    def test_canonical_sidecar_keeps_lyrics_last_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "song.txt"
            write_canonical_sidecar(
                path,
                caption="bright synth pop",
                genre="synthpop",
                lyrics="[Verse]\nhello world",
                bpm=120,
                bpm_source="local_overrode_lm",
                lm_bpm=100,
                local_bpm=120,
                keyscale="C major",
                key_source="lm_agrees_with_local",
                lm_keyscale="C major",
                local_keyscale="C major",
                timesignature="4",
                is_instrumental=False,
                custom_tag="garytone",
            )

            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.splitlines()[-2:], ["lyrics: [Verse]", "hello world"])
            self.assertIn("keyscale: C major\n", text)
            self.assertIn("timesignature: 4\n", text)
            parsed = parse_key_value_sidecar(path)
            self.assertEqual(parsed["caption"], "bright synth pop")
            self.assertEqual(parsed["lyrics"], "[Verse]\nhello world")
            self.assertEqual(parsed["custom_tag"], "garytone")
            self.assertEqual(parsed["bpm_source"], "local_overrode_lm")
            self.assertEqual(parsed["lm_bpm"], "100")
            self.assertEqual(parsed["local_bpm"], "120")
            self.assertEqual(parsed["key_source"], "lm_agrees_with_local")
            self.assertEqual(parsed["lm_keyscale"], "C major")
            self.assertEqual(parsed["local_keyscale"], "C major")

    def test_dataset_builder_reads_metadata_and_wav_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "demo_bpm_128.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(b"\x00\x00" * 8000)
            write_canonical_sidecar(
                audio.with_suffix(".txt"),
                caption="minimal house",
                genre="house",
                lyrics="",
                bpm=128,
                is_instrumental=True,
            )

            output = root / "run" / "dataset.json"
            result = build_dataset_json(
                root,
                output,
                name="demo",
                trigger="careybeat",
                genre_ratio=30,
                instrumental_default=True,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            sample = payload["samples"][0]
            self.assertEqual(result["samples"], 1)
            self.assertEqual(payload["metadata"]["genre_ratio"], 30)
            self.assertEqual(sample["caption"], "minimal house")
            self.assertEqual(sample["lyrics"], "[Instrumental]")
            self.assertEqual(sample["custom_tag"], "careybeat")
            self.assertAlmostEqual(sample["duration"], 1.0, places=2)

    def test_legacy_plain_txt_is_treated_as_lyrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "voice.wav"
            audio.write_bytes(b"not-real-audio")
            audio.with_suffix(".txt").write_text("[Verse]\nplain lyrics\n", encoding="utf-8")

            output = root / "dataset.json"
            build_dataset_json(root, output, name="voice")
            sample = json.loads(output.read_text(encoding="utf-8"))["samples"][0]
            self.assertEqual(sample["lyrics"], "[Verse]\nplain lyrics")
            self.assertFalse(sample["is_instrumental"])

    def test_key_and_signature_aliases_feed_ace_native_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "aliased.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(b"\x00\x00" * 800)
            audio.with_suffix(".txt").write_text(
                "\n".join(
                    [
                        "caption: angular instrumental rock",
                        "genre: Math rock",
                        "bpm: 100",
                        "key: D minor",
                        "signature: 3",
                        "is_instrumental: true",
                        "lyrics: [Instrumental]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output = root / "dataset.json"
            build_dataset_json(root, output, name="aliases")
            sample = json.loads(output.read_text(encoding="utf-8"))["samples"][0]
            self.assertEqual(sample["keyscale"], "D minor")
            self.assertEqual(sample["timesignature"], "3")

    def test_instrumental_sidecar_preserves_bracket_only_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "instrumental.txt"
            write_canonical_sidecar(
                path,
                caption="lo-fi hip hop beat",
                lyrics="[Intro]\n[Synth melody starts]\n[Outro]",
                is_instrumental=True,
            )

            text = path.read_text(encoding="utf-8")
            self.assertIn("is_instrumental: true\n", text)
            self.assertEqual(
                text.splitlines()[-3:],
                ["lyrics: [Intro]", "[Synth melody starts]", "[Outro]"],
            )


if __name__ == "__main__":
    unittest.main()
