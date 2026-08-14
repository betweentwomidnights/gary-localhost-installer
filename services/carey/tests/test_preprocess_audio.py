from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

CAREY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAREY_DIR))

from acestep.training.dataset_builder_modules.preprocess_audio import (  # noqa: E402
    load_audio_stereo,
)


class PreprocessAudioTests(unittest.TestCase):
    def test_flac_decoding_does_not_require_torchcodec(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audio_path = Path(temp) / "training.flac"
            sample_rate = 48_000
            sf.write(
                audio_path,
                np.zeros((sample_rate, 1), dtype=np.float32),
                sample_rate,
                format="FLAC",
            )

            with patch(
                "acestep.audio_loading.torchaudio.load",
                side_effect=RuntimeError("TorchCodec is required"),
            ) as torchaudio_load:
                audio, source_rate = load_audio_stereo(
                    str(audio_path), sample_rate, max_duration=0.5
                )

            torchaudio_load.assert_not_called()
            self.assertEqual(source_rate, sample_rate)
            self.assertEqual(tuple(audio.shape), (2, sample_rate // 2))


if __name__ == "__main__":
    unittest.main()
