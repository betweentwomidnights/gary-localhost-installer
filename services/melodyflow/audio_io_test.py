"""Tests for MelodyFlow's SoundFile-backed request audio I/O."""

import tempfile
import unittest
from pathlib import Path

import soundfile as sf
import torch

from audio_io import load_audio, save_audio


class MelodyFlowAudioIoTests(unittest.TestCase):
    def test_loads_flac_resamples_and_expands_mono_to_stereo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "input.flac"
            sf.write(path, torch.linspace(-0.5, 0.5, 16000).numpy(), 16000)

            waveform = load_audio(path, target_sr=32000, device="cpu")

            self.assertEqual(waveform.shape, (1, 2, 32000))
            self.assertTrue(torch.equal(waveform[:, 0], waveform[:, 1]))

    def test_writes_stereo_pcm_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "output.wav"
            waveform = torch.stack(
                [torch.linspace(-0.25, 0.25, 100), torch.zeros(100)]
            )

            save_audio(path, waveform, sample_rate=32000)

            info = sf.info(path)
            self.assertEqual(info.samplerate, 32000)
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.frames, 100)
            self.assertEqual(info.subtype, "PCM_16")

    def test_rejects_batched_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "output.wav"
            with self.assertRaisesRegex(ValueError, "Expected"):
                save_audio(path, torch.zeros(1, 2, 100), sample_rate=32000)


if __name__ == "__main__":
    unittest.main()
