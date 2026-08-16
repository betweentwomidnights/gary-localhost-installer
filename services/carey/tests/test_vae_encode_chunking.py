"""The VAE encode chunk has to size itself from free memory, not installed.

A machine can report 36 GB total while another service holds most of it. Asking
for a 30 second chunk there is how we ended up failing to place a 2.24 GiB
allocation on a card that still showed 10 GB free.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

CAREY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAREY_DIR))

from acestep.core.generation.handler.memory_utils import MemoryUtilsMixin  # noqa: E402


class Handler(MemoryUtilsMixin):
    def __init__(self, device: str = "cuda") -> None:
        self.device = device


FREE_VRAM = "acestep.core.generation.handler.memory_utils.get_effective_free_vram_gb"


class EncodeChunkSizeTests(unittest.TestCase):
    def test_plenty_of_room_keeps_the_thirty_second_chunk(self) -> None:
        with patch(FREE_VRAM, return_value=30.0):
            self.assertEqual(Handler()._get_auto_encode_chunk_size(), 48000 * 30)

    def test_a_busy_card_gets_a_shorter_chunk(self) -> None:
        # The reported failure: 36 GB installed, ~10 GB actually free.
        with patch(FREE_VRAM, return_value=10.31):
            self.assertEqual(Handler()._get_auto_encode_chunk_size(), 48000 * 15)

    def test_an_unmeasurable_card_is_left_alone(self) -> None:
        # 0 means we could not read it, which is not the same as being full.
        with patch(FREE_VRAM, return_value=0.0):
            self.assertEqual(Handler()._get_auto_encode_chunk_size(), 48000 * 30)

    def test_a_failing_probe_does_not_break_encoding(self) -> None:
        with patch(FREE_VRAM, side_effect=RuntimeError("no hip device")):
            self.assertEqual(Handler()._get_auto_encode_chunk_size(), 48000 * 30)

    def test_the_env_override_wins(self) -> None:
        with patch(FREE_VRAM, return_value=10.31), patch.dict(
            "os.environ", {"ACESTEP_VAE_ENCODE_CHUNK_SIZE": "480000"}
        ):
            self.assertEqual(Handler()._get_auto_encode_chunk_size(), 480000)

    def test_a_junk_override_falls_back_to_the_measured_answer(self) -> None:
        with patch(FREE_VRAM, return_value=30.0), patch.dict(
            "os.environ", {"ACESTEP_VAE_ENCODE_CHUNK_SIZE": "lots"}
        ):
            self.assertEqual(Handler()._get_auto_encode_chunk_size(), 48000 * 30)

    def test_non_cuda_devices_keep_the_default(self) -> None:
        with patch(FREE_VRAM, return_value=1.0):
            self.assertEqual(Handler("cpu")._get_auto_encode_chunk_size(), 48000 * 30)


if __name__ == "__main__":
    unittest.main()
