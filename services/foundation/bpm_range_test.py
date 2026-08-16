"""Tests for Foundation-1's host tempo validation."""

import unittest

from bpm_range import MAX_HOST_BPM, MIN_HOST_BPM, HostBpmError, resolve_host_bpm


class ResolveHostBpmTests(unittest.TestCase):
    def test_accepts_the_tempo_range_a_daw_can_report(self):
        for value in (MIN_HOST_BPM, 90, 128.5, 174, 300, MAX_HOST_BPM):
            self.assertEqual(resolve_host_bpm(value), float(value))

    def test_accepts_a_numeric_string(self):
        self.assertEqual(resolve_host_bpm("128"), 128.0)

    def test_rejects_a_bogus_host_tempo_with_the_value_in_the_message(self):
        with self.assertRaises(HostBpmError) as caught:
            resolve_host_bpm(3159345)

        message = str(caught.exception)
        self.assertIn("3159345", message)
        self.assertIn("20-999", message)

    def test_rejects_tempos_outside_the_range(self):
        for value in (0, -120, 19.9, 1000):
            with self.assertRaises(HostBpmError):
                resolve_host_bpm(value)

    def test_rejects_non_finite_tempos(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(HostBpmError):
                resolve_host_bpm(value)

    def test_rejects_a_non_numeric_tempo_instead_of_raising_type_error(self):
        with self.assertRaisesRegex(HostBpmError, "must be a number"):
            resolve_host_bpm("fast")

    def test_missing_tempo_uses_the_default_when_one_is_given(self):
        self.assertEqual(resolve_host_bpm(None, default=120.0), 120.0)

        with self.assertRaisesRegex(HostBpmError, "required"):
            resolve_host_bpm(None)


if __name__ == "__main__":
    unittest.main()
