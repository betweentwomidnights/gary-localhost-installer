"""Regression tests for ACE-Step seed handoff in the inference orchestrator."""

import unittest

import torch

from acestep.inference import GenerationConfig, GenerationParams, generate_music


class FakeDitHandler:
    """Small no-model stand-in for testing inference-layer seed plumbing."""

    def __init__(self):
        self.prepare_seed_calls = []
        self.generate_music_calls = []
        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0

    def prepare_seeds(self, actual_batch_size, seed, use_random_seed):
        self.prepare_seed_calls.append(
            {
                "actual_batch_size": actual_batch_size,
                "seed": seed,
                "use_random_seed": use_random_seed,
            }
        )
        if use_random_seed:
            seeds = [1000 + idx for idx in range(actual_batch_size)]
        else:
            parsed = []
            if isinstance(seed, str):
                for raw in seed.split(","):
                    raw = raw.strip()
                    if raw:
                        parsed.append(int(float(raw)))
            elif seed is not None:
                parsed.append(int(seed))
            seeds = parsed[:actual_batch_size]
            while len(seeds) < actual_batch_size:
                seeds.append(2000 + len(seeds))
        return seeds, ", ".join(str(seed) for seed in seeds)

    def generate_music(self, **kwargs):
        self.generate_music_calls.append(kwargs)
        batch_size = int(kwargs.get("batch_size") or 1)
        return {
            "success": True,
            "status_message": "ok",
            "extra_outputs": {},
            "audios": [
                {
                    "tensor": torch.zeros(2, 16),
                    "sample_rate": 48000,
                }
                for _ in range(batch_size)
            ],
        }


class InferenceSeedHandoffTests(unittest.TestCase):
    def _generate(self, handler, *, params=None, config=None):
        return generate_music(
            handler,
            None,
            params=params
            or GenerationParams(
                caption="test",
                lyrics="",
                enable_normalization=False,
                use_cot_metas=False,
                use_cot_caption=False,
                use_cot_language=False,
            ),
            config=config or GenerationConfig(batch_size=1, use_random_seed=True),
            save_dir=None,
        )

    def test_random_seed_is_resolved_once_and_locked_for_handler(self):
        handler = FakeDitHandler()

        result = self._generate(
            handler,
            config=GenerationConfig(batch_size=1, use_random_seed=True),
        )

        self.assertTrue(result.success)
        self.assertEqual(handler.prepare_seed_calls[0]["use_random_seed"], True)
        self.assertEqual(handler.generate_music_calls[0]["use_random_seed"], False)
        self.assertEqual(handler.generate_music_calls[0]["seed"], "1000")
        self.assertEqual(result.audios[0]["params"]["seed"], 1000)

    def test_explicit_config_seed_remains_locked_for_handler(self):
        handler = FakeDitHandler()

        result = self._generate(
            handler,
            config=GenerationConfig(batch_size=1, use_random_seed=False, seeds=[42]),
        )

        self.assertTrue(result.success)
        self.assertEqual(handler.prepare_seed_calls[0]["seed"], "42")
        self.assertEqual(handler.prepare_seed_calls[0]["use_random_seed"], False)
        self.assertEqual(handler.generate_music_calls[0]["use_random_seed"], False)
        self.assertEqual(handler.generate_music_calls[0]["seed"], "42")
        self.assertEqual(result.audios[0]["params"]["seed"], 42)

    def test_params_seed_is_used_when_config_seed_is_omitted(self):
        handler = FakeDitHandler()

        result = self._generate(
            handler,
            params=GenerationParams(
                caption="test",
                lyrics="",
                seed=77,
                enable_normalization=False,
                use_cot_metas=False,
                use_cot_caption=False,
                use_cot_language=False,
            ),
            config=GenerationConfig(batch_size=1, use_random_seed=False, seeds=None),
        )

        self.assertTrue(result.success)
        self.assertEqual(handler.prepare_seed_calls[0]["seed"], "77")
        self.assertEqual(handler.generate_music_calls[0]["seed"], "77")
        self.assertEqual(result.audios[0]["params"]["seed"], 77)


if __name__ == "__main__":
    unittest.main()
