import unittest

from inference_profiles import (
    resolve_audio2audio_inference_settings,
    resolve_inference_settings,
)


class InferenceProfilesTest(unittest.TestCase):
    def test_default_preserves_gary_fallback(self):
        self.assertEqual(resolve_inference_settings({}), {
            "inference_profile": "gary_fallback",
            "sampler_type": "dpmpp-2m-sde",
            "sigma_min": 0.5,
            "sigma_max": 50.0,
            "rho": 1.0,
            "steps": 100,
            "guidance_scale": 7.0,
        })

    def test_royalcities_profile(self):
        settings = resolve_inference_settings({"inference_profile": "royalcities"})
        self.assertEqual(settings["sampler_type"], "dpmpp-3m-sde")
        self.assertEqual(settings["sigma_min"], 0.01)
        self.assertEqual(settings["sigma_max"], 100.0)

    def test_explicit_values_override_profile(self):
        settings = resolve_inference_settings({
            "inference_profile": "royalcities",
            "sampler": "k-heun",
            "sigma_min": 0.2,
            "sigma_max": 20,
            "rho": 2,
            "steps": 64,
            "guidance_scale": 5.5,
        })
        self.assertEqual(settings, {
            "inference_profile": "royalcities",
            "sampler_type": "k-heun",
            "sigma_min": 0.2,
            "sigma_max": 20.0,
            "rho": 2.0,
            "steps": 64,
            "guidance_scale": 5.5,
        })

    def test_audio2audio_reports_effective_sigma_max(self):
        settings = resolve_audio2audio_inference_settings(
            {"inference_profile": "royalcities"}, 0.25
        )
        self.assertEqual(settings["configured_sigma_max"], 100.0)
        self.assertEqual(settings["sigma_max"], 0.25)

    def test_invalid_settings_are_rejected(self):
        invalid_payloads = [
            {"inference_profile": "unknown"},
            {"inference_profile": []},
            {"sampler_type": "unknown"},
            {"sampler_type": []},
            {"steps": 0},
            {"steps": 1.5},
            {"sigma_min": 2, "sigma_max": 1},
            {"rho": 0},
            {"guidance_scale": -1},
            {"guidance_scale": True},
            {"rho": float("inf")},
            {"sampler": "k-heun", "sampler_type": "k-lms"},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    resolve_inference_settings(payload)


if __name__ == "__main__":
    unittest.main()
