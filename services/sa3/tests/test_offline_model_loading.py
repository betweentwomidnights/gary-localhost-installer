import os
import unittest
from unittest.mock import patch

import api
from stable_audio_3 import model_configs


class OfflineModelLoadingTests(unittest.TestCase):
    def test_configuring_auth_only_normalizes_environment(self):
        with patch.dict(
            os.environ,
            {
                "HF_TOKEN": "",
                "HUGGING_FACE_HUB_TOKEN": "offline-token",
            },
        ):
            api.configure_hf_auth()

            self.assertEqual(os.environ["HF_TOKEN"], "offline-token")
            self.assertEqual(
                os.environ["HUGGING_FACE_HUB_TOKEN"], "offline-token"
            )

    def test_cached_model_resolution_never_calls_download(self):
        config = model_configs.ModelConfig(
            "example/offline-model", "model_config.json", "model.safetensors"
        )
        cached_paths = ["cached/model_config.json", "cached/model.safetensors"]

        with patch.object(
            model_configs,
            "try_to_load_from_cache",
            side_effect=cached_paths,
        ), patch.object(model_configs, "hf_hub_download") as download:
            resolved = config.resolve()

        self.assertEqual(resolved, tuple(cached_paths))
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
