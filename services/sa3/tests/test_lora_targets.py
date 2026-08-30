import tempfile
import unittest
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from stable_audio_3 import StableAudioModel
from stable_audio_3.models.lora import (
    LoRAParametrization,
    add_lora,
    get_lora_state_dict,
    merge_lora,
    save_lora_safetensors,
)
from stable_audio_3.models.lora.loader import (
    _resolve_lora_targets,
    load_and_apply_loras,
)


RANK = 4
ALPHA = 4.0


def lora_config(index=0):
    return {
        nn.Linear: {
            "weight": partial(
                LoRAParametrization.from_linear,
                rank=RANK,
                lora_alpha=ALPHA,
                adapter_type="lora",
                lora_index=index,
            )
        }
    }


class Half(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))

    def forward(self, value):
        return self.layers(value)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Half()
        self.decoder = Half()


class Pretransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Autoencoder()


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Half()
        self.conditioner = Half()
        self.pretransform = Pretransform()


def save_checkpoint(path: Path, target="dit", seed=0):
    source = Half()
    add_lora(source, lora_config())
    state = get_lora_state_dict(source)
    generator = torch.Generator().manual_seed(seed)
    for key, value in state.items():
        if "lora_B" in key:
            state[key] = torch.randn(value.shape, generator=generator, dtype=value.dtype)
    config = {"rank": RANK, "alpha": ALPHA, "adapter_type": "lora"}
    if target != "dit":
        config["target"] = target
    save_lora_safetensors(state, config, path)
    return state


class LoraTargetTests(unittest.TestCase):
    def test_decoder_resolves_through_full_model_pretransform(self):
        model = FullModel()
        modules, prefixes = _resolve_lora_targets(model, "diffusion_cond", "decoder")
        self.assertEqual(modules, [model.pretransform.model.decoder])
        self.assertEqual(prefixes, ["decoder."])

    def test_decoder_loaded_after_dit_uses_decoder_slot_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dit = root / "style.safetensors"
            decoder = root / "decoder.safetensors"
            save_checkpoint(dit, seed=1)
            expected = save_checkpoint(decoder, target="decoder", seed=2)
            model = FullModel()

            load_and_apply_loras(
                model, [str(dit), str(decoder)], "diffusion_cond"
            )

            loaded = get_lora_state_dict(model.pretransform.model.decoder)
            self.assertTrue(loaded)
            self.assertTrue(
                any(".parametrizations.weight.0.lora_B" in key for key in loaded)
            )
            self.assertFalse(any(".parametrizations.weight.1." in key for key in loaded))
            for key, value in expected.items():
                self.assertIn(key, loaded)
                torch.testing.assert_close(loaded[key], value.half().to(loaded[key].dtype))

    def test_merged_decoder_adapter_changes_output_without_live_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            decoder_path = Path(directory) / "decoder.safetensors"
            save_checkpoint(decoder_path, target="decoder", seed=3)
            model = FullModel()
            value = torch.randn(2, 8)
            before = model.pretransform.model.decoder(value).clone()

            load_and_apply_loras(model, [str(decoder_path)], "diffusion_cond")
            merge_lora(model.pretransform.model.decoder)
            after = model.pretransform.model.decoder(value)

            self.assertFalse(torch.allclose(before, after))
            self.assertFalse(get_lora_state_dict(model.pretransform.model.decoder))

    def test_weight_norm_layers_are_skipped(self):
        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.plain = nn.Linear(8, 8)
                self.normed = nn.utils.weight_norm(nn.Linear(8, 8))

        model = Mixed()
        add_lora(model, lora_config())
        keys = get_lora_state_dict(model)
        self.assertTrue(any(key.startswith("plain.") for key in keys))
        self.assertFalse(any(key.startswith("normed.") for key in keys))

    def test_strength_control_reaches_decoder(self):
        model = FullModel()
        add_lora(model.pretransform.model.decoder, lora_config(index=7))
        wrapper = object.__new__(StableAudioModel)
        wrapper.model = model

        wrapper.set_lora_strength(0.25, lora_index=7)

        strengths = [
            value
            for name, value in model.pretransform.model.decoder.named_buffers()
            if name.endswith("lora_strength")
        ]
        self.assertTrue(strengths)
        self.assertTrue(all(float(value) == 0.25 for value in strengths))


if __name__ == "__main__":
    unittest.main()
