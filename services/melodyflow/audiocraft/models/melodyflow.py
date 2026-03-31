# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Main model for using MelodyFlow. This will combine all the required components
and provide easy access to the generation API.
"""

import typing as tp
from audiocraft.utils.autocast import TorchAutocast
import torch

from .genmodel import BaseGenModel
from ..modules.conditioners import ConditioningAttributes
from ..utils.utils import vae_sample
from .loaders import load_compression_model, load_dit_model_melodyflow


class MelodyFlow(BaseGenModel):
    """MelodyFlow main model with convenient generation API.
    Args:
       See MelodyFlow class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_generation_params()
        self.set_editing_params()
        if self.device.type == 'cpu' or self.device.type == 'mps':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device.type, dtype=torch.bfloat16)

    @staticmethod
    def get_pretrained(name: str = 'facebook/melodyflow-t24-30secs', device=None):
        # TODO complete the list of pretrained models
        """
        """
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        compression_model = load_compression_model(name, device=device)

        def _remove_weight_norm(module):
            if hasattr(module, "conv"):
                if hasattr(module.conv, "conv"):
                    torch.nn.utils.parametrize.remove_parametrizations(
                        module.conv.conv, "weight"
                    )
            if hasattr(module, "convtr"):
                if hasattr(module.convtr, "convtr"):
                    torch.nn.utils.parametrize.remove_parametrizations(
                        module.convtr.convtr, "weight"
                    )

        def _clear_weight_norm(module):
            _remove_weight_norm(module)
            for child in module.children():
                _clear_weight_norm(child)

        compression_model.to('cpu')
        _clear_weight_norm(compression_model)
        compression_model.to(device)

        lm = load_dit_model_melodyflow(name, device=device)

        kwargs = {'name': name, 'compression_model': compression_model, 'lm': lm}
        return MelodyFlow(**kwargs)

    def set_generation_params(
        self,
        solver: str = "midpoint",
        steps: int = 64,
        duration: float = 10.0,
    ) -> tp.Dict[str, torch.Tensor]:
        """Set regularized inversion parameters for MelodyFlow.

        Args:
            solver (str, optional): ODE solver, either euler or midpoint.
            steps (int, optional): number of inference steps.
        """
        self.generation_params = {
            'solver': solver,
            'steps': steps,
            'duration': duration,
        }

    def set_editing_params(
        self,
        solver: str = "euler",
        steps: int = 25,
        target_flowstep: float = 0.0,
        regularize: bool = True,
        regularize_iters: int = 4,
        keep_last_k_iters: int = 2,
        lambda_kl: float = 0.2,
    ) -> tp.Dict[str, torch.Tensor]:
        """Set regularized inversion parameters for MelodyFlow.

        Args:
            solver (str, optional): ODE solver, either euler or midpoint.
            steps (int, optional): number of inference steps.
            target_flowstep (float): Target flow step.
            regularize (bool): Regularize each solver step.
            regularize_iters (int, optional): Number of regularization iterations.
            keep_last_k_iters (int, optional): Number of meaningful regularization iterations for moving average computation.
            lambda_kl (float, optional): KL regularization loss weight.
        """
        self.editing_params = {
            'solver': solver,
            'steps': steps,
            'target_flowstep': target_flowstep,
            'regularize': regularize,
            'regularize_iters': regularize_iters,
            'keep_last_k_iters': keep_last_k_iters,
            'lambda_kl': lambda_kl,
        }

    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Generate Audio from tokens."""
        assert waveform.dim() == 3
        with torch.no_grad():
            latent_sequence = self.compression_model.encode(waveform)[0].squeeze(1)
        return latent_sequence

    def generate_audio(self, gen_tokens: torch.Tensor) -> torch.Tensor:
        """Generate Audio from tokens."""
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            if self.lm.latent_mean.shape[1] != gen_tokens.shape[1]:
                # tokens directly emanate from the VAE encoder
                mean, scale = gen_tokens.chunk(2, dim=1)
                gen_tokens = vae_sample(mean, scale)
            else:
                # tokens emanate from the generator
                gen_tokens = gen_tokens * (self.lm.latent_std + 1e-5) + self.lm.latent_mean
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio

    def generate_unconditional(self, num_samples: int, progress: bool = False,
                               return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                        tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples in an unconditional manner.

        Args:
            num_samples (int): Number of samples to be generated.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        descriptions: tp.List[tp.Optional[str]] = [None] * num_samples
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes=attributes,
                                       prompt_tokens=prompt_tokens,
                                       progress=progress,
                                       **self.generation_params,
                                       )
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate(self, descriptions: tp.List[str], progress: bool = False, return_tokens: bool = False) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes=attributes,
                                       prompt_tokens=prompt_tokens,
                                       progress=progress,
                                       **self.generation_params,
                                       )
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def edit(self,
             prompt_tokens: torch.Tensor,
             descriptions: tp.List[str],
             src_descriptions: tp.Optional[tp.List[str]] = None,
             progress: bool = False,
             return_tokens: bool = False,
             ) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text.

        Args:
            prompt_tokens (torch.Tensor, optional): Audio prompt used as initial latent sequence.
            descriptions (list of str): A list of strings used as editing conditioning.
            inversion (str): Inversion method (either ddim or fm_renoise)
            target_flowstep (float): Target flow step pivot in [0, 1[.
            steps (int): number of solver steps.
            src_descriptions (list of str): A list of strings used as conditioning during latent inversion.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
            return_tokens (bool): Whether to return the generated tokens.
        """
        empty_attributes, no_tokens = self._prepare_tokens_and_attributes(
            [""] if src_descriptions is None else src_descriptions, None)
        assert no_tokens is None
        edit_attributes, no_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert no_tokens is None

        inversion_params = self.editing_params.copy()
        override_total_steps = inversion_params["steps"] * (
            inversion_params["regularize_iters"] + 1) if inversion_params["regularize"] else inversion_params["steps"] * 2
        current_step_offset: int = 0

        def _progress_callback(elapsed_steps: int, total_steps: int):
            elapsed_steps += current_step_offset
            if self._progress_callback is not None:
                self._progress_callback(elapsed_steps, override_total_steps)
            else:
                print(f'{elapsed_steps: 6d} / {override_total_steps: 6d}', end='\r')

        intermediate_tokens = self._generate_tokens(attributes=empty_attributes,
                                                    prompt_tokens=prompt_tokens,
                                                    source_flowstep=1.0,
                                                    progress=progress,
                                                    callback=_progress_callback,
                                                    **inversion_params,
                                                    )
        if intermediate_tokens.shape[0] < len(descriptions):
            intermediate_tokens = intermediate_tokens.repeat(len(descriptions)//intermediate_tokens.shape[0], 1, 1)
        current_step_offset += inversion_params["steps"] * (
            inversion_params["regularize_iters"]) if inversion_params["regularize"] else inversion_params["steps"]
        inversion_params.pop("regularize")
        final_tokens = self._generate_tokens(attributes=edit_attributes,
                                             prompt_tokens=intermediate_tokens,
                                             source_flowstep=inversion_params.pop("target_flowstep"),
                                             target_flowstep=1.0,
                                             progress=progress,
                                             callback=_progress_callback,
                                             **inversion_params,)
        if return_tokens:
            return self.generate_audio(final_tokens), final_tokens
        return self.generate_audio(final_tokens)

    def _generate_tokens(self,
                         attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor],
                         progress: bool = False,
                         callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                         **kwargs) -> torch.Tensor:
        """Generate continuous audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (here text).
            prompt_tokens (torch.Tensor, optional): Audio prompt used as initial latent sequence.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        generate_params = kwargs.copy()
        total_gen_len = prompt_tokens.shape[-1] if prompt_tokens is not None else int(
            generate_params.pop('duration') * self.frame_rate)
        current_step_offset: int = 0

        def _progress_callback(elapsed_steps: int, total_steps: int):
            elapsed_steps += current_step_offset
            if self._progress_callback is not None:
                self._progress_callback(elapsed_steps, total_steps)
            else:
                print(f'{elapsed_steps: 6d} / {total_steps: 6d}', end='\r')

        if progress and callback is None:
            callback = _progress_callback

        assert total_gen_len <= int(self.max_duration * self.frame_rate)

        with self.autocast:
            gen_tokens = self.lm.generate(
                prompt=prompt_tokens,
                conditions=attributes,
                callback=callback,
                max_gen_len=total_gen_len,
                **generate_params,
            )

        return gen_tokens
