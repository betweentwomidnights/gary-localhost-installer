# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import math
import typing as tp

import torch
from torch import nn

from ..utils import utils
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..utils.renoise import noise_regularization
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
)
from ..modules.activations import get_activation_fn


logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


def get_init_fn(
    method: str, input_dim: int, init_depth: tp.Optional[int] = None
) -> partial[torch.Tensor]:
    """LM layer initialization.
    Inspired from xlformers: https://github.com/fairinternal/xlformers

    Args:
        method (str): Method name for init function. Valid options are:
            'gaussian', 'uniform'.
        input_dim (int): Input dimension of the initialized module.
        init_depth (Optional[int]): Optional init depth value used to rescale
            the standard deviation if defined.
    """
    # Compute std
    std = 1 / math.sqrt(input_dim)
    # Rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == "gaussian":
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == "uniform":
        bound = math.sqrt(3) * std  # ensure the standard deviation is `std`
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(
    m: torch.nn.Module,
    method: str,
    init_depth: tp.Optional[int] = None,
    zero_bias_init: bool = False,
) -> None:
    """Wrapper around ``get_init_fn`` for proper initialization of LM modules.

    Args:
        m (nn.Module): Module to initialize.
        method (str): Method name for the init function.
        init_depth (Optional[int]): Optional init depth value used to rescale
            the standard deviation if defined.
        zero_bias_init (bool): Whether to initialize the bias to 0 or not.
    """
    if isinstance(m, torch.nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        init_fn(m.weight)


class TimestepEmbedding(torch.nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, bias_proj: bool = False, **kwargs) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_size, hidden_size, bias=bias_proj, **kwargs),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=bias_proj, **kwargs),
        )
        self.frequency_embedding_size: int = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FlowModel(torch.nn.Module):
    """Transformer-based flow model operating on continuous audio latents.

    Args:
        condition_provider (MusicConditioningProvider): Conditioning provider from metadata.
        fuser (ConditionFuser): Fuser handling the fusing of conditions with language model input.
        dim (int): Dimension of the transformer encoder.
        latent_dim (int): Dimension of the latent audio representation.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_first (bool): Use pre-norm instead of post-norm.
        bias_proj (bool): Use bias for output projections.
        weight_init (str, optional): Method for weight initialization.
        depthwise_init (str, optional): Method for depthwise weight initialization.
        zero_bias_init (bool): If true and bias in Linears, initialize bias to zeros.
        cfg_coef (float): Classifier-free guidance coefficient.
        **kwargs: Additional parameters for the transformer encoder.
    """
    def __init__(
        self,
        condition_provider: ConditioningProvider,
        fuser: ConditionFuser,
        dim: int = 128,
        latent_dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = 'layer_norm',
        norm_first: bool = False,
        bias_proj: bool = False,
        weight_init: tp.Optional[str] = None,
        depthwise_init: tp.Optional[str] = None,
        zero_bias_init: bool = False,
        cfg_coef: float = 4.0,
        **kwargs
    ):
        super().__init__()
        # self.cfg = cfg
        # self.device = device
        self.latent_dim = latent_dim
        self.timestep_embedder = TimestepEmbedding(dim, bias_proj=bias_proj, device=kwargs["device"])
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first, **kwargs)
        self.condition_provider: ConditioningProvider = condition_provider
        self.fuser: ConditionFuser = fuser
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim, device=kwargs["device"])
        self.in_proj = torch.nn.Linear(latent_dim, dim, bias=bias_proj, device=kwargs["device"])
        self.out_proj = torch.nn.Linear(dim, latent_dim, bias=bias_proj, device=kwargs["device"])
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self.cfg_coef = cfg_coef
        # statistics of latent audio representations (tracked during codec training)
        self.register_buffer(
            "latent_mean", torch.zeros(1, latent_dim, 1, device=kwargs["device"])
        )
        self.register_buffer(
            "latent_std", torch.ones(1, latent_dim, 1, device=kwargs["device"])
        )

    def _init_weights(
        self,
        weight_init: tp.Optional[str],
        depthwise_init: tp.Optional[str],
        zero_bias_init: bool,
    ) -> None:
        """Initialization of the transformer module weights.

        This initialization schema is still experimental and may be subject to changes.

        Args:
            weight_init (Optional[str]): Weight initialization strategy. See ``get_init_fn`` for valid options.
            depthwise_init (Optional[str]): Depwthwise initialization strategy. The following options are valid:
                'current' where the depth corresponds to the current layer index or 'global' where the total number
                of layer is used as depth. If not set, no depthwise initialization strategy is used.
            zero_bias_init (bool): Whether to initialize bias to zero or not.
        """
        assert depthwise_init is None or depthwise_init in ["current", "global"]
        assert (
            depthwise_init is None or weight_init is not None
        ), "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert (
            not zero_bias_init or weight_init is not None
        ), "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        # for emb_layer in self.emb:
        init_layer(
            self.in_proj,
            method=weight_init,
            init_depth=None,
            zero_bias_init=zero_bias_init,
        )

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == "current":
                depth = layer_idx + 1
            elif depthwise_init == "global":
                depth = len(self.transformer.layers)
            init_fn = partial(
                init_layer,
                method=weight_init,
                init_depth=depth,
                zero_bias_init=zero_bias_init,
            )
            tr_layer.apply(init_fn)

        # for linear in self.linears:
        init_layer(
            self.out_proj,
            method=weight_init,
            init_depth=None,
            zero_bias_init=zero_bias_init,
        )

        for layer in self.timestep_embedder.mlp:
            if isinstance(layer, torch.nn.Linear):
                init_layer(
                    layer,
                    method=weight_init,
                    init_depth=None,
                    zero_bias_init=zero_bias_init,
                )

        for linear in self.transformer.skip_projections:
            init_layer(
                linear,
                method=weight_init,
                init_depth=None,
                zero_bias_init=zero_bias_init,
            )

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, condition_src: torch.Tensor, condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Rescale input so that its std remains 1 whatever the flow step t.
        x = z / torch.sqrt(torch.pow(t.unsqueeze(-1), 2) + torch.pow(1 - t.unsqueeze(-1), 2))
        x = self.in_proj(x.permute(0, 2, 1))
        x = self.transformer(
            x,
            cross_attention_src=condition_src,
            cross_attention_mask=condition_mask,
            timestep_embedding=self.timestep_embedder(t),
        )
        if self.out_norm is not None:
            x = self.out_norm(x)
        # Rescale output so that the DiT output std remains 1.
        x = self.out_proj(x).permute(0, 2, 1) * math.sqrt(2)
        return x

    @torch.no_grad()
    def generate(self,
                 prompt: tp.Optional[torch.Tensor] = None,
                 conditions: tp.List[ConditioningAttributes] = [],
                 solver: str = "euler",
                 steps: int = 16,
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 source_flowstep: int = 0.0,
                 target_flowstep: int = 1.0,
                 regularize: bool = False,
                 regularize_iters: int = 4,
                 keep_last_k_iters: int = 2,
                 lambda_kl: float = 0.2,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                 sway_coefficient: float = -0.8,
                ) -> torch.Tensor:
        """Generate tokens by running the ODE solver from source flow step to target flow step, either given a prompt or unconditionally.

        Args:
            prompt (torch.Tensor, optional): Prompt tokens of shape [B, K, T] (for editing).
            conditions (list of ConditioningAttributes, optional): List of conditions.
            solver (str): ODE solver (either euler or midpoint)
            steps (int): number of solver steps.
            num_samples (int, optional): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            source_flowstep (float): Source flow step (0 for generation, 1 for inversion).
            target_flowstep (float): Target flow step (1 for generation, 0 for inversion).
            regularize (bool): Regularize each solver step.
            regularize_iters (int): Number of regularization iterations.
            keep_last_k_iters (int): Number of meaningful regularization iterations for moving average computation.
            lambda_kl (float): KL regularization loss weight.
            sway_coefficient (float): sway sampling coefficient (https://arxiv.org/pdf/2410.06885), should be set to 0 for uniform.
            callback (Callback, optional): Callback function to report generation progress.
        Returns:
            torch.Tensor: Generated tokens.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        assert solver in ["euler", "midpoint"], "Supported ODE solvers are either euler or midpoint!"
        assert not (regularize and solver == "midpoint"), "Latent regularization is only supported with euler solver!"
        assert keep_last_k_iters <= regularize_iters

        # Checking all input shapes are consistent.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]

        if prompt is not None and self.latent_mean.shape[1] != prompt.shape[1]:
            # tokens directly emanate from the VAE encoder
            mean, scale = prompt.chunk(2, dim=1)
            gen_tokens = utils.vae_sample(mean, scale)
            prompt = (gen_tokens - self.latent_mean) / (self.latent_std + 1e-5)

        # below we create set of conditions: one conditional and one unconditional
        # to do that we merge the regular condition together with the null condition
        # we then do 1 forward pass instead of 2.
        # the reason for that is two-fold:
        # 1. it is about x2 faster than doing 2 forward passes
        # 2. avoid the streaming API treating the 2 passes as part of different time steps
        # We also support doing two different passes, in particular to ensure that
        # the padding structure is exactly the same between train and test.
        # With a batch size of 1, this can be slower though.
        cfg_conditions: CFGConditions
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            conditions = conditions + null_conditions
            tokenized = self.condition_provider.tokenize(conditions)
            cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        gen_sequence = torch.randn(num_samples, self.latent_dim, max_gen_len, device=device) if prompt is None else prompt
        B = gen_sequence.shape[0]

        if solver == "midpoint":
            assert steps % 2 == 0, "Midpoint solver can only run with even number of steps"
            next_sequence: tp.Optional[torch.Tensor] = None
        if not regularize:
            regularize_iters = 1
            keep_last_k_iters = 0
        else:
            avg_regularized_velocity = torch.zeros_like(gen_sequence)
        regularization_iters_threshold = regularize_iters - keep_last_k_iters
        cfg_coef = 0.0 if target_flowstep < source_flowstep else self.cfg_coef  # prevent divergence during latent inversion
        if target_flowstep > source_flowstep:
            schedule = torch.arange(0, 1+1e-5, 1/steps)
        else:
            schedule = torch.arange(1, 0-1e-5, -1/steps)
        schedule = schedule + sway_coefficient * (torch.cos(math.pi * 0.5 * schedule) - 1 + schedule)
        if target_flowstep > source_flowstep:
            schedule = schedule * (target_flowstep-source_flowstep) + source_flowstep
        else:
            schedule = schedule * (source_flowstep-target_flowstep) + target_flowstep

        for idx, current_flowstep in enumerate(schedule[:-1]):
            delta_t = schedule[idx+1] - current_flowstep
            input_sequence = gen_sequence
            if solver == "midpoint" and idx % 2 == 1:
                input_sequence = next_sequence
            for jdx in range(regularize_iters):
                should_compute_kl = jdx >= regularization_iters_threshold
                if should_compute_kl:
                    regularizing_sequence = (
                                1 - schedule[idx+1]
                            ) * torch.randn_like(input_sequence) + schedule[idx+1] * prompt + 1e-5 * torch.randn_like(input_sequence)
                    input_sequence = torch.cat(
                        [
                            input_sequence,
                            regularizing_sequence,
                        ],
                        dim=0,
                    )
                if cfg_coef == 0.0:
                    predicted_velocity = self.forward(input_sequence,
                                            torch.tensor([schedule[idx+1] if regularize else current_flowstep], device=device).expand(B * (1+should_compute_kl), 1),
                                            cfg_conditions['description'][0][:B].repeat(1+should_compute_kl, 1, 1),
                                            torch.log(cfg_conditions['description'][1][:B].unsqueeze(1).unsqueeze(1).repeat(1+should_compute_kl, 1, 1, 1)))
                    velocity = predicted_velocity[:B]
                    if should_compute_kl:
                        regularizing_velocity = predicted_velocity[B:]
                else:
                    predicted_velocity = self.forward(input_sequence.repeat(2, 1, 1),
                                            torch.tensor([schedule[idx+1] if regularize else current_flowstep], device=device).expand(B*2*(1 + should_compute_kl), 1),
                                            cfg_conditions['description'][0].repeat_interleave(1 + should_compute_kl, dim=0),
                                            torch.log(cfg_conditions['description'][1].unsqueeze(1).unsqueeze(1)).repeat_interleave(1 + should_compute_kl, dim=0))
                    if should_compute_kl:
                        velocity = (1 + cfg_coef) * predicted_velocity[:B] - cfg_coef * predicted_velocity[2*B:3*B]
                        regularizing_velocity = (1 + cfg_coef) * predicted_velocity[B:2*B] - cfg_coef * predicted_velocity[3*B:4*B]
                    else:
                        velocity = (1 + cfg_coef) * predicted_velocity[:B] - cfg_coef * predicted_velocity[B:2*B]
                if should_compute_kl:
                    regularized_velocity = noise_regularization(
                        velocity,
                        regularizing_velocity,
                        lambda_kl=lambda_kl,
                        lambda_ac=0.0,
                        num_reg_steps=4,
                        num_ac_rolls=5,
                    )
                    avg_regularized_velocity += (
                        regularized_velocity
                        * jdx
                        / (
                            keep_last_k_iters * regularization_iters_threshold
                            + sum(range(keep_last_k_iters))
                        )
                    )
                if should_compute_kl:
                    input_sequence = gen_sequence + regularized_velocity * delta_t
                else:
                    input_sequence = gen_sequence + velocity * delta_t
                if callback is not None:
                    callback(1 + idx * regularize_iters + jdx, steps * regularize_iters)
            if regularize:
                velocity = avg_regularized_velocity
                avg_regularized_velocity = torch.zeros_like(gen_sequence)
            if solver == "midpoint":
                if idx % 2 == 0:
                    next_sequence = gen_sequence + velocity * delta_t
                else:
                    gen_sequence = gen_sequence + velocity * (schedule[idx+1] - schedule[idx-1])
            else:
                gen_sequence = gen_sequence + velocity * delta_t
        return gen_sequence
