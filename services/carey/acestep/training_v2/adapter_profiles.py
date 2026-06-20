"""Static, architecture-level adapter profiles for ACE-Step.

The balanced profile is an independent summary of projection-family capacity
observed in user-owned instrumental and vocal adapter artifacts.  It does not
implement or reproduce Fisher analysis.  Layer-specific rankings are
deliberately ignored so the profile generalizes across datasets and the
24-layer base / 32-layer XL decoders.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


ATTENTION_PROFILE = "attention"
BALANCED_PROFILE = "balanced"
MODULE_PROFILE_CHOICES = (ATTENTION_PROFILE, BALANCED_PROFILE)


# Relative to the user-selected reference rank.  At rank 64 these become:
# self Q/K/V/O = 16/24/80/56, cross Q/K/V/O = 64/40/32/48,
# MLP gate/up/down = 40/48/48.
_BALANCED_RANK_MULTIPLIERS = {
    "cross_attn.k_proj": 5 / 8,
    "cross_attn.o_proj": 3 / 4,
    "cross_attn.q_proj": 1.0,
    "cross_attn.v_proj": 1 / 2,
    "mlp.down_proj": 3 / 4,
    "mlp.gate_proj": 5 / 8,
    "mlp.up_proj": 3 / 4,
    "self_attn.k_proj": 3 / 8,
    "self_attn.o_proj": 7 / 8,
    "self_attn.q_proj": 1 / 4,
    "self_attn.v_proj": 5 / 4,
}


def build_balanced_projection_profile(
    base_rank: int,
    base_alpha: int,
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """Build PEFT target, rank, and alpha patterns for the balanced profile.

    Alpha scales with each family rank so every projection preserves the
    user's requested ``alpha / rank`` ratio.
    """
    if base_rank <= 0 or base_alpha <= 0:
        raise ValueError("base rank and alpha must be greater than zero")

    rank_pattern = {
        name: max(1, round(base_rank * multiplier))
        for name, multiplier in _BALANCED_RANK_MULTIPLIERS.items()
    }
    alpha_pattern = {
        name: max(1, round(base_alpha * rank / base_rank))
        for name, rank in rank_pattern.items()
    }
    return list(_BALANCED_RANK_MULTIPLIERS), rank_pattern, alpha_pattern
