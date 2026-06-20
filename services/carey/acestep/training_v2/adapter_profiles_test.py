from __future__ import annotations

import pytest

from acestep.training_v2.adapter_profiles import build_balanced_projection_profile


def test_balanced_profile_matches_empirical_rank_64_family_budget() -> None:
    targets, ranks, alphas = build_balanced_projection_profile(64, 128)

    assert len(targets) == 11
    assert ranks == {
        "cross_attn.k_proj": 40,
        "cross_attn.o_proj": 48,
        "cross_attn.q_proj": 64,
        "cross_attn.v_proj": 32,
        "mlp.down_proj": 48,
        "mlp.gate_proj": 40,
        "mlp.up_proj": 48,
        "self_attn.k_proj": 24,
        "self_attn.o_proj": 56,
        "self_attn.q_proj": 16,
        "self_attn.v_proj": 80,
    }
    assert all(alphas[name] == rank * 2 for name, rank in ranks.items())


def test_balanced_profile_preserves_non_default_alpha_ratio() -> None:
    _targets, ranks, alphas = build_balanced_projection_profile(32, 32)
    assert alphas == ranks


@pytest.mark.parametrize("rank,alpha", [(0, 128), (64, 0), (-1, 128)])
def test_balanced_profile_rejects_invalid_reference_values(rank: int, alpha: int) -> None:
    with pytest.raises(ValueError):
        build_balanced_projection_profile(rank, alpha)
