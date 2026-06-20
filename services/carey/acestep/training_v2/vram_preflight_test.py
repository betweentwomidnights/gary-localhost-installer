from __future__ import annotations

from acestep.training_v2.vram_preflight import estimate_vram_preflight


def estimate(
    *,
    total_mb: float,
    free_mb: float,
    trainable_millions: float,
    duration: float,
):
    trainable = int(trainable_millions * 1_000_000)
    return estimate_vram_preflight(
        total_mb=total_mb,
        free_mb=free_mb,
        allocated_mb=total_mb - free_mb,
        reserved_mb=total_mb - free_mb,
        trainable_parameters=trainable,
        trainable_bytes=trainable * 2,
        max_duration=duration,
        batch_size=1,
        gradient_checkpointing=True,
    )


def test_rank_128_balanced_is_blocked_without_encoder_offload_headroom() -> None:
    result = estimate(
        total_mb=8151,
        free_mb=2600,
        trainable_millions=116.244,
        duration=300,
    )
    assert not result.safe
    assert result.margin_mb < 0


def test_rank_128_balanced_can_pass_after_real_encoder_offload() -> None:
    result = estimate(
        total_mb=8151,
        free_mb=4200,
        trainable_millions=116.244,
        duration=240,
    )
    assert result.safe
    assert result.margin_mb > 0


def test_sixteen_gb_has_comfortable_headroom_for_rank_128() -> None:
    result = estimate(
        total_mb=16384,
        free_mb=11000,
        trainable_millions=116.244,
        duration=300,
    )
    assert result.safe
    assert result.margin_mb > 5000


def test_disabling_checkpointing_increases_activation_reserve() -> None:
    common = dict(
        total_mb=8151,
        free_mb=4000,
        allocated_mb=4000,
        reserved_mb=4000,
        trainable_parameters=58_000_000,
        trainable_bytes=116_000_000,
        max_duration=240,
        batch_size=1,
    )
    checkpointed = estimate_vram_preflight(**common, gradient_checkpointing=True)
    uncheckpointed = estimate_vram_preflight(**common, gradient_checkpointing=False)
    assert uncheckpointed.activation_mb > checkpointed.activation_mb
    assert uncheckpointed.required_headroom_mb > checkpointed.required_headroom_mb
