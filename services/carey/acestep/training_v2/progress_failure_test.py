from __future__ import annotations

from acestep.training_v2.ui import TrainingUpdate
from acestep.training_v2.ui.progress import (
    TrainingStats,
    _process_structured,
    _process_tuple,
)


def test_structured_failure_is_preserved_for_cli_exit_status() -> None:
    stats = TrainingStats()
    update = TrainingUpdate(
        0,
        0.0,
        "[FAIL] VRAM preflight requires 3200 MiB free",
        kind="fail",
    )

    _process_structured(update, stats)

    assert stats.failed
    assert stats.failure_message == update.msg


def test_legacy_failure_tuple_is_preserved_for_cli_exit_status() -> None:
    stats = TrainingStats()

    _process_tuple(0, 0.0, "  [FAIL] No trainable parameters found", stats)

    assert stats.failed
    assert "No trainable parameters" in stats.failure_message


def test_completion_update_preserves_the_last_completed_epoch() -> None:
    stats = TrainingStats(max_epochs=5)

    _process_structured(
        TrainingUpdate(
            5,
            0.6572,
            "[OK] Epoch 5/5 in 37.1s, Loss: 0.6572",
            kind="epoch",
            epoch=5,
            max_epochs=5,
        ),
        stats,
    )
    _process_structured(
        TrainingUpdate(
            5,
            0.6572,
            "[OK] Training complete!",
            kind="complete",
        ),
        stats,
    )

    assert stats.current_epoch == 5
    assert stats.current_step == 5
