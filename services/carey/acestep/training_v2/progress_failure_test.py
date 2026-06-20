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
