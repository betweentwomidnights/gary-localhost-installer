from __future__ import annotations

import pytest

from acestep.training_v2.checkpoint_selection import SmoothedBestCheckpointTracker


def test_tracker_activates_at_warmup_and_uses_ma5() -> None:
    tracker = SmoothedBestCheckpointTracker(enabled=True, start_epoch=3)

    assert not tracker.update(1, 9.0).active
    assert not tracker.update(2, 8.0).active

    first = tracker.update(3, 5.0)
    assert first.activated
    assert first.is_new_best
    assert first.best_epoch == 3

    tracker.update(4, 4.0)
    tracker.update(5, 3.0)
    tracker.update(6, 2.0)
    fifth = tracker.update(7, 1.0)
    assert fifth.smoothed_loss == pytest.approx(3.0)
    assert fifth.best_epoch == 7


def test_tracker_honors_min_delta() -> None:
    tracker = SmoothedBestCheckpointTracker(
        enabled=True,
        start_epoch=1,
        window_size=1,
        min_delta=0.001,
    )
    assert tracker.update(1, 1.0).is_new_best
    assert not tracker.update(2, 0.9995).is_new_best
    assert tracker.update(3, 0.998).is_new_best


def test_tracker_can_be_disabled() -> None:
    tracker = SmoothedBestCheckpointTracker(enabled=False, start_epoch=1)
    assert not tracker.update(100, 0.1).active
