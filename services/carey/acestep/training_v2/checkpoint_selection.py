"""Small, framework-independent helpers for best-checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BestCheckpointDecision:
    activated: bool
    active: bool
    is_new_best: bool
    smoothed_loss: float | None
    best_loss: float
    best_epoch: int


class SmoothedBestCheckpointTracker:
    """Track the lowest moving-average epoch loss after a warm-up epoch."""

    def __init__(
        self,
        *,
        enabled: bool,
        start_epoch: int,
        window_size: int = 5,
        min_delta: float = 0.001,
    ) -> None:
        self.enabled = enabled and start_epoch > 0
        self.start_epoch = start_epoch
        self.window_size = max(1, window_size)
        self.min_delta = max(0.0, min_delta)
        self.active = False
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.recent_losses: list[float] = []

    def update(self, epoch: int, epoch_loss: float) -> BestCheckpointDecision:
        activated = False
        if self.enabled and not self.active and epoch >= self.start_epoch:
            self.active = True
            self.best_loss = float("inf")
            self.best_epoch = 0
            self.recent_losses.clear()
            activated = True

        if not self.active:
            return BestCheckpointDecision(
                activated=False,
                active=False,
                is_new_best=False,
                smoothed_loss=None,
                best_loss=self.best_loss,
                best_epoch=self.best_epoch,
            )

        self.recent_losses.append(float(epoch_loss))
        if len(self.recent_losses) > self.window_size:
            self.recent_losses.pop(0)
        smoothed_loss = sum(self.recent_losses) / len(self.recent_losses)
        is_new_best = smoothed_loss < self.best_loss - self.min_delta
        if is_new_best:
            self.best_loss = smoothed_loss
            self.best_epoch = epoch

        return BestCheckpointDecision(
            activated=activated,
            active=True,
            is_new_best=is_new_best,
            smoothed_loss=smoothed_loss,
            best_loss=self.best_loss,
            best_epoch=self.best_epoch,
        )
