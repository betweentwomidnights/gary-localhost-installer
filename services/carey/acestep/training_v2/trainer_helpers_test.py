from __future__ import annotations

import torch
import torch.nn as nn

from acestep.training_v2.trainer_helpers import offload_non_decoder


class RecordingModule(nn.Module):
    def __init__(self, *, trainable: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2), requires_grad=trainable)
        self.to_calls: list[object] = []

    def to(self, *args: object, **kwargs: object) -> "RecordingModule":
        self.to_calls.append(args[0] if args else kwargs.get("device"))
        return self


class AceTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = RecordingModule(trainable=True)
        self.encoder = RecordingModule()
        self.tokenizer = RecordingModule()
        self.detokenizer = RecordingModule()
        self.null_condition_emb = nn.Parameter(torch.ones(1), requires_grad=False)


def test_offload_moves_actual_ace_components_but_preserves_training_state() -> None:
    model = AceTrainingModel()
    null_device = model.null_condition_emb.device

    count = offload_non_decoder(model)

    assert count == 3
    assert model.encoder.to_calls == ["cpu"]
    assert model.tokenizer.to_calls == ["cpu"]
    assert model.detokenizer.to_calls == ["cpu"]
    assert model.decoder.to_calls == []
    assert model.null_condition_emb.device == null_device


def test_offload_refuses_to_move_trainable_non_decoder_component() -> None:
    model = AceTrainingModel()
    model.encoder.weight.requires_grad = True

    count = offload_non_decoder(model)

    assert count == 2
    assert model.encoder.to_calls == []
    assert model.tokenizer.to_calls == ["cpu"]
    assert model.detokenizer.to_calls == ["cpu"]
