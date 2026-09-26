import contextlib
from types import SimpleNamespace

import torch

from analysis.steering_comparison import _fixed_token_profile


class _ProfileModel:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask, position_ids, past_key_values=None,
                 use_cache=False, **_kwargs):
        self.calls.append({
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "position_ids": position_ids.clone(),
            "past_key_values": past_key_values,
        })
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        logits[..., 1] = 1.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=0 if past_key_values is None else past_key_values + 1,
        )


def test_fixed_token_profile_supplies_left_padding_safe_position_ids():
    model = _ProfileModel()
    ctx = SimpleNamespace(
        device=torch.device("cpu"),
        model=SimpleNamespace(llama=model),
    )
    encoded = {
        "input_ids": torch.tensor([[0, 0, 5], [4, 5, 6]]),
        "attention_mask": torch.tensor([[0, 0, 1], [1, 1, 1]]),
    }

    report = _fixed_token_profile(
        ctx, encoded, contextlib.nullcontext, tokens=3,
    )

    assert report["generated_tokens_per_sequence"] == 3
    assert len(model.calls) == 3
    torch.testing.assert_close(
        model.calls[0]["position_ids"],
        torch.tensor([[0, 0, 0], [0, 1, 2]]),
    )
    torch.testing.assert_close(
        model.calls[1]["position_ids"],
        torch.tensor([[1], [3]]),
    )
    torch.testing.assert_close(
        model.calls[2]["position_ids"],
        torch.tensor([[2], [4]]),
    )

