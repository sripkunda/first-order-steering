"""Regression tests for local weight-curvature layer selection."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from analysis import herd_gain_fit as fit


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)

    def forward(self, value):
        return self.linear(value)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = Projection()

    def forward(self, hidden):
        return self.proj(hidden).square()


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])

    def forward(self, input_ids, attention_mask, use_cache=False):
        hidden = input_ids.float().unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=hidden)


def evaluator(adapted_layers, selected_layers):
    lm = ToyLM()
    model = SimpleNamespace(llama=lm, _ctx={'active': False}, _adapted_linears={
        f'layer{i}.proj': layer.proj for i, layer in enumerate(lm.layers)
    })
    keys = [f'layer{i}.proj' for i in adapted_layers]
    factor = SimpleNamespace(A=torch.ones(1, 1), B=torch.ones(1, 1), scale=1.0)
    baseline = SimpleNamespace(axes=['a', 'b'], keys=keys,
        factors={axis: dict.fromkeys(keys, factor) for axis in ('a', 'b')},
        prepared_factor=lambda *args: factor)

    class Interactions:
        subsets = [(0, 1)]
        scale = 1.0

        def branch(self, subset, key):
            if key not in keys:
                raise KeyError(key)
            return factor

    return fit.ResidualEvaluator(model, None, baseline, Interactions(), torch.device('cpu'),
        8, 0.1, {'ss': 1.0, 'static': 1.0}, torch.ones(1), selected_layers)


class CurvatureLayerSelectionTest(unittest.TestCase):
    def test_curvature_only_touches_selected_adapter_layers(self):
        def encode(tokenizer, prompts, device, max_length):
            return {'input_ids': torch.tensor([[1, 2], [1, 3]], device=device),
                    'attention_mask': torch.ones(2, 2, dtype=torch.long, device=device)}

        with patch.object(fit, '_decoder_layers', lambda current: current.layers), \
             patch.object(fit, 'encode', encode):
            for adapted, selected in [([1], [1]), ([0, 1], [0]), ([0, 1], None)]:
                with self.subTest(adapted=adapted, selected=selected):
                    current = evaluator(adapted, selected)
                    ss, static = current._curvature(torch.tensor([[0.0], [1.0]]), ['a', 'b'], seed=19)
                    layers = 2 if selected is None else 1
                    self.assertEqual(ss.shape, (2, layers * 2))
                    self.assertEqual(static.shape, (2, layers))
                    point = 0.1 + 0.8 * torch.rand((2,), generator=torch.Generator().manual_seed(19))
                    expected = torch.tensor([point.sum(), point.sum() + point.prod()])
                    torch.testing.assert_close(ss, expected[:, None].expand_as(ss), rtol=1e-5, atol=1e-6)
                    self.assertTrue(torch.isfinite(static).all())
                    self.assertFalse(any(module._forward_hooks for module in current.model._adapted_linears.values()))
                    self.assertFalse(any(layer._forward_hooks or layer._forward_pre_hooks for layer in current.layers))
