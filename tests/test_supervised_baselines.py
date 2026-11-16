"""Checks for the retained supervised steering baselines."""

import unittest

import torch

from analysis.steering_comparison import CLASSICAL_METHODS, _controller_fit
from analysis.supervised_baselines.contrastive_caa import select_pairs
from analysis.supervised_baselines.run import BiPOVector, LoFiT, LoReFT


class SupervisedBaselineTest(unittest.TestCase):
    def test_adapter_difference_has_separate_name(self):
        self.assertIn("adapter_caa", CLASSICAL_METHODS)
        self.assertNotIn("caa", CLASSICAL_METHODS)
        positive = torch.tensor([[3.0, 2.0], [5.0, 4.0]])
        negative = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        controller, _ = _controller_fit("adapter_caa", positive, negative, {})
        torch.testing.assert_close(controller.vector, torch.tensor([3.0, 2.0]))

    def test_rank_two_reft_and_bipo_shapes(self):
        hidden = torch.randn(2, 3, 8)
        self.assertEqual(LoReFT(8, 2).delta(hidden).shape, hidden.shape)
        self.assertEqual(BiPOVector(8).delta(hidden).shape, (8,))

    def test_lofit_selects_top_fraction(self):
        controller = LoFiT([(4, 2), (4, 2)])
        with torch.no_grad():
            controller.scales[0].zero_()
            controller.scales[1].zero_()
            controller.scales[1][2].fill_(10)
        self.assertEqual(controller.select_heads(0.03), [(1, 2)])
        self.assertEqual(controller.stage, "bias")

    def test_contrastive_caa_pair_filter(self):
        rows = [{"prompt": "train", "messages": [{"role": "user", "content": "Hi"},
                 {"role": "assistant", "content": "- A"}],
                 "original_messages": [{"role": "user", "content": "Hi"},
                 {"role": "assistant", "content": "A"}]}]
        selected = select_pairs(rows, set(), 1, 42)
        self.assertEqual(len(selected), 1)
        self.assertNotEqual(selected[0][1][-1], selected[0][2][-1])


if __name__ == "__main__":
    unittest.main()
