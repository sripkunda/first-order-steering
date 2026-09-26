from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from analysis.activation import _constant_least_squares_statistics
from analysis.herd_merging import compositional_teacher_log_probs
from analysis.independent import (
    MultiAffineInteractionHooks,
    MultiAffineInteractions,
    load_independent_loras,
)


def _write_adapter(root, axis: str, scale: float) -> None:
    directory = root / axis
    directory.mkdir()
    (directory / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "toy/base", "r": 2, "lora_alpha": 2,
    }))
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    save_file({
        f"{prefix}.lora_A.weight": scale * torch.eye(2, 3),
        f"{prefix}.lora_B.weight": torch.ones(4, 2),
    }, directory / "adapter_model.safetensors")


def test_task_arithmetic_and_herd_interactions_preserve_pure_axes(tmp_path):
    _write_adapter(tmp_path, "a", 1.0)
    _write_adapter(tmp_path, "b", 2.0)
    baseline = load_independent_loras(
        tmp_path, ["a", "b"], ["layer0.q_proj"], "toy/base",
    )
    torch.testing.assert_close(
        baseline.task_arithmetic([0.25, 0.75])["layer0.q_proj"],
        0.25 * baseline.dense_axis("a")["layer0.q_proj"]
        + 0.75 * baseline.dense_axis("b")["layer0.q_proj"],
    )

    interactions = MultiAffineInteractions(baseline, rank=2, seed=0)
    branch = interactions.branch((0, 1), "layer0.q_proj")
    with torch.no_grad():
        branch.A.copy_(torch.eye(2, 3))
        branch.B.fill_(1.0)

    class AdaptedProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4, bias=False)
            torch.nn.init.zeros_(self.linear.weight)

        def forward(self, value):
            return self.linear(value)

    projection = AdaptedProjection()
    model = type("Toy", (), {"_adapted_linears": {"layer0.q_proj": projection}})()
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    inputs = torch.ones(3, 1, 3)
    with MultiAffineInteractionHooks(model, interactions, values):
        outputs = projection(inputs)
    torch.testing.assert_close(outputs[:2], torch.zeros_like(outputs[:2]))
    assert outputs[2].abs().sum() > 0


def test_opinion_pool_has_exact_origin_and_pure_axis_boundaries():
    log_probs = torch.log_softmax(torch.tensor([[
        [2.0, 0.0, -1.0], [-1.0, 3.0, 0.0], [0.0, -2.0, 4.0],
    ]]), dim=-1)
    strengths = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    teacher = compositional_teacher_log_probs(log_probs, strengths)
    torch.testing.assert_close(teacher[:, 0], log_probs[:, 0])
    torch.testing.assert_close(teacher[:, 1], log_probs[:, 1])
    torch.testing.assert_close(teacher[:, 2], log_probs[:, 2])
    expected = torch.log_softmax(log_probs[:, 1] + log_probs[:, 2] - log_probs[:, 0], dim=-1)
    torch.testing.assert_close(teacher[:, 3], expected)


def test_static_least_squares_vector_is_the_prompt_mean():
    fields = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
    mean, target_energy, mean_energy, residual_energy = _constant_least_squares_statistics(
        fields.sum(0), fields.square().sum(), len(fields),
    )
    torch.testing.assert_close(mean, torch.tensor([2.0, 2.0]))
    assert target_energy == 10.0
    assert mean_energy == 8.0
    assert residual_energy == 2.0
