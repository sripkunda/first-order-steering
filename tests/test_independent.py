from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from analysis.compositional_generation import (
    _CompositionalLayerSourceCapture,
    interaction_geometry,
)
from analysis.herd_merging import compositional_teacher_log_probs
from analysis.independent import (
    MultiAffineInteractionHooks,
    MultiAffineInteractions,
    load_independent_loras,
)


def _write_axis(root: Path, axis: str, multiplier: float, layer: int = 0):
    target = root / axis
    target.mkdir(parents=True)
    (target / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "base/model", "r": 2, "lora_alpha": 4,
        "rank_pattern": {}, "alpha_pattern": {},
    }))
    prefix = f"base_model.model.model.language_model.layers.{layer}.self_attn.q_proj"
    save_file({
        f"{prefix}.lora_A.weight": multiplier * torch.eye(2, 3),
        f"{prefix}.lora_B.weight": torch.ones(4, 2),
        # This must never be interpreted as a language-model adapter site.
        "base_model.model.model.visual.blocks.0.attn.q_proj.lora_A.weight": torch.ones(2, 3),
        "base_model.model.model.visual.blocks.0.attn.q_proj.lora_B.weight": torch.ones(4, 2),
    }, target / "adapter_model.safetensors")


def test_independent_loader_filters_vision_and_composes_continuous_task_arithmetic(tmp_path):
    _write_axis(tmp_path, "a", 1.0)
    _write_axis(tmp_path, "b", 2.0)
    loaded = load_independent_loras(
        tmp_path, ["a", "b"], ["layer0.q_proj"], "base/model",
    )
    assert loaded.keys == ["layer0.q_proj"]
    expected = 0.25 * loaded.dense_axis("a")["layer0.q_proj"]
    expected += 0.75 * loaded.dense_axis("b")["layer0.q_proj"]
    torch.testing.assert_close(
        loaded.task_arithmetic([0.25, 0.75])["layer0.q_proj"], expected,
    )


def test_sparse_single_layer_axes_are_zero_outside_their_own_layer(tmp_path):
    _write_axis(tmp_path, "a", 1.0, layer=0)
    _write_axis(tmp_path, "b", 2.0, layer=1)
    expected_keys = ["layer0.q_proj", "layer1.q_proj"]
    loaded = load_independent_loras(
        tmp_path, ["a", "b"], expected_keys, "base/model", allow_sparse=True,
    )
    assert loaded.keys == expected_keys
    merged = loaded.task_arithmetic([1.0, 1.0])
    torch.testing.assert_close(merged["layer0.q_proj"], loaded.dense_axis("a")["layer0.q_proj"])
    torch.testing.assert_close(merged["layer1.q_proj"], loaded.dense_axis("b")["layer1.q_proj"])

    try:
        load_independent_loras(tmp_path, ["a", "b"], expected_keys, "base/model")
    except ValueError as error:
        assert "attachment mismatch" in str(error)
    else:
        raise AssertionError("Strict loading unexpectedly accepted sparse adapters")


def test_uniform_single_layer_set_is_accepted_as_model_projection_subset(tmp_path):
    _write_axis(tmp_path, "a", 1.0, layer=13)
    _write_axis(tmp_path, "b", 2.0, layer=13)
    loaded = load_independent_loras(tmp_path, ["a", "b"])
    assert loaded.validate_uniform_subset([
        "layer0.q_proj", "layer13.q_proj", "layer15.q_proj",
    ]) == ["layer13.q_proj"]


def test_distributional_teacher_has_exact_base_and_pure_axis_boundaries():
    teacher = torch.log_softmax(torch.tensor([[
        [2.0, 0.0, -1.0],
        [-1.0, 3.0, 0.0],
        [0.0, -2.0, 4.0],
    ]]), dim=-1)
    values = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = compositional_teacher_log_probs(teacher, values)
    torch.testing.assert_close(target[:, 0], teacher[:, 0])
    torch.testing.assert_close(target[:, 1], teacher[:, 1])
    torch.testing.assert_close(target[:, 2], teacher[:, 2])
    expected_joint = torch.log_softmax(teacher[:, 1] + teacher[:, 2] - teacher[:, 0], dim=-1)
    torch.testing.assert_close(target[:, 3], expected_joint)

    # The same identities must hold independently at every shared
    # autoregressive-prefix context.
    contextual = teacher.unsqueeze(1).repeat(1, 3, 1, 1)
    contextual[:, 1] = torch.log_softmax(contextual[:, 1] + 0.2, dim=-1)
    contextual_target = compositional_teacher_log_probs(contextual, values)
    assert contextual_target.shape == (1, 4, 3, 3)
    torch.testing.assert_close(contextual_target[:, 0], contextual[:, :, 0])
    torch.testing.assert_close(contextual_target[:, 1], contextual[:, :, 1])
    torch.testing.assert_close(contextual_target[:, 2], contextual[:, :, 2])


class _FakeAdapted(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = _FakeAdapted()
        self._adapted_linears = {"layer0.q_proj": self.projection}

    def forward_base(self, input_ids, attention_mask=None):
        return self.projection(input_ids)


def test_multi_affine_interactions_vanish_on_pure_axes_and_round_trip(tmp_path):
    _write_axis(tmp_path, "a", 1.0)
    _write_axis(tmp_path, "b", 2.0)
    loaded = load_independent_loras(tmp_path, ["a", "b"], ["layer0.q_proj"], "base/model")
    interactions = MultiAffineInteractions(loaded, rank=2, seed=7)
    branch = interactions.branch((0, 1), "layer0.q_proj")
    with torch.no_grad():
        branch.A.copy_(torch.eye(2, 3))
        branch.B.fill_(1.0)
        interactions.subset_gains.fill_(0.25)

    model = _FakeModel()
    x = torch.ones(3, 1, 3)
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    with MultiAffineInteractionHooks(model, interactions, values):
        output = model.forward_base(x)
    torch.testing.assert_close(output[0], torch.zeros_like(output[0]))
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    assert output[2].abs().sum() > 0

    with MultiAffineInteractionHooks(
        model, interactions, values,
        subset_gains=torch.tensor([[1.0], [1.0], [0.0]]),
    ):
        suppressed = model.forward_base(x)
    torch.testing.assert_close(suppressed, torch.zeros_like(suppressed))

    artifact = tmp_path / "artifact"
    interactions.save_artifact(artifact, {"test": True})
    restored = MultiAffineInteractions.load_artifact(artifact, loaded)
    for key, value in interactions.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_factor_first_cached_in_inference_mode_can_participate_in_backward(tmp_path):
    _write_axis(tmp_path, "a", 1.0)
    loaded = load_independent_loras(tmp_path, ["a"], ["layer0.q_proj"], "base/model")
    with torch.inference_mode():
        cached = loaded.prepared_factor(
            "a", "layer0.q_proj", torch.device("cpu"), torch.float32,
        )
    assert not cached.A.is_inference()
    assert not cached.B.is_inference()

    x = torch.ones(1, 3, requires_grad=True)
    output = torch.nn.functional.linear(
        torch.nn.functional.linear(x, cached.A), cached.B,
    ).sum()
    output.backward()
    assert x.grad is not None


def test_interaction_geometry_reports_nonzero_correction_ratio(tmp_path):
    _write_axis(tmp_path, "a", 1.0)
    _write_axis(tmp_path, "b", 2.0)
    loaded = load_independent_loras(tmp_path, ["a", "b"], ["layer0.q_proj"], "base/model")
    interactions = MultiAffineInteractions(loaded, rank=2, seed=3)
    with torch.no_grad():
        branch = interactions.branch((0, 1), "layer0.q_proj")
        branch.A.copy_(torch.eye(2, 3))
        branch.B.fill_(0.5)
    report = interaction_geometry(loaded, interactions)
    subset = report["subsets"]["a+b"]
    assert subset["interaction_frobenius"] > 0
    assert subset["interaction_to_task_sum_ratio"] > 0
    assert report["binary_combinations"]["a+b"]["interaction_frobenius"] > 0


def test_complete_layer_source_capture_includes_fitted_interactions(tmp_path):
    _write_axis(tmp_path, "a", 1.0)
    _write_axis(tmp_path, "b", 2.0)
    loaded = load_independent_loras(tmp_path, ["a", "b"], ["layer0.q_proj"], "base/model")
    interactions = MultiAffineInteractions(loaded, rank=2, seed=3)
    with torch.no_grad():
        branch = interactions.branch((0, 1), "layer0.q_proj")
        branch.A.copy_(torch.eye(2, 3))
        branch.B.fill_(0.5)

    projection = _FakeAdapted()

    class Layer(torch.nn.Module):
        def forward(self, hidden_states):
            residual = torch.nn.functional.pad(hidden_states, (0, 1))
            return residual + torch.tanh(projection(hidden_states))

    layer = Layer()

    class Model:
        _adapted_linears = {"layer0.q_proj": projection}

        @staticmethod
        def forward_base(input_ids, attention_mask=None):
            return layer(input_ids)

    ctx = SimpleNamespace(model=Model(), layers=[layer])
    hidden = torch.ones(1, 1, 3)

    def capture_field():
        capture = _CompositionalLayerSourceCapture(
            ctx, loaded, interactions, torch.tensor([[1.0, 1.0]]),
        )
        try:
            ctx.model.forward_base(hidden)
        finally:
            capture.close()
        return capture.fields[0]

    with_interaction = capture_field()
    component_capture = _CompositionalLayerSourceCapture(
        ctx, loaded, interactions,
        branch_rows=torch.tensor([[1.0, 1.0, 1.0]]),
    )
    try:
        ctx.model.forward_base(hidden)
    finally:
        component_capture.close()
    assert torch.allclose(with_interaction, component_capture.fields[0])
    with torch.no_grad():
        branch.B.zero_()
    without_interaction = capture_field()
    assert (with_interaction - without_interaction).abs().sum() > 0
