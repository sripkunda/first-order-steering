from __future__ import annotations

import itertools
from types import SimpleNamespace

import torch

from analysis.activation import (
    _IndependentLayerSourceJacobianCapture,
    _LayerSourceJacobianCapture,
    _StaticProjectionInjector,
    _analytical_taylor_factors_multi,
    _constant_least_squares_statistics,
    analytical_taylor_factors,
    analytical_taylor_coefficients,
    default_axis_directions,
    grid_axis_directions,
)
from analysis.common import AnchoredNeuralMergerWeightAdapter, active_factors
from analysis.independent import IndependentFactor, IndependentLoRAs, IndependentLoRAHooks
from steered_finetuner.model import AdaptedLinear


def _adapter() -> AnchoredNeuralMergerWeightAdapter:
    torch.manual_seed(7)
    adapter = AnchoredNeuralMergerWeightAdapter(
        d_in=6, d_out=5, signal_dim=2, rank=3, adapter_hidden_dim=7,
    ).float()
    with torch.no_grad():
        for gate in adapter.axis_gates:
            gate[2].weight.normal_(std=0.2)
            gate[2].bias.normal_(std=0.2)
        adapter.axis_B.normal_()
        adapter.neutral_B.normal_()
        adapter.interaction_B.normal_()
    return adapter


def test_analytical_coefficients_match_exact_first_derivative():
    adapter = _adapter()
    direction = torch.tensor([0.3, 0.7])
    epsilon = 1e-3
    zero = torch.zeros(1, 2)
    plus = adapter._all_coefficients((epsilon * direction).reshape(1, -1))[0]
    minus = adapter._all_coefficients((-epsilon * direction).reshape(1, -1))[0]
    numerical = (plus - minus) / (2.0 * epsilon)
    _, analytical, _ = analytical_taylor_coefficients(adapter, direction, 0.0, 1)
    torch.testing.assert_close(analytical, numerical, atol=2e-4, rtol=2e-3)


def test_analytical_coefficients_match_exact_second_derivative():
    adapter = _adapter()
    direction = torch.tensor([0.4, 0.6])
    epsilon = 1e-2
    zero_signal = torch.zeros(1, 2)
    zero = adapter._all_coefficients(zero_signal)[0]
    plus = adapter._all_coefficients((epsilon * direction).reshape(1, -1))[0]
    minus = adapter._all_coefficients((-epsilon * direction).reshape(1, -1))[0]
    numerical = (plus - 2.0 * zero + minus) / (epsilon * epsilon)
    _, _, analytical = analytical_taylor_coefficients(adapter, direction, 0.0, 2)
    torch.testing.assert_close(analytical, numerical, atol=3e-3, rtol=1e-2)


def test_first_order_factor_path_is_exact_for_initial_linear_axis_gate():
    adapter = AnchoredNeuralMergerWeightAdapter(
        d_in=6, d_out=5, signal_dim=2, rank=3, adapter_hidden_dim=7,
    ).float()
    with torch.no_grad():
        adapter.axis_B.normal_()
        adapter.neutral_B.normal_()
        adapter.interaction_B.normal_()
    strength = 0.73
    direction = [1.0, 0.0]
    coefficients, _, _ = analytical_taylor_coefficients(adapter, direction, strength, 1)
    exact = adapter._all_coefficients(torch.tensor([[strength, 0.0]]))[0]
    torch.testing.assert_close(coefficients, exact)


def test_analytical_coefficients_support_bfloat16_adapters():
    adapter = _adapter().bfloat16()
    value, first, second = analytical_taylor_coefficients(
        adapter, [0.5, 0.5], strength=1.0, order=2,
    )
    assert value.dtype == torch.float32
    assert first.isfinite().all()
    assert second.isfinite().all()


def test_default_directions_cover_every_binary_corner():
    directions = default_axis_directions(["a", "b", "c"])
    assert len(directions) == 8
    assert directions["neutral"] == [0.0, 0.0, 0.0]
    assert directions["a+b+c"] == [1.0, 1.0, 1.0]
    assert {tuple(value) for value in directions.values()} == {
        tuple(float(bit) for bit in corner)
        for corner in itertools.product((0, 1), repeat=3)
    }


def test_grid_directions_cover_every_cartesian_point():
    directions = grid_axis_directions(["a", "b", "c"], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert len(directions) == 125
    assert directions["a=0,b=0,c=0"] == [0.0, 0.0, 0.0]
    assert directions["a=1,b=1,c=1"] == [1.0, 1.0, 1.0]


def test_constant_least_squares_is_the_sample_mean():
    samples = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
    mean, target_energy, mean_energy, residual_energy = _constant_least_squares_statistics(
        samples.sum(0), samples.square().sum(), len(samples),
    )
    torch.testing.assert_close(mean, torch.tensor([2.0, 2.0]))
    assert target_energy == 10.0
    assert mean_energy == 8.0
    assert residual_energy == 2.0


def test_direction_batched_taylor_factors_match_single_direction_factors():
    adapter = _adapter()
    ctx = SimpleNamespace(adapters=[("layer0.q_proj", adapter)])
    directions = [[1.0, 0.0], [0.0, 1.0]]
    multi_A, multi_B = _analytical_taylor_factors_multi(
        ctx, directions, strength=0.7, order=2, repeats=2,
    )["layer0.q_proj"]
    for direction_index, direction in enumerate(directions):
        single_A, single_B = analytical_taylor_factors(
            ctx, direction, strength=0.7, order=2, batch_size=2,
        )["layer0.q_proj"]
        start = 2 * direction_index
        torch.testing.assert_close(multi_A[start:start + 2], single_A)
        torch.testing.assert_close(multi_B[start:start + 2], single_B)


def test_static_projection_injector_broadcasts_distinct_batch_vectors():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2, bias=False)

        def forward(self, value):
            return self.linear(value)

    projection = Projection()
    model = SimpleNamespace(_adapted_linears={"layer0.q_proj": projection})
    values = torch.randn(2, 4, 3)
    vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    baseline = projection(values)
    injector = _StaticProjectionInjector(model, {"layer0.q_proj": vectors})
    try:
        steered = projection(values)
    finally:
        injector.close()
    torch.testing.assert_close(steered - baseline, vectors.unsqueeze(1).expand(-1, 4, -1))


def _layer_with_factors(layer, model, hidden, factors):
    with active_factors(model, factors):
        return layer(hidden)


def test_layer_source_capture_matches_exact_complete_layer_jvp():
    torch.manual_seed(11)
    adapter = AnchoredNeuralMergerWeightAdapter(
        d_in=4, d_out=4, signal_dim=2, rank=2, adapter_hidden_dim=5,
    ).float()
    with torch.no_grad():
        adapter.neutral_B.normal_(std=0.1)
        adapter.axis_B.normal_(std=0.1)
    wrapped = AdaptedLinear(torch.nn.Linear(4, 4, bias=False), adapter)

    class ToyLayer(torch.nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def forward(self, hidden_states):
            value = self.linear(torch.nn.functional.layer_norm(hidden_states, (4,)))
            return hidden_states + torch.tanh(value)

    layer = ToyLayer(wrapped)
    model = SimpleNamespace(_ctx={"active": False, "precomputed": {}})
    wrapped._ctx = model._ctx
    wrapped._ctx_key = "layer0.q_proj"
    ctx = SimpleNamespace(
        model=model,
        layers=[layer],
        adapters=[("layer0.q_proj", adapter)],
    )
    hidden = torch.randn(3, 2, 4)
    direction = torch.tensor([0.4, -0.2])
    zero = torch.zeros(3, 2)
    basis = direction.reshape(1, -1).expand(3, -1)
    neutral = {"layer0.q_proj": adapter.compute_AB(zero)}

    def factors_at(alpha):
        return {"layer0.q_proj": adapter.compute_AB(alpha * basis)}

    primal_expected, derivative_expected = torch.autograd.functional.jvp(
        lambda alpha: _layer_with_factors(layer, model, hidden, factors_at(alpha)),
        (torch.zeros(()),), (torch.ones(()),), strict=True,
    )
    # The anchored path can be nonlinear, so obtain its exact derivative at
    # zero rather than subtracting the finite-strength endpoint factors.
    (_A0, _B0), (dA, dB) = torch.autograd.functional.jvp(
        lambda alpha: adapter.compute_AB(alpha * basis),
        (torch.zeros(()),), (torch.ones(()),), strict=False,
    )
    tangents = {"layer0.q_proj": (dA, dB)}
    capture = _LayerSourceJacobianCapture(ctx, neutral, tangents)
    try:
        with active_factors(model, neutral):
            primal = layer(hidden)
    finally:
        capture.close()
    torch.testing.assert_close(primal, primal_expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(capture.fields[0], derivative_expected, atol=1e-5, rtol=1e-5)


def test_independent_layer_source_capture_matches_exact_complete_layer_jvp(tmp_path):
    torch.manual_seed(13)
    wrapped = AdaptedLinear(
        torch.nn.Linear(4, 4, bias=False),
        AnchoredNeuralMergerWeightAdapter(
            d_in=4, d_out=4, signal_dim=2, rank=2, adapter_hidden_dim=5,
        ).float(),
    )

    class ToyLayer(torch.nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def forward(self, hidden_states):
            value = self.linear(torch.nn.functional.layer_norm(hidden_states, (4,)))
            return hidden_states + torch.tanh(value)

    layer = ToyLayer(wrapped)
    model = SimpleNamespace(
        _ctx={"active": False, "precomputed": {}},
        _adapted_linears={"layer0.q_proj": wrapped},
    )
    wrapped._ctx = model._ctx
    wrapped._ctx_key = "layer0.q_proj"
    axes = ["a", "b"]
    factors = {}
    for axis in axes:
        factors[axis] = {
            "layer0.q_proj": IndependentFactor(
                A=torch.randn(2, 4), B=torch.randn(4, 2), scale=0.5,
            )
        }
    baseline = IndependentLoRAs(tmp_path, axes, "toy", factors)
    ctx = SimpleNamespace(model=model, layers=[layer])
    hidden = torch.randn(3, 2, 4)
    direction = torch.tensor([0.4, -0.2]).reshape(1, -1).expand(3, -1)

    def at(alpha):
        with IndependentLoRAHooks(model, baseline, alpha * direction):
            return layer(hidden)

    primal_expected, derivative_expected = torch.autograd.functional.jvp(
        at, (torch.zeros(()),), (torch.ones(()),), strict=True,
    )
    capture = _IndependentLayerSourceJacobianCapture(ctx, baseline, direction)
    try:
        primal = layer(hidden)
    finally:
        capture.close()
    torch.testing.assert_close(primal, primal_expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(capture.fields[0], derivative_expected, atol=1e-5, rtol=1e-5)
