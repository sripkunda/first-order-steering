from __future__ import annotations

import contextlib
import logging
import math
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from .common import AnalysisContext, active_factors, chunked, encode, layer_from_key, signal
from .independent import IndependentLoRAHooks, IndependentLoRAs, load_independent_loras

LOG = logging.getLogger(__name__)

ACTIVATION_GENERATION_METHODS = {
    "hera_weight",
    "hera_jacobian",
    "hera_static",
    "independent_weight",
    "independent_jacobian",
    "independent_static",
}


class _CaptureLayer:
    def __init__(self, layer):
        self.output = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.output = output[0] if isinstance(output, tuple) else output

    def close(self):
        self.handle.remove()


class _TokenwiseInjector:
    def __init__(self, layers, fields: Mapping[int, torch.Tensor], strength: float,
                 last_token_only: bool = False, steer_prefill: bool = True):
        self.fields = fields
        self.strength = strength
        self.last_token_only = last_token_only
        self.steer_prefill = steer_prefill
        # ``generate`` first runs a full prompt prefill, then one cached decode
        # step at a time.  Shape alone cannot reliably identify the prefill (a
        # one-token prompt is valid), so mark its completion at the final decoder
        # layer instead.  This leaves every later autoregressive step steered.
        self._prefill_pending = not steer_prefill
        self.handles = [layers[layer].register_forward_hook(self._hook(layer)) for layer in fields]
        self._prefill_complete_handle = None
        if not steer_prefill:
            self._prefill_complete_handle = layers[-1].register_forward_hook(
                self._finish_prefill, always_call=True,
            )

    def _finish_prefill(self, _module, _inputs, output):
        self._prefill_pending = False
        return output

    def _hook(self, layer):
        def inject(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if self._prefill_pending:
                return output
            field = self.fields[layer].to(value)
            if field.ndim == 2 and value.ndim == 3:
                field = field.unsqueeze(1)
            if self.last_token_only and value.ndim == 3:
                updated = value.clone()
                if field.ndim == 3:
                    field = field[:, -1]
                updated[:, -1] = updated[:, -1] + self.strength * field
            else:
                updated = value + self.strength * field
            return (updated, *output[1:]) if isinstance(output, tuple) else updated
        return inject

    def close(self):
        for handle in self.handles:
            handle.remove()
        if self._prefill_complete_handle is not None:
            self._prefill_complete_handle.remove()


class _LayerSourceJacobianCapture:
    """Capture exact local layer-output signal JVPs in one model pass.

    Each decoder layer receives a zero input tangent and dual LoRA factors
    carrying the requested signal direction.  Forward AD then applies the
    ordinary chain rule through the complete layer (normalization, attention,
    softmax, MLP gates, and residual path).  Resetting the hidden-state tangent
    at every layer isolates the direct source produced by that layer's adapted
    weights, matching ``exact_local_directional_fields`` without rerunning the
    transformer once per layer.
    """

    def __init__(self, ctx: AnalysisContext, neutral_factors, tangent_factors):
        self.ctx = ctx
        self.neutral_factors = neutral_factors
        self.tangent_factors = tangent_factors
        self.fields: Dict[int, torch.Tensor] = {}
        self._levels = {}
        self._layer_keys = {
            layer: [key for key, _adapter in ctx.adapters if key.startswith(f"layer{layer}.")]
            for layer in range(len(ctx.layers))
        }
        self._pre_handles = [
            layer.register_forward_pre_hook(self._pre_hook(index), with_kwargs=True)
            for index, layer in enumerate(ctx.layers)
        ]
        self._post_handles = [
            layer.register_forward_hook(
                self._post_hook(index), with_kwargs=True, always_call=True,
            )
            for index, layer in enumerate(ctx.layers)
        ]

    @staticmethod
    def _replace_hidden(args, kwargs, hidden):
        if args:
            return (hidden, *args[1:]), kwargs
        if "hidden_states" in kwargs:
            updated = dict(kwargs)
            updated["hidden_states"] = hidden
            return args, updated
        raise RuntimeError("Decoder layer did not receive a hidden-state input")

    def _pre_hook(self, layer_index):
        def hook(_module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError("Decoder layer did not receive hidden states")
            level = torch.autograd.forward_ad.dual_level()
            level.__enter__()
            self._levels[layer_index] = level
            try:
                dual_hidden = torch.autograd.forward_ad.make_dual(
                    hidden, torch.zeros_like(hidden),
                )
                for key in self._layer_keys[layer_index]:
                    neutral_A, neutral_B = self.neutral_factors[key]
                    tangent_A, tangent_B = self.tangent_factors[key]
                    self.ctx.model._ctx["precomputed"][key] = (
                        torch.autograd.forward_ad.make_dual(
                            neutral_A.contiguous(), tangent_A.contiguous(),
                        ),
                        torch.autograd.forward_ad.make_dual(
                            neutral_B.contiguous(), tangent_B.contiguous(),
                        ),
                    )
                return self._replace_hidden(args, kwargs, dual_hidden)
            except Exception:
                self._restore(layer_index)
                raise
        return hook

    def _restore(self, layer_index):
        for key in self._layer_keys[layer_index]:
            self.ctx.model._ctx["precomputed"][key] = self.neutral_factors[key]
        level = self._levels.pop(layer_index, None)
        if level is not None:
            level.__exit__(None, None, None)

    def _post_hook(self, layer_index):
        def hook(_module, _args, _kwargs, output):
            if output is None:
                self._restore(layer_index)
                return None
            value = output[0] if isinstance(output, tuple) else output
            try:
                primal, tangent = torch.autograd.forward_ad.unpack_dual(value)
                if tangent is None:
                    tangent = torch.zeros_like(primal)
                self.fields[layer_index] = tangent.detach()
                if isinstance(output, tuple):
                    return (primal, *output[1:])
                return primal
            finally:
                self._restore(layer_index)
        return hook

    def close(self):
        for handle in self._pre_handles + self._post_handles:
            handle.remove()
        for layer_index in list(self._levels):
            self._restore(layer_index)


class _IndependentLayerSourceJacobianCapture:
    """Capture complete-layer source Jacobians for independent task addition.

    Independent LoRAs have the exactly linear path
    ``W(s) = W_base + sum_i s_i Delta W_i``.  At each decoder layer we reset
    the incoming hidden-state tangent to zero, seed the independent-LoRA
    coefficients with the requested directional tangent, and let forward AD
    propagate that source through the entire decoder block.  This is the
    placement-matched task-addition analogue of ``_LayerSourceJacobianCapture``.
    """

    def __init__(self, ctx: AnalysisContext, baseline: IndependentLoRAs,
                 direction_rows: torch.Tensor):
        if direction_rows.ndim != 2 or direction_rows.shape[1] != len(baseline.axes):
            raise ValueError(
                "Independent direction rows must have shape [batch, axes]"
            )
        self.ctx = ctx
        self.baseline = baseline
        self.direction_rows = direction_rows
        self.fields: Dict[int, torch.Tensor] = {}
        self._levels = {}
        self._projection_handles = defaultdict(list)
        self._layer_keys = {
            layer: [key for key in baseline.keys if key.startswith(f"layer{layer}.")]
            for layer in range(len(ctx.layers))
        }
        self._pre_handles = [
            layer.register_forward_pre_hook(self._pre_hook(index), with_kwargs=True)
            for index, layer in enumerate(ctx.layers)
        ]
        self._post_handles = [
            layer.register_forward_hook(
                self._post_hook(index), with_kwargs=True, always_call=True,
            )
            for index, layer in enumerate(ctx.layers)
        ]

    @staticmethod
    def _replace_hidden(args, kwargs, hidden):
        return _LayerSourceJacobianCapture._replace_hidden(args, kwargs, hidden)

    def _pre_hook(self, layer_index):
        def hook(_module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError("Decoder layer did not receive hidden states")
            level = torch.autograd.forward_ad.dual_level()
            level.__enter__()
            self._levels[layer_index] = level
            try:
                dual_hidden = torch.autograd.forward_ad.make_dual(
                    hidden, torch.zeros_like(hidden),
                )
                primal = torch.zeros(
                    hidden.shape[0], len(self.baseline.axes),
                    device=hidden.device, dtype=hidden.dtype,
                )
                tangent = self.direction_rows.to(device=hidden.device, dtype=hidden.dtype)
                coefficients = torch.autograd.forward_ad.make_dual(primal, tangent)
                for key in self._layer_keys[layer_index]:
                    adapted = self.ctx.model._adapted_linears[key]
                    weight = adapted.linear.weight
                    factors = [
                        (self.baseline.prepared_factor(axis, key, weight.device, weight.dtype)
                         if key in self.baseline.factors[axis] else None)
                        for axis in self.baseline.axes
                    ]
                    self._projection_handles[layer_index].append(
                        adapted.register_forward_hook(
                            IndependentLoRAHooks._hook(factors, coefficients)
                        )
                    )
                return self._replace_hidden(args, kwargs, dual_hidden)
            except Exception:
                self._restore(layer_index)
                raise
        return hook

    def _restore(self, layer_index):
        for handle in self._projection_handles.pop(layer_index, []):
            handle.remove()
        level = self._levels.pop(layer_index, None)
        if level is not None:
            level.__exit__(None, None, None)

    def _post_hook(self, layer_index):
        def hook(_module, _args, _kwargs, output):
            if output is None:
                self._restore(layer_index)
                return None
            value = output[0] if isinstance(output, tuple) else output
            try:
                primal, tangent = torch.autograd.forward_ad.unpack_dual(value)
                if tangent is None:
                    tangent = torch.zeros_like(primal)
                self.fields[layer_index] = tangent.detach()
                return (primal, *output[1:]) if isinstance(output, tuple) else primal
            finally:
                self._restore(layer_index)
        return hook

    def close(self):
        for handle in self._pre_handles + self._post_handles:
            handle.remove()
        for layer_index in list(self._levels):
            self._restore(layer_index)


class _ProjectionInputCapture:
    def __init__(self, model, attention_mask: torch.Tensor | None = None,
                 token_count: int | None = None):
        self.inputs = {}
        self.attention_mask = attention_mask
        self.token_count = token_count
        self.handles = [
            module.register_forward_pre_hook(self._hook(key))
            for key, module in model._adapted_linears.items()
        ]

    def _hook(self, key):
        def capture(_module, inputs):
            value = inputs[0]
            if self.attention_mask is not None and self.token_count is not None:
                value = _selected_projection_inputs(
                    value, self.attention_mask, self.token_count,
                )
            # Retain only the requested token rows. Keeping every layer's full
            # B x T activation would waste several GB during an 80-prompt fit.
            self.inputs[key] = value.detach()
        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()


class _StaticProjectionInjector:
    """Add one context-independent vector at every adapted projection output."""
    def __init__(self, model, vectors: Mapping[str, torch.Tensor]):
        self.handles = []
        for key, vector in vectors.items():
            module = model._adapted_linears[key]
            prepared = vector.to(
                device=module.linear.weight.device, dtype=module.linear.weight.dtype,
                non_blocking=True,
            )
            self.handles.append(module.register_forward_hook(self._hook(prepared)))

    @staticmethod
    def _hook(vector):
        def inject(_module, _inputs, output):
            if vector.ndim == 1:
                return output + vector
            if output.ndim == 3:
                return output + vector.unsqueeze(1)
            if output.ndim == 2 and vector.shape[0] == output.shape[0]:
                return output + vector
            raise ValueError(
                f"Static-vector batch {tuple(vector.shape)} cannot broadcast to "
                f"projection output {tuple(output.shape)}."
            )
        return inject

    def close(self):
        for handle in self.handles:
            handle.remove()


def _direction_signal(scalar: torch.Tensor, direction: torch.Tensor, batch_size: int) -> torch.Tensor:
    return scalar.reshape(1, 1) * direction.reshape(1, -1).expand(batch_size, -1)


def _gelu_derivative(value: torch.Tensor) -> torch.Tensor:
    """Derivative of torch.nn.GELU(approximate='none')."""
    return (0.5 * (1.0 + torch.erf(value / math.sqrt(2.0))) +
            value * torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi))


def _axis_gate_at_zero(adapter, axis: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return g_i(0) and dg_i/dt(0) analytically in FP32."""
    gate = adapter.axis_gates[axis]
    first, second = gate[0], gate[2]
    bias = first.bias.float()
    hidden = torch.nn.functional.gelu(bias)
    value = torch.nn.functional.linear(hidden, second.weight.float(), second.bias.float())
    hidden_derivative = _gelu_derivative(bias) * first.weight[:, 0].float()
    derivative = torch.nn.functional.linear(hidden_derivative, second.weight.float(), None)
    return value, derivative


def analytical_taylor_coefficients(adapter, direction: Sequence[float], strength: float,
                                     order: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return coefficient value, first derivative, and second derivative at s=0.

    Coefficients correspond exactly to ``adapter._basis_factors()``.  Only the
    small merger gates are evaluated here; no transformer JVP is involved.
    """
    if order not in (1, 2):
        raise ValueError("analytical Taylor order must be 1 or 2")
    device = adapter.axis_A.device
    direction_tensor = torch.as_tensor(direction, device=device, dtype=torch.float32)
    if direction_tensor.numel() != adapter.signal_dim:
        raise ValueError(f"Expected {adapter.signal_dim} steering coordinates, got {direction_tensor.numel()}.")
    base_parts = [torch.ones(adapter.rank, device=device)]
    first_parts = [torch.zeros(adapter.rank, device=device)]
    second_parts = [torch.zeros(adapter.rank, device=device)]
    for axis in range(adapter.signal_dim):
        gate_zero, gate_prime_zero = _axis_gate_at_zero(adapter, axis)
        coordinate = direction_tensor[axis]
        base_parts.append(torch.zeros(adapter.rank, device=device))
        first_parts.append(coordinate * (torch.ones_like(gate_zero) + gate_zero))
        second_parts.append(
            coordinate.square() * (-2.0 * gate_zero + 2.0 * gate_prime_zero)
        )
    if adapter.interaction_A is not None:
        interaction_gate = adapter.interaction_gate
        zero_signal = torch.zeros(1, adapter.signal_dim, device=device, dtype=torch.float32)
        interaction_hidden = torch.nn.functional.linear(
            zero_signal, interaction_gate[0].weight.float(), interaction_gate[0].bias.float(),
        )
        interaction_at_zero = torch.nn.functional.linear(
            torch.nn.functional.gelu(interaction_hidden),
            interaction_gate[2].weight.float(), interaction_gate[2].bias.float(),
        ).flatten()
        pair_sum = sum(
            direction_tensor[first] * direction_tensor[second]
            for first in range(adapter.signal_dim)
            for second in range(first + 1, adapter.signal_dim)
        )
        base_parts.append(torch.zeros(adapter.rank, device=device))
        first_parts.append(torch.zeros(adapter.rank, device=device))
        # q(alpha*d)''|_0 = 2 sum_{i<j} d_i d_j.  Since q(0)=q'(0)=0,
        # derivatives of the interaction MLP do not enter at second order.
        second_parts.append(2.0 * pair_sum * interaction_at_zero)
    base = torch.cat(base_parts)
    first = torch.cat(first_parts)
    second = torch.cat(second_parts)
    value = base + strength * first
    if order == 2:
        value = value + 0.5 * strength * strength * second
    return value, first, second


def analytical_taylor_factors(ctx: AnalysisContext, direction: Sequence[float], strength: float,
                              order: int, batch_size: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Compile W(0)+alpha W'(0)[d]+alpha^2 W''(0)[d,d]/2 for every projection."""
    factors = {}
    for key, adapter in ctx.adapters:
        coefficients, _, _ = analytical_taylor_coefficients(adapter, direction, strength, order)
        A_basis, B_basis = adapter._basis_factors()
        A = A_basis.unsqueeze(0).expand(batch_size, -1, -1)
        B = (B_basis.float() * coefficients.reshape(1, -1)).to(B_basis)
        factors[key] = (A, B.unsqueeze(0).expand(batch_size, -1, -1))
    return factors


def _analytical_taylor_factors_multi(ctx: AnalysisContext,
                                     directions: Sequence[Sequence[float]], strength: float,
                                     order: int, repeats: int) -> dict:
    """Compile distinct Taylor factors for a direction-major generation batch."""
    factors = {}
    for key, adapter in ctx.adapters:
        coefficients = torch.stack([
            analytical_taylor_coefficients(adapter, direction, strength, order)[0]
            for direction in directions
        ]).repeat_interleave(repeats, dim=0)
        A_basis, B_basis = adapter._basis_factors()
        factors[key] = (
            A_basis.unsqueeze(0).expand(coefficients.shape[0], -1, -1),
            B_basis.unsqueeze(0) * coefficients.to(B_basis).unsqueeze(1),
        )
    return factors


def _coefficient_deltas(ctx: AnalysisContext, directions: Mapping[str, Sequence[float]],
                        strength: float, orders: Sequence[int]) -> dict:
    result = {}
    for name, direction in directions.items():
        for order in orders:
            for key, adapter in ctx.adapters:
                value, _, _ = analytical_taylor_coefficients(adapter, direction, strength, order)
                base, _, _ = analytical_taylor_coefficients(adapter, direction, 0.0, order)
                result[(name, order, key)] = (value - base).detach()
    return result


def _selected_projection_inputs(value: torch.Tensor, attention_mask: torch.Tensor,
                                token_count: int) -> torch.Tensor:
    if token_count < 1:
        raise ValueError("static_fit_tokens must be positive")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    last = (positions * attention_mask.long()).amax(1, keepdim=True)
    selected = attention_mask.bool() & (positions > last - token_count)
    return value[selected]


def _constant_least_squares_statistics(total: torch.Tensor, squared_norm_total: torch.Tensor,
                                       count: int) -> Tuple[torch.Tensor, float, float, float]:
    """Return the optimal constant and its target, mean, and residual energies."""
    if count < 1:
        raise ValueError("A static least-squares fit requires at least one activation sample.")
    mean = total / count
    mean_energy = float(mean.square().sum())
    target_energy = float(squared_norm_total / count)
    # E||a-c*||^2 = E||a||^2-||E[a]||^2. Clamp only roundoff.
    residual_energy = max(0.0, target_energy - mean_energy)
    return mean, target_energy, mean_energy, residual_energy


def fit_static_projection_vectors(ctx: AnalysisContext, prompts: Sequence[dict],
                                  directions: Mapping[str, Sequence[float]], strength: float,
                                  orders: Sequence[int], batch_size: int,
                                  max_prompt_tokens: int, token_count: int) -> Tuple[dict, dict]:
    """Least-squares fit of a context-independent vector at every projection.

    The field tuple for one layer lives in the direct sum of its projection
    output spaces.  Euclidean least squares in each component gives its sample
    mean.  Projection-normalized and energy-weighted coherence are reported to
    evaluate this static compression without letting wide MLP projections
    dominate solely because of dimensionality.
    """
    if not prompts:
        return {}, {}
    deltas = _coefficient_deltas(ctx, directions, strength, orders)
    conditions = [(name, order) for name in directions for order in orders]
    stacked_deltas = {
        key: torch.stack([deltas[(name, order, key)] for name, order in conditions])
        for key, _adapter in ctx.adapters
    }
    sums = {}
    squared_norm_sums = {}
    counts = defaultdict(int)
    prompt_list = list(prompts)
    for batch_index, batch in enumerate(chunked(prompt_list, batch_size)):
        LOG.info("Static analytical field fit batch %d/%d", batch_index + 1,
                 math.ceil(len(prompt_list) / batch_size))
        encoded = encode(ctx.tokenizer, [item["prompt"] for item in batch],
                         ctx.device, max_prompt_tokens)
        neutral_signal = torch.zeros(
            len(batch), len(ctx.axes), device=ctx.device, dtype=ctx.dtype,
        )
        capture = _ProjectionInputCapture(
            ctx.model, encoded["attention_mask"], token_count,
        )
        try:
            with torch.no_grad(), active_factors(
                ctx.model, ctx.model.lora.precompute_all(neutral_signal),
            ):
                ctx.model.llama(**encoded, use_cache=False)
        finally:
            capture.close()
        for key, adapter in ctx.adapters:
            selected = capture.inputs[key].float()
            A_basis, B_basis = adapter._basis_factors()
            reduced = selected @ A_basis.float().T
            # Evaluate every direction/order in one contraction for this
            # projection rather than launching thousands of tiny GEMMs.
            field = adapter.scale * torch.einsum(
                "mr,cr,or->cmo", reduced, stacked_deltas[key].to(reduced), B_basis.float(),
            )
            field_sum = field.sum(1)
            field_squared = field.square().sum((1, 2))
            sums[key] = field_sum if key not in sums else sums[key] + field_sum
            squared_norm_sums[key] = (
                field_squared if key not in squared_norm_sums
                else squared_norm_sums[key] + field_squared
            )
            counts[key] += field.shape[1]

    vectors, metrics = {}, {}
    condition_indices = {condition: index for index, condition in enumerate(conditions)}
    for name in directions:
        for order in orders:
            by_projection = {}
            layer_values = defaultdict(list)
            current_vectors = {}
            condition_index = condition_indices[(name, order)]
            for key, _adapter in ctx.adapters:
                count = counts[key]
                mean, target_energy, mean_energy, residual_energy = (
                    _constant_least_squares_statistics(
                        sums[key][condition_index], squared_norm_sums[key][condition_index], count,
                    )
                )
                coherence = mean_energy / target_energy if target_energy > 1e-20 else 1.0
                current_vectors[key] = mean.detach().cpu()
                row = {
                    "sample_count": count,
                    "target_field_rms_l2": math.sqrt(max(target_energy, 0.0)),
                    "static_vector_l2": math.sqrt(max(mean_energy, 0.0)),
                    "least_squares_residual_rms_l2": math.sqrt(residual_energy),
                    "constant_field_coherence": coherence,
                }
                by_projection[key] = row
                layer_values[layer_from_key(key)].append((target_energy, mean_energy, residual_energy))
            by_layer = {}
            for layer, values in sorted(layer_values.items()):
                target_total = sum(value[0] for value in values)
                mean_total = sum(value[1] for value in values)
                by_layer[str(layer)] = {
                    "energy_weighted_constant_field_coherence": (
                        mean_total / target_total if target_total > 1e-20 else 1.0
                    ),
                    "mean_projection_normalized_coherence": sum(
                        value[1] / value[0] if value[0] > 1e-20 else 1.0
                        for value in values
                    ) / len(values),
                    "product_space_relative_residual": math.sqrt(
                        sum(value[2] for value in values) / target_total
                    ) if target_total > 1e-20 else 0.0,
                }
            all_values = [value for values in layer_values.values() for value in values]
            target_total = sum(value[0] for value in all_values)
            mean_total = sum(value[1] for value in all_values)
            vectors[(name, order)] = current_vectors
            metrics[(name, order)] = {
                "global": {
                    "energy_weighted_constant_field_coherence": (
                        mean_total / target_total if target_total > 1e-20 else 1.0
                    ),
                    "mean_projection_normalized_coherence": sum(
                        value[1] / value[0] if value[0] > 1e-20 else 1.0
                        for value in all_values
                    ) / len(all_values),
                    "product_space_relative_residual": math.sqrt(
                        sum(value[2] for value in all_values) / target_total
                    ) if target_total > 1e-20 else 0.0,
                },
                "by_layer": by_layer,
                "by_projection": by_projection,
            }
    return vectors, metrics


def _resolve_static_axis_layers(
    ctx: AnalysisContext,
    layer_indices: Sequence[int] | None,
    layers_by_axis: Mapping[str, int | Sequence[int]] | None,
) -> dict[int, list[int]]:
    if layer_indices is not None and layers_by_axis:
        raise ValueError("Use only one of static_layers and static_layers_by_axis")
    if layers_by_axis:
        unknown = sorted(set(layers_by_axis) - set(ctx.axes))
        missing = sorted(set(ctx.axes) - set(layers_by_axis))
        if unknown or missing:
            raise ValueError(
                f"Axis-specific static layers have unknown={unknown}, missing={missing}"
            )
        result = {}
        for axis_index, axis in enumerate(ctx.axes):
            raw = layers_by_axis[axis]
            values = [raw] if isinstance(raw, int) else list(raw)
            result[axis_index] = sorted({int(value) for value in values})
    else:
        values = list(range(len(ctx.layers))) if layer_indices is None else list(layer_indices)
        result = {axis_index: values for axis_index in range(len(ctx.axes))}
    for axis_index, values in result.items():
        if not values or any(layer < 0 or layer >= len(ctx.layers) for layer in values):
            raise ValueError(
                f"Static layers for {ctx.axes[axis_index]!r} must be in "
                f"[0, {len(ctx.layers) - 1}]; got {values}"
            )
    return result


def fit_static_layer_vectors(
    ctx: AnalysisContext,
    prompts: Sequence[dict],
    batch_size: int,
    max_prompt_tokens: int,
    token_count: int,
    layer_indices: Sequence[int] | None = None,
    layers_by_axis: Mapping[str, int | Sequence[int]] | None = None,
) -> Tuple[dict, dict]:
    """Fit composable residual vectors to complete-layer Jacobian columns.

    Each axis column is computed in a direction-batched forward-mode pass. The
    Euclidean least-squares optimum at a layer is its calibration-sample mean.
    """
    if not prompts:
        return {}, {}
    if token_count < 1:
        raise ValueError("static_fit_tokens must be positive")
    axis_count = len(ctx.axes)
    axis_layers = _resolve_static_axis_layers(ctx, layer_indices, layers_by_axis)
    selected_layer_set = set().union(*map(set, axis_layers.values()))
    basis = torch.eye(axis_count, dtype=torch.float32)
    sums: dict[tuple[int, int], torch.Tensor] = {}
    squared: dict[tuple[int, int], torch.Tensor] = {}
    counts = defaultdict(int)
    batches = list(chunked(list(prompts), batch_size))
    for batch_index, batch in enumerate(batches):
        LOG.info("Static complete-layer Jacobian fit batch %d/%d", batch_index + 1, len(batches))
        text = [item["prompt"] for _axis in range(axis_count) for item in batch]
        encoded = encode(ctx.tokenizer, text, ctx.device, max_prompt_tokens)
        rows = basis.repeat_interleave(len(batch), dim=0)
        prepared = _analytical_layer_factor_tangents(ctx, rows, len(text))
        _cache, fields = analytical_layer_directional_fields(
            ctx, encoded, rows, prepared_factors=prepared,
        )
        mask = encoded["attention_mask"]
        for layer, field in fields.items():
            if layer not in selected_layer_set:
                continue
            for axis in range(axis_count):
                if layer not in axis_layers[axis]:
                    continue
                start, stop = axis * len(batch), (axis + 1) * len(batch)
                selected = _selected_projection_inputs(
                    field[start:stop], mask[start:stop], token_count,
                ).float()
                key = (axis, layer)
                value_sum = selected.sum(0).cpu()
                value_squared = selected.square().sum().cpu()
                sums[key] = value_sum if key not in sums else sums[key] + value_sum
                squared[key] = (value_squared if key not in squared
                                else squared[key] + value_squared)
                counts[key] += selected.shape[0]

    vectors = {axis: {} for axis in ctx.axes}
    metrics = {axis: {"by_layer": {}} for axis in ctx.axes}
    for axis_index, axis_name in enumerate(ctx.axes):
        totals = []
        for layer in axis_layers[axis_index]:
            key = (axis_index, layer)
            mean, target, mean_energy, residual = _constant_least_squares_statistics(
                sums[key], squared[key], counts[key],
            )
            vectors[axis_name][layer] = mean
            totals.append((target, mean_energy, residual))
            metrics[axis_name]["by_layer"][str(layer)] = {
                "sample_count": counts[key],
                "target_field_rms_l2": math.sqrt(max(target, 0.0)),
                "static_vector_l2": math.sqrt(max(mean_energy, 0.0)),
                "least_squares_residual_rms_l2": math.sqrt(residual),
                "constant_field_coherence": mean_energy / target if target > 1e-20 else 1.0,
            }
        target = sum(value[0] for value in totals)
        mean_energy = sum(value[1] for value in totals)
        residual = sum(value[2] for value in totals)
        metrics[axis_name]["global"] = {
            "selected_layers": axis_layers[axis_index],
            "energy_weighted_constant_field_coherence": (
                mean_energy / target if target > 1e-20 else 1.0
            ),
            "mean_layer_normalized_coherence": sum(
                value[1] / value[0] if value[0] > 1e-20 else 1.0 for value in totals
            ) / len(totals),
            "residual_direct_sum_relative_error": (
                math.sqrt(residual / target) if target > 1e-20 else 0.0
            ),
        }
    return vectors, metrics


def fit_independent_static_layer_vectors(
    ctx: AnalysisContext,
    baseline: IndependentLoRAs,
    prompts: Sequence[dict],
    batch_size: int,
    max_prompt_tokens: int,
    token_count: int,
    layer_indices: Sequence[int] | None = None,
    layers_by_axis: Mapping[str, int | Sequence[int]] | None = None,
) -> Tuple[dict, dict]:
    """Fit residual-space static vectors to independent task-addition sources.

    This is the exact placement-matched counterpart of
    ``fit_static_layer_vectors``: both fits average the complete decoder-layer
    source Jacobian, not individual projection outputs.
    """
    if not prompts:
        return {}, {}
    if token_count < 1:
        raise ValueError("static_fit_tokens must be positive")
    axis_count = len(ctx.axes)
    axis_layers = _resolve_static_axis_layers(ctx, layer_indices, layers_by_axis)
    selected_layer_set = set().union(*map(set, axis_layers.values()))
    basis = torch.eye(axis_count, dtype=torch.float32)
    sums: dict[tuple[int, int], torch.Tensor] = {}
    squared: dict[tuple[int, int], torch.Tensor] = {}
    counts = defaultdict(int)
    batches = list(chunked(list(prompts), batch_size))
    for batch_index, batch in enumerate(batches):
        LOG.info(
            "Independent static complete-layer fit batch %d/%d",
            batch_index + 1, len(batches),
        )
        text = [item["prompt"] for _axis in range(axis_count) for item in batch]
        encoded = encode(ctx.tokenizer, text, ctx.device, max_prompt_tokens)
        rows = basis.repeat_interleave(len(batch), dim=0)
        _cache, fields = independent_layer_directional_fields(
            ctx, baseline, encoded, rows,
        )
        mask = encoded["attention_mask"]
        for layer, field in fields.items():
            if layer not in selected_layer_set:
                continue
            for axis in range(axis_count):
                if layer not in axis_layers[axis]:
                    continue
                start, stop = axis * len(batch), (axis + 1) * len(batch)
                selected = _selected_projection_inputs(
                    field[start:stop], mask[start:stop], token_count,
                ).float()
                key = (axis, layer)
                value_sum = selected.sum(0).cpu()
                value_squared = selected.square().sum().cpu()
                sums[key] = value_sum if key not in sums else sums[key] + value_sum
                squared[key] = (
                    value_squared if key not in squared else squared[key] + value_squared
                )
                counts[key] += selected.shape[0]

    vectors = {axis: {} for axis in ctx.axes}
    metrics = {axis: {"by_layer": {}} for axis in ctx.axes}
    for axis_index, axis_name in enumerate(ctx.axes):
        totals = []
        for layer in axis_layers[axis_index]:
            key = (axis_index, layer)
            mean, target, mean_energy, residual = _constant_least_squares_statistics(
                sums[key], squared[key], counts[key],
            )
            vectors[axis_name][layer] = mean
            totals.append((target, mean_energy, residual))
            metrics[axis_name]["by_layer"][str(layer)] = {
                "sample_count": counts[key],
                "target_field_rms_l2": math.sqrt(max(target, 0.0)),
                "static_vector_l2": math.sqrt(max(mean_energy, 0.0)),
                "least_squares_residual_rms_l2": math.sqrt(residual),
                "constant_field_coherence": mean_energy / target if target > 1e-20 else 1.0,
            }
        target = sum(value[0] for value in totals)
        mean_energy = sum(value[1] for value in totals)
        residual = sum(value[2] for value in totals)
        metrics[axis_name]["global"] = {
            "selected_layers": axis_layers[axis_index],
            "energy_weighted_constant_field_coherence": (
                mean_energy / target if target > 1e-20 else 1.0
            ),
            "mean_layer_normalized_coherence": sum(
                value[1] / value[0] if value[0] > 1e-20 else 1.0
                for value in totals
            ) / len(totals),
            "residual_direct_sum_relative_error": (
                math.sqrt(residual / target) if target > 1e-20 else 0.0
            ),
        }
    return vectors, metrics


def generate_static_layer_vectors(
    ctx: AnalysisContext,
    prompts: Sequence[str],
    direction_items: Sequence[Tuple[str, Sequence[float]]],
    axis_vectors: Mapping[str, Mapping[int, torch.Tensor]],
    max_prompt_tokens: int,
    max_new_tokens: int,
    anchored_neutral: bool = True,
) -> Dict[str, List[str]]:
    """Generate with linearly composed static residual vectors at all layers."""
    names = [name for name, _direction in direction_items]
    expanded = _direction_major_prompts(prompts, len(direction_items))
    encoded = encode(ctx.tokenizer, expanded, ctx.device, max_prompt_tokens)
    selected_layers = sorted(set.union(*(
        set(axis_vectors[axis]) for axis in ctx.axes
    )))
    if not selected_layers:
        raise ValueError("Static vectors do not contain a layer shared by every axis")
    fields = {
        layer: torch.stack([
            sum(
                float(value) * axis_vectors[axis][layer]
                for axis, value in zip(ctx.axes, direction)
                if layer in axis_vectors[axis]
            )
            for _name, direction in direction_items
        ]).repeat_interleave(len(prompts), dim=0)
        for layer in selected_layers
    }
    neutral_signal = torch.zeros(
        len(expanded), len(ctx.axes), device=ctx.device, dtype=ctx.dtype,
    )
    injector = _TokenwiseInjector(ctx.layers, fields, strength=1.0)
    reference = (
        active_factors(ctx.model, ctx.model.lora.precompute_all(neutral_signal))
        if anchored_neutral else contextlib.nullcontext()
    )
    try:
        with torch.no_grad(), reference:
            output = ctx.model.llama.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            )
    finally:
        injector.close()
    return _partition_generations(
        ctx, output, encoded["input_ids"].shape[1], names, len(prompts),
    )


def fit_independent_static_vectors(
    ctx: AnalysisContext,
    baseline: IndependentLoRAs,
    prompts: Sequence[dict],
    directions: Mapping[str, Sequence[float]],
    strength: float,
    batch_size: int,
    max_prompt_tokens: int,
    token_count: int,
    condition_batch_size: int = 8,
) -> Tuple[dict, dict]:
    """Fit static vectors to task-arithmetic fields at every independent-LoRA site."""
    names = list(directions)
    coefficients = torch.tensor(
        [[strength * value for value in directions[name]] for name in names],
        dtype=torch.float32,
    )
    sums = {
        key: torch.zeros(len(names), baseline.factors[baseline.axes[0]][key].B.shape[0])
        for key in baseline.keys
    }
    squared = {key: torch.zeros(len(names)) for key in baseline.keys}
    counts = defaultdict(int)
    for batch_index, batch in enumerate(chunked(list(prompts), batch_size)):
        LOG.info("Independent static-field fit batch %d/%d", batch_index + 1,
                 math.ceil(len(prompts) / batch_size))
        encoded = encode(
            ctx.tokenizer, [item["prompt"] for item in batch],
            ctx.device, max_prompt_tokens,
        )
        capture = _ProjectionInputCapture(ctx.model, encoded["attention_mask"], token_count)
        try:
            with torch.no_grad():
                ctx.model.llama(**encoded, use_cache=False)
        finally:
            capture.close()
        for key in baseline.keys:
            selected = capture.inputs[key].float()
            axis_fields = []
            for axis in baseline.axes:
                factor = baseline.prepared_factor(axis, key, selected.device, selected.dtype)
                reduced = selected @ factor.A.T
                axis_fields.append(factor.scale * (reduced @ factor.B.T))
            axis_fields = torch.stack(axis_fields, dim=1)
            for start in range(0, len(names), condition_batch_size):
                stop = min(start + condition_batch_size, len(names))
                field = torch.einsum(
                    "ck,mko->cmo", coefficients[start:stop].to(selected), axis_fields,
                )
                sums[key][start:stop] += field.sum(1).cpu()
                squared[key][start:stop] += field.square().sum((1, 2)).cpu()
            counts[key] += selected.shape[0]

    vectors = {name: {} for name in names}
    metrics = {}
    for condition, name in enumerate(names):
        layer_values = defaultdict(list)
        by_projection = {}
        for key in baseline.keys:
            mean, target, mean_energy, residual = _constant_least_squares_statistics(
                sums[key][condition], squared[key][condition], counts[key],
            )
            vectors[name][key] = mean
            layer_values[layer_from_key(key)].append((target, mean_energy, residual))
            by_projection[key] = {
                "sample_count": counts[key],
                "target_field_rms_l2": math.sqrt(max(target, 0.0)),
                "static_vector_l2": math.sqrt(max(mean_energy, 0.0)),
                "least_squares_residual_rms_l2": math.sqrt(residual),
                "constant_field_coherence": mean_energy / target if target > 1e-20 else 1.0,
            }
        by_layer = {}
        for layer, values in sorted(layer_values.items()):
            target = sum(value[0] for value in values)
            mean_energy = sum(value[1] for value in values)
            residual = sum(value[2] for value in values)
            by_layer[str(layer)] = {
                "energy_weighted_constant_field_coherence": (
                    mean_energy / target if target > 1e-20 else 1.0
                ),
                "product_space_relative_residual": (
                    math.sqrt(residual / target) if target > 1e-20 else 0.0
                ),
            }
        all_values = [value for values in layer_values.values() for value in values]
        target = sum(value[0] for value in all_values)
        mean_energy = sum(value[1] for value in all_values)
        residual = sum(value[2] for value in all_values)
        metrics[name] = {
            "global": {
                "energy_weighted_constant_field_coherence": (
                    mean_energy / target if target > 1e-20 else 1.0
                ),
                "product_space_relative_residual": (
                    math.sqrt(residual / target) if target > 1e-20 else 0.0
                ),
            },
            "by_layer": by_layer,
            "by_projection": by_projection,
        }
    return vectors, metrics


def _local_factors(ctx: AnalysisContext, neutral, direction_signal: torch.Tensor, layer: int):
    result = dict(neutral)
    changed = ctx.model.lora.precompute_all(direction_signal)
    prefix = f"layer{layer}."
    result.update({key: value for key, value in changed.items() if key.startswith(prefix)})
    return result


def exact_local_directional_fields(ctx: AnalysisContext, encoded: Mapping[str, torch.Tensor],
                                   direction: Sequence[float]) -> Tuple[dict, Dict[int, torch.Tensor]]:
    """Exact local JVP d h_l / d alpha for s=alpha*direction at alpha=0."""
    batch_size = encoded["input_ids"].shape[0]
    direction_tensor = torch.tensor(direction, device=ctx.device, dtype=ctx.dtype)
    neutral_signal = torch.zeros(batch_size, len(ctx.axes), device=ctx.device, dtype=ctx.dtype)
    neutral = ctx.model.lora.precompute_all(neutral_signal)
    fields = {}
    scalar0 = torch.zeros((), device=ctx.device, dtype=ctx.dtype)
    for layer_index, layer in enumerate(ctx.layers):
        LOG.info("    exact local JVP layer %d/%d", layer_index + 1, len(ctx.layers))

        def layer_output(scalar: torch.Tensor):
            factors = _local_factors(
                ctx, neutral, _direction_signal(scalar, direction_tensor, batch_size), layer_index,
            )
            capture = _CaptureLayer(layer)
            try:
                with active_factors(ctx.model, factors):
                    ctx.model.llama(**encoded, use_cache=False)
                return capture.output
            finally:
                capture.close()

        try:
            _, derivative = torch.autograd.functional.jvp(
                layer_output, (scalar0,), (torch.ones_like(scalar0),),
                create_graph=False, strict=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Exact local signal JVP failed. Use eager attention and reduce the prompt/generation batch size; "
                "this utility intentionally has no finite-difference fallback."
            ) from exc
        fields[layer_index] = derivative.detach()
    return neutral, fields


def _analytical_layer_factor_tangents(ctx: AnalysisContext, direction: Sequence[float] | torch.Tensor,
                                      batch_size: int) -> Tuple[dict, dict]:
    """Compile neutral factors and their exact Jacobian contraction J_s[d]."""
    zero = torch.zeros(batch_size, len(ctx.axes), device=ctx.device, dtype=ctx.dtype)
    neutral = ctx.model.lora.precompute_all(zero)
    direction_tensor = torch.as_tensor(direction, dtype=torch.float32)
    if direction_tensor.ndim == 1:
        linearized = analytical_taylor_factors(
            ctx, direction_tensor.tolist(), strength=1.0, order=1, batch_size=batch_size,
        )
    elif direction_tensor.ndim == 2 and tuple(direction_tensor.shape) == (batch_size, len(ctx.axes)):
        linearized = {}
        rows = direction_tensor.tolist()
        for key, adapter in ctx.adapters:
            coefficients = torch.stack([
                analytical_taylor_coefficients(adapter, row, 1.0, 1)[0]
                for row in rows
            ])
            A_basis, B_basis = adapter._basis_factors()
            linearized[key] = (
                A_basis.unsqueeze(0).expand(batch_size, -1, -1),
                B_basis.unsqueeze(0) * coefficients.to(B_basis).unsqueeze(1),
            )
    else:
        raise ValueError(
            f"Direction must have shape ({len(ctx.axes)},) or "
            f"({batch_size}, {len(ctx.axes)}), got {tuple(direction_tensor.shape)}"
        )
    tangents = {
        key: (linearized[key][0] - neutral[key][0],
              linearized[key][1] - neutral[key][1])
        for key in neutral
    }
    return neutral, tangents


def analytical_layer_directional_fields(
    ctx: AnalysisContext,
    model_inputs: Mapping[str, torch.Tensor],
    direction: Sequence[float] | torch.Tensor,
    past_key_values=None,
    prepared_factors: Tuple[dict, dict] | None = None,
) -> Tuple[object, Dict[int, torch.Tensor]]:
    """Compute every local layer-output source JVP in one analytical pass."""
    batch_size = model_inputs["input_ids"].shape[0]
    neutral, tangents = (prepared_factors if prepared_factors is not None else
                         _analytical_layer_factor_tangents(ctx, direction, batch_size))
    capture = _LayerSourceJacobianCapture(ctx, neutral, tangents)
    try:
        with torch.no_grad(), active_factors(ctx.model, neutral):
            outputs = ctx.model.llama(
                **model_inputs, past_key_values=past_key_values, use_cache=True,
            )
    except RuntimeError as exc:
        raise RuntimeError(
            "Analytical layer-output Jacobian propagation failed. Use eager attention; "
            "the active attention kernel must support forward-mode AD."
        ) from exc
    finally:
        capture.close()
    if len(capture.fields) != len(ctx.layers):
        raise RuntimeError(
            f"Captured {len(capture.fields)}/{len(ctx.layers)} layer Jacobian fields"
        )
    return outputs.past_key_values, capture.fields


def independent_layer_directional_fields(
    ctx: AnalysisContext,
    baseline: IndependentLoRAs,
    model_inputs: Mapping[str, torch.Tensor],
    direction: Sequence[float] | torch.Tensor,
    past_key_values=None,
) -> Tuple[object, Dict[int, torch.Tensor]]:
    """Compute complete-layer source Jacobians for independent task addition."""
    batch_size = model_inputs["input_ids"].shape[0]
    rows = torch.as_tensor(direction, dtype=torch.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1).expand(batch_size, -1)
    if tuple(rows.shape) != (batch_size, len(baseline.axes)):
        raise ValueError(
            f"Independent direction must have shape ({len(baseline.axes)},) or "
            f"({batch_size}, {len(baseline.axes)}), got {tuple(rows.shape)}"
        )
    capture = _IndependentLayerSourceJacobianCapture(ctx, baseline, rows)
    try:
        with torch.no_grad():
            outputs = ctx.model.llama(
                **model_inputs, past_key_values=past_key_values, use_cache=True,
            )
    except RuntimeError as exc:
        raise RuntimeError(
            "Independent task-addition layer Jacobian propagation failed. Use eager "
            "attention; the active attention kernel must support forward-mode AD."
        ) from exc
    finally:
        capture.close()
    if len(capture.fields) != len(ctx.layers):
        raise RuntimeError(
            f"Captured {len(capture.fields)}/{len(ctx.layers)} independent layer fields"
        )
    return outputs.past_key_values, capture.fields


def _text_position_ids(attention_mask: torch.Tensor, last_token_only: bool) -> torch.Tensor:
    positions = attention_mask.long().cumsum(-1) - 1
    positions = positions.masked_fill(attention_mask == 0, 0)
    return positions[:, -1:] if last_token_only else positions


def _fast_layer_jacobian_rows(
    ctx: AnalysisContext,
    prompts: Sequence[str],
    direction_rows: torch.Tensor,
    strength: float,
    max_prompt_tokens: int,
    max_new_tokens: int,
    token_average: int,
    ignore_eos: bool = False,
):
    """Cached two-pass decoding with exact local layer-output Jacobians.

    A neutral analytical forward-mode pass obtains all local layer sources.
    A second ordinary pass injects those fields and produces steered logits.
    Separate KV caches preserve the neutral reference and steered trajectory.
    """
    encoded = encode(ctx.tokenizer, prompts, ctx.device, max_prompt_tokens)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prefix = input_ids.shape[1]
    batch_size = input_ids.shape[0]
    finished = torch.zeros(batch_size, device=ctx.device, dtype=torch.bool)
    eos_ids = _eos_ids(ctx.model)
    fallback_stop = next(iter(eos_ids), ctx.tokenizer.eos_token_id)
    norms = []
    neutral_cache = None
    steered_cache = None
    current_ids = input_ids
    prepared = _analytical_layer_factor_tangents(ctx, direction_rows, batch_size)

    for step in range(max_new_tokens):
        LOG.info("  analytical layer-Jacobian decode step %d/%d, batch=%d",
                 step + 1, max_new_tokens, batch_size)
        first = step == 0
        position_ids = _text_position_ids(attention_mask, last_token_only=not first)
        current = {
            "input_ids": current_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        neutral_cache, fields = analytical_layer_directional_fields(
            ctx, current, direction_rows, neutral_cache, prepared,
        )
        scaled_fields = {layer: strength * value for layer, value in fields.items()}
        norms.append({
            "decode_step": step,
            "by_layer": _field_norm_report(
                scaled_fields,
                attention_mask if first else attention_mask[:, -1:],
                token_average,
            ),
        })

        neutral, _tangents = prepared
        injector = _TokenwiseInjector(ctx.layers, scaled_fields, strength=1.0)
        try:
            with torch.no_grad(), active_factors(ctx.model, neutral):
                outputs = ctx.model.llama(
                    **current, past_key_values=steered_cache, use_cache=True,
                )
        finally:
            injector.close()
        steered_cache = outputs.past_key_values
        next_token = outputs.logits[:, -1].argmax(-1)
        if fallback_stop is not None:
            next_token = torch.where(
                finished, torch.full_like(next_token, fallback_stop), next_token,
            )
        newly_finished = torch.zeros_like(finished)
        if not ignore_eos:
            for token_id in eos_ids:
                newly_finished |= next_token == token_id
            finished |= newly_finished
        input_ids = torch.cat((input_ids, next_token.unsqueeze(1)), dim=1)
        attention_mask = torch.cat(
            (attention_mask, (~finished | newly_finished).long().unsqueeze(1)), dim=1,
        )
        current_ids = next_token.unsqueeze(1)
        if not ignore_eos and bool(finished.all()):
            break

    generations = [ctx.tokenizer.decode(row[prefix:], skip_special_tokens=True) for row in input_ids]
    return generations, norms


def _fast_layer_jacobian_batch(
    ctx: AnalysisContext,
    prompts: Sequence[str],
    direction: Sequence[float],
    strength: float,
    max_prompt_tokens: int,
    max_new_tokens: int,
    token_average: int,
):
    rows = torch.tensor(direction, dtype=torch.float32).reshape(1, -1).expand(len(prompts), -1)
    return _fast_layer_jacobian_rows(
        ctx, prompts, rows, strength, max_prompt_tokens, max_new_tokens, token_average,
    )


def fast_layer_jacobian_multi_batch(
    ctx: AnalysisContext,
    prompts: Sequence[str],
    direction_items: Sequence[Tuple[str, Sequence[float]]],
    strength: float,
    max_prompt_tokens: int,
    max_new_tokens: int,
    token_average: int = 1,
    fixed_token_work: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, List[dict]]]:
    """Direction-batched complete-layer Jacobian generation."""
    names = [name for name, _direction in direction_items]
    expanded = _direction_major_prompts(prompts, len(direction_items))
    rows = torch.tensor(
        [direction for _name, direction in direction_items], dtype=torch.float32,
    ).repeat_interleave(len(prompts), dim=0)
    generations, norms = _fast_layer_jacobian_rows(
        ctx, expanded, rows, strength, max_prompt_tokens, max_new_tokens, token_average,
        fixed_token_work,
    )
    prompt_count = len(prompts)
    by_name = {
        name: generations[index * prompt_count:(index + 1) * prompt_count]
        for index, name in enumerate(names)
    }
    norm_by_name = {name: [] for name in names}
    for step in norms:
        by_layer = step["by_layer"]
        for index, name in enumerate(names):
            start, stop = index * prompt_count, (index + 1) * prompt_count
            norm_by_name[name].append({
                "decode_step": step["decode_step"],
                "by_layer": {
                    layer: {
                        "full_batch_token_frobenius": math.sqrt(sum(
                            value * value
                            for value in values["full_token_frobenius_by_sequence"][start:stop]
                        )),
                        "full_token_frobenius_by_sequence": (
                            values["full_token_frobenius_by_sequence"][start:stop]
                        ),
                        "mean_last_k_token_jacobian_l2_by_sequence": (
                            values["mean_last_k_token_jacobian_l2_by_sequence"][start:stop]
                        ),
                    }
                    for layer, values in by_layer.items()
                },
            })
    return by_name, norm_by_name


def _fast_independent_layer_jacobian_rows(
    ctx: AnalysisContext,
    baseline: IndependentLoRAs,
    prompts: Sequence[str],
    direction_rows: torch.Tensor,
    strength: float,
    max_prompt_tokens: int,
    max_new_tokens: int,
    token_average: int,
):
    """Cached two-pass complete-layer Jacobian steering for task addition."""
    encoded = encode(ctx.tokenizer, prompts, ctx.device, max_prompt_tokens)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prefix = input_ids.shape[1]
    batch_size = input_ids.shape[0]
    finished = torch.zeros(batch_size, device=ctx.device, dtype=torch.bool)
    eos_ids = _eos_ids(ctx.model)
    fallback_stop = next(iter(eos_ids), ctx.tokenizer.eos_token_id)
    norms = []
    neutral_cache = None
    steered_cache = None
    current_ids = input_ids

    for step in range(max_new_tokens):
        LOG.info(
            "  independent layer-Jacobian decode step %d/%d, batch=%d",
            step + 1, max_new_tokens, batch_size,
        )
        first = step == 0
        position_ids = _text_position_ids(attention_mask, last_token_only=not first)
        current = {
            "input_ids": current_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        neutral_cache, fields = independent_layer_directional_fields(
            ctx, baseline, current, direction_rows, neutral_cache,
        )
        scaled_fields = {layer: strength * value for layer, value in fields.items()}
        norms.append({
            "decode_step": step,
            "by_layer": _field_norm_report(
                scaled_fields,
                attention_mask if first else attention_mask[:, -1:],
                token_average,
            ),
        })

        injector = _TokenwiseInjector(ctx.layers, scaled_fields, strength=1.0)
        try:
            with torch.no_grad():
                outputs = ctx.model.llama(
                    **current, past_key_values=steered_cache, use_cache=True,
                )
        finally:
            injector.close()
        steered_cache = outputs.past_key_values
        next_token = outputs.logits[:, -1].argmax(-1)
        if fallback_stop is not None:
            next_token = torch.where(
                finished, torch.full_like(next_token, fallback_stop), next_token,
            )
        newly_finished = torch.zeros_like(finished)
        for token_id in eos_ids:
            newly_finished |= next_token == token_id
        finished |= newly_finished
        input_ids = torch.cat((input_ids, next_token.unsqueeze(1)), dim=1)
        attention_mask = torch.cat(
            (attention_mask, (~finished | newly_finished).long().unsqueeze(1)), dim=1,
        )
        current_ids = next_token.unsqueeze(1)
        if bool(finished.all()):
            break

    generations = [
        ctx.tokenizer.decode(row[prefix:], skip_special_tokens=True) for row in input_ids
    ]
    return generations, norms


def fast_independent_layer_jacobian_multi_batch(
    ctx: AnalysisContext,
    baseline: IndependentLoRAs,
    prompts: Sequence[str],
    direction_items: Sequence[Tuple[str, Sequence[float]]],
    strength: float,
    max_prompt_tokens: int,
    max_new_tokens: int,
    token_average: int = 1,
) -> Tuple[Dict[str, List[str]], Dict[str, List[dict]]]:
    """Direction-batched task-addition complete-layer Jacobian generation."""
    names = [name for name, _direction in direction_items]
    expanded = _direction_major_prompts(prompts, len(direction_items))
    rows = torch.tensor(
        [direction for _name, direction in direction_items], dtype=torch.float32,
    ).repeat_interleave(len(prompts), dim=0)
    generations, norms = _fast_independent_layer_jacobian_rows(
        ctx, baseline, expanded, rows, strength, max_prompt_tokens,
        max_new_tokens, token_average,
    )
    prompt_count = len(prompts)
    by_name = {
        name: generations[index * prompt_count:(index + 1) * prompt_count]
        for index, name in enumerate(names)
    }
    norm_by_name = {name: [] for name in names}
    for step in norms:
        for index, name in enumerate(names):
            start, stop = index * prompt_count, (index + 1) * prompt_count
            norm_by_name[name].append({
                "decode_step": step["decode_step"],
                "by_layer": {
                    layer: {
                        "full_batch_token_frobenius": math.sqrt(sum(
                            value * value
                            for value in values["full_token_frobenius_by_sequence"][start:stop]
                        )),
                        "full_token_frobenius_by_sequence": (
                            values["full_token_frobenius_by_sequence"][start:stop]
                        ),
                        "mean_last_k_token_jacobian_l2_by_sequence": (
                            values["mean_last_k_token_jacobian_l2_by_sequence"][start:stop]
                        ),
                    }
                    for layer, values in step["by_layer"].items()
                },
            })
    return by_name, norm_by_name


def _last_k_field_norms(field: torch.Tensor, attention_mask: torch.Tensor, count: int) -> List[float]:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    last = (positions * attention_mask.long()).amax(1, keepdim=True)
    selected = attention_mask.bool() & (positions > last - count)
    norms = field.float().norm(dim=-1)
    return ((norms * selected).sum(-1) / selected.sum(-1).clamp_min(1)).cpu().tolist()


def _field_norm_report(fields: Mapping[int, torch.Tensor], attention_mask: torch.Tensor,
                       token_average: int) -> dict:
    return {
        str(layer): {
            "full_batch_token_frobenius": float(field.float().norm()),
            "full_token_frobenius_by_sequence": (
                field.float().flatten(1).norm(dim=1).cpu().tolist()
            ),
            "mean_last_k_token_jacobian_l2_by_sequence": _last_k_field_norms(
                field, attention_mask, token_average,
            ),
        }
        for layer, field in fields.items()
    }


def _eos_ids(model) -> set:
    value = getattr(model.llama.generation_config, "eos_token_id", None)
    if value is None:
        return set()
    return set(value if isinstance(value, (list, tuple)) else [value])


def _dynamic_batch(ctx: AnalysisContext, prompts: Sequence[str], direction: Sequence[float], strength: float,
                   max_prompt_tokens: int, max_new_tokens: int, token_average: int):
    encoded = encode(ctx.tokenizer, prompts, ctx.device, max_prompt_tokens)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prefix = input_ids.shape[1]
    finished = torch.zeros(input_ids.shape[0], device=ctx.device, dtype=torch.bool)
    eos_ids = _eos_ids(ctx.model)
    fallback_stop = next(iter(eos_ids), ctx.tokenizer.eos_token_id)
    norms = []
    for step in range(max_new_tokens):
        LOG.info("  tokenwise first-order decode step %d/%d, batch=%d", step + 1,
                 max_new_tokens, len(prompts))
        current = {"input_ids": input_ids, "attention_mask": attention_mask}
        neutral, fields = exact_local_directional_fields(ctx, current, direction)
        norms.append({"decode_step": step, "by_layer": _field_norm_report(fields, attention_mask, token_average)})
        injector = _TokenwiseInjector(ctx.layers, fields, strength)
        try:
            with torch.no_grad(), active_factors(ctx.model, neutral):
                logits = ctx.model.llama(**current, use_cache=False).logits
        finally:
            injector.close()
        next_token = logits[:, -1].argmax(-1)
        if fallback_stop is not None:
            next_token = torch.where(finished, torch.full_like(next_token, fallback_stop), next_token)
        newly_finished = torch.zeros_like(finished)
        for token_id in eos_ids:
            newly_finished |= next_token == token_id
        finished |= newly_finished
        input_ids = torch.cat((input_ids, next_token.unsqueeze(1)), dim=1)
        attention_mask = torch.cat((attention_mask, (~finished | newly_finished).long().unsqueeze(1)), dim=1)
        if bool(finished.all()):
            break
    generations = [ctx.tokenizer.decode(row[prefix:], skip_special_tokens=True) for row in input_ids]
    return generations, norms


def _learned_batch(ctx: AnalysisContext, prompts: Sequence[str], direction: Sequence[float], strength: float,
                   max_prompt_tokens: int, max_new_tokens: int) -> List[str]:
    encoded = encode(ctx.tokenizer, prompts, ctx.device, max_prompt_tokens)
    values = [strength * value for value in direction]
    s = signal(values, ctx.device, ctx.dtype, len(prompts))
    with torch.no_grad(), active_factors(ctx.model, ctx.model.lora.precompute_all(s)):
        output = ctx.model.llama.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    prefix = encoded["input_ids"].shape[1]
    return [ctx.tokenizer.decode(row[prefix:], skip_special_tokens=True) for row in output]


def _direction_major_prompts(prompts: Sequence[str], count: int) -> List[str]:
    return [prompt for _ in range(count) for prompt in prompts]


def _partition_generations(ctx: AnalysisContext, output: torch.Tensor, prefix: int,
                           names: Sequence[str], prompts_per_direction: int) -> Dict[str, List[str]]:
    decoded = [ctx.tokenizer.decode(row[prefix:], skip_special_tokens=True) for row in output]
    return {
        name: decoded[index * prompts_per_direction:(index + 1) * prompts_per_direction]
        for index, name in enumerate(names)
    }


def _learned_multi_batch(ctx: AnalysisContext, prompts: Sequence[str],
                         direction_items: Sequence[Tuple[str, Sequence[float]]], strength: float,
                         max_prompt_tokens: int, max_new_tokens: int) -> Dict[str, List[str]]:
    names = [item[0] for item in direction_items]
    expanded = _direction_major_prompts(prompts, len(direction_items))
    encoded = encode(ctx.tokenizer, expanded, ctx.device, max_prompt_tokens)
    values = torch.tensor(
        [[strength * value for value in direction] for _name, direction in direction_items],
        device=ctx.device, dtype=ctx.dtype,
    ).repeat_interleave(len(prompts), dim=0)
    with torch.no_grad(), active_factors(ctx.model, ctx.model.lora.precompute_all(values)):
        output = ctx.model.llama.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    return _partition_generations(
        ctx, output, encoded["input_ids"].shape[1], names, len(prompts),
    )


def _independent_multi_batch(ctx: AnalysisContext, baseline: IndependentLoRAs,
                             prompts: Sequence[str],
                             direction_items: Sequence[Tuple[str, Sequence[float]]],
                             strength: float, max_prompt_tokens: int,
                             max_new_tokens: int) -> Dict[str, List[str]]:
    names = [item[0] for item in direction_items]
    expanded = _direction_major_prompts(prompts, len(direction_items))
    encoded = encode(ctx.tokenizer, expanded, ctx.device, max_prompt_tokens)
    values = torch.tensor(
        [[strength * value for value in direction] for _name, direction in direction_items],
        dtype=ctx.dtype, device=ctx.device,
    ).repeat_interleave(len(prompts), dim=0)
    with torch.no_grad(), IndependentLoRAHooks(ctx.model, baseline, values):
        output = ctx.model.llama.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    return _partition_generations(
        ctx, output, encoded["input_ids"].shape[1], names, len(prompts),
    )


def _analytical_batch(ctx: AnalysisContext, prompts: Sequence[str], direction: Sequence[float],
                      strength: float, order: int, max_prompt_tokens: int,
                      max_new_tokens: int) -> List[str]:
    """Generate with the analytically Taylor-expanded weight path and normal KV caching."""
    encoded = encode(ctx.tokenizer, prompts, ctx.device, max_prompt_tokens)
    factors = analytical_taylor_factors(ctx, direction, strength, order, len(prompts))
    with torch.no_grad(), active_factors(ctx.model, factors):
        output = ctx.model.llama.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    prefix = encoded["input_ids"].shape[1]
    return [ctx.tokenizer.decode(row[prefix:], skip_special_tokens=True) for row in output]


def _analytical_multi_batch(ctx: AnalysisContext, prompts: Sequence[str],
                            direction_items: Sequence[Tuple[str, Sequence[float]]],
                            strength: float, order: int, max_prompt_tokens: int,
                            max_new_tokens: int) -> Dict[str, List[str]]:
    names = [item[0] for item in direction_items]
    expanded = _direction_major_prompts(prompts, len(direction_items))
    encoded = encode(ctx.tokenizer, expanded, ctx.device, max_prompt_tokens)
    factors = _analytical_taylor_factors_multi(
        ctx, [item[1] for item in direction_items], strength, order, len(prompts),
    )
    with torch.no_grad(), active_factors(ctx.model, factors):
        output = ctx.model.llama.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    return _partition_generations(
        ctx, output, encoded["input_ids"].shape[1], names, len(prompts),
    )


def _static_projection_batch(ctx: AnalysisContext, prompts: Sequence[str],
                             vectors: Mapping[str, torch.Tensor], max_prompt_tokens: int,
                             max_new_tokens: int) -> List[str]:
    encoded = encode(ctx.tokenizer, prompts, ctx.device, max_prompt_tokens)
    neutral = torch.zeros(len(prompts), len(ctx.axes), device=ctx.device, dtype=ctx.dtype)
    injector = _StaticProjectionInjector(ctx.model, vectors)
    try:
        with torch.no_grad(), active_factors(ctx.model, ctx.model.lora.precompute_all(neutral)):
            output = ctx.model.llama.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            )
    finally:
        injector.close()
    prefix = encoded["input_ids"].shape[1]
    return [ctx.tokenizer.decode(row[prefix:], skip_special_tokens=True) for row in output]


def _static_projection_multi_batch(ctx: AnalysisContext, prompts: Sequence[str],
                                   direction_items: Sequence[Tuple[str, Sequence[float]]],
                                   vectors_by_name: Mapping[str, Mapping[str, torch.Tensor]],
                                   max_prompt_tokens: int,
                                   max_new_tokens: int,
                                   anchored_neutral: bool = True) -> Dict[str, List[str]]:
    names = [item[0] for item in direction_items]
    expanded = _direction_major_prompts(prompts, len(direction_items))
    encoded = encode(ctx.tokenizer, expanded, ctx.device, max_prompt_tokens)
    combined_vectors = {
        key: torch.stack([vectors_by_name[name][key] for name in names]).repeat_interleave(
            len(prompts), dim=0,
        )
        for key, _adapter in ctx.adapters
    }
    injector = _StaticProjectionInjector(ctx.model, combined_vectors)
    try:
        if anchored_neutral:
            neutral = torch.zeros(len(expanded), len(ctx.axes), device=ctx.device, dtype=ctx.dtype)
            reference = active_factors(ctx.model, ctx.model.lora.precompute_all(neutral))
        else:
            reference = contextlib.nullcontext()
        with torch.no_grad(), reference:
            output = ctx.model.llama.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            )
    finally:
        injector.close()
    return _partition_generations(
        ctx, output, encoded["input_ids"].shape[1], names, len(prompts),
    )


def tokenwise_autoregressive_report(ctx: AnalysisContext, prompts: Sequence[dict],
                                    directions: Mapping[str, Sequence[float]], strength: float = 1.0,
                                    generation_batch_size: int = 4, max_prompt_tokens: int = 512,
                                    max_new_tokens: int = 32, jacobian_token_average: int = 1,
                                    method: str = "layer_jacobian",
                                    analytical_orders: Sequence[int] = (1, 2),
                                    static_fit_prompt_count: int = 80,
                                    generation_prompt_offset: int = 0,
                                    static_fit_batch_size: int = 8,
                                    static_fit_tokens: int = 1,
                                    direction_batch_size: int = 8,
                                    true_baseline: Path | None = None,
                                    generation_methods: Sequence[str] | None = None,
                                    static_layers: Sequence[int] | None = None,
                                    static_layers_by_axis: Mapping[str, int | Sequence[int]] | None = None) -> dict:
    if jacobian_token_average < 1:
        raise ValueError("jacobian_token_average must be positive")
    aliases = {"analytical": "matrixwise_taylor", "both": "all"}
    method = aliases.get(method, method)
    if method not in {"layer_jacobian", "matrixwise_taylor", "exact_tokenwise", "all"}:
        raise ValueError(
            "method must be layer_jacobian, matrixwise_taylor, exact_tokenwise, or all"
        )
    orders = sorted(set(analytical_orders))
    if not orders or any(order not in (1, 2) for order in orders):
        raise ValueError("analytical_orders must contain one or both of 1 and 2")
    if direction_batch_size < 1:
        raise ValueError("direction_batch_size must be positive")
    selected_methods = set(generation_methods or ACTIVATION_GENERATION_METHODS)
    unknown_methods = selected_methods - ACTIVATION_GENERATION_METHODS
    if unknown_methods:
        raise ValueError(
            f"Unknown activation generation methods: {sorted(unknown_methods)}"
        )
    if any(name.startswith("independent_") for name in selected_methods) and true_baseline is None:
        raise ValueError("Independent activation methods require --true_baseline")
    if static_fit_prompt_count < 0 or static_fit_prompt_count >= len(prompts):
        raise ValueError("static_fit_prompt_count must be nonnegative and smaller than the prompt count")
    if generation_prompt_offset < 0:
        raise ValueError("generation_prompt_offset must be nonnegative")
    generation_start = static_fit_prompt_count + generation_prompt_offset
    if generation_start >= len(prompts):
        raise ValueError(
            "static_fit_prompt_count + generation_prompt_offset must be smaller than the prompt count"
        )
    fit_prompts = list(prompts[:static_fit_prompt_count])
    generation_prompts = list(prompts[generation_start:])
    independent = None
    independent_static_vectors, independent_static_metrics = {}, {}
    if true_baseline is not None:
        independent = load_independent_loras(
            true_baseline, ctx.axes, [key for key, _adapter in ctx.adapters],
            ctx.config.base_model, allow_sparse=True,
        )
    static_vectors, static_metrics = {}, {}
    layer_static_vectors, layer_static_metrics = {}, {}
    if fit_prompts and method in {"layer_jacobian", "all"} and "hera_static" in selected_methods:
        LOG.info("Fitting static complete-layer vectors on %d disjoint prompts", len(fit_prompts))
        layer_static_vectors, layer_static_metrics = fit_static_layer_vectors(
            ctx, fit_prompts, static_fit_batch_size, max_prompt_tokens, static_fit_tokens,
            static_layers,
            static_layers_by_axis,
        )
    if fit_prompts and method in {"matrixwise_taylor", "all"}:
        LOG.info("Fitting static projection vectors on %d disjoint prompts", len(fit_prompts))
        static_vectors, static_metrics = fit_static_projection_vectors(
            ctx, fit_prompts, directions, strength, orders, static_fit_batch_size,
            max_prompt_tokens, static_fit_tokens,
        )
    if fit_prompts and independent is not None and "independent_static" in selected_methods:
        LOG.info(
            "Fitting independent task-arithmetic complete-layer static vectors on %d prompts",
            len(fit_prompts),
        )
        independent_static_vectors, independent_static_metrics = (
            fit_independent_static_layer_vectors(
                ctx, independent, fit_prompts, static_fit_batch_size,
                max_prompt_tokens, static_fit_tokens, static_layers,
                static_layers_by_axis,
            )
        )
    direction_items = list(directions.items())
    result = {}
    for name, direction in direction_items:
        if len(direction) != len(ctx.axes):
            raise ValueError(f"Direction {name!r} has {len(direction)} coordinates; expected {len(ctx.axes)}.")
        result[name] = {
            "direction": list(direction),
            "strength": strength,
            "static_fit": {
                f"taylor_order_{order}": static_metrics[(name, order)]
                for order in orders if (name, order) in static_metrics
            },
            "static_complete_layer_fit": {
                axis: layer_static_metrics[axis]
                for axis, value in zip(ctx.axes, direction)
                if value and axis in layer_static_metrics
            },
            "independent_task_arithmetic_static_fit": (
                {
                    axis: independent_static_metrics[axis]
                    for axis, value in zip(ctx.axes, direction)
                    if value and axis in independent_static_metrics
                } if independent is not None else None
            ),
            "examples": [],
            "jacobian_norms": [],
        }
    prompt_batches = list(chunked(generation_prompts, generation_batch_size))
    for prompt_batch_index, batch in enumerate(prompt_batches):
        text = [item["prompt"] for item in batch]
        for direction_batch_index, direction_batch in enumerate(
            chunked(direction_items, direction_batch_size)
        ):
            names = [item[0] for item in direction_batch]
            LOG.info(
                "Activation generation: prompt batch %d/%d, direction batch %d/%d (%s)",
                prompt_batch_index + 1, len(prompt_batches), direction_batch_index + 1,
                math.ceil(len(direction_items) / direction_batch_size), ", ".join(names),
            )
            actual = (
                _learned_multi_batch(
                    ctx, text, direction_batch, strength, max_prompt_tokens, max_new_tokens,
                ) if "hera_weight" in selected_methods else None
            )
            independent_actual = (
                _independent_multi_batch(
                    ctx, independent, text, direction_batch, strength,
                    max_prompt_tokens, max_new_tokens,
                ) if independent is not None and "independent_weight" in selected_methods else None
            )
            analytical = {
                order: _analytical_multi_batch(
                    ctx, text, direction_batch, strength, order,
                    max_prompt_tokens, max_new_tokens,
                )
                for order in orders
            } if method in {"matrixwise_taylor", "all"} else {}
            static = {}
            if method in {"matrixwise_taylor", "all"} and static_vectors:
                for order in orders:
                    vectors_by_name = {
                        name: static_vectors[(name, order)]
                        for name in names if (name, order) in static_vectors
                    }
                    if len(vectors_by_name) == len(names):
                        static[order] = _static_projection_multi_batch(
                            ctx, text, direction_batch, vectors_by_name,
                            max_prompt_tokens, max_new_tokens,
                        )
            layer_static = None
            if (method in {"layer_jacobian", "all"} and layer_static_vectors
                    and "hera_static" in selected_methods):
                layer_static = generate_static_layer_vectors(
                    ctx, text, direction_batch, layer_static_vectors,
                    max_prompt_tokens, max_new_tokens,
                )
            independent_static = None
            if (independent is not None and independent_static_vectors
                    and "independent_static" in selected_methods):
                independent_static = generate_static_layer_vectors(
                    ctx, text, direction_batch, independent_static_vectors,
                    max_prompt_tokens, max_new_tokens,
                    anchored_neutral=False,
                )
            layer_values, layer_norms_by_name = {}, {}
            if method in {"layer_jacobian", "all"} and "hera_jacobian" in selected_methods:
                layer_values, layer_norms_by_name = fast_layer_jacobian_multi_batch(
                    ctx, text, direction_batch, strength, max_prompt_tokens,
                    max_new_tokens, jacobian_token_average,
                )
            independent_layer_values, independent_layer_norms_by_name = {}, {}
            if (independent is not None and method in {"layer_jacobian", "all"}
                    and "independent_jacobian" in selected_methods):
                independent_layer_values, independent_layer_norms_by_name = (
                    fast_independent_layer_jacobian_multi_batch(
                        ctx, independent, text, direction_batch, strength,
                        max_prompt_tokens, max_new_tokens, jacobian_token_average,
                    )
                )
            for name, direction in direction_batch:
                layer_approximate = layer_values.get(name)
                layer_norms = layer_norms_by_name.get(name, [])
                approximate, exact_norms = None, []
                if method in {"exact_tokenwise", "all"}:
                    approximate, norms = _dynamic_batch(
                        ctx, text, direction, strength, max_prompt_tokens,
                        max_new_tokens, jacobian_token_average,
                    )
                    exact_norms = norms
                offset = len(result[name]["examples"])
                global_offset = generation_prompt_offset + offset
                for index, item in enumerate(batch):
                    example = {
                        "prompt_index": global_offset + index,
                        "source_index": item.get("source_index"),
                        "prompt": item["prompt"],
                    }
                    if actual is not None:
                        example["learned_weight_steering"] = actual[name][index]
                    if independent_actual is not None:
                        example["independent_task_arithmetic_weight_steering"] = (
                            independent_actual[name][index]
                        )
                    if independent_static is not None:
                        example["independent_static_complete_layer_jacobian_ls"] = (
                            independent_static[name][index]
                        )
                    if name in independent_layer_values:
                        example["independent_task_arithmetic_layer_output_jacobian_steering"] = (
                            independent_layer_values[name][index]
                        )
                    for order, values in analytical.items():
                        example[f"analytical_matrixwise_taylor_order_{order}"] = values[name][index]
                    for order, values in static.items():
                        example[f"static_projection_ls_taylor_order_{order}"] = values[name][index]
                    if layer_approximate is not None:
                        example["analytical_layer_output_jacobian_steering"] = (
                            layer_approximate[index]
                        )
                    if layer_static is not None:
                        example["static_complete_layer_jacobian_ls"] = layer_static[name][index]
                    if approximate is not None:
                        example["exact_jvp_tokenwise_activation_steering"] = approximate[index]
                    result[name]["examples"].append(example)
                if layer_norms:
                    result[name]["jacobian_norms"].append({
                        "method": "analytical_layer_output_jacobian",
                        "prompt_indices": list(range(global_offset, global_offset + len(batch))),
                        "steps": layer_norms,
                    })
                if exact_norms:
                    result[name]["jacobian_norms"].append({
                        "method": "exact_jvp_tokenwise_control",
                        "prompt_indices": list(range(global_offset, global_offset + len(batch))),
                        "steps": exact_norms,
                    })
                if name in independent_layer_norms_by_name:
                    result[name]["jacobian_norms"].append({
                        "method": "independent_task_arithmetic_layer_output_jacobian",
                        "prompt_indices": list(range(global_offset, global_offset + len(batch))),
                        "steps": independent_layer_norms_by_name[name],
                    })
    return {
        "definition": (
            "Layer-Jacobian mode analytically propagates the exact adapter-factor tangent through "
            "each complete decoder layer and injects the resulting layer-output source field. It "
            "computes all layer fields in one neutral forward-mode pass and uses a second cached "
            "forward for steering. Matrixwise-Taylor mode is retained only as a projection-level "
            "ablation. Exact-tokenwise mode is the legacy full-model JVP validation control."
        ),
        "analytical_definition": (
            "For layer-Jacobian steering, closed-form adapter factor tangents seed forward-mode "
            "differentiation through each complete decoder block. The resulting local layer-output "
            "source U_l(x)d is injected in the residual stream. For the separately labeled "
            "matrixwise-Taylor ablation only, each adapted weight is replaced by "
            "W(0)+alpha DsW(0)[d]+alpha^2 DssW(0)[d,d]/2."
        ),
        "jacobian_definition": (
            "The reported field norm is the norm of the signal-Jacobian directional product J_s h_l d; "
            "it is not an input-embedding Jacobian or a finite difference."
        ),
        "axes": ctx.axes,
        "true_baseline": str(independent.root) if independent is not None else None,
        "method": method,
        "analytical_taylor_orders": orders,
        "direction_batch_size": direction_batch_size,
        "static_layers": (list(static_layers) if static_layers is not None else None),
        "static_layers_by_axis": (
            dict(static_layers_by_axis) if static_layers_by_axis else None
        ),
        "generation_methods": sorted(selected_methods),
        "split": {
            "static_fit_prompts": len(fit_prompts),
            "generation_prompt_offset": generation_prompt_offset,
            "generation_prompts": len(generation_prompts),
            "generation_source_range": [
                generation_start,
                generation_start + len(generation_prompts),
            ],
        },
        "static_fit_definition": (
            "The primary static fit averages each coordinate column U_l,i(x,t) over disjoint "
            "calibration prompts, yielding one residual vector per layer and axis. Arbitrary "
            "directions compose exactly as sum_i s_i c_l,i. When independent LoRAs are supplied, "
            "the same complete-layer fit is performed independently at the raw-base task-addition "
            "origin. Separately labeled matrixwise-Taylor static fits are projection-level ablations."
        ),
        "jacobian_token_average": jacobian_token_average,
        "directions": result,
    }


def default_axis_directions(axes: Sequence[str]) -> Dict[str, List[float]]:
    """Return every binary corner of the steering cube, including neutral."""
    result = {}
    for corner in itertools.product((0.0, 1.0), repeat=len(axes)):
        active = [axis for axis, value in zip(axes, corner) if value]
        name = "neutral" if not active else "+".join(active)
        result[name] = list(corner)
    return result


def grid_axis_directions(axes: Sequence[str], levels: Sequence[float]) -> Dict[str, List[float]]:
    """Return the complete Cartesian steering grid with explicit coordinate names."""
    return {
        ",".join(f"{axis}={float(value):g}" for axis, value in zip(axes, point)): list(point)
        for point in itertools.product(tuple(float(value) for value in levels), repeat=len(axes))
    }
