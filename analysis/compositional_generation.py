"""Generation and static-steering evaluation for HeRD merging LoRAs."""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import AutoTokenizer

from .activation import (
    _TokenwiseInjector,
    _constant_least_squares_statistics,
    _direction_major_prompts,
    _partition_generations,
    _resolve_static_axis_layers,
    _selected_projection_inputs,
    default_axis_directions,
)
from .common import (
    AnalysisContext,
    ROOT,
    chunked,
    encode,
    json_dump,
    layer_from_key,
    read_prompts,
)
from .herd_merging import _axis_order
from .independent import (
    IndependentFactor,
    IndependentLoRAHooks,
    MultiAffineInteractionHooks,
    MultiAffineInteractions,
    load_independent_loras,
)
from .generation import _generate_merged

from config import TrainingConfig  # noqa: E402
from model import build_model  # noqa: E402


LOG = logging.getLogger("compositional_generation")


def load_compositional_context(independent_root: Path, interaction_dir: Path,
                               axes_value: str | None, device_name: str,
                               max_prompt_tokens: int,
                               attn_implementation: str = "sdpa"):
    if axes_value is None:
        interaction_config = interaction_dir / "interaction_config.json"
        if not interaction_config.is_file():
            raise FileNotFoundError(f"Missing interaction metadata: {interaction_config}")
        recorded_axes = json.loads(interaction_config.read_text()).get("axes")
        if not isinstance(recorded_axes, list) or not recorded_axes:
            raise ValueError(f"Interaction artifact has no canonical axis order: {interaction_config}")
        axes = [str(axis) for axis in recorded_axes]
    else:
        axes = _axis_order(independent_root, axes_value)
    baseline = load_independent_loras(independent_root, axes)
    interactions = MultiAffineInteractions.load_artifact(interaction_dir, baseline)
    cfg = TrainingConfig(
        base_model=baseline.base_model,
        signal_dim=len(axes),
        adapter_architecture="anchored_neural_merger",
        lora_rank=1,
        adapter_hidden_dim=1,
        freeze_base_model=True,
        bf16=True,
    )
    model = build_model(
        cfg, device=device_name, attn_implementation=attn_implementation,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if hasattr(model.llama.config, "use_cache"):
        model.llama.config.use_cache = False
    expected = sorted(model._adapted_linears)
    fitted_keys = baseline.validate_uniform_subset(expected)
    LOG.info("Compositional model uses %d shared adapter projections: %s", len(fitted_keys), fitted_keys)
    device = torch.device(device_name)
    interactions.to(device=device, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(baseline.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = max_prompt_tokens
    ctx = AnalysisContext(
        interaction_dir.resolve(), -1, axes, cfg, model, tokenizer, device,
        next(model.lora.parameters()).dtype,
    )
    return ctx, baseline, interactions


def compositional_weight_generations(ctx: AnalysisContext, baseline, interactions,
                                     prompts: Sequence[dict], generation_batch_size: int,
                                     direction_batch_size: int, max_prompt_tokens: int,
                                     max_new_tokens: int,
                                     include_task_arithmetic: bool = False,
                                     merge_methods: Sequence[str] = (),
                                     task_arithmetic_only: bool = False,
                                     pure_axes_only: bool = False) -> dict:
    directions = default_axis_directions(ctx.axes)
    if pure_axes_only:
        directions = {
            name: direction for name, direction in directions.items()
            if sum(float(value) != 0.0 for value in direction) <= 1
        }
    items = list(directions.items())
    fitted_generated = {name: [] for name in directions}
    task_arithmetic_generated = ({name: [] for name in directions}
                                 if include_task_arithmetic or task_arithmetic_only else None)
    prompt_batches = list(chunked(list(prompts), generation_batch_size))
    for prompt_index, batch in enumerate(prompt_batches, 1):
        texts = [item["prompt"] for item in batch]
        for direction_index, direction_batch in enumerate(
            chunked(items, direction_batch_size), 1,
        ):
            LOG.info(
                "Fitted-weight generation prompt batch %d/%d, direction batch %d/%d",
                prompt_index, len(prompt_batches), direction_index,
                math.ceil(len(items) / direction_batch_size),
            )
            variants = [] if task_arithmetic_only else ["fitted"]
            if include_task_arithmetic or task_arithmetic_only:
                variants.append("task_arithmetic")
            expanded_items = [
                (f"{variant}::{name}", direction, variant)
                for variant in variants
                for name, direction in direction_batch
            ]
            expanded = _direction_major_prompts(texts, len(expanded_items))
            encoded = encode(ctx.tokenizer, expanded, ctx.device, max_prompt_tokens)
            task_values = torch.tensor(
                [direction for _label, direction, _variant in expanded_items],
                device=ctx.device, dtype=ctx.dtype,
            ).repeat_interleave(len(texts), dim=0)
            interaction_values = torch.tensor(
                [direction if variant == "fitted" else [0.0] * len(ctx.axes)
                 for _label, direction, variant in expanded_items],
                device=ctx.device, dtype=ctx.dtype,
            ).repeat_interleave(len(texts), dim=0)
            with torch.no_grad(), IndependentLoRAHooks(ctx.model, baseline, task_values), \
                    MultiAffineInteractionHooks(ctx.model, interactions, interaction_values):
                output = ctx.model.llama.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                )
            partitioned = _partition_generations(
                ctx, output, encoded["input_ids"].shape[1],
                [label for label, _direction, _variant in expanded_items], len(texts),
            )
            for name, _direction in direction_batch:
                if not task_arithmetic_only:
                    fitted_generated[name].extend(partitioned[f"fitted::{name}"])
                if task_arithmetic_generated is not None:
                    task_arithmetic_generated[name].extend(
                        partitioned[f"task_arithmetic::{name}"]
                    )

    merge_generated = {
        method: {name: [] for name in directions} for method in merge_methods
    }
    prompt_text = [item["prompt"] for item in prompts]
    for method in merge_methods:
        for name, direction in items:
            active = [index for index, value in enumerate(direction) if float(value) != 0.0]
            LOG.info("Independent merge generation: method=%s direction=%s", method, name)
            updates = baseline.merge(method, active)
            merge_generated[method][name] = _generate_merged(
                ctx, prompt_text, updates, generation_batch_size,
                max_prompt_tokens, max_new_tokens, 0.0, anchored_neutral=False,
            )

    return {
        "definition": (
            "Exact independent task arithmetic plus fitted static multi-affine "
            "interaction LoRAs at every adapted projection. When requested, the "
            "paired task-arithmetic control is generated in the same batch on the "
            "identical prompts and binary directions."
        ),
        "axes": ctx.axes,
        "combination_count": len(directions),
        "prompt_count": len(prompts),
        "directions": {
            name: {
                "direction": direction,
                "examples": [
                    {
                        "prompt_index": index,
                        "source_index": prompts[index].get("source_index"),
                        "prompt": prompts[index]["prompt"],
                        **({
                            "fitted_interaction_weight_generation": (
                                fitted_generated[name][index]
                            ),
                        } if not task_arithmetic_only else {}),
                        **({
                            "independent_task_arithmetic_weight_generation": (
                                task_arithmetic_generated[name][index]
                            ),
                        } if task_arithmetic_generated is not None else {}),
                        **{
                            f"independent_{method}_weight_generation":
                            merge_generated[method][name][index]
                            for method in merge_methods
                        },
                    }
                    for index in range(len(prompts))
                ],
            }
            for name, direction in directions.items()
        },
        "independent_merge_methods": list(merge_methods),
    }


class _CompositionalLayerSourceCapture:
    """Complete-layer JVP sources for task vectors plus HeRD mergings.

    The primal path is the raw base model. A forward-mode scalar multiplies
    each requested static weight direction. Incoming hidden tangents are reset
    at each decoder layer, so the captured tangent is the direct complete-layer
    source after normalization, attention, residual mixing, and the gated MLP.
    """

    def __init__(self, ctx: AnalysisContext, baseline, interactions,
                 direction_rows: torch.Tensor | None = None,
                 branch_rows: torch.Tensor | None = None):
        if (direction_rows is None) == (branch_rows is None):
            raise ValueError("Provide exactly one of direction_rows or branch_rows")
        self.ctx = ctx
        self.baseline = baseline
        self.interactions = interactions
        self.direction_rows = direction_rows
        self.branch_rows = branch_rows
        self.fields = {}
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
        if args:
            return (hidden, *args[1:]), kwargs
        if "hidden_states" in kwargs:
            updated = dict(kwargs)
            updated["hidden_states"] = hidden
            return args, updated
        raise RuntimeError("Decoder layer did not receive hidden states")

    @staticmethod
    def _projection_hook(branches, coefficients):
        def add(_module, inputs, output):
            x = inputs[0]
            correction = torch.zeros_like(output)
            shape = (coefficients.shape[0],) + (1,) * (output.ndim - 1)
            for branch_index, (A, B, scale) in enumerate(branches):
                current = torch.nn.functional.linear(
                    torch.nn.functional.linear(x, A.to(x)), B.to(x),
                )
                correction = correction + scale * current * coefficients[:, branch_index].reshape(shape)
            return output + correction
        return add

    def _branch_tangents(self, hidden):
        if self.branch_rows is not None:
            return self.branch_rows.to(device=hidden.device, dtype=hidden.dtype)
        values = self.direction_rows.to(device=hidden.device, dtype=hidden.dtype)
        parts = [values[:, axis] for axis in range(len(self.baseline.axes))]
        parts.extend(values[:, list(subset)].prod(-1) for subset in self.interactions.subsets)
        return torch.stack(parts, dim=1)

    def _pre_hook(self, layer_index):
        def hook(_module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError("Decoder layer did not receive hidden states")
            level = torch.autograd.forward_ad.dual_level()
            level.__enter__()
            self._levels[layer_index] = level
            try:
                dual_hidden = torch.autograd.forward_ad.make_dual(hidden, torch.zeros_like(hidden))
                tangent = self._branch_tangents(hidden)
                coefficients = torch.autograd.forward_ad.make_dual(
                    torch.zeros_like(tangent), tangent,
                )
                for key in self._layer_keys[layer_index]:
                    adapted = self.ctx.model._adapted_linears[key]
                    weight = adapted.linear.weight
                    branches = []
                    for axis in self.baseline.axes:
                        factor = self.baseline.prepared_factor(
                            axis, key, weight.device, weight.dtype,
                        )
                        branches.append((factor.A, factor.B, factor.scale))
                    for subset in self.interactions.subsets:
                        branch = self.interactions.branch(subset, key)
                        branches.append((
                            branch.A, branch.B,
                            self.interactions.scale * self.interactions.subset_gain(subset),
                        ))
                    handle = adapted.register_forward_hook(
                        self._projection_hook(branches, coefficients)
                    )
                    self._projection_handles[layer_index].append(handle)
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
                self.fields[layer_index] = (
                    torch.zeros_like(primal) if tangent is None else tangent.detach()
                )
                return (primal, *output[1:]) if isinstance(output, tuple) else primal
            finally:
                self._restore(layer_index)
        return hook

    def close(self):
        for handle in self._pre_handles + self._post_handles:
            handle.remove()
        for layer_index in list(self._levels):
            self._restore(layer_index)


def fit_compositional_static_vectors(ctx: AnalysisContext, baseline, interactions,
                                     prompts: Sequence[dict], batch_size: int,
                                     max_prompt_tokens: int, token_count: int,
                                     static_layers_by_axis: Mapping[str, int | Sequence[int]] | None = None,
                                     static_layer: int | None = None):
    if token_count < 1:
        raise ValueError("static_fit_tokens must be positive")
    directions = default_axis_directions(ctx.axes)
    components = [
        (axis, (index,)) for index, axis in enumerate(ctx.axes)
    ] + [
        ("+".join(ctx.axes[index] for index in subset), tuple(subset))
        for subset in interactions.subsets
    ]
    if static_layer is not None:
        if static_layers_by_axis is not None:
            raise ValueError("Use only one of static_layer and static_layers_by_axis")
        static_layers_by_axis = {axis: int(static_layer) for axis in ctx.axes}
    axis_layers = _resolve_static_axis_layers(ctx, None, static_layers_by_axis)
    component_layers = {
        name: sorted(set().union(*(set(axis_layers[index]) for index in subset)))
        for name, subset in components
    }
    sums = {}
    squared = {}
    counts = defaultdict(int)
    batches = list(chunked(list(prompts), batch_size))
    for batch_index, batch in enumerate(batches, 1):
        LOG.info("Fitted-interaction static fit batch %d/%d", batch_index, len(batches))
        text = _direction_major_prompts([item["prompt"] for item in batch], len(components))
        encoded = encode(ctx.tokenizer, text, ctx.device, max_prompt_tokens)
        branch_rows = torch.eye(len(components), dtype=torch.float32).repeat_interleave(
            len(batch), dim=0,
        )
        capture = _CompositionalLayerSourceCapture(
            ctx, baseline, interactions, branch_rows=branch_rows,
        )
        try:
            with torch.no_grad():
                ctx.model.forward_base(**encoded)
        finally:
            capture.close()
        mask = encoded["attention_mask"]
        for layer, field in capture.fields.items():
            for component_index, (name, _subset) in enumerate(components):
                if layer not in component_layers[name]:
                    continue
                start = component_index * len(batch)
                stop = start + len(batch)
                selected = _selected_projection_inputs(
                    field[start:stop], mask[start:stop], token_count,
                ).float()
                key = (name, layer)
                value_sum = selected.sum(0).cpu()
                value_squared = selected.square().sum().cpu()
                sums[key] = value_sum if key not in sums else sums[key] + value_sum
                squared[key] = value_squared if key not in squared else squared[key] + value_squared
                counts[key] += selected.shape[0]

    component_vectors = {name: {} for name, _subset in components}
    component_metrics = {
        name: {"subset": list(subset), "by_layer": {}}
        for name, subset in components
    }
    for name, _subset in components:
        totals = []
        for layer in component_layers[name]:
            key = (name, layer)
            mean, target, mean_energy, residual = _constant_least_squares_statistics(
                sums[key], squared[key], counts[key],
            )
            component_vectors[name][layer] = mean
            totals.append((target, mean_energy, residual))
            component_metrics[name]["by_layer"][str(layer)] = {
                "sample_count": counts[key],
                "target_field_rms_l2": math.sqrt(max(target, 0.0)),
                "static_vector_l2": math.sqrt(max(mean_energy, 0.0)),
                "least_squares_residual_rms_l2": math.sqrt(residual),
                "constant_field_coherence": mean_energy / target if target > 1e-20 else 1.0,
            }
        target = sum(value[0] for value in totals)
        mean_energy = sum(value[1] for value in totals)
        residual = sum(value[2] for value in totals)
        component_metrics[name]["global"] = {
            "selected_layers": component_layers[name],
            "energy_weighted_constant_field_coherence": (
                mean_energy / target if target > 1e-20 else 1.0
            ),
            "residual_direct_sum_relative_error": (
                math.sqrt(residual / target) if target > 1e-20 else 0.0
            ),
        }
    selected_layers = sorted(set().union(*(set(values) for values in component_layers.values())))
    vectors = {}
    task_vectors = {}
    metrics = {}
    for direction_name, direction in directions.items():
        coefficients = {
            name: (
                float(direction[subset[0]]) if len(subset) == 1 else
                float(math.prod(direction[index] for index in subset))
            )
            for name, subset in components
        }
        vectors[direction_name] = {}
        task_vectors[direction_name] = {}
        for layer in selected_layers:
            template = next(
                component_vectors[component_name][layer]
                for component_name, _subset in components
                if layer in component_vectors[component_name]
            )
            total = torch.zeros_like(template)
            task_total = torch.zeros_like(template)
            for component_name, _subset in components:
                if layer in component_vectors[component_name]:
                    total = total + coefficients[component_name] * component_vectors[component_name][layer]
                    if len(_subset) == 1:
                        task_total = task_total + coefficients[component_name] * component_vectors[component_name][layer]
            vectors[direction_name][layer] = total
            task_vectors[direction_name][layer] = task_total
        metrics[direction_name] = {
            "parameterization": (
                "selected_layer_multi_affine_components"
                if static_layers_by_axis else "all_layer_multi_affine_components"
            ),
            "static_layers_by_axis": (
                {axis: axis_layers[index] for index, axis in enumerate(ctx.axes)}
                if static_layers_by_axis else None
            ),
            "component_coefficients": coefficients,
            "by_layer": {
                str(layer): {"static_vector_l2": float(vectors[direction_name][layer].norm())}
                for layer in selected_layers
            },
        }
    return directions, vectors, task_vectors, metrics, component_metrics


def save_static_vector_artifact(path: Path, ctx: AnalysisContext, directions, vectors,
                                task_vectors, metrics, component_metrics) -> None:
    """Persist compiled residual-space vectors, rather than only their norms.

    Static-generation JSON reports intentionally remain compact.  This artifact is
    the companion for geometry diagnostics: it contains the actual constant
    residual vectors indexed by complete direction and decoder layer.  All tensors
    are detached CPU FP32 values, so it is safe to inspect without loading a model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialize = lambda source: {
        name: {
            str(layer): value.detach().float().cpu().contiguous()
            for layer, value in by_layer.items()
        }
        for name, by_layer in source.items()
    }
    torch.save({
        "format": "compositional_static_residual_vectors_v1",
        "definition": (
            "Constant least-squares residual-stream steering vectors.  "
            "vector[direction][layer] is the vector added at the output of "
            "that decoder layer."
        ),
        "axes": list(ctx.axes),
        "directions": {name: [float(value) for value in direction]
                       for name, direction in directions.items()},
        "fitted_interaction_vectors": serialize(vectors),
        "task_arithmetic_vectors": serialize(task_vectors),
        "static_metrics": metrics,
        "component_static_fit": component_metrics,
    }, path)
    LOG.info("Saved compiled static vectors to %s", path)


def static_generations(ctx: AnalysisContext, prompts: Sequence[dict], directions,
                       vectors, task_vectors, metrics, component_metrics, generation_batch_size: int,
                       direction_batch_size: int, max_prompt_tokens: int,
                       max_new_tokens: int, include_task_arithmetic: bool = True,
                       last_token_only: bool = False,
                       steer_prefill: bool = True) -> dict:
    items = list(directions.items())
    generated = {name: [] for name in directions}
    task_generated = {name: [] for name in directions}
    prompt_batches = list(chunked(list(prompts), generation_batch_size))
    for prompt_index, batch in enumerate(prompt_batches, 1):
        texts = [item["prompt"] for item in batch]
        for direction_index, direction_batch in enumerate(
            chunked(items, direction_batch_size), 1,
        ):
            LOG.info(
                "Static generation prompt batch %d/%d, direction batch %d/%d",
                prompt_index, len(prompt_batches), direction_index,
                math.ceil(len(items) / direction_batch_size),
            )
            names = [name for name, _direction in direction_batch]
            selected_layers = sorted(vectors[names[0]])
            variants = ("fitted", "task") if include_task_arithmetic else ("fitted",)
            variant_names = [f"{variant}::{name}" for variant in variants for name in names]
            encoded = encode(
                ctx.tokenizer, _direction_major_prompts(texts, len(variant_names)),
                ctx.device, max_prompt_tokens,
            )
            fields = {}
            for layer in selected_layers:
                rows = [vectors[name][layer] for name in names]
                if include_task_arithmetic:
                    rows += [task_vectors[name][layer] for name in names]
                fields[layer] = torch.stack(rows).repeat_interleave(len(texts), dim=0)
            injector = _TokenwiseInjector(
                ctx.layers, fields, strength=1.0,
                last_token_only=last_token_only,
                steer_prefill=steer_prefill,
            )
            try:
                with torch.no_grad():
                    output = ctx.model.llama.generate(
                        **encoded, max_new_tokens=max_new_tokens,
                        do_sample=False, use_cache=True,
                    )
            finally:
                injector.close()
            partitioned = _partition_generations(
                ctx, output, encoded["input_ids"].shape[1], variant_names, len(texts),
            )
            for name in names:
                generated[name].extend(partitioned[f"fitted::{name}"])
                if include_task_arithmetic:
                    task_generated[name].extend(partitioned[f"task::{name}"])
    return {
        "definition": (
            "Prompt-independent residual vectors for every axis task adapter and "
            "HeRD merging R_S at the configured decoder layers. At direction "
            "s, each retained layer receives sum_i s_i c_{l,i} plus "
            "sum_S prod_{i in S}(s_i) c_{l,S}."
        ),
        "static_parameterization": metrics[next(iter(metrics))]["parameterization"],
        "static_layers_by_axis": metrics[next(iter(metrics))]["static_layers_by_axis"],
        "steer_prefill": steer_prefill,
        "component_static_fit": component_metrics,
        "axes": ctx.axes,
        "combination_count": len(directions),
        "prompt_count": len(prompts),
        "directions": {
            name: {
                "direction": direction,
                "static_fit": metrics[name],
                "examples": [
                    {
                        "prompt_index": index,
                        "source_index": prompts[index].get("source_index"),
                        "prompt": prompts[index]["prompt"],
                        "fitted_interaction_static_generation": generated[name][index],
                        **({
                            "independent_task_arithmetic_static_generation": task_generated[name][index],
                        } if include_task_arithmetic else {}),
                    }
                    for index in range(len(prompts))
                ],
            }
            for name, direction in directions.items()
        },
    }


def _inner(first, second) -> float:
    A1, B1, scale1 = first
    A2, B2, scale2 = second
    return float(scale1 * scale2 * (
        (B1.float().T @ B2.float()) * (A1.float() @ A2.float().T)
    ).sum())


def _norm_sq(branches) -> float:
    return sum(_inner(first, second) for first in branches for second in branches)


def _singular_values(A: torch.Tensor, B: torch.Tensor, scale: float) -> list[float]:
    _q_b, r_b = torch.linalg.qr(B.float(), mode="reduced")
    _q_a, r_a = torch.linalg.qr(A.float().T, mode="reduced")
    return [float(value) for value in torch.linalg.svdvals(scale * (r_b @ r_a.T))]


def interaction_geometry(baseline, interactions) -> dict:
    subset_stats = {}
    projection_stats = {}
    for subset in interactions.subsets:
        label = "+".join(baseline.axes[index] for index in subset)
        by_layer_sq = defaultdict(float)
        total_sq = 0.0
        ta_sq = 0.0
        inner_total = 0.0
        for key in baseline.keys:
            branch = interactions.branch(subset, key)
            current = (
                branch.A.detach().cpu(), branch.B.detach().cpu(),
                float(interactions.scale * interactions.subset_gain(subset)),
            )
            axes = [
                (baseline.factors[baseline.axes[index]][key].A,
                 baseline.factors[baseline.axes[index]][key].B,
                 baseline.factors[baseline.axes[index]][key].scale)
                for index in subset
            ]
            current_sq = _norm_sq([current])
            current_ta_sq = _norm_sq(axes)
            current_inner = sum(_inner(current, axis) for axis in axes)
            total_sq += current_sq
            ta_sq += current_ta_sq
            inner_total += current_inner
            by_layer_sq[layer_from_key(key)] += current_sq
            singular = _singular_values(*current)
            projection_stats[f"{label}:{key}"] = {
                "frobenius": math.sqrt(max(current_sq, 0.0)),
                "rms_entry": math.sqrt(max(current_sq, 0.0) / (branch.A.shape[1] * branch.B.shape[0])),
                "singular_values": singular,
                "numerical_rank_1e-4": sum(
                    value > (singular[0] * 1e-4 if singular else 0.0) for value in singular
                ),
            }
        subset_stats[label] = {
            "subset": list(subset),
            "subset_gain": float(interactions.subset_gain(subset)),
            "interaction_frobenius": math.sqrt(max(total_sq, 0.0)),
            "corresponding_task_sum_frobenius": math.sqrt(max(ta_sq, 0.0)),
            "interaction_to_task_sum_ratio": math.sqrt(max(total_sq, 0.0) / ta_sq) if ta_sq else 0.0,
            "cosine_with_task_sum": inner_total / math.sqrt(total_sq * ta_sq) if total_sq and ta_sq else 0.0,
            "by_layer_frobenius": {
                str(layer): math.sqrt(max(value, 0.0)) for layer, value in sorted(by_layer_sq.items())
            },
        }

    subset_labels = ["+".join(baseline.axes[index] for index in subset)
                     for subset in interactions.subsets]
    pairwise_subset_cosine = {}
    for first_index, first_subset in enumerate(interactions.subsets):
        first_label = subset_labels[first_index]
        first_sq = subset_stats[first_label]["interaction_frobenius"] ** 2
        for second_index in range(first_index + 1, len(interactions.subsets)):
            second_subset = interactions.subsets[second_index]
            second_label = subset_labels[second_index]
            second_sq = subset_stats[second_label]["interaction_frobenius"] ** 2
            cross = 0.0
            for key in baseline.keys:
                first = interactions.branch(first_subset, key)
                second = interactions.branch(second_subset, key)
                cross += _inner(
                    (first.A.detach().cpu(), first.B.detach().cpu(),
                     float(interactions.scale * interactions.subset_gain(first_subset))),
                    (second.A.detach().cpu(), second.B.detach().cpu(),
                     float(interactions.scale * interactions.subset_gain(second_subset))),
                )
            pairwise_subset_cosine[f"{first_label} :: {second_label}"] = (
                cross / math.sqrt(first_sq * second_sq) if first_sq and second_sq else 0.0
            )

    combinations = {}
    directions = default_axis_directions(baseline.axes)
    for name, values in directions.items():
        active = [index for index, value in enumerate(values) if value]
        # Cross-projection matrices are orthogonal coordinates in the direct sum;
        # compute each site's terms separately rather than mixing their shapes.
        ta_sq = correction_sq = cross = 0.0
        for key in baseline.keys:
            ta_site = [(baseline.factors[baseline.axes[index]][key].A,
                        baseline.factors[baseline.axes[index]][key].B,
                        baseline.factors[baseline.axes[index]][key].scale)
                       for index in active]
            correction_site = [(interactions.branch(subset, key).A.detach().cpu(),
                                interactions.branch(subset, key).B.detach().cpu(),
                                float(interactions.scale * interactions.subset_gain(subset)))
                               for subset in interactions.subsets
                               if all(values[index] for index in subset)]
            ta_sq += _norm_sq(ta_site)
            correction_sq += _norm_sq(correction_site)
            cross += sum(_inner(first, second) for first in ta_site for second in correction_site)
        full_sq = ta_sq + correction_sq + 2.0 * cross
        combinations[name] = {
            "direction": values,
            "task_arithmetic_frobenius": math.sqrt(max(ta_sq, 0.0)),
            "interaction_frobenius": math.sqrt(max(correction_sq, 0.0)),
            "full_update_frobenius": math.sqrt(max(full_sq, 0.0)),
            "interaction_to_task_arithmetic_ratio": (
                math.sqrt(max(correction_sq, 0.0) / ta_sq) if ta_sq else 0.0
            ),
            "interaction_to_full_update_ratio": (
                math.sqrt(max(correction_sq, 0.0) / full_sq) if full_sq else 0.0
            ),
            "cosine_task_arithmetic_interaction": (
                cross / math.sqrt(ta_sq * correction_sq) if ta_sq and correction_sq else 0.0
            ),
        }
    return {
        "definition": "Exact low-rank direct-sum matrix geometry; no dense matrices are materialized.",
        "axes": baseline.axes,
        "rank": interactions.rank,
        "subsets": subset_stats,
        "pairwise_subset_cosine": pairwise_subset_cosine,
        "binary_combinations": combinations,
        "by_projection": projection_stats,
    }


def _common(parser):
    parser.add_argument("--independent_root", type=Path, required=True)
    parser.add_argument("--interaction_dir", type=Path,
                        help="Defaults to INDEPENDENT_ROOT/compositional-interactions.")
    parser.add_argument("--axes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf_token")


def _json_object(value: str | None):
    if value is None:
        return None
    stripped = value.lstrip()
    if stripped.startswith("{"):
        payload = json.loads(value)
    else:
        candidate = Path(value)
        payload = json.loads(candidate.read_text())
    if not isinstance(payload, dict):
        raise ValueError("--static_layers_by_axis must be a JSON object or JSON file")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fitted distributional interactions.")
    commands = parser.add_subparsers(dest="command", required=True)
    geometry = commands.add_parser("geometry")
    _common(geometry)

    for name in ("weight", "static"):
        current = commands.add_parser(name)
        _common(current)
        current.add_argument("--prompts_file", type=Path, default=ROOT / "datasets" / "test_prompts.jsonl")
        current.add_argument("--num_prompts", type=int, default=100)
        current.add_argument("--prompt_offset", type=int, default=0,
                            help="Skip this many shuffled prompts before the generation split.")
        current.add_argument("--generation_batch_size", type=int, default=8)
        current.add_argument("--direction_batch_size", type=int, default=8)
        current.add_argument("--max_prompt_tokens", type=int, default=128)
        current.add_argument("--max_new_tokens", type=int, default=128)
        current.add_argument("--device", default="cuda:0")
        current.add_argument("--seed", type=int, default=42)
        if name == "weight":
            current.add_argument(
                "--update_fitted_only", action="store_true",
                help=("Generate only fitted_interaction_weight_generation and update "
                      "that field in an existing --output file, preserving all previously "
                      "generated task-arithmetic/TIES/DARE/KNoTs controls."),
            )
            current.add_argument(
                "--include_task_arithmetic", action="store_true",
                help=("Generate a paired exact task-arithmetic control on every "
                      "prompt and direction in the same batched forward."),
            )
            current.add_argument(
                "--merge_methods", default="",
                help=("Comma-separated independent adapter merges additionally generated: "
                      "ties,dare_task_arithmetic,dare_ties,knots_ties."),
            )
            current.add_argument(
                "--task_arithmetic_only", action="store_true",
                help="Generate only exact independent-LoRA task arithmetic.",
            )
            current.add_argument(
                "--pure_axes_only", action="store_true",
                help="Generate only neutral and single-active-axis directions.",
            )
        if name == "static":
            current.add_argument(
                "--update_fitted_only", action="store_true",
                help=("Generate only fitted_interaction_static_generation and update "
                      "that field in an existing --output file."),
            )
            current.add_argument("--fit_prompts", type=int, default=100)
            current.add_argument(
                "--fit_prompts_file", type=Path,
                help=("Optional calibration-only prompt file. When set, static vectors "
                      "are fitted from this file and --prompts_file is reserved entirely "
                      "for held-out generation."),
            )
            current.add_argument("--static_fit_batch_size", type=int, default=8)
            current.add_argument("--static_fit_tokens", type=int, default=1)
            current.add_argument(
                "--static_vectors_output", type=Path,
                help=("Optional .pt artifact receiving the compiled residual-space "
                      "static vectors for sparsity and orthogonality diagnostics."),
            )
            current.add_argument(
                "--compile_only", action="store_true",
                help=("Compile and optionally save static vectors, but do not run "
                      "autoregressive generations."),
            )
            current.add_argument(
                "--steer-prefill", action=argparse.BooleanOptionalAction, default=True,
                help=("Apply static residual vectors during the initial prompt prefill. "
                      "Disable for a diagnostic that leaves the prompt/cache base-model "
                      "conditioned while retaining injection at every generated token."),
            )
            current.add_argument(
                "--static_layers_by_axis",
                help=("JSON object or file assigning each axis one residual layer or a list. "
                      "An interaction component is retained at the union of its constituent "
                      "axes' selected layers."),
            )
            current.add_argument(
                "--static_layer", type=int,
                help=("Retain every axis and interaction component at exactly this one "
                      "zero-indexed residual layer. Intended for a complete set of LoRAs "
                      "trained at the same layer."),
            )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    root = args.independent_root.resolve()
    artifact = (args.interaction_dir or root / "compositional-interactions").resolve()
    if args.command == "geometry":
        axes = _axis_order(root, args.axes)
        baseline = load_independent_loras(root, axes)
        interactions = MultiAffineInteractions.load_artifact(artifact, baseline)
        payload = interaction_geometry(baseline, interactions)
    else:
        fit_count = args.fit_prompts if args.command == "static" else 0
        total = args.num_prompts + args.prompt_offset + (
            0 if args.command == "static" and args.fit_prompts_file else fit_count
        )
        prompts = read_prompts(args.prompts_file, total, args.seed)
        ctx, baseline, interactions = load_compositional_context(
            root, artifact, args.axes, args.device, args.max_prompt_tokens,
            attn_implementation="eager" if args.command == "static" else "sdpa",
        )
        if args.command == "weight":
            generation_prompts = prompts[args.prompt_offset:args.prompt_offset + args.num_prompts]
            payload = compositional_weight_generations(
                ctx, baseline, interactions, generation_prompts, args.generation_batch_size,
                args.direction_batch_size, args.max_prompt_tokens, args.max_new_tokens,
                include_task_arithmetic=(
                    args.include_task_arithmetic and not args.update_fitted_only
                ),
                merge_methods=([] if args.update_fitted_only else
                    [item for item in args.merge_methods.split(",") if item]),
                task_arithmetic_only=args.task_arithmetic_only,
                pure_axes_only=args.pure_axes_only,
            )
            payload["split"] = {
                "generation_prompt_offset": args.prompt_offset,
                "generation_prompts": len(generation_prompts),
            }
        else:
            if args.fit_prompts_file:
                fit_prompts = read_prompts(
                    args.fit_prompts_file, args.fit_prompts, args.seed,
                )
                generation_start = args.prompt_offset
            else:
                fit_prompts = prompts[:args.fit_prompts]
                generation_start = args.fit_prompts + args.prompt_offset
            generation_prompts = prompts[generation_start:generation_start + args.num_prompts]
            fit_text = {" ".join(item["prompt"].split()) for item in fit_prompts}
            generation_text = {
                " ".join(item["prompt"].split()) for item in generation_prompts
            }
            overlap = fit_text & generation_text
            if overlap:
                raise ValueError(
                    f"Static calibration overlaps {len(overlap)} generation prompts"
                )
            directions, vectors, task_vectors, metrics, component_metrics = fit_compositional_static_vectors(
                ctx, baseline, interactions, fit_prompts, args.static_fit_batch_size,
                args.max_prompt_tokens, args.static_fit_tokens,
                _json_object(args.static_layers_by_axis), args.static_layer,
            )
            if args.static_vectors_output:
                save_static_vector_artifact(
                    args.static_vectors_output, ctx, directions, vectors, task_vectors,
                    metrics, component_metrics,
                )
            if args.compile_only:
                payload = {
                    "definition": (
                        "Compiled fitted-interaction and task-arithmetic static "
                        "residual vectors; no autoregressive generations were run."
                    ),
                    "axes": ctx.axes,
                    "directions": directions,
                    "static_parameterization": metrics[next(iter(metrics))]["parameterization"],
                    "static_layers_by_axis": metrics[next(iter(metrics))]["static_layers_by_axis"],
                    "component_static_fit": component_metrics,
                    "static_vector_artifact": (
                        str(args.static_vectors_output.resolve())
                        if args.static_vectors_output else None
                    ),
                }
            else:
                payload = static_generations(
                    ctx, generation_prompts, directions, vectors, task_vectors, metrics, component_metrics,
                    args.generation_batch_size, args.direction_batch_size,
                    args.max_prompt_tokens, args.max_new_tokens,
                    include_task_arithmetic=not args.update_fitted_only,
                    steer_prefill=args.steer_prefill,
                )
            payload["split"] = {
                "static_fit_prompts": len(fit_prompts),
                "static_fit_prompts_file": (
                    str(args.fit_prompts_file.resolve()) if args.fit_prompts_file else None
                ),
                "generation_prompt_offset_after_fit": args.prompt_offset,
                "generation_prompts": len(generation_prompts),
            }
    result = {
        "independent_root": str(root),
        "interaction_dir": str(artifact),
        "command": args.command,
        "analysis": payload,
    }
    if (args.command in {"weight", "static"} and args.update_fitted_only
            and args.output.is_file()):
        previous = json.loads(args.output.read_text())
        old_directions = previous.get("analysis", {}).get("directions", {})
        new_directions = result["analysis"]["directions"]
        if set(old_directions) != set(new_directions):
            raise ValueError("Cannot update fitted-only output: direction sets differ")
        for name, new_direction in new_directions.items():
            old_examples = old_directions[name].get("examples", [])
            new_examples = new_direction.get("examples", [])
            if len(old_examples) != len(new_examples):
                raise ValueError(
                    f"Cannot update fitted-only output for {name}: prompt counts differ"
                )
            for old, new in zip(old_examples, new_examples):
                if (old.get("source_index"), old.get("prompt")) != (
                    new.get("source_index"), new.get("prompt")
                ):
                    raise ValueError(
                        f"Cannot update fitted-only output for {name}: prompts differ"
                    )
                field = (
                    "fitted_interaction_weight_generation" if args.command == "weight"
                    else "fitted_interaction_static_generation"
                )
                old[field] = new[field]
        previous["interaction_dir"] = result["interaction_dir"]
        previous["analysis"]["definition"] = result["analysis"]["definition"]
        previous["analysis"]["split"] = result["analysis"].get("split")
        result = previous
        LOG.info("Updated only fitted-interaction fields; preserved baseline generations")
    json_dump(args.output, result)
    LOG.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
