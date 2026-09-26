from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch

from .activation import (fast_layer_jacobian_multi_batch, fit_static_layer_vectors,
                         generate_static_layer_vectors)
from .common import (AnalysisContext, chunked, encode, encode_messages,
                     signal_grid)
from .independent import IndependentLoRAHooks, load_independent_loras
from .steering_baselines import (ActivationController, MultiAxisControllerHooks,
                                 MultiMethodControllerHooks,
                                 fit_caa, fit_linear_act, fit_mimic, fit_recontrol,
                                 fit_repe, fit_odesteer,
                                 standardized_mean_separation)

LOG = logging.getLogger(__name__)

CLASSICAL_METHODS = ("caa", "repe", "mimic", "re_control", "linear_act", "odesteer")
OUR_METHODS = ("ours_static", "ours_autoregressive_jacobian")
ALL_METHODS = CLASSICAL_METHODS + OUR_METHODS


class _LayerCapture:
    def __init__(self, layers: Sequence[torch.nn.Module]):
        self.values: dict[int, torch.Tensor] = {}
        self.handles = [layer.register_forward_hook(self._hook(index))
                        for index, layer in enumerate(layers)]

    def _hook(self, index: int):
        def capture(_module, _inputs, output):
            self.values[index] = (output[0] if isinstance(output, tuple) else output).detach()
        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()


def _last_token(value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    indices = attention_mask.long().sum(1).sub(1).clamp_min(0)
    # Inputs are left padded, so the final active token is also the final tensor
    # position. Keep the general gather for tokenizer portability.
    if bool((attention_mask[:, -1] == 1).all()):
        return value[:, -1].float()
    return value[torch.arange(len(value), device=value.device), indices].float()


def _endpoint_activations(ctx: AnalysisContext, independent, prompts: Sequence[dict],
                          batch_size: int, max_prompt_tokens: int) -> tuple[dict, dict]:
    """Return raw-base and one-independent-axis endpoint residual states."""
    neutral = defaultdict(list)
    positive = {axis: defaultdict(list) for axis in range(len(ctx.axes))}
    batches = list(chunked(list(prompts), batch_size))
    for batch_index, batch in enumerate(batches):
        LOG.info("Baseline activation cache batch %d/%d", batch_index + 1, len(batches))
        encoded = encode(ctx.tokenizer, [item["prompt"] for item in batch],
                         ctx.device, max_prompt_tokens)
        capture = _LayerCapture(ctx.layers)
        try:
            with torch.no_grad():
                ctx.model.llama(**encoded, use_cache=False)
        finally:
            capture.close()
        for layer, value in capture.values.items():
            neutral[layer].append(_last_token(value, encoded["attention_mask"]).cpu())
        for axis in range(len(ctx.axes)):
            values = torch.zeros(len(batch), len(ctx.axes), device=ctx.device, dtype=ctx.dtype)
            values[:, axis] = 1
            capture = _LayerCapture(ctx.layers)
            try:
                with torch.no_grad(), IndependentLoRAHooks(ctx.model, independent, values):
                    ctx.model.llama(**encoded, use_cache=False)
            finally:
                capture.close()
            for layer, value in capture.values.items():
                positive[axis][layer].append(
                    _last_token(value, encoded["attention_mask"]).cpu()
                )
    return (
        {layer: torch.cat(values) for layer, values in neutral.items()},
        {axis: {layer: torch.cat(values) for layer, values in layers.items()}
         for axis, layers in positive.items()},
    )


def _messages(value) -> list[dict] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, dict) and isinstance(item.get("role"), str)
               and isinstance(item.get("content"), str) for item in value):
        return None
    return value


def _odesteer_contrastive_activations(
    ctx: AnalysisContext,
    sample_count: int,
    batch_size: int,
    max_prompt_tokens: int,
    seed: int,
) -> tuple[dict, dict, dict]:
    """Collect paper-style positive/negative conversation activations."""
    from datasets import load_dataset

    specs = {
        (spec["axis"] if isinstance(spec, dict) else spec.axis):
        (spec["dataset"] if isinstance(spec, dict) else spec.dataset)
        for spec in ctx.config.axis_datasets
    }
    negative = {axis: defaultdict(list) for axis in range(len(ctx.axes))}
    positive = {axis: defaultdict(list) for axis in range(len(ctx.axes))}
    report = {}
    for axis, axis_name in enumerate(ctx.axes):
        if axis_name not in specs:
            raise ValueError(f"No configured contrastive dataset for axis {axis_name!r}")
        dataset = load_dataset(specs[axis_name], split=ctx.config.dataset_split)
        order = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed + axis),
        ).tolist()
        pairs = []
        for index in order:
            row = dataset[index]
            status = str(row.get("_status", "")).strip().lower()
            failed = row.get("generation_failed", False)
            if status == "failed" or failed is True or str(failed).lower() in {"true", "1", "yes"}:
                continue
            current_positive = _messages(row.get("messages"))
            current_negative = _messages(row.get("original_messages"))
            if current_positive is None or current_negative is None:
                continue
            pairs.append((current_positive, current_negative))
            if len(pairs) == sample_count:
                break
        if len(pairs) < sample_count:
            raise ValueError(
                f"Axis {axis_name!r} supplied only {len(pairs)}/{sample_count} valid ODESteer pairs"
            )
        batches = list(chunked(pairs, batch_size))
        for batch_index, batch in enumerate(batches):
            LOG.info(
                "ODESteer contrastive activations: axis %s, batch %d/%d",
                axis_name, batch_index + 1, len(batches),
            )
            negatives = [item[1] for item in batch]
            positives = [item[0] for item in batch]
            encoded = encode_messages(
                ctx.tokenizer, negatives + positives, ctx.device, max_prompt_tokens,
                add_generation_prompt=False,
            )
            capture = _LayerCapture(ctx.layers)
            try:
                with torch.no_grad():
                    ctx.model.llama(**encoded, use_cache=False)
            finally:
                capture.close()
            count = len(batch)
            for layer, value in capture.values.items():
                selected = _last_token(value, encoded["attention_mask"]).cpu()
                negative[axis][layer].append(selected[:count])
                positive[axis][layer].append(selected[count:])
        report[axis_name] = {
            "dataset": specs[axis_name],
            "samples": len(pairs),
            "positive_column": "messages",
            "negative_column": "original_messages",
        }
    return (
        {axis: {layer: torch.cat(values) for layer, values in layers.items()}
         for axis, layers in negative.items()},
        {axis: {layer: torch.cat(values) for layer, values in layers.items()}
         for axis, layers in positive.items()},
        report,
    )


def _controller_fit(method: str, positive: torch.Tensor, negative: torch.Tensor,
                    method_kwargs: Mapping) -> tuple[ActivationController, dict]:
    start = time.perf_counter()
    if method == "caa":
        controller, details = fit_caa(positive, negative), {}
    elif method == "repe":
        controller = fit_repe(positive, negative)
        mean_delta = positive.float().mean(0) - negative.float().mean(0)
        # PCA fixes only orientation; project the endpoint mean displacement to
        # choose a data-derived full-strength magnitude without test tuning.
        controller.vector.mul_(float(controller.vector @ mean_delta))
        details = {}
    elif method == "mimic":
        controller, details = fit_mimic(positive, negative), {}
    elif method == "linear_act":
        controller, details = fit_linear_act(positive, negative), {}
    elif method == "re_control":
        controller, details = fit_recontrol(positive, negative, **method_kwargs)
    elif method == "odesteer":
        controller, details = fit_odesteer(positive, negative, **method_kwargs)
    else:
        raise ValueError(f"Unknown activation baseline {method!r}")
    details["fit_wall_seconds"] = time.perf_counter() - start
    return controller, details


def _fit_controllers(ctx: AnalysisContext, neutral: Mapping[int, torch.Tensor],
                     positive: Mapping[int, Mapping[int, torch.Tensor]],
                     methods: Sequence[str], validation_fraction: float,
                     controller_kwargs: Mapping[str, Mapping],
                     selected_layers: Mapping[str, int] | None = None) -> tuple[dict, dict]:
    count = len(next(iter(neutral.values())))
    validation_count = max(1, round(validation_fraction * count))
    training_count = count - validation_count
    if training_count < 2:
        raise ValueError("Controller fitting needs at least two training prompts.")
    controllers = {method: defaultdict(dict) for method in methods}
    report = {method: {} for method in methods}
    for axis, axis_name in enumerate(ctx.axes):
        layer_scores = {
            layer: standardized_mean_separation(
                positive[axis][layer][:training_count], neutral[layer][:training_count],
            )
            for layer in neutral
        }
        selected_layer = (
            int(selected_layers[axis_name]) if selected_layers is not None
            else max(layer_scores, key=layer_scores.get)
        )
        if selected_layer not in layer_scores:
            raise ValueError(f"CAA-calibrated layer {selected_layer} is unavailable for {axis_name}")
        train_positive = positive[axis][selected_layer][:training_count]
        train_negative = neutral[selected_layer][:training_count]
        validation_positive = positive[axis][selected_layer][training_count:]
        validation_negative = neutral[selected_layer][training_count:]
        LOG.info("Axis %s selected residual layer %d (separation %.4g)",
                 axis_name, selected_layer, layer_scores[selected_layer])
        for method in methods:
            controller, details = _controller_fit(
                method, train_positive, train_negative,
                {
                    **controller_kwargs.get(method, {}),
                    **({"seed": int(controller_kwargs.get(method, {}).get("seed", 42)) + axis}
                       if method in {"re_control", "odesteer"} else {}),
                },
            )
            controller.to(ctx.device, ctx.dtype)
            with torch.no_grad() if method != "re_control" else contextlib.nullcontext():
                predicted = validation_negative.to(ctx.device, dtype=ctx.dtype)
                displacement = controller.displacement(predicted).float().cpu()
            target = validation_positive - validation_negative
            residual = displacement - target
            relative = float(residual.norm() / target.norm().clamp_min(1e-12))
            cosine = float(torch.nn.functional.cosine_similarity(
                displacement.flatten(), target.flatten(), dim=0,
            ))
            controllers[method][selected_layer][axis] = controller
            report[method][axis_name] = {
                "selected_layer": selected_layer,
                "training_prompts": training_count,
                "validation_prompts": validation_count,
                "validation_relative_endpoint_field_error": relative,
                "validation_endpoint_field_cosine": cosine,
                "layer_selection_standardized_mean_separation": {
                    str(layer): score for layer, score in sorted(layer_scores.items())
                },
                **details,
            }
    return {method: dict(values) for method, values in controllers.items()}, report


def _fit_odesteer_controllers(
    ctx: AnalysisContext,
    negative: Mapping[int, Mapping[int, torch.Tensor]],
    positive: Mapping[int, Mapping[int, torch.Tensor]],
    validation_fraction: float,
    kwargs: Mapping,
    selected_layers: Mapping[int, int],
) -> tuple[dict, dict]:
    """Fit each ODESteer barrier at its calibration-only CAA-optimal layer."""
    controllers = defaultdict(dict)
    report = {}
    for axis, axis_name in enumerate(ctx.axes):
        selected_layer = int(selected_layers[axis])
        if selected_layer not in negative[axis]:
            available = sorted(negative[axis])
            raise ValueError(
                f"ODESteer layer {selected_layer} for axis {axis_name!r} is unavailable; "
                f"expected one of {available[0]}..{available[-1]}."
            )
        count = len(next(iter(negative[axis].values())))
        validation_count = max(1, round(validation_fraction * count))
        training_count = count - validation_count
        if training_count < 2:
            raise ValueError("ODESteer fitting needs at least two training pairs per axis")
        layer_separations = {
            layer: standardized_mean_separation(
                positive[axis][layer][:training_count],
                negative[axis][layer][:training_count],
            )
            for layer in negative[axis]
        }
        train_positive = positive[axis][selected_layer][:training_count]
        train_negative = negative[axis][selected_layer][:training_count]
        validation_positive = positive[axis][selected_layer][training_count:]
        validation_negative = negative[axis][selected_layer][training_count:]
        LOG.info(
            "ODESteer axis %s using fixed residual layer %d (separation %.4g)",
            axis_name, selected_layer, layer_separations[selected_layer],
        )
        start = time.perf_counter()
        controller, details = fit_odesteer(
            train_positive, train_negative,
            **{**kwargs, "seed": int(kwargs.get("seed", 42)) + axis},
        )
        details["fit_wall_seconds"] = time.perf_counter() - start
        controller.to(ctx.device, ctx.dtype)
        with torch.no_grad():
            validation_positive_device = validation_positive.to(ctx.device)
            validation_negative_device = validation_negative.to(ctx.device)
            positive_score = controller._barrier(validation_positive_device).cpu()
            negative_score = controller._barrier(validation_negative_device).cpu()
            displacement = controller.displacement(validation_negative_device).float()
            steered_negative = validation_negative_device.float() + displacement
            steered_score = controller._barrier(steered_negative).cpu()
            target = validation_positive_device.float() - validation_negative_device.float()
            displacement_norm = displacement.norm(dim=-1)
            activation_norm = validation_negative_device.float().norm(dim=-1)
            target_cosine = torch.nn.functional.cosine_similarity(
                displacement, target, dim=-1,
            ).cpu()
        validation_accuracy = float(torch.cat(
            (positive_score >= 0, negative_score < 0),
        ).float().mean())
        controllers[selected_layer][axis] = controller
        report[axis_name] = {
            "selected_layer": selected_layer,
            "layer_selection": "calibration_only_caa_standardized_mean_separation",
            "training_pairs": training_count,
            "validation_pairs": validation_count,
            "validation_barrier_accuracy": validation_accuracy,
            "validation_positive_score_mean": float(positive_score.mean()),
            "validation_negative_score_mean": float(negative_score.mean()),
            "validation_steered_negative_score_mean": float(steered_score.mean()),
            "validation_barrier_increase_mean": float((steered_score - negative_score).mean()),
            "validation_barrier_increase_fraction": float(
                (steered_score >= negative_score - 1e-6).float().mean()
            ),
            "validation_ode_displacement_l2_mean": float(displacement_norm.mean().cpu()),
            "validation_activation_l2_mean": float(activation_norm.mean().cpu()),
            "validation_relative_displacement_mean": float(
                (displacement_norm / activation_norm.clamp_min(1e-12)).mean().cpu()
            ),
            "validation_endpoint_displacement_cosine_mean": float(target_cosine.mean()),
            "layer_selection_standardized_mean_separation": {
                str(layer): score for layer, score in sorted(layer_separations.items())
            },
            **details,
        }
    return dict(controllers), report


def _directions(axes: Sequence[str], levels: Sequence[float]) -> list[tuple[str, list[float]]]:
    return [
        (",".join(f"{axis}={float(value):g}" for axis, value in zip(axes, point)), list(point))
        for point in signal_grid(len(axes), levels)
    ]


def _decode(ctx: AnalysisContext, output: torch.Tensor, prefix: int,
            names: Sequence[str], prompt_count: int) -> dict[str, list[str]]:
    decoded = [ctx.tokenizer.decode(row[prefix:], skip_special_tokens=True) for row in output]
    return {name: decoded[index * prompt_count:(index + 1) * prompt_count]
            for index, name in enumerate(names)}


def _expanded_batch(ctx: AnalysisContext, prompts: Sequence[str], direction_items,
                    max_prompt_tokens: int):
    expanded = [prompt for _name, _direction in direction_items for prompt in prompts]
    encoded = encode(ctx.tokenizer, expanded, ctx.device, max_prompt_tokens)
    strengths = torch.tensor(
        [direction for _name, direction in direction_items], device=ctx.device, dtype=ctx.dtype,
    ).repeat_interleave(len(prompts), dim=0)
    return encoded, strengths


def _generate_controller(ctx: AnalysisContext, prompts: Sequence[str], direction_items,
                         controllers, max_prompt_tokens: int, max_new_tokens: int) -> dict:
    encoded, strengths = _expanded_batch(ctx, prompts, direction_items, max_prompt_tokens)
    with torch.no_grad(), MultiAxisControllerHooks(ctx.layers, controllers, strengths):
        output = ctx.model.llama.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    return _decode(ctx, output, encoded["input_ids"].shape[1],
                   [item[0] for item in direction_items], len(prompts))


def _generate_controller_methods(ctx: AnalysisContext, prompts: Sequence[str], direction_items,
                                 method_names: Sequence[str], controllers: Mapping,
                                 max_prompt_tokens: int, max_new_tokens: int) -> dict:
    # Layout is method-major, then direction-major, then prompt-major. Hugging
    # Face greedy generation preserves this row ordering.
    one_encoded, one_strengths = _expanded_batch(
        ctx, prompts, direction_items, max_prompt_tokens,
    )
    rendered_ids = one_encoded["input_ids"].repeat(len(method_names), 1)
    rendered_mask = one_encoded["attention_mask"].repeat(len(method_names), 1)
    encoded = {"input_ids": rendered_ids, "attention_mask": rendered_mask}
    strengths = one_strengths.repeat(len(method_names), 1)
    rows_per_method = len(one_strengths)
    with torch.no_grad(), MultiMethodControllerHooks(
        ctx.layers, [controllers[name] for name in method_names], strengths, rows_per_method,
    ):
        output = ctx.model.llama.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    decoded = [ctx.tokenizer.decode(row[encoded["input_ids"].shape[1]:], skip_special_tokens=True)
               for row in output]
    result = {}
    direction_names = [item[0] for item in direction_items]
    for method_index, method in enumerate(method_names):
        block = decoded[method_index * rows_per_method:(method_index + 1) * rows_per_method]
        result[method] = {
            name: block[index * len(prompts):(index + 1) * len(prompts)]
            for index, name in enumerate(direction_names)
        }
    return result


def _axis_static_vectors(ctx: AnalysisContext, fit_prompts: Sequence[dict], batch_size: int,
                         max_prompt_tokens: int, tokens: int) -> tuple[dict, dict]:
    return fit_static_layer_vectors(
        ctx, fit_prompts, batch_size, max_prompt_tokens, tokens,
    )


def _generate_ours_static(ctx: AnalysisContext, prompts: Sequence[str], direction_items,
                          axis_vectors: Mapping[str, Mapping[int, torch.Tensor]],
                          max_prompt_tokens: int, max_new_tokens: int) -> dict:
    return generate_static_layer_vectors(
        ctx, prompts, direction_items, axis_vectors, max_prompt_tokens, max_new_tokens,
    )


def _generate_ours_jacobian(ctx: AnalysisContext, prompts: Sequence[str], direction_items,
                            max_prompt_tokens: int, max_new_tokens: int) -> dict:
    generations, _norms = fast_layer_jacobian_multi_batch(
        ctx, prompts, direction_items, 1.0, max_prompt_tokens, max_new_tokens,
    )
    return generations


def _cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fixed_token_profile(ctx: AnalysisContext, encoded: Mapping[str, torch.Tensor],
                         intervention, tokens: int) -> dict:
    """Profile fixed-length greedy decoding, separating prefill from decode."""
    if tokens < 1:
        raise ValueError("profile tokens must be positive")
    _cuda_sync(ctx.device)
    if ctx.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(ctx.device)
        start_allocated = torch.cuda.memory_allocated(ctx.device)
    else:
        start_allocated = 0
    wall_start = time.perf_counter()
    decode_seconds = []
    with intervention(), torch.no_grad():
        # Supply text position IDs explicitly.  In particular, Qwen-VL caches
        # ``rope_deltas`` on the model object; after a generation with another
        # batch shape, a hand-written cache loop that omits position_ids can
        # accidentally reuse that stale tensor.  Explicit 1-D text positions
        # are valid for ordinary causal models and Qwen-VL expands them across
        # its three RoPE dimensions.
        attention_mask = encoded["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        prefill_inputs = dict(encoded)
        prefill_inputs["position_ids"] = position_ids
        _cuda_sync(ctx.device)
        start = time.perf_counter()
        outputs = ctx.model.llama(**prefill_inputs, use_cache=True)
        _cuda_sync(ctx.device)
        prefill = time.perf_counter() - start
        token = outputs.logits[:, -1].argmax(-1, keepdim=True)
        cache = outputs.past_key_values
        for _ in range(tokens - 1):
            attention_mask = torch.cat((attention_mask, torch.ones_like(token)), dim=1)
            # The appended token's zero-based text position is the number of
            # active tokens preceding it.  This remains correct for left-padded
            # batches with unequal prompt lengths.
            next_position_ids = attention_mask.long().sum(-1, keepdim=True) - 1
            _cuda_sync(ctx.device)
            start = time.perf_counter()
            outputs = ctx.model.llama(
                input_ids=token, attention_mask=attention_mask,
                position_ids=next_position_ids,
                past_key_values=cache, use_cache=True,
            )
            _cuda_sync(ctx.device)
            decode_seconds.append(time.perf_counter() - start)
            token = outputs.logits[:, -1].argmax(-1, keepdim=True)
            cache = outputs.past_key_values
    _cuda_sync(ctx.device)
    wall = time.perf_counter() - wall_start
    peak = (torch.cuda.max_memory_allocated(ctx.device) if ctx.device.type == "cuda" else 0)
    return {
        "wall_seconds": wall,
        "prefill_seconds": prefill,
        "decode_seconds": sum(decode_seconds),
        "mean_decode_token_seconds": (
            sum(decode_seconds) / len(decode_seconds) if decode_seconds else None
        ),
        "generated_tokens_per_sequence": tokens,
        "throughput_tokens_per_second": encoded["input_ids"].shape[0] * tokens / wall,
        "peak_allocated_bytes": int(peak),
        "incremental_peak_allocated_bytes": int(max(0, peak - start_allocated)),
    }


def _mean_profile(rows: Sequence[Mapping]) -> dict:
    numeric = {key for row in rows for key, value in row.items()
               if isinstance(value, (int, float)) and value is not None}
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in sorted(numeric)}


def _profile_dynamic_methods(ctx: AnalysisContext, prompts: Sequence[dict], controllers,
                             repetitions: int, warmup: int, batch_size: int,
                             max_prompt_tokens: int, tokens: int) -> dict:
    prompt_text = [item["prompt"] for item in prompts[:batch_size]]
    encoded = encode(ctx.tokenizer, prompt_text, ctx.device, max_prompt_tokens)
    direction = [1.0] * len(ctx.axes)
    strengths = torch.ones(len(prompt_text), len(ctx.axes), device=ctx.device, dtype=ctx.dtype)

    def recontrol_context():
        return MultiAxisControllerHooks(ctx.layers, controllers["re_control"], strengths)

    def profile_ours():
        _cuda_sync(ctx.device)
        if ctx.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(ctx.device)
            start_allocated = torch.cuda.memory_allocated(ctx.device)
        else:
            start_allocated = 0
        start = time.perf_counter()
        fast_layer_jacobian_multi_batch(
            ctx, prompt_text, [("all", direction)], 1.0,
            max_prompt_tokens, tokens, fixed_token_work=True,
        )
        _cuda_sync(ctx.device)
        wall = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(ctx.device) if ctx.device.type == "cuda" else 0
        return {
            "wall_seconds": wall,
            "generated_tokens_per_sequence": tokens,
            "throughput_tokens_per_second": len(prompt_text) * tokens / wall,
            "peak_allocated_bytes": int(peak),
            "incremental_peak_allocated_bytes": int(max(0, peak - start_allocated)),
        }

    for _ in range(warmup):
        profile_ours()
        _fixed_token_profile(ctx, encoded, recontrol_context, tokens)
    ours_rows = [profile_ours() for _ in range(repetitions)]
    recontrol_rows = [
        _fixed_token_profile(ctx, encoded, recontrol_context, tokens)
        for _ in range(repetitions)
    ]
    result = {
        "ours_autoregressive_jacobian": {
            "runs": ours_rows,
            "mean": _mean_profile(ours_rows),
            "two_pass_complete_layer_jacobian": True,
        },
        "re_control": {"runs": recontrol_rows, "mean": _mean_profile(recontrol_rows)},
    }
    result["protocol"] = {
        "batch_size": len(prompt_text),
        "fixed_generated_tokens_per_sequence": tokens,
        "warmup_runs": warmup,
        "measured_runs": repetitions,
        "cuda_synchronized": ctx.device.type == "cuda",
        "same_encoded_prompts": True,
        "eos_ignored_for_fixed_work": True,
    }
    return result


def steering_method_comparison(
    ctx: AnalysisContext,
    prompts: Sequence[dict],
    true_baseline: Path,
    methods: Sequence[str] = ALL_METHODS,
    levels: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    fit_prompt_count: int = 80,
    validation_fraction: float = 0.2,
    activation_batch_size: int = 8,
    static_fit_batch_size: int = 4,
    generation_batch_size: int = 8,
    direction_batch_size: int = 8,
    method_batch_size: int = 5,
    max_prompt_tokens: int = 512,
    max_new_tokens: int = 64,
    static_fit_tokens: int = 1,
    profile_repetitions: int = 5,
    profile_warmup: int = 1,
    profile_batch_size: int = 8,
    profile_tokens: int = 32,
    recontrol_hidden: int = 256,
    recontrol_epochs: int = 20,
    recontrol_batch_size: int = 512,
    recontrol_learning_rate: float = 1e-3,
    recontrol_step_size: float = 0.1,
    recontrol_iterations: int = 3,
    odesteer_features: int = 8000,
    odesteer_gamma: float = 0.1,
    odesteer_c0: float = 1.0,
    odesteer_time: float = 14.0,
    odesteer_steps: int = 10,
    odesteer_logistic_c: float = 1.0,
    odesteer_logistic_steps: int = 1000,
    odesteer_feature_batch_size: int = 256,
    odesteer_training_samples: int = 1000,
    odesteer_layer: int | str | None = "caa",
    caa_layers_by_axis: Mapping[str, int] | None = None,
    seed: int = 42,
) -> dict:
    methods = tuple(methods)
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown steering comparison methods: {unknown}")
    if fit_prompt_count < 4 or fit_prompt_count >= len(prompts):
        raise ValueError("fit_prompt_count must leave generation prompts and contain at least four prompts")
    fit_prompts, generation_prompts = list(prompts[:fit_prompt_count]), list(prompts[fit_prompt_count:])
    independent = load_independent_loras(
        true_baseline, ctx.axes, [key for key, _adapter in ctx.adapters], ctx.config.base_model,
        allow_sparse=True,
    )
    classical = [method for method in methods if method in CLASSICAL_METHODS]
    endpoint_classical = [method for method in classical if method != "odesteer"]
    controllers, controller_report = {}, {}
    odesteer_data_report = None
    need_caa_calibration = bool(endpoint_classical) or odesteer_layer == "caa"
    neutral = positive = None
    if need_caa_calibration:
        neutral, positive = _endpoint_activations(
            ctx, independent, fit_prompts, activation_batch_size, max_prompt_tokens,
        )
    if endpoint_classical:
        controllers, controller_report = _fit_controllers(
            ctx, neutral, positive, endpoint_classical, validation_fraction, {
                "re_control": {
                    "hidden": recontrol_hidden,
                    "epochs": recontrol_epochs,
                    "batch_size": recontrol_batch_size,
                    "learning_rate": recontrol_learning_rate,
                    "step_size": recontrol_step_size,
                    "iterations": recontrol_iterations,
                    "seed": seed,
                },
            }, selected_layers=caa_layers_by_axis,
        )
    if "odesteer" in classical:
        if odesteer_layer is None:
            raise ValueError(
                "ODESteer requires --odesteer_layer caa or an explicit layer."
            )
        if odesteer_layer == "caa":
            if caa_layers_by_axis is not None:
                odesteer_layers = {
                    axis: int(caa_layers_by_axis[axis_name])
                    for axis, axis_name in enumerate(ctx.axes)
                }
            else:
                assert neutral is not None and positive is not None
                count = len(next(iter(neutral.values())))
                validation_count = max(1, round(validation_fraction * count))
                training_count = count - validation_count
                odesteer_layers = {
                    axis: max(
                        neutral,
                        key=lambda layer: standardized_mean_separation(
                            positive[axis][layer][:training_count],
                            neutral[layer][:training_count],
                        ),
                    )
                    for axis in range(len(ctx.axes))
                }
            LOG.info(
                "ODESteer reuses calibration-only CAA layers: %s",
                {ctx.axes[axis]: layer for axis, layer in odesteer_layers.items()},
            )
        else:
            odesteer_layers = {axis: int(odesteer_layer) for axis in range(len(ctx.axes))}
        ode_negative, ode_positive, odesteer_data_report = _odesteer_contrastive_activations(
            ctx, odesteer_training_samples, activation_batch_size,
            max_prompt_tokens, seed,
        )
        ode_controllers, ode_report = _fit_odesteer_controllers(
            ctx, ode_negative, ode_positive, validation_fraction, {
                "feature_count": odesteer_features,
                "gamma": odesteer_gamma,
                "c0": odesteer_c0,
                "integration_time": odesteer_time,
                "steps": odesteer_steps,
                "logistic_c": odesteer_logistic_c,
                "logistic_steps": odesteer_logistic_steps,
                "feature_batch_size": odesteer_feature_batch_size,
                "seed": seed,
            }, odesteer_layers,
        )
        controllers["odesteer"] = ode_controllers
        controller_report["odesteer"] = ode_report
    axis_static_vectors, static_metrics = {}, {}
    if "ours_static" in methods:
        LOG.info("Fitting our static all-site vectors on %d prompts", len(fit_prompts))
        axis_static_vectors, static_metrics = _axis_static_vectors(
            ctx, fit_prompts, static_fit_batch_size, max_prompt_tokens, static_fit_tokens,
        )
    direction_items = _directions(ctx.axes, levels)
    generations = {name: {"direction": direction, "examples": []}
                   for name, direction in direction_items}
    generation_batches = list(chunked(generation_prompts, generation_batch_size))
    for prompt_batch_index, prompt_batch in enumerate(generation_batches):
        prompt_text = [item["prompt"] for item in prompt_batch]
        for direction_batch_index, directions in enumerate(chunked(direction_items, direction_batch_size)):
            LOG.info("Steering comparison prompt batch %d/%d, direction batch %d/%d",
                     prompt_batch_index + 1, len(generation_batches), direction_batch_index + 1,
                     math.ceil(len(direction_items) / direction_batch_size))
            outputs = {}
            for method_batch in chunked(classical, method_batch_size):
                outputs.update(_generate_controller_methods(
                    ctx, prompt_text, directions, method_batch, controllers,
                    max_prompt_tokens, max_new_tokens,
                ))
            if "ours_static" in methods:
                outputs["ours_static"] = _generate_ours_static(
                    ctx, prompt_text, directions, axis_static_vectors,
                    max_prompt_tokens, max_new_tokens,
                )
            if "ours_autoregressive_jacobian" in methods:
                outputs["ours_autoregressive_jacobian"] = _generate_ours_jacobian(
                    ctx, prompt_text, directions, max_prompt_tokens, max_new_tokens,
                )
            for direction_name, _direction in directions:
                offset = len(generations[direction_name]["examples"])
                for index, item in enumerate(prompt_batch):
                    generations[direction_name]["examples"].append({
                        "prompt_index": offset + index,
                        "source_index": item.get("source_index"),
                        "prompt": item["prompt"],
                        "generations": {method: outputs[method][direction_name][index]
                                        for method in methods},
                        "judge": {"per_axis_satisfied": None, "overall_quality": None,
                                  "best_method": None, "notes": None},
                    })
    profiling = None
    if {"re_control", "ours_autoregressive_jacobian"}.issubset(methods):
        try:
            profiling = _profile_dynamic_methods(
                ctx, generation_prompts, controllers, profile_repetitions, profile_warmup,
                profile_batch_size, max_prompt_tokens, profile_tokens,
            )
            profiling["status"] = "completed"
        except (RuntimeError, ValueError) as exc:
            # Profiling is an auxiliary measurement performed after every
            # expensive generation has completed.  Preserve those results if
            # an architecture-specific cache path still cannot be profiled.
            LOG.exception("Dynamic-method profiling failed; preserving completed generations")
            profiling = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return {
        "axes": ctx.axes,
        "methods": list(methods),
        "levels": list(levels),
        "combination_count": len(direction_items),
        "true_baseline": str(independent.root),
        "caa_layers_by_axis": (
            {axis: int(layer) for axis, layer in caa_layers_by_axis.items()}
            if caa_layers_by_axis is not None else None
        ),
        "split": {
            "controller_and_static_fit_prompts": len(fit_prompts),
            "controller_activation_batch_size": activation_batch_size,
            "static_jacobian_batch_size": static_fit_batch_size,
            "controller_internal_validation_fraction": validation_fraction,
            "odesteer_contrastive_training_samples_per_axis": (
                odesteer_training_samples if "odesteer" in methods else None
            ),
            "odesteer_layer_protocol": odesteer_layer if "odesteer" in methods else None,
            "odesteer_layers_by_axis": (
                {ctx.axes[axis]: layer for axis, layer in odesteer_layers.items()}
                if "odesteer" in methods else None
            ),
            "held_out_generation_prompts": len(generation_prompts),
        },
        "method_definitions": {
            "caa": "Per-axis residual mean difference; independent endpoint fields compose additively.",
            "repe": "Per-axis PC1 of paired independent endpoint differences, oriented and scaled by the training mean displacement.",
            "mimic": "Per-axis affine Gaussian optimal-transport map gated to activations classified as negative; maps compose additively.",
            "re_control": "Per-axis value MLP fitted to independent endpoint labels; its activation gradient is recomputed at every autoregressive forward.",
            "linear_act": "Per-axis diagonal linear activation-transport map fitted from independent endpoints; maps compose additively.",
            "odesteer": "Per-axis nonlinear log-density-ratio barrier fitted with degree-two Polynomial Count Sketch features; ten normalized-gradient Euler steps steer the final token at the selected residual layer.",
            "ours_static": "Context-independent residual vectors fitted at every decoder-layer output to the complete-layer source Jacobian; axis vectors compose linearly.",
            "ours_autoregressive_jacobian": "Exact adapter tangents propagated through every complete decoder block; token-dependent source vectors are injected at every layer output using two cached passes per token.",
        },
        "fairness_notes": [
            "All methods use the same prompts, steering grid, greedy decoding, and token budget.",
            "Classical baselines are fitted only from standalone independent-LoRA endpoint trajectories and do not use anchored-merger activations.",
            "Endpoint-derived classical baselines select a residual layer using training-split standardized mean separation; no generation prompt participates in selection.",
            "RE-Control uses independent endpoint labels as the value target. This preserves its learned-gradient controller but is a proxy-reward adaptation, not the original paper's external human-preference reward model.",
            "ODESteer reuses the calibration-only CAA-optimal residual layer for each axis unless an explicit shared layer is requested. Completed contrastive conversations are encoded without an appended generation header, and only the current final token is steered. Multi-axis conditions add independently solved per-axis ODE displacements from the same incoming activation; the original paper evaluates one behavioral contrast at a time.",
            "Ours operates at every decoder-layer residual output, while each published classical baseline selects one residual layer per axis; placement remains an explicit part of the comparison.",
        ],
        "controller_fit": controller_report,
        "odesteer_contrastive_data": odesteer_data_report,
        "ours_static_fit": static_metrics,
        "profiling": profiling,
        "comparisons": generations,
    }
