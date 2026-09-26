"""Constrain HeRD merging directions with a joint LM/Gauss--Newton fit.

The independent LoRAs and the previously learned low-rank interaction
directions remain fixed.  Four deployable subset gains (for three axes) are
fit against a residual vector containing distributional composition,
paired-dataset marginal behavior, exact local signal curvature, prompt-wise
variation of the local steering field, and normalized interaction magnitude.

All fitting examples come from the configured axis datasets.  A configurable
evaluation prompt set is loaded only to exclude overlap; it is never used as
fitting or validation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, set_seed

from .common import ROOT, encode, read_prompts
from .herd_merging import (
    TeacherBatch,
    _axis_order,
    _kl_per_example,
    _multi_axis_corners,
    cache_teacher_batches,
    compositional_teacher_log_probs,
)
from .independent import (
    IndependentLoRAHooks,
    MultiAffineInteractionHooks,
    MultiAffineInteractions,
    load_independent_loras,
)

# common.py places steered_finetuner on sys.path.
from config import TrainingConfig  # noqa: E402
from dataset import _json, _map_axis_rows, collate_fn  # noqa: E402
from model import _decoder_layers, build_model  # noqa: E402


LOG = logging.getLogger("constrained_interaction_gains")


def per_example_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy for each sequence, ignoring masked labels."""
    batch_size, sequence_length, vocabulary_size = logits.shape
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].to(logits.device).contiguous()
    token_losses = F.cross_entropy(
        shifted_logits.view(-1, vocabulary_size),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(batch_size, sequence_length - 1)
    mask = (shifted_labels != -100).to(token_losses.dtype)
    return (token_losses * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


@dataclass
class PairExample:
    axis: int
    prompt: str
    encoded: dict[str, torch.Tensor]


def _prompt_digest(values: Sequence[str]) -> str:
    return hashlib.sha256("\n\0\n".join(values).encode()).hexdigest()


def _normalize_prompt(value: str) -> str:
    return " ".join(value.strip().split())


def _row_prompt(row: Mapping) -> str | None:
    messages = _json(row.get("original_messages"))
    if not isinstance(messages, list):
        return None
    # Use the last user request preceding the first assistant response.  This
    # matches the single-prompt generation evaluation used by the analysis
    # suite while retaining a deterministic overlap key.
    latest = None
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "assistant":
            break
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            latest = message["content"]
    return latest.strip() if latest and latest.strip() else None


def _infer_axis_datasets(root: Path, axes: Sequence[str], overrides: Sequence[str]) -> dict[str, str]:
    result = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError("--axis_dataset must be AXIS=DATASET")
        axis, dataset = item.split("=", 1)
        result[axis.strip()] = dataset.strip()
    for axis in axes:
        if axis in result:
            continue
        path = root / axis / "training_config.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot infer the dataset for {axis!r}: missing {path}; use --axis_dataset"
            )
        config = json.loads(path.read_text())
        dataset = config.get("trained_dataset")
        if not dataset:
            dataset = next(
                (item.get("dataset") for item in config.get("axis_datasets", [])
                 if item.get("axis") == axis),
                None,
            )
        if not dataset:
            raise ValueError(f"Cannot infer the dataset for axis {axis!r}")
        result[axis] = str(dataset)
    unexpected = sorted(set(result) - set(axes))
    if unexpected:
        raise ValueError(f"Dataset overrides contain unknown axes: {unexpected}")
    return result


def _mapped_examples(row: Mapping, tokenizer, max_seq_len: int, axis: int) -> list[PairExample]:
    columns = set(row)
    batch = {key: [row.get(key)] for key in columns}
    # The mapper requires original_messages/messages and handles both status
    # conventions itself.
    output = _map_axis_rows(
        batch, tokenizer=tokenizer, max_seq_len=max_seq_len, axis_index=axis,
    )
    prompt = _row_prompt(row)
    if prompt is None:
        return []
    examples = []
    for index in range(len(output["dim_idx"])):
        encoded = {
            key: torch.tensor(output[key][index], dtype=torch.long)
            for key in (
                "pos_input_ids", "pos_attention_mask", "pos_labels",
                "neg_input_ids", "neg_attention_mask", "neg_labels",
            )
        }
        encoded["dim_idx"] = torch.tensor(output["dim_idx"][index], dtype=torch.long)
        examples.append(PairExample(axis, prompt, encoded))
    return examples


def load_pair_splits(root: Path, axes: Sequence[str], datasets: Mapping[str, str], tokenizer,
                     fit_per_axis: int, validation_per_axis: int, max_seq_len: int,
                     heldout_prompts: set[str], split: str, seed: int,
                     hf_token: str | None) -> tuple[list[PairExample], list[PairExample], dict]:
    from datasets import load_dataset

    fit, validation, metadata = [], [], {}
    required = fit_per_axis + validation_per_axis
    for axis_index, axis in enumerate(axes):
        kwargs = {"token": hf_token} if hf_token else {}
        raw = load_dataset(datasets[axis], split=split, **kwargs)
        missing = {"original_messages", "messages"} - set(raw.column_names)
        if missing:
            raise ValueError(f"{datasets[axis]} is missing columns {sorted(missing)}")
        raw = raw.shuffle(seed=seed + axis_index)
        accepted, excluded = [], 0
        for row in raw:
            prompt = _row_prompt(row)
            if prompt is None:
                continue
            if _normalize_prompt(prompt) in heldout_prompts:
                excluded += 1
                continue
            accepted.extend(_mapped_examples(row, tokenizer, max_seq_len, axis_index))
            if len(accepted) >= required:
                break
        if len(accepted) < required:
            raise ValueError(
                f"Axis {axis!r} produced {len(accepted)} valid non-overlapping pairs; "
                f"need {required}"
            )
        validation.extend(accepted[:validation_per_axis])
        fit.extend(accepted[validation_per_axis:required])
        metadata[axis] = {
            "dataset": datasets[axis],
            "fit_pairs": fit_per_axis,
            "validation_pairs": validation_per_axis,
            "excluded_evaluation_prompt_matches": excluded,
        }
        LOG.info(
            "Axis %s: %d fit pairs, %d validation pairs, %d held-out overlaps excluded",
            axis, fit_per_axis, validation_per_axis, excluded,
        )
    random.Random(seed + 101).shuffle(fit)
    random.Random(seed + 103).shuffle(validation)
    return fit, validation, metadata


def _pair_batch(examples: Sequence[PairExample], tokenizer) -> tuple[dict, list[str]]:
    values = [item.encoded for item in examples]
    batch = collate_fn(values, pad_token_id=tokenizer.pad_token_id or 0)
    return batch, [item.prompt for item in examples]


def _repeat_rows(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))


def _reverse_directional_derivatives(function, primal: torch.Tensor,
                                     tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact J[d] and H[d,d] using reverse-over-reverse autodiff.

    For vector-valued y(s), introduce an auxiliary cotangent c.  Since
    <c, Jd> is scalar, differentiating it with respect to c recovers Jd.
    Differentiating once more with respect to s along d and then with respect
    to c recovers the vector-valued H[d,d].  This avoids forward-AD kernel
    requirements while retaining exact derivatives.
    """
    signal = primal.detach().requires_grad_(True)
    output = function(signal)
    cotangent = torch.zeros_like(output, requires_grad=True)
    vjp = torch.autograd.grad(
        output, signal, grad_outputs=cotangent,
        create_graph=True, retain_graph=True,
    )[0]
    first_pairing = (vjp * tangent).sum()
    first = torch.autograd.grad(
        first_pairing, cotangent, create_graph=False, retain_graph=True,
    )[0]
    directional_gradient = torch.autograd.grad(
        first_pairing, signal, create_graph=True, retain_graph=True,
    )[0]
    second_pairing = (directional_gradient * tangent).sum()
    second = torch.autograd.grad(
        second_pairing, cotangent, create_graph=False, retain_graph=False,
    )[0]
    return first.detach(), second.detach()


class ResidualEvaluator:
    def __init__(self, model, tokenizer, baseline, interactions, device: torch.device,
                 max_prompt_tokens: int, interior_margin: float,
                 weights: Mapping[str, float], interaction_ratios: torch.Tensor,
                 curvature_layers: Sequence[int] | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.baseline = baseline
        self.interactions = interactions
        self.device = device
        self.max_prompt_tokens = max_prompt_tokens
        self.interior_margin = interior_margin
        self.weights = dict(weights)
        self.interaction_ratios = interaction_ratios.to(device)
        self.corners = _multi_axis_corners(len(baseline.axes), device)
        self.layers = list(_decoder_layers(model.llama))
        self.curvature_layers = (
            list(range(len(self.layers)))
            if curvature_layers is None else sorted(set(curvature_layers))
        )
        invalid = [
            layer for layer in self.curvature_layers
            if layer < 0 or layer >= len(self.layers)
        ]
        if invalid:
            raise ValueError(
                f"Curvature layers {invalid} are outside [0, {len(self.layers) - 1}]"
            )
        if not self.curvature_layers:
            raise ValueError("At least one curvature layer is required")
        self.layer_keys = {
            layer: [
                key for key in baseline.keys
                if key.startswith(f"layer{layer}.")
            ]
            for layer in self.curvature_layers
        }
        missing = [layer for layer, keys in self.layer_keys.items() if not keys]
        if missing:
            raise ValueError(f"Selected curvature layers have no adapted projections: {missing}")

    @staticmethod
    def _scaled(residual: torch.Tensor, weight: float) -> torch.Tensor:
        flat = residual.float().flatten(1)
        return math.sqrt(weight) * flat / math.sqrt(max(1, flat.shape[1]))

    def _forward(self, ids, mask, signals, gains, *, labels=None,
                 output_hidden_states: bool = False):
        self.model._ctx["active"] = False
        with IndependentLoRAHooks(self.model, self.baseline, signals), \
                MultiAffineInteractionHooks(
                    self.model, self.interactions, signals, subset_gains=gains,
                ):
            return self.model.llama(
                input_ids=ids, attention_mask=mask, labels=labels,
                output_hidden_states=output_hidden_states, use_cache=False,
            )

    @torch.inference_mode()
    def _kl(self, candidates: torch.Tensor, teacher_batch: TeacherBatch) -> torch.Tensor:
        M, B, C = candidates.shape[0], teacher_batch.size, self.corners.shape[0]
        positions = int(teacher_batch.log_probs.shape[1])
        ids = teacher_batch.input_ids.to(self.device).repeat_interleave(C, 0)
        mask = teacher_batch.attention_mask.to(self.device).repeat_interleave(C, 0)
        signals = self.corners.repeat(B, 1)
        ids, mask, signals = _repeat_rows(ids, M), _repeat_rows(mask, M), _repeat_rows(signals, M)
        gains = candidates.repeat_interleave(B * C, 0)
        output = self._forward(ids, mask, signals, gains)
        student = F.log_softmax(output.logits[:, -positions:].float(), -1).reshape(
            M, B, C, positions, -1,
        )
        target = compositional_teacher_log_probs(
            teacher_batch.log_probs.to(self.device), self.corners,
        ).unsqueeze(0)
        kl = _kl_per_example(target, student).clamp_min(0)
        # 1/2 ||sqrt(2 KL)||^2 equals the mean KL after group normalization.
        return torch.sqrt(2.0 * kl + 1e-12)

    @torch.inference_mode()
    def _marginal(self, candidates: torch.Tensor, pair_batch: dict) -> torch.Tensor:
        M = candidates.shape[0]
        ids_pos = pair_batch["pos_input_ids"].to(self.device)
        ids_neg = pair_batch["neg_input_ids"].to(self.device)
        mask_pos = pair_batch["pos_attention_mask"].to(self.device)
        mask_neg = pair_batch["neg_attention_mask"].to(self.device)
        labels_pos = pair_batch["pos_labels"].to(self.device)
        labels_neg = pair_batch["neg_labels"].to(self.device)
        axes = pair_batch["dim_idx"].to(self.device)
        B, K = axes.shape[0], len(self.baseline.axes)
        length = max(ids_pos.shape[1], ids_neg.shape[1])

        def pad(value, amount, fill):
            return F.pad(value, (0, amount), value=fill) if amount else value

        ids = torch.cat([
            pad(ids_pos, length - ids_pos.shape[1], self.tokenizer.pad_token_id or 0),
            pad(ids_neg, length - ids_neg.shape[1], self.tokenizer.pad_token_id or 0),
        ])
        mask = torch.cat([
            pad(mask_pos, length - mask_pos.shape[1], 0),
            pad(mask_neg, length - mask_neg.shape[1], 0),
        ])
        labels = torch.cat([
            pad(labels_pos, length - labels_pos.shape[1], -100),
            pad(labels_neg, length - labels_neg.shape[1], -100),
        ])
        pure_zero = torch.zeros(B, K, device=self.device)
        pure_one = pure_zero.clone()
        pure_one[torch.arange(B, device=self.device), axes] = 1
        background_zero = torch.ones(B, K, device=self.device)
        background_zero[torch.arange(B, device=self.device), axes] = 0
        background_one = torch.ones(B, K, device=self.device)
        conditions = [pure_zero, pure_one, background_zero, background_one]
        condition_signals = torch.cat([
            torch.cat([condition, condition], 0) for condition in conditions
        ])
        ids = torch.cat([ids] * 4)
        mask = torch.cat([mask] * 4)
        labels = torch.cat([labels] * 4)
        ids, mask, labels = _repeat_rows(ids, M), _repeat_rows(mask, M), _repeat_rows(labels, M)
        signals = _repeat_rows(condition_signals, M)
        gains = candidates.repeat_interleave(8 * B, 0)
        output = self._forward(ids, mask, signals, gains)
        ce = per_example_ce(output.logits, labels).reshape(M, 4, 2, B)
        contrast = ce[:, :, 0] - ce[:, :, 1]
        amplitude = contrast[:, 1] - contrast[:, 0]
        marginal = contrast[:, 3] - contrast[:, 2]
        scale = amplitude.detach().abs().clamp_min(0.25)
        return (marginal - amplitude.detach()) / scale

    def _curvature(self, candidates: torch.Tensor, prompts: Sequence[str], seed: int):
        """Exact local J_s and J_s J_s residuals at random interior points.

        For layer ell, every preceding layer runs at a fixed sampled signal s.
        Only layer ell sees the differentiable signal argument.  Consequently
        its input h_{ell-1}(x, s) has zero signal tangent, and the nested JVPs
        are derivatives of the single decoder-block map F_ell with that input
        held fixed.  Prompt variation of the first JVP is the directly relevant
        condition for compressing the source field to a static residual vector.
        """
        M, B, K = candidates.shape[0], len(prompts), len(self.baseline.axes)
        if B < 2 and self.weights.get("static", 0) > 0:
            raise ValueError(
                "The static-field residual requires at least two curvature prompts"
            )
        encoded = encode(self.tokenizer, prompts, self.device, self.max_prompt_tokens)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        span = 1.0 - 2.0 * self.interior_margin
        point = self.interior_margin + span * torch.rand((1, K), generator=generator)
        point = point.to(self.device).expand(B, -1)
        direction = torch.randint(0, 2, (1, K), generator=generator).float().mul_(2).sub_(1)
        direction = (direction / math.sqrt(K)).to(self.device).expand(B, -1)

        ids = _repeat_rows(encoded["input_ids"], M)
        mask = _repeat_rows(encoded["attention_mask"], M)
        signals = _repeat_rows(point, M)
        tangent = _repeat_rows(direction, M)
        gains = candidates.repeat_interleave(B, 0)
        all_keys = set(self.baseline.keys)
        ss_values, static_values = [], []
        for layer_index in self.curvature_layers:
            local_keys = self.layer_keys[layer_index]
            fixed_keys = sorted(all_keys - set(local_keys))

            def layer_output(local_signal: torch.Tensor) -> torch.Tensor:
                captured = {}

                def stop_layer_gradient(_module, _inputs, output):
                    value = output[0] if isinstance(output, tuple) else output
                    captured["value"] = value
                    # Later layers are irrelevant to this local derivative.
                    # Detaching the value passed onward prevents constructing
                    # a needless suffix graph while preserving normal forward
                    # execution and architecture-specific bookkeeping.
                    detached = value.detach()
                    return ((detached, *output[1:])
                            if isinstance(output, tuple) else detached)

                capture_handle = self.layers[layer_index].register_forward_hook(
                    stop_layer_gradient,
                )
                self.model._ctx["active"] = False
                try:
                    with IndependentLoRAHooks(
                        self.model, self.baseline, signals, keys=fixed_keys,
                    ), MultiAffineInteractionHooks(
                        self.model, self.interactions, signals,
                        subset_gains=gains, keys=fixed_keys,
                    ), IndependentLoRAHooks(
                        self.model, self.baseline, local_signal, keys=local_keys,
                    ), MultiAffineInteractionHooks(
                        self.model, self.interactions, local_signal,
                        subset_gains=gains, keys=local_keys,
                    ):
                        self.model.llama(
                            input_ids=ids, attention_mask=mask,
                            output_hidden_states=False, use_cache=False,
                        )
                finally:
                    capture_handle.remove()
                if "value" not in captured:
                    raise RuntimeError(f"Failed to capture decoder layer {layer_index}")
                return captured["value"][:, -1]

            with torch.enable_grad():
                first, second = _reverse_directional_derivatives(
                    layer_output, signals, tangent,
                )
            first = first.float().reshape(M, B, -1)
            second = second.float().reshape(M, B, -1)
            first_norm = first.norm(dim=-1).clamp_min(1e-6)
            ss_values.append(second.norm(dim=-1) / first_norm)

            mean = first.mean(dim=1, keepdim=True)
            rms = first.square().sum(dim=-1).mean(dim=1).sqrt().clamp_min(1e-6)
            static_values.append((first - mean).norm(dim=-1) / rms.unsqueeze(1))
        return torch.stack(ss_values, 1), torch.stack(static_values, 1)

    def __call__(self, candidates: torch.Tensor, teacher_batch: TeacherBatch,
                 pair_batch: dict, curvature_prompts: Sequence[str], seed: int):
        candidates = candidates.to(self.device, dtype=torch.float32)
        groups = {}
        if self.weights["kl"] > 0:
            groups["kl"] = self._kl(candidates, teacher_batch)
        if self.weights["marginal"] > 0:
            groups["marginal"] = self._marginal(candidates, pair_batch)
        if self.weights["ss"] > 0 or self.weights["static"] > 0:
            ss, static = self._curvature(candidates, curvature_prompts, seed)
            if self.weights["ss"] > 0:
                groups["ss"] = ss
            if self.weights["static"] > 0:
                groups["static"] = static
        if self.weights["norm"] > 0:
            groups["norm"] = candidates * self.interaction_ratios.unsqueeze(0)
        if not groups:
            raise ValueError("At least one constrained-fit residual weight must be positive")
        residuals = [self._scaled(groups[name], self.weights[name]) for name in groups]
        combined = torch.cat(residuals, 1)
        diagnostics = {
            name: group.float().flatten(1).square().mean(1).sqrt().cpu()
            for name, group in groups.items()
        }
        return combined.cpu(), diagnostics


def _interaction_ratios(baseline, interactions) -> torch.Tensor:
    values = []
    for subset in interactions.subsets:
        interaction_sq = task_sq = 0.0
        for key in baseline.keys:
            branch = interactions.branch(subset, key)
            dense = interactions.scale * branch.B.detach().float() @ branch.A.detach().float()
            interaction_sq += float(dense.square().sum())
            task = sum(
                baseline.factors[baseline.axes[index]][key].dense()
                for index in subset
            )
            task_sq += float(task.float().square().sum())
        values.append(math.sqrt(interaction_sq / max(task_sq, 1e-20)))
    return torch.tensor(values, dtype=torch.float32)


def fit_gains(evaluator: ResidualEvaluator, fit_teacher: Sequence[TeacherBatch],
              fit_pairs: Sequence[PairExample], tokenizer, steps: int,
              pair_batch_size: int, curvature_prompts: int,
              finite_difference: float, damping: float, trust_radius: float,
              max_gain: float, seed: int) -> tuple[torch.Tensor, list[dict]]:
    subset_count = len(evaluator.interactions.subsets)
    gains = torch.zeros(subset_count, dtype=torch.float64)
    history = []
    pair_batches = [
        fit_pairs[start:start + pair_batch_size]
        for start in range(0, len(fit_pairs), pair_batch_size)
    ]
    current_damping, current_radius = damping, trust_radius
    for step in range(steps):
        teacher = fit_teacher[step % len(fit_teacher)]
        examples = pair_batches[step % len(pair_batches)]
        paired, prompts = _pair_batch(examples, tokenizer)
        prompts = prompts[:curvature_prompts]
        candidates = [gains]
        for index in range(subset_count):
            offset = torch.zeros_like(gains)
            offset[index] = finite_difference
            candidates.extend([gains + offset, gains - offset])
        candidate_tensor = torch.stack(candidates).float()
        residual, diagnostics = evaluator(
            candidate_tensor, teacher, paired, prompts, seed + 1009 * step,
        )
        base = residual[0].double()
        jacobian = torch.stack([
            (residual[1 + 2 * index] - residual[2 + 2 * index]).double()
            / (2 * finite_difference)
            for index in range(subset_count)
        ], 1)
        normal = jacobian.T @ jacobian + current_damping * torch.eye(subset_count)
        gradient = jacobian.T @ base
        try:
            delta = torch.linalg.solve(normal, -gradient)
        except torch.linalg.LinAlgError:
            delta = torch.linalg.lstsq(normal, -gradient).solution
        delta_norm = float(delta.norm())
        if delta_norm > current_radius:
            delta *= current_radius / delta_norm
        trial = (gains + delta).clamp(0.0, max_gain)
        trial_residual, trial_diagnostics = evaluator(
            trial.float().unsqueeze(0), teacher, paired, prompts, seed + 1009 * step,
        )
        base_loss = 0.5 * float(base.square().sum())
        trial_loss = 0.5 * float(trial_residual[0].double().square().sum())
        accepted = trial_loss < base_loss
        if accepted:
            gains = trial
            current_damping = max(current_damping / 2.0, 1e-6)
            current_radius = min(current_radius * 1.5, math.sqrt(subset_count))
        else:
            current_damping = min(current_damping * 10.0, 1e6)
            current_radius = max(current_radius / 2.0, 1e-3)
        record = {
            "step": step + 1,
            "accepted": accepted,
            "base_objective": base_loss,
            "trial_objective": trial_loss,
            "damping": current_damping,
            "trust_radius": current_radius,
            "delta_norm": delta_norm,
            "gains": gains.tolist(),
            "base_residual_rms": {name: float(value[0]) for name, value in diagnostics.items()},
            "trial_residual_rms": {
                name: float(value[0]) for name, value in trial_diagnostics.items()
            },
        }
        history.append(record)
        LOG.info(
            "Constrained fit %d/%d: accepted=%s objective %.6f -> %.6f gains=%s",
            step + 1, steps, accepted, base_loss, trial_loss,
            ",".join(f"{value:.4f}" for value in gains),
        )
    return gains.float(), history


def evaluate_split(evaluator: ResidualEvaluator, gains: torch.Tensor,
                   batches: Sequence[TeacherBatch], pairs: Sequence[PairExample], tokenizer,
                   pair_batch_size: int, curvature_prompts: int, seed: int) -> dict:
    totals, count = {}, 0
    pair_batches = [
        pairs[start:start + pair_batch_size]
        for start in range(0, len(pairs), pair_batch_size)
    ]
    for index, teacher in enumerate(batches):
        examples = pair_batches[index % len(pair_batches)]
        paired, prompts = _pair_batch(examples, tokenizer)
        _residual, diagnostics = evaluator(
            gains.unsqueeze(0), teacher, paired, prompts[:curvature_prompts], seed + index,
        )
        for name, value in diagnostics.items():
            totals[name] = totals.get(name, 0.0) + float(value[0])
        count += 1
    return {name: value / count for name, value in totals.items()}


def _layer_indices(value: str) -> list[int]:
    try:
        layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Curvature layers must be comma-separated integers"
        ) from error
    if not layers or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError(
            "Curvature layers must be nonnegative comma-separated integers"
        )
    return layers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent_root", type=Path, required=True)
    parser.add_argument("--interaction_dir", type=Path,
                        help="Defaults to INDEPENDENT_ROOT/compositional-interactions.")
    parser.add_argument("--output_dir", type=Path,
                        help="Defaults to INDEPENDENT_ROOT/compositional-interactions-constrained.")
    parser.add_argument("--axes")
    parser.add_argument("--axis_dataset", action="append", default=[], metavar="AXIS=DATASET")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--fit_pairs_per_axis", type=int, default=100)
    parser.add_argument("--validation_pairs_per_axis", type=int, default=20)
    parser.add_argument("--pair_batch_size", type=int, default=2)
    parser.add_argument("--teacher_batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_prompt_tokens", type=int, default=128)
    parser.add_argument("--prefix_contexts", type=int, default=2)
    parser.add_argument("--evaluation_prompts_file", type=Path,
                        default=ROOT / "datasets" / "test_prompts.jsonl")
    parser.add_argument("--heldout_evaluation_prompts", type=int, default=100)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--finite_difference", type=float, default=0.02)
    parser.add_argument("--lm_damping", type=float, default=0.1)
    parser.add_argument("--trust_radius", type=float, default=0.5)
    parser.add_argument("--max_gain", type=float, default=1.0)
    parser.add_argument("--interior_margin", type=float, default=0.1,
                        help="Sample local-JVP base points uniformly from [margin,1-margin]^J.")
    parser.add_argument("--signal_delta", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--input_radius", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--curvature_prompts", type=int, default=4)
    parser.add_argument(
        "--curvature_layers", type=_layer_indices,
        help=("Zero-indexed comma-separated decoder layers used by both exact local "
              "signal-curvature and prompt-variation residuals. Defaults to all layers."),
    )
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--marginal_weight", type=float, default=1.0)
    parser.add_argument("--ss_weight", type=float, default=0.1)
    parser.add_argument("--static_weight", type=float,
                        help="Weight on prompt variation of the exact local steering JVP.")
    parser.add_argument("--xs_weight", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--interaction_norm_weight", type=float, default=0.05)
    parser.add_argument(
        "--kl_improvement_retention", type=float, default=0.8,
        help=("Select the minimum-interaction-norm held-out gain candidate that "
              "retains at least this fraction of the best KL improvement over "
              "zero interactions."),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if min(args.fit_pairs_per_axis, args.validation_pairs_per_axis, args.steps,
           args.pair_batch_size, args.teacher_batch_size, args.curvature_prompts) < 1:
        raise ValueError("Pair counts, steps, and batch sizes must be positive")
    if not 0 <= args.interior_margin < 0.5:
        raise ValueError("--interior_margin must be in [0, 0.5)")
    if not 0 <= args.kl_improvement_retention <= 1:
        raise ValueError("--kl_improvement_retention must be in [0, 1]")
    if args.static_weight is not None and args.xs_weight is not None:
        raise ValueError("Use --static_weight; do not also pass the deprecated --xs_weight")
    static_weight = (
        args.static_weight if args.static_weight is not None
        else args.xs_weight if args.xs_weight is not None
        else 0.1
    )
    if args.xs_weight is not None:
        LOG.warning("--xs_weight is deprecated; interpreting it as --static_weight")
    if static_weight > 0 and min(args.pair_batch_size, args.curvature_prompts) < 2:
        raise ValueError(
            "The prompt-variation residual requires --pair_batch_size and "
            "--curvature_prompts to be at least 2"
        )
    residual_weights = {
        "kl": args.kl_weight,
        "marginal": args.marginal_weight,
        "ss": args.ss_weight,
        "static": static_weight,
        "norm": args.interaction_norm_weight,
    }
    if any(weight < 0 for weight in residual_weights.values()):
        raise ValueError("Constrained-fit residual weights must be nonnegative")
    if not any(weight > 0 for weight in residual_weights.values()):
        raise ValueError("At least one constrained-fit residual weight must be positive")

    root = args.independent_root.resolve()
    source = (args.interaction_dir or root / "compositional-interactions").resolve()
    output = (args.output_dir or root / "compositional-interactions-constrained").resolve()
    if source == output:
        raise ValueError("Use a distinct output directory so the source directions remain recoverable")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite")
        shutil.rmtree(output)

    axes = _axis_order(root, args.axes)
    baseline = load_independent_loras(root, axes)
    interactions = MultiAffineInteractions.load_artifact(source, baseline)
    interactions.subset_gains.fill_(1.0)
    datasets = _infer_axis_datasets(root, axes, args.axis_dataset)
    device = torch.device(args.device)
    set_seed(args.seed)

    cfg = TrainingConfig(
        base_model=baseline.base_model, signal_dim=len(axes),
        adapter_architecture="anchored_neural_merger", lora_rank=1,
        adapter_hidden_dim=1, freeze_base_model=True, bf16=False,
    )
    model = build_model(cfg, device=args.device, attn_implementation="eager")
    LOG.info("Constrained gain fit uses FP32 for stable exact second derivatives")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fitted_keys = baseline.validate_uniform_subset(sorted(model._adapted_linears))
    LOG.info("Constrained gain fit uses %d shared adapter projections: %s", len(fitted_keys), fitted_keys)
    interactions.to(device=device, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(baseline.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    heldout_records = read_prompts(
        args.evaluation_prompts_file, args.heldout_evaluation_prompts, args.seed,
    )
    heldout = {_normalize_prompt(item["prompt"]) for item in heldout_records}
    fit_pairs, validation_pairs, dataset_metadata = load_pair_splits(
        root, axes, datasets, tokenizer, args.fit_pairs_per_axis,
        args.validation_pairs_per_axis, args.max_seq_len, heldout,
        args.dataset_split, args.seed, args.hf_token or os.environ.get("HF_TOKEN"),
    )

    # Dataset prompts, rather than test prompts, supply all teacher contexts.
    fit_contexts = list(dict.fromkeys(item.prompt for item in fit_pairs))
    validation_contexts = list(dict.fromkeys(item.prompt for item in validation_pairs))
    fit_teacher = cache_teacher_batches(
        model, tokenizer, baseline, fit_contexts, device, args.teacher_batch_size,
        args.max_prompt_tokens, args.prefix_contexts, "dataset-fit",
    )
    validation_teacher = cache_teacher_batches(
        model, tokenizer, baseline, validation_contexts, device, args.teacher_batch_size,
        args.max_prompt_tokens, args.prefix_contexts, "dataset-validation",
    )
    ratios = _interaction_ratios(baseline, interactions)
    weights = residual_weights
    requested_curvature_layers = args.curvature_layers
    LOG.info(
        "Active residuals: %s; curvature layers: %s",
        [name for name, weight in weights.items() if weight > 0],
        requested_curvature_layers if requested_curvature_layers is not None else "all",
    )
    evaluator = ResidualEvaluator(
        model, tokenizer, baseline, interactions, device, args.max_prompt_tokens,
        args.interior_margin, weights, ratios,
        curvature_layers=requested_curvature_layers,
    )
    gains, history = fit_gains(
        evaluator, fit_teacher, fit_pairs, tokenizer, args.steps,
        args.pair_batch_size, args.curvature_prompts, args.finite_difference,
        args.lm_damping, args.trust_radius, args.max_gain, args.seed,
    )
    # LM steps use changing minibatches.  Select the deployable gains on one
    # fixed held-out split, including the exact task-arithmetic point eta=0.
    # This prevents the last stochastic accepted step from being mistaken for
    # the best generalizing gain vector.
    candidate_gains = [torch.zeros_like(gains)]
    candidate_gains.extend(
        torch.tensor(item["gains"], dtype=gains.dtype)
        for item in history if item["accepted"]
    )
    candidate_gains.append(gains)
    unique_candidates = []
    seen_candidates = set()
    for candidate in candidate_gains:
        key = tuple(round(float(value), 10) for value in candidate)
        if key not in seen_candidates:
            seen_candidates.add(key)
            unique_candidates.append(candidate)

    validation_candidates = []
    for index, candidate in enumerate(unique_candidates):
        diagnostics = evaluate_split(
            evaluator, candidate, validation_teacher, validation_pairs, tokenizer,
            args.pair_batch_size, args.curvature_prompts, args.seed + 9001,
        )
        objective = 0.5 * sum(
            weights[name] * value * value
            for name, value in diagnostics.items()
        )
        validation_candidates.append({
            "gains": candidate.tolist(),
            "objective": objective,
            "residual_rms": diagnostics,
        })
        LOG.info(
            "Held-out gain candidate %d/%d: objective=%.6f gains=%s",
            index + 1, len(unique_candidates), objective,
            ",".join(f"{float(value):.4f}" for value in candidate),
        )
    zero_kl = validation_candidates[0]["residual_rms"]["kl"]
    best_kl = min(item["residual_rms"]["kl"] for item in validation_candidates)
    kl_limit = zero_kl - args.kl_improvement_retention * (zero_kl - best_kl)
    eligible = [
        item for item in validation_candidates
        if item["residual_rms"]["kl"] <= kl_limit + 1e-12
    ]
    selected = min(
        eligible,
        key=lambda item: (item["residual_rms"].get("norm", 0.0), item["objective"]),
    )
    gains = torch.tensor(selected["gains"], dtype=torch.float32)
    validation = selected["residual_rms"]
    LOG.info(
        "Selected minimum-norm held-out gains: objective=%.6f KL=%.6f "
        "limit=%.6f retained=%.2f gains=%s",
        selected["objective"], selected["residual_rms"]["kl"], kl_limit,
        args.kl_improvement_retention, gains.tolist(),
    )
    interactions.subset_gains.copy_(gains.to(interactions.subset_gains))
    source_config = json.loads((source / "interaction_config.json").read_text())
    objective_name = (
        "joint_distributional_marginal_and_steerability_lm"
        if weights["marginal"] > 0
        else "joint_distributional_and_steerability_lm"
    )
    metadata = {
        "objective": objective_name,
        "parameterization": source_config.get("parameterization"),
        "pure_axis_endpoints": "exact_independent_lora",
        "source_interaction_dir": str(source),
        "source_interaction_config_sha256": hashlib.sha256(
            (source / "interaction_config.json").read_bytes()
        ).hexdigest(),
        "fit": {
            "optimizer": "bounded_relinearized_levenberg_marquardt",
            "optimized_parameters": "one scalar gain per multi-axis interaction subset",
            "dataset_split": args.dataset_split,
            "axis_datasets": dataset_metadata,
            "fit_pairs_per_axis": args.fit_pairs_per_axis,
            "validation_pairs_per_axis": args.validation_pairs_per_axis,
            "fit_context_sha256": _prompt_digest(fit_contexts),
            "validation_context_sha256": _prompt_digest(validation_contexts),
            "heldout_evaluation_prompts": args.heldout_evaluation_prompts,
            "heldout_evaluation_prompt_sha256": _prompt_digest(
                [item["prompt"] for item in heldout_records]
            ),
            "evaluation_prompts_used_for_optimization": False,
            "steps": args.steps,
            "finite_difference": args.finite_difference,
            "initial_damping": args.lm_damping,
            "initial_trust_radius": args.trust_radius,
            "max_gain": args.max_gain,
            "kl_improvement_retention": args.kl_improvement_retention,
            "heldout_kl_limit": kl_limit,
            "interior_margin": args.interior_margin,
            "signal_derivative_method": "exact_reverse_over_reverse_directional_autodiff",
            "signal_derivative_scope": "single_decoder_layer_with_fixed_input",
            "static_residual": "normalized_prompt_variance_of_local_signal_jvp",
            "curvature_layers": evaluator.curvature_layers,
            "active_residuals": [name for name, weight in weights.items() if weight > 0],
            "weights": weights,
            "seed": args.seed,
        },
        "validation_residual_rms": validation,
        "validation_gain_candidates": validation_candidates,
        "unscaled_interaction_to_task_ratios": ratios.tolist(),
    }
    interactions.save_artifact(output, metadata)
    (output / "gain_fit_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "static_fit_prompts.jsonl").write_text("".join(
        json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n"
        for prompt in fit_contexts
    ))
    LOG.info("Saved constrained HeRD mergings to %s", output)
    LOG.info("Final subset gains: %s", gains.tolist())
    LOG.info("Validation residual RMS: %s", validation)


if __name__ == "__main__":
    main()
