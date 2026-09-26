"""Fit static multi-affine interaction LoRAs to distributional composition.

The standalone adapters remain immutable and are applied by exact continuous
task arithmetic. Only interaction branches R_S, |S| >= 2, are optimized:

    Delta W(s) = sum_i s_i tau_i + sum_S prod_{i in S} s_i R_S.

At a shared prompt context, the teacher is the normalized log-opinion pool

    log q_s = log p_0 + sum_i s_i (log p_i - log p_0) - log Z_s.

This file is intentionally a standalone runner. It consumes ordinary PEFT
LoRA directories and writes an additional artifact beneath the same root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, set_seed

from .common import ROOT, encode, read_prompts
from .independent import (
    IndependentLoRAHooks,
    MultiAffineInteractionHooks,
    MultiAffineInteractions,
    interaction_subsets,
    load_independent_loras,
)

# common.py places steered_finetuner on sys.path.
from config import TrainingConfig  # noqa: E402
from model import build_model  # noqa: E402


LOG = logging.getLogger("distributional_interactions")


@dataclass
class TeacherBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    # [prompt, autoregressive context, base + axes, vocabulary], stored as FP32
    # on CPU. Keeping the teacher in FP32 matters because the objective sums
    # over the full vocab.
    log_probs: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.input_ids.shape[0])


def _axis_order(root: Path, requested: str | None) -> list[str]:
    if requested:
        axes = [item.strip() for item in requested.split(",") if item.strip()]
        if not axes:
            raise ValueError("--axes did not contain any names")
        return axes

    configs = sorted(root.glob("*/training_config.json"))
    if not configs:
        raise FileNotFoundError(
            f"Cannot infer axes: no AXIS/training_config.json files beneath {root}"
        )
    first = json.loads(configs[0].read_text())
    configured = [item["axis"] for item in first.get("axis_datasets", [])]
    axes = [axis for axis in configured if (root / axis).is_dir()]
    if not axes:
        axes = sorted(path.parent.name for path in configs)
    return axes


def _normalize_prompt(value: str) -> str:
    return " ".join(value.strip().split())


def _axis_datasets(root: Path, axes: Sequence[str], overrides: Sequence[str]) -> dict[str, str]:
    result = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError("--axis_dataset must be AXIS=DATASET")
        axis, dataset = (part.strip() for part in item.split("=", 1))
        if not axis or not dataset:
            raise ValueError("--axis_dataset must contain nonempty AXIS and DATASET")
        result[axis] = dataset
    for axis in axes:
        if axis in result:
            continue
        config_path = root / axis / "training_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Cannot infer dataset for {axis!r}: missing {config_path}"
            )
        config = json.loads(config_path.read_text())
        dataset = config.get("trained_dataset") or next(
            (item.get("dataset") for item in config.get("axis_datasets", [])
             if item.get("axis") == axis),
            None,
        )
        if not dataset:
            raise ValueError(f"Cannot infer dataset for axis {axis!r}")
        result[axis] = str(dataset)
    unknown = sorted(set(result) - set(axes))
    if unknown:
        raise ValueError(f"Dataset overrides contain unknown axes: {unknown}")
    return result


def _row_prompt(row: Mapping) -> str | None:
    direct = row.get("prompt")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    messages = row.get("original_messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return None
    if not isinstance(messages, list):
        return None
    latest = None
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "assistant":
            break
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            latest = message["content"]
    return latest.strip() if latest and latest.strip() else None


def dataset_prompt_splits(
    datasets: Mapping[str, str],
    axes: Sequence[str],
    split: str,
    fit_per_axis: int,
    validation_per_axis: int,
    excluded_prompts: set[str],
    seed: int,
    hf_token: str | None,
) -> tuple[list[str], list[str], dict]:
    """Balanced, disjoint dataset contexts with evaluation prompts removed."""
    from datasets import load_dataset

    fit, validation, metadata = [], [], {}
    globally_used = set(excluded_prompts)
    needed = fit_per_axis + validation_per_axis
    for axis_index, axis in enumerate(axes):
        kwargs = {"token": hf_token} if hf_token else {}
        raw = load_dataset(datasets[axis], split=split, **kwargs).shuffle(
            seed=seed + axis_index,
        )
        accepted, excluded, duplicates = [], 0, 0
        for row in raw:
            status = str(row.get("_status", "")).strip().lower()
            if status and status != "success":
                continue
            prompt = _row_prompt(row)
            if prompt is None:
                continue
            key = _normalize_prompt(prompt)
            if key in excluded_prompts:
                excluded += 1
                continue
            if key in globally_used:
                duplicates += 1
                continue
            globally_used.add(key)
            accepted.append(prompt)
            if len(accepted) >= needed:
                break
        if len(accepted) < needed:
            raise ValueError(
                f"Axis {axis!r} supplied {len(accepted)} unique non-evaluation prompts; "
                f"need {needed}"
            )
        validation.extend(accepted[:validation_per_axis])
        fit.extend(accepted[validation_per_axis:needed])
        metadata[axis] = {
            "dataset": datasets[axis],
            "fit_prompts": fit_per_axis,
            "validation_prompts": validation_per_axis,
            "excluded_evaluation_overlaps": excluded,
            "excluded_cross_axis_duplicates": duplicates,
        }
    random.Random(seed + 101).shuffle(fit)
    random.Random(seed + 103).shuffle(validation)
    return fit, validation, metadata


def _multi_axis_corners(axis_count: int, device: torch.device) -> torch.Tensor:
    values = []
    for mask in range(1 << axis_count):
        row = [(mask >> axis) & 1 for axis in range(axis_count)]
        if sum(row) >= 2:
            values.append(row)
    return torch.tensor(values, dtype=torch.float32, device=device)


def compositional_teacher_log_probs(
    teacher_log_probs: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Return log q_s for every prompt/signal pair.

    Args:
        teacher_log_probs: [B, 1+k, V] or [B,P,1+k,V], with p0 first.
        values: [C, k].
    Returns:
        [B,C,V] or [B,C,P,V] normalized log probabilities.
    """
    if teacher_log_probs.ndim not in (3, 4) or values.ndim != 2:
        raise ValueError("teacher_log_probs must be [B,1+k,V] or [B,P,1+k,V]")
    squeeze_position = teacher_log_probs.ndim == 3
    if squeeze_position:
        teacher_log_probs = teacher_log_probs.unsqueeze(1)
    if teacher_log_probs.shape[2] != values.shape[1] + 1:
        raise ValueError("teacher axis count does not match steering values")
    base = teacher_log_probs[:, :, 0].float()
    effects = teacher_log_probs[:, :, 1:].float() - base.unsqueeze(2)
    unnormalized = base.unsqueeze(1) + torch.einsum(
        "ca,bpav->bcpv", values.float(), effects,
    )
    result = F.log_softmax(unnormalized, dim=-1)
    return result.squeeze(2) if squeeze_position else result


def _tail_log_probs(output, positions: int) -> torch.Tensor:
    return F.log_softmax(output.logits[:, -positions:, :].float(), dim=-1)


@torch.inference_mode()
def cache_teacher_batches(
    model,
    tokenizer,
    baseline,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_prompt_tokens: int,
    prefix_contexts: int,
    label: str,
) -> list[TeacherBatch]:
    batches = []
    total = (len(prompts) + batch_size - 1) // batch_size
    zero = torch.zeros(1, len(baseline.axes), device=device)
    for batch_index, start in enumerate(range(0, len(prompts), batch_size), 1):
        current = prompts[start:start + batch_size]
        encoded = encode(tokenizer, current, device, max_prompt_tokens)
        B = len(current)
        # One short neutral greedy rollout supplies shared autoregressive
        # prefixes. Every teacher is then evaluated on precisely the same
        # tokens, so axis effects are never confounded by different histories.
        for _ in range(prefix_contexts - 1):
            next_token = model.forward_base(**encoded).logits[:, -1].argmax(-1, keepdim=True)
            encoded["input_ids"] = torch.cat([encoded["input_ids"], next_token], dim=1)
            encoded["attention_mask"] = torch.cat([
                encoded["attention_mask"],
                torch.ones(B, 1, device=device, dtype=encoded["attention_mask"].dtype),
            ], dim=1)
        distributions = [_tail_log_probs(model.forward_base(**encoded), prefix_contexts)]
        for axis in range(len(baseline.axes)):
            values = zero.expand(B, -1).clone()
            values[:, axis] = 1.0
            with IndependentLoRAHooks(model, baseline, values):
                distributions.append(_tail_log_probs(model.forward_base(**encoded), prefix_contexts))
        batches.append(TeacherBatch(
            input_ids=encoded["input_ids"].cpu(),
            attention_mask=encoded["attention_mask"].cpu(),
            log_probs=torch.stack(distributions, dim=2).cpu(),
        ))
        LOG.info("Teacher cache %s: batch %d/%d", label, batch_index, total)
    return batches


def _repeat_encoded(batch: TeacherBatch, repeats: int, device: torch.device) -> dict:
    return {
        "input_ids": batch.input_ids.to(device).repeat_interleave(repeats, dim=0),
        "attention_mask": batch.attention_mask.to(device).repeat_interleave(repeats, dim=0),
    }


def _student_log_probs(model, baseline, interactions, batch: TeacherBatch,
                       values: torch.Tensor, device: torch.device) -> torch.Tensor:
    B, C = batch.size, values.shape[0]
    positions = int(batch.log_probs.shape[1])
    encoded = _repeat_encoded(batch, C, device)
    signals = values.repeat(B, 1)
    with IndependentLoRAHooks(model, baseline, signals):
        if interactions is None:
            output = model.forward_base(**encoded)
        else:
            with MultiAffineInteractionHooks(model, interactions, signals):
                output = model.forward_base(**encoded)
    vocabulary = output.logits.shape[-1]
    return _tail_log_probs(output, positions).reshape(B, C, positions, vocabulary)


def _kl_per_example(target_log_probs: torch.Tensor,
                    student_log_probs: torch.Tensor) -> torch.Tensor:
    return (target_log_probs.exp() * (target_log_probs - student_log_probs.float())).sum(-1)


@torch.inference_mode()
def evaluate(
    model,
    baseline,
    interactions,
    batches: Sequence[TeacherBatch],
    corners: torch.Tensor,
    device: torch.device,
) -> dict:
    totals = torch.zeros(corners.shape[0], dtype=torch.float64)
    counts = torch.zeros(corners.shape[0], dtype=torch.float64)
    for batch in batches:
        teacher = compositional_teacher_log_probs(batch.log_probs.to(device), corners)
        student = _student_log_probs(model, baseline, interactions, batch, corners, device)
        kl = _kl_per_example(teacher, student).double().cpu()
        totals += kl.sum(dim=(0, 2))
        counts += batch.size * kl.shape[2]
    means = totals / counts.clamp_min(1)
    return {
        "mean_kl": float(means.mean()),
        "by_corner": {
            ",".join(str(int(value)) for value in corner.tolist()): float(mean)
            for corner, mean in zip(corners.cpu(), means)
        },
    }


def _training_values(corners: torch.Tensor, random_count: int,
                     generator: torch.Generator) -> torch.Tensor:
    if random_count <= 0:
        return corners
    random_values = torch.rand(
        random_count, corners.shape[1], generator=generator, device="cpu"
    ).to(corners.device)
    # Exclude nearly pure samples: those provide no interaction gradient and
    # their endpoints are already exact by construction.
    strongest = torch.topk(random_values, k=2, dim=1).values
    keep = strongest[:, 1] >= 0.05
    return torch.cat([corners, random_values[keep]], dim=0)


def fit_interactions(
    model,
    baseline,
    interactions,
    fit_batches: Sequence[TeacherBatch],
    validation_batches: Sequence[TeacherBatch],
    device: torch.device,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    random_directions: int,
    seed: int,
    log_every: int,
    validation_every: int,
) -> tuple[dict, dict, list[dict]]:
    corners = _multi_axis_corners(len(baseline.axes), device)
    before = evaluate(model, baseline, None, validation_batches, corners, device)
    LOG.info("Validation task-arithmetic KL before fit: %.6f", before["mean_kl"])

    # The zero-initialized interaction model is exactly task arithmetic.  Keep
    # it as an explicit candidate so a noisy/overfit interaction fit can never
    # silently make the deployable artifact worse on held-out composition KL.
    best_validation = float(before["mean_kl"])
    best_step = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in interactions.state_dict().items()
    }

    interactions.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        interactions.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    python_rng = random.Random(seed + 23)
    history = []
    started = time.time()
    for step in range(1, steps + 1):
        batch = fit_batches[python_rng.randrange(len(fit_batches))]
        values = _training_values(corners, random_directions, generator)
        teacher = compositional_teacher_log_probs(batch.log_probs.to(device), values)
        student = _student_log_probs(model, baseline, interactions, batch, values, device)
        loss = _kl_per_example(teacher, student).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(clip_grad_norm_(interactions.parameters(), 1.0))
        optimizer.step()

        should_validate = (
            step == steps or step % validation_every == 0
        )
        validation = None
        if should_validate:
            validation = evaluate(
                model, baseline, interactions, validation_batches, corners, device,
            )
            if validation["mean_kl"] < best_validation:
                best_validation = float(validation["mean_kl"])
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in interactions.state_dict().items()
                }

        if step == 1 or step % log_every == 0 or step == steps:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": gradient_norm,
                "elapsed_seconds": time.time() - started,
                "validation_mean_kl": (
                    None if validation is None else float(validation["mean_kl"])
                ),
                "best_validation_mean_kl": best_validation,
                "best_step": best_step,
            }
            history.append(record)
            LOG.info(
                "Interaction fit step %d/%d: KL=%.6f grad=%.4f elapsed=%.1fs",
                step, steps, record["loss"], gradient_norm, record["elapsed_seconds"],
            )

    interactions.load_state_dict(best_state)
    interactions.to(device=device, dtype=torch.float32)
    after = evaluate(model, baseline, interactions, validation_batches, corners, device)
    if best_step == 0:
        LOG.warning(
            "No interaction checkpoint improved held-out KL; restoring exact task "
            "arithmetic (zero interactions)."
        )
    LOG.info(
        "Validation selected-interaction KL: %.6f (best step %d; task arithmetic %.6f)",
        after["mean_kl"], best_step, before["mean_kl"],
    )
    return before, after, history


def _prompt_digest(prompts: Sequence[str]) -> str:
    return hashlib.sha256("\n\0\n".join(prompts).encode()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit static multi-affine interaction LoRAs to full-distribution composition."
    )
    parser.add_argument("--independent_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path,
                        help="Defaults to INDEPENDENT_ROOT/compositional-interactions.")
    parser.add_argument("--axes", help="Comma-separated axis order; inferred from training metadata by default.")
    parser.add_argument("--axis_dataset", action="append", default=[], metavar="AXIS=DATASET")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--fit_prompts_per_axis", type=int, default=100)
    parser.add_argument("--validation_prompts_per_axis", type=int, default=20)
    parser.add_argument(
        "--validation_prompt_jsonl", type=Path, action="append", required=True,
        help="Repeatable evaluation-prompt JSONL excluded from fitting and validation.",
    )
    parser.add_argument("--teacher_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_tokens", type=int, default=128)
    parser.add_argument("--prefix_contexts", type=int, default=4,
                        help="Shared base-rollout contexts per prompt, including the prompt boundary.")
    parser.add_argument("--interaction_rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--random_directions", type=int, default=4,
                        help="Continuous directions added to all binary multi-axis corners per step.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument(
        "--validation_every", type=int, default=10,
        help="Select the deployable checkpoint by held-out KL at this step interval.",
    )
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if args.fit_prompts_per_axis < 1 or args.validation_prompts_per_axis < 1:
        raise ValueError("Per-axis fit and validation prompt counts must both be positive")
    if (args.steps < 1 or args.teacher_batch_size < 1 or args.prefix_contexts < 1
            or args.validation_every < 1):
        raise ValueError(
            "steps, teacher_batch_size, prefix_contexts, and validation_every must be positive"
        )

    independent_root = args.independent_root.resolve()
    output = (args.output_dir or independent_root / "compositional-interactions").resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)

    axes = _axis_order(independent_root, args.axes)
    baseline = load_independent_loras(independent_root, axes)
    datasets = _axis_datasets(independent_root, axes, args.axis_dataset)
    device = torch.device(args.device)
    set_seed(args.seed)
    LOG.info("Independent adapters: %s; base=%s", axes, baseline.base_model)
    LOG.info("Interaction subsets: %s", interaction_subsets(len(axes)))

    cfg = TrainingConfig(
        base_model=baseline.base_model,
        signal_dim=len(axes),
        adapter_architecture="anchored_neural_merger",
        lora_rank=1,
        adapter_hidden_dim=1,
        freeze_base_model=True,
        bf16=True,
    )
    model = build_model(cfg, device=args.device, attn_implementation="sdpa")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    expected_keys = sorted(model._adapted_linears)
    fitted_keys = baseline.validate_uniform_subset(expected_keys)
    LOG.info(
        "Fitting interactions at %d/%d model projections: %s",
        len(fitted_keys), len(expected_keys), fitted_keys,
    )

    tokenizer = AutoTokenizer.from_pretrained(baseline.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    excluded_prompts = {
        _normalize_prompt(item["prompt"])
        for path in args.validation_prompt_jsonl
        for item in read_prompts(path)
    }
    fit_prompts, validation_prompts, dataset_metadata = dataset_prompt_splits(
        datasets, axes, args.dataset_split,
        args.fit_prompts_per_axis, args.validation_prompts_per_axis,
        excluded_prompts, args.seed, args.hf_token or os.environ.get("HF_TOKEN"),
    )
    LOG.info(
        "Dataset-only interaction split: fit=%d validation=%d excluded-evaluation=%d",
        len(fit_prompts), len(validation_prompts), len(excluded_prompts),
    )

    fit_batches = cache_teacher_batches(
        model, tokenizer, baseline, fit_prompts, device,
        args.teacher_batch_size, args.max_prompt_tokens, args.prefix_contexts, "fit",
    )
    validation_batches = cache_teacher_batches(
        model, tokenizer, baseline, validation_prompts, device,
        args.teacher_batch_size, args.max_prompt_tokens, args.prefix_contexts, "validation",
    )
    interactions = MultiAffineInteractions(baseline, args.interaction_rank, args.seed)
    before, after, history = fit_interactions(
        model, baseline, interactions, fit_batches, validation_batches, device,
        args.steps, args.learning_rate, args.weight_decay, args.random_directions,
        args.seed, args.log_every, args.validation_every,
    )

    metadata: Mapping = {
        "objective": "full_distribution_log_opinion_pool_kl",
        "parameterization": (
            "sum_i s_i tau_i + sum_{|S|>=2} prod_{i in S}(s_i) R_S"
        ),
        "pure_axis_endpoints": "exact_independent_lora",
        "fit": {
            "source": "axis_training_datasets",
            "dataset_split": args.dataset_split,
            "axis_datasets": dataset_metadata,
            "fit_prompts_per_axis": args.fit_prompts_per_axis,
            "validation_prompts_per_axis": args.validation_prompts_per_axis,
            "fit_prompts": len(fit_prompts),
            "validation_prompts": len(validation_prompts),
            "excluded_evaluation_prompt_count": len(excluded_prompts),
            "validation_prompt_jsonl": [str(path) for path in args.validation_prompt_jsonl],
            "evaluation_prompts_used": False,
            "prompt_sha256": _prompt_digest(fit_prompts + validation_prompts),
            "max_prompt_tokens": args.max_prompt_tokens,
            "prefix_contexts_per_prompt": args.prefix_contexts,
            "prefix_source": "shared_base_greedy_rollout",
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "random_directions_per_step": args.random_directions,
            "validation_every": args.validation_every,
            "seed": args.seed,
        },
        "validation": {
            "task_arithmetic": before,
            "fitted_interactions": after,
            "mean_kl_improvement": before["mean_kl"] - after["mean_kl"],
        },
    }
    interactions.save_artifact(output, metadata)
    (output / "fit_history.json").write_text(json.dumps(history, indent=2) + "\n")
    LOG.info("Saved deployable interactions to %s", output)


if __name__ == "__main__":
    main()
