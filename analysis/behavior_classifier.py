"""Train per-axis DeBERTa classifiers and score steering generations.

Training labels are paired and explicit:

    original_messages -> 0
    messages          -> 1

All splitting occurs at the prompt/pair group level. Prompts supplied through
``--validation_prompt_jsonl`` are excluded before splitting, preventing the
generation benchmark from leaking into classifier training or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

from .common import json_dump, read_prompts


LOG = logging.getLogger("behavior_classifier")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class AxisSpec:
    axis: str
    dataset: str


@dataclass
class PairRecord:
    pair_id: str
    prompt_key: str
    negative_text: str
    positive_text: str


@dataclass
class LabeledRecord:
    text: str
    label: int
    pair_id: str
    prompt_group: str | None = None


def normalize_prompt(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                value = item.get("text", item.get("content", ""))
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def normalize_messages(value) -> list[dict] | None:
    if not isinstance(value, list) or not value:
        return None
    result = []
    for message in value:
        if not isinstance(message, Mapping):
            return None
        role = str(message.get("role", "")).strip().lower()
        content = _content_text(message.get("content")).strip()
        if not role or not content:
            continue
        result.append({"role": role, "content": content})
    return result or None


def format_conversation(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n\n".join(
        f"[{str(message['role']).upper()}]\n{message['content'].strip()}"
        for message in messages
    )


def format_generation(prompt: str, response: str) -> str:
    return f"[USER]\n{prompt.strip()}\n\n[ASSISTANT]\n{response.strip()}"


def prompt_candidates(messages: Sequence[Mapping[str, str]]) -> set[str]:
    values = {
        normalize_prompt(message["content"])
        for message in messages
        if message["role"] in {"user", "system"}
    }
    joined = normalize_prompt("\n".join(
        message["content"] for message in messages
        if message["role"] in {"user", "system"}
    ))
    if joined:
        values.add(joined)
    return {value for value in values if value}


def validation_prompt_set(paths: Sequence[Path]) -> set[str]:
    result = set()
    for path in paths:
        for item in read_prompts(path):
            result.add(normalize_prompt(item["prompt"]))
    return result


def _failed_row(row: Mapping) -> bool:
    status = str(row.get("_status", "")).strip().lower()
    if status and status != "success":
        return True
    failed = row.get("generation_failed", False)
    return bool(failed)


def load_axis_pairs(spec: AxisSpec, split: str, excluded_prompts: set[str],
                    hf_token: str | None, max_pairs: int | None = None):
    from datasets import load_dataset

    kwargs = {"token": hf_token} if hf_token else {}
    dataset = load_dataset(spec.dataset, split=split, **kwargs)
    required = {"original_messages", "messages"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{spec.dataset} is missing columns {sorted(missing)}")
    pairs = []
    removed_overlap = removed_failed = removed_invalid = 0
    for index, row in enumerate(dataset):
        if _failed_row(row):
            removed_failed += 1
            continue
        negative = normalize_messages(row.get("original_messages"))
        positive = normalize_messages(row.get("messages"))
        if negative is None or positive is None:
            removed_invalid += 1
            continue
        canonical_prompt = normalize_prompt(str(row.get("prompt", "")))
        candidates = prompt_candidates(negative) | prompt_candidates(positive)
        if canonical_prompt:
            candidates.add(canonical_prompt)
        if candidates & excluded_prompts:
            removed_overlap += 1
            continue
        prompt_key = canonical_prompt or "\n".join(sorted(candidates))
        raw_id = row.get("prompt_id", prompt_key or index)
        pair_id = hashlib.sha256(f"{spec.axis}:{raw_id}:{index}".encode()).hexdigest()
        pairs.append(PairRecord(
            pair_id=pair_id,
            prompt_key=prompt_key,
            negative_text=format_conversation(negative),
            positive_text=format_conversation(positive),
        ))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    if not pairs:
        raise ValueError(f"Axis {spec.axis!r} produced no classifier pairs")
    return pairs, {
        "retained_pairs": len(pairs),
        "removed_validation_prompt_overlap": removed_overlap,
        "removed_failed": removed_failed,
        "removed_invalid": removed_invalid,
    }


def grouped_split(pairs: Sequence[PairRecord], validation_fraction: float,
                  test_fraction: float, seed: int):
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must lie in (0,1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below one")
    # A repeated prompt must remain in one split even when it has multiple
    # dataset rows/pair identifiers.
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair.prompt_key].append(pair)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(len(keys) * test_fraction))
    n_validation = max(1, round(len(keys) * validation_fraction))
    if n_test + n_validation >= len(keys):
        raise ValueError("Not enough prompt groups for the requested split")
    split_keys = {
        "test": set(keys[:n_test]),
        "validation": set(keys[n_test:n_test + n_validation]),
        "train": set(keys[n_test + n_validation:]),
    }
    result = {}
    for name, selected in split_keys.items():
        current = []
        for key in selected:
            current.extend(groups[key])
        result[name] = current
    return result


def labeled_pairs(pairs: Sequence[PairRecord]) -> list[LabeledRecord]:
    records = []
    for pair in pairs:
        records.append(LabeledRecord(pair.negative_text, 0, pair.pair_id, pair.prompt_key))
        records.append(LabeledRecord(pair.positive_text, 1, pair.pair_id, pair.prompt_key))
    return records


class TextDataset(Dataset):
    def __init__(self, records: Sequence[LabeledRecord], tokenizer, max_length: int):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        encoded = self.tokenizer(
            record.text, truncation=True, max_length=self.max_length,
        )
        encoded["labels"] = record.label
        encoded["record_index"] = index
        return encoded


class PreTokenizedDataset(Dataset):
    def __init__(self, records: Sequence[LabeledRecord], tokenizer, max_length: int,
                 tokenization_batch_size: int = 2048, description: str = "records"):
        self.records = list(records)
        self.features = []
        total = len(self.records)
        LOG.info(
            "Tokenizing %s: %d records in parent-process batches of %d",
            description, total, tokenization_batch_size,
        )
        for start in range(0, total, tokenization_batch_size):
            stop = min(start + tokenization_batch_size, total)
            encoded = tokenizer(
                [record.text for record in self.records[start:stop]],
                truncation=True, max_length=max_length, padding=False,
            )
            for offset, record in enumerate(self.records[start:stop]):
                feature = {
                    key: torch.tensor(values[offset], dtype=torch.int32)
                    for key, values in encoded.items()
                }
                feature["labels"] = record.label
                feature["record_index"] = start + offset
                self.features.append(feature)
            if stop == total or (start // tokenization_batch_size + 1) % 10 == 0:
                LOG.info("Tokenized %s: %d/%d", description, stop, total)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # The collator removes record_index, so never expose the stored mapping
        # itself for mutation.
        return dict(self.features[index])


class ClassifierCollator:
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    def __call__(self, features):
        indices = torch.tensor([item.pop("record_index") for item in features])
        batch = self.base(features)
        batch["record_index"] = indices
        return batch


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def predict_tokenized(model, tokenizer, dataset: PreTokenizedDataset, device: torch.device,
                      batch_size: int, description: str = "evaluation",
                      log_every: int = 25):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=ClassifierCollator(tokenizer), num_workers=0,
        pin_memory=device.type == "cuda",
    )
    scores = np.empty(len(dataset), dtype=np.float64)
    model.eval()
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, 1):
        indices = batch.pop("record_index").numpy()
        labels = batch.pop("labels", None)
        output = model(**_move_batch(batch, device))
        current = torch.softmax(output.logits.float(), dim=-1)[:, 1].cpu().numpy()
        scores[indices] = current
        if batch_index % log_every == 0 or batch_index == len(loader):
            processed = min(batch_index * batch_size, len(dataset))
            elapsed = time.perf_counter() - started
            rate = processed / elapsed
            remaining = (len(dataset) - processed) / rate if rate else float("inf")
            LOG.info(
                "%s inference: %d/%d (%.1f examples/s, ETA %.1f min)",
                description, processed, len(dataset), rate, remaining / 60.0,
            )
    return scores


def predict_records(model, tokenizer, records: Sequence[LabeledRecord], device: torch.device,
                    max_length: int, batch_size: int, num_workers: int = 0,
                    tokenization_batch_size: int = 2048,
                    description: str = "evaluation records",
                    inference_log_every: int = 25):
    dataset = PreTokenizedDataset(
        records, tokenizer, max_length, tokenization_batch_size, description,
    )
    return predict_tokenized(
        model, tokenizer, dataset, device, batch_size, description,
        inference_log_every,
    )


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1], denominator,
        out=np.zeros_like(denominator), where=denominator > 0,
    )
    best_f1 = np.max(f1)
    candidates = np.flatnonzero(np.isclose(f1, best_f1))
    best = candidates[np.argmin(np.abs(thresholds[candidates] - 0.5))]
    return float(thresholds[best])


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 15) -> float:
    total = len(labels)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = ((scores >= edges[index]) &
                    (scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]))
        if selected.any():
            value += selected.mean() * abs(scores[selected].mean() - labels[selected].mean())
    return float(value)


def paired_accuracy(records: Sequence[LabeledRecord], scores: np.ndarray) -> float:
    pairs = defaultdict(dict)
    for record, score in zip(records, scores):
        pairs[record.pair_id][record.label] = float(score)
    valid = [value for value in pairs.values() if set(value) == {0, 1}]
    return float(np.mean([item[1] > item[0] for item in valid])) if valid else float("nan")


def binary_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float,
                   records: Sequence[LabeledRecord] | None = None) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0,
    )
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    result = {
        "count": int(len(labels)),
        "positive_fraction": float(labels.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "auprc": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "true_positive_rate": float(recall),
        "true_negative_rate": float(tn / (tn + fp)) if tn + fp else None,
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
        "f1": float(f1),
        "brier": float(brier_score_loss(labels, scores)),
        "ece_15": expected_calibration_error(labels, scores, 15),
        "confusion_matrix": matrix.tolist(),
        "mean_score_negative": float(scores[labels == 0].mean()) if np.any(labels == 0) else None,
        "mean_score_positive": float(scores[labels == 1].mean()) if np.any(labels == 1) else None,
    }
    if records is not None:
        result["paired_accuracy"] = paired_accuracy(records, scores)
    return result


def grouped_bootstrap_ci(records: Sequence[LabeledRecord], scores: np.ndarray,
                         threshold: float, repetitions: int, seed: int):
    if repetitions <= 0:
        return {}
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.prompt_group or record.pair_id].append(index)
    keys = list(groups)
    rng = random.Random(seed)
    values = defaultdict(list)
    for _ in range(repetitions):
        sampled = [rng.choice(keys) for _ in keys]
        indices = [index for key in sampled for index in groups[key]]
        labels = [records[index].label for index in indices]
        current = binary_metrics(labels, scores[indices], threshold)
        for metric in ("auroc", "auprc", "balanced_accuracy", "f1"):
            if current[metric] is not None:
                values[metric].append(current[metric])
    return {
        metric: {
            "lower_2.5": float(np.quantile(current, 0.025)),
            "upper_97.5": float(np.quantile(current, 0.975)),
        }
        for metric, current in values.items() if current
    }


def train_axis_classifier(spec: AxisSpec, args, excluded_prompts: set[str], device: torch.device):
    axis_dir = args.output_dir / spec.axis
    if axis_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Classifier output exists: {axis_dir}")
        shutil.rmtree(axis_dir)
    pairs, filtering = load_axis_pairs(
        spec, args.dataset_split, excluded_prompts,
        args.hf_token or os.environ.get("HF_TOKEN"), args.max_pairs,
    )
    split_pairs = grouped_split(
        pairs, args.validation_fraction, args.test_fraction, args.seed,
    )
    records = {name: labeled_pairs(value) for name, value in split_pairs.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, trust_remote_code=True, dtype=torch.float32,
    ).float().to(device)
    train_dataset = PreTokenizedDataset(
        records["train"], tokenizer, args.max_length,
        args.tokenization_batch_size, f"axis {spec.axis} train split",
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        collate_fn=ClassifierCollator(tokenizer), num_workers=0,
        pin_memory=device.type == "cuda",
    )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, round(total_updates * args.warmup_ratio), total_updates,
    )
    use_bf16 = device.type == "cuda" and args.bf16
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if parameter_dtypes != {torch.float32}:
        raise RuntimeError(
            f"Classifier master weights must be FP32; loaded {sorted(map(str, parameter_dtypes))}"
        )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    LOG.info(
        "Axis %s classifier precision: FP32 master weights, %s%s",
        spec.axis, "BF16 autocast" if use_bf16 else "FP32 forward/backward",
        " with TF32 matmuls" if device.type == "cuda" and not use_bf16 else "",
    )
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for batch_index, batch in enumerate(loader, 1):
            batch.pop("record_index")
            batch = _move_batch(batch, device)
            group_start = ((batch_index - 1) // args.gradient_accumulation_steps
                           * args.gradient_accumulation_steps)
            group_size = min(
                args.gradient_accumulation_steps, len(loader) - group_start,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                raw_loss = model(**batch).loss
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(
                    f"Non-finite classifier loss for axis {spec.axis!r} at epoch "
                    f"{epoch + 1}, batch {batch_index}. Run in FP32 (the default); "
                    "BF16 is an explicit experimental opt-in for DeBERTa."
                )
            loss = raw_loss / group_size
            loss.backward()
            running += float(raw_loss.detach())
            if batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(loader):
                clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    LOG.info(
                        "Axis %s epoch %d/%d update %d/%d loss=%.4f",
                        spec.axis, epoch + 1, args.epochs, global_step, total_updates,
                        running / batch_index,
                    )

    validation_scores = predict_records(
        model, tokenizer, records["validation"], device,
        args.max_length, args.eval_batch_size, 0, args.tokenization_batch_size,
        f"axis {spec.axis} validation split",
    )
    validation_labels = np.asarray([item.label for item in records["validation"]])
    threshold = select_threshold(validation_labels, validation_scores)
    test_scores = predict_records(
        model, tokenizer, records["test"], device,
        args.max_length, args.eval_batch_size, 0, args.tokenization_batch_size,
        f"axis {spec.axis} test split",
    )
    validation = binary_metrics(
        validation_labels, validation_scores, threshold, records["validation"],
    )
    test = binary_metrics(
        [item.label for item in records["test"]], test_scores, threshold, records["test"],
    )
    test["bootstrap_95_ci"] = grouped_bootstrap_ci(
        records["test"], test_scores, threshold, args.bootstrap_repetitions,
        args.seed + 31,
    )
    axis_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(axis_dir, safe_serialization=True)
    tokenizer.save_pretrained(axis_dir)
    metadata = {
        "axis": spec.axis,
        "dataset": spec.dataset,
        "model_name": args.model_name,
        "positive_source": "messages",
        "negative_source": "original_messages",
        "input_format": "role-tagged full conversation",
        "truncation_side": "left",
        "validation_prompt_jsonl": [str(path) for path in args.validation_prompt_jsonl],
        "filtering": filtering,
        "split_pairs": {name: len(value) for name, value in split_pairs.items()},
        "threshold": threshold,
        "validation": validation,
        "test": test,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "tokenization_batch_size": args.tokenization_batch_size,
            "seed": args.seed,
            "optimizer_updates": global_step,
            "precision": "bf16_autocast" if use_bf16 else "fp32_tf32",
        },
    }
    (axis_dir / "classifier_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def parse_axis_specs(values: Sequence[str]) -> list[AxisSpec]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError("--axis_dataset values must be AXIS=DATASET")
        axis, dataset = value.split("=", 1)
        axis, dataset = axis.strip(), dataset.strip()
        if not _SAFE_NAME.fullmatch(axis) or not dataset:
            raise ValueError(f"Invalid axis dataset specification: {value!r}")
        result.append(AxisSpec(axis, dataset))
    if not result:
        raise ValueError("At least one --axis_dataset is required")
    return result


def _direction_dict(value, axes: Sequence[str]) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {axis: float(value.get(axis, 0.0)) for axis in axes}
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != len(axes):
            raise ValueError("Generation direction length does not match axes")
        return {axis: float(item) for axis, item in zip(axes, value)}
    raise ValueError(f"Unsupported generation direction: {value!r}")


_EXAMPLE_METADATA = {"prompt_index", "source_index", "prompt", "judge"}


def _generation_fields(example: Mapping) -> dict[str, str]:
    if isinstance(example.get("generations"), Mapping):
        return {
            str(name): str(value) for name, value in example["generations"].items()
            if isinstance(value, str)
        }
    return {
        str(name): str(value) for name, value in example.items()
        if name not in _EXAMPLE_METADATA and isinstance(value, str)
    }


def _streamed_conditions(path: Path, schema: str):
    import ijson

    if schema == "directions":
        with path.open("rb") as handle:
            for name, item in ijson.kvitems(handle, "analysis.directions"):
                yield name, item["direction"], item.get("examples", [])
    elif schema == "comparisons_dict":
        with path.open("rb") as handle:
            for name, item in ijson.kvitems(handle, "analysis.comparisons"):
                yield name, item["direction"], item.get("examples", [])
    elif schema == "comparisons_list":
        with path.open("rb") as handle:
            for item in ijson.items(handle, "analysis.comparisons.item"):
                yield item.get("label", "condition"), item["combination"], item.get("examples", [])
    else:
        raise ValueError(f"Unknown streaming schema {schema!r}")


def _streamed_axes(path: Path) -> list[str]:
    import ijson

    with path.open("rb") as handle:
        return [str(item) for item in ijson.items(handle, "analysis.axes.item")]


def extract_generation_records(source: Mapping, base_dir: Path,
                               loading_mode: str = "auto") -> list[dict]:
    path = Path(source["path"])
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    source_mode = str(source.get("loading", loading_mode))
    if source_mode not in {"auto", "eager", "stream"}:
        raise ValueError(f"Unknown generation JSON loading mode {source_mode!r}")
    streaming = (
        source_mode == "stream" or
        (source_mode == "auto" and
         path.stat().st_size >= int(source.get("stream_threshold_bytes", 128 << 20)))
    )
    analysis = None
    if streaming:
        axes = list(source.get("axes", [])) or _streamed_axes(path)
        schema = str(source.get("schema", "directions"))
        conditions = _streamed_conditions(path, schema)
        LOG.info("Streaming generation source %s with schema %s", path, schema)
    else:
        if source_mode == "eager":
            import orjson

            LOG.info("Eager-loading generation source %s (%.2f GiB)",
                     path, path.stat().st_size / (1 << 30))
            payload = orjson.loads(path.read_bytes())
        else:
            payload = json.loads(path.read_text())
        analysis = payload.get("analysis", payload)
        axes = list(analysis.get("axes", source.get("axes", [])))
        conditions = []
        if isinstance(analysis.get("comparisons"), Mapping):
            for name, item in analysis["comparisons"].items():
                conditions.append((name, item["direction"], item.get("examples", [])))
        elif isinstance(analysis.get("comparisons"), list):
            for item in analysis["comparisons"]:
                conditions.append((item.get("label", "condition"), item["combination"],
                                   item.get("examples", [])))
        elif isinstance(analysis.get("directions"), Mapping):
            for name, item in analysis["directions"].items():
                conditions.append((name, item["direction"], item.get("examples", [])))
        else:
            raise ValueError(f"Unsupported generation JSON schema: {path}")
    if not axes:
        raise ValueError(f"Cannot determine axes for {path}")
    include = set(source.get("include_methods", []))
    exclude = set(source.get("exclude_methods", []))
    aliases = source.get("method_aliases", {})
    benchmark = str(source.get("benchmark", path.stem))
    family = str(source.get("family", "unspecified"))
    binary_only = bool(source.get("binary_corners_only", False))
    prompt_index_min = source.get("prompt_index_min")
    prompt_index_max = source.get("prompt_index_max")
    result = []
    for condition_name, raw_direction, examples in conditions:
        direction = _direction_dict(raw_direction, axes)
        if binary_only and not all(
            math.isclose(value, 0.0) or math.isclose(value, 1.0)
            for value in direction.values()
        ):
            continue
        for example in examples:
            prompt = str(example.get("prompt", ""))
            if not prompt:
                continue
            prompt_index = example.get("prompt_index")
            if prompt_index_min is not None and (
                prompt_index is None or int(prompt_index) < int(prompt_index_min)
            ):
                continue
            if prompt_index_max is not None and (
                prompt_index is None or int(prompt_index) >= int(prompt_index_max)
            ):
                continue
            for raw_method, response in _generation_fields(example).items():
                if include and raw_method not in include:
                    continue
                if raw_method in exclude:
                    continue
                method = str(aliases.get(raw_method, raw_method))
                result.append({
                    "benchmark": benchmark,
                    "family": family,
                    "source_file": str(path),
                    "method": method,
                    "raw_method": raw_method,
                    "condition": str(condition_name),
                    "direction": direction,
                    "prompt_index": prompt_index,
                    "source_index": example.get("source_index"),
                    "prompt": prompt,
                    "response": response,
                })
    return result


def load_manifest(path: Path, loading_mode: str = "auto") -> list[dict]:
    payload = json.loads(path.read_text())
    sources = payload.get("sources", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(sources, list) or not sources:
        raise ValueError("Generation manifest must contain a nonempty sources list")
    records = []
    for source in sources:
        records.extend(extract_generation_records(source, path.parent, loading_mode))
    if not records:
        raise ValueError("Generation manifest selected no generations")
    return records


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def correlation_metrics(levels: np.ndarray, scores: np.ndarray) -> dict:
    if len(levels) < 2 or np.std(levels) == 0 or np.std(scores) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(levels, scores)[0, 1]),
        "spearman": float(np.corrcoef(_rankdata(levels), _rankdata(scores))[0, 1]),
    }


def _prompt_identity(record: Mapping) -> tuple:
    return (record.get("source_index"), record.get("prompt_index"), record["prompt"])


def monotonicity_report(records: Sequence[dict], axis: str, axes: Sequence[str]) -> dict:
    """Compare adjacent levels while holding prompt and all other axes fixed."""
    groups = defaultdict(list)
    other_axes = [name for name in axes if name != axis]
    for record in records:
        background = tuple(record["direction"].get(name, 0.0) for name in other_axes)
        groups[(_prompt_identity(record), background)].append(record)
    comparisons = violations = 0
    deltas = []
    for values in groups.values():
        values.sort(key=lambda item: item["direction"].get(axis, 0.0))
        for left, right in zip(values, values[1:]):
            left_level = left["direction"].get(axis, 0.0)
            right_level = right["direction"].get(axis, 0.0)
            if np.isclose(left_level, right_level):
                continue
            delta = float(right["axis_scores"][axis] - left["axis_scores"][axis])
            comparisons += 1
            violations += int(delta < 0)
            deltas.append(delta)
    return {
        "adjacent_comparisons": comparisons,
        "violation_count": violations,
        "violation_rate": violations / comparisons if comparisons else None,
        "mean_adjacent_score_change": float(np.mean(deltas)) if deltas else None,
    }


def generation_bootstrap_ci(records: Sequence[dict], axis: str, threshold: float,
                            repetitions: int, seed: int) -> dict:
    endpoint = [record for record in records if (
        np.isclose(record["direction"].get(axis, 0.0), 0.0) or
        np.isclose(record["direction"].get(axis, 0.0), 1.0)
    )]
    if repetitions <= 0 or not endpoint:
        return {}
    groups = defaultdict(list)
    for record in endpoint:
        groups[_prompt_identity(record)].append(record)
    keys = list(groups)
    rng = random.Random(seed)
    values = defaultdict(list)
    for _ in range(repetitions):
        sampled = [rng.choice(keys) for _ in keys]
        current = [record for key in sampled for record in groups[key]]
        labels = [round(record["direction"].get(axis, 0.0)) for record in current]
        scores = [record["axis_scores"][axis] for record in current]
        metrics = binary_metrics(labels, scores, threshold)
        for metric in ("auroc", "auprc", "balanced_accuracy", "f1"):
            if metrics[metric] is not None:
                values[metric].append(metrics[metric])
    return {
        metric: {
            "lower_2.5": float(np.quantile(current, 0.025)),
            "upper_97.5": float(np.quantile(current, 0.975)),
        }
        for metric, current in values.items() if current
    }


def condition_summaries(records: Sequence[dict], axes: Sequence[str],
                        classifier_metadata: Mapping[str, Mapping]) -> dict:
    groups = defaultdict(list)
    for record in records:
        direction_key = tuple(record["direction"].get(axis, 0.0) for axis in axes)
        groups[(record["condition"], direction_key)].append(record)
    result = {}
    for (condition, direction_key), current in groups.items():
        exact = []
        desired_logs = []
        binary_corner = all(np.isclose(target, 0.0) or np.isclose(target, 1.0)
                            for target in direction_key)
        for record in current if binary_corner else []:
            correct = []
            log_terms = []
            for axis, target in zip(axes, direction_key):
                score = min(max(float(record["axis_scores"][axis]), 1e-7), 1 - 1e-7)
                predicted = score >= float(classifier_metadata[axis]["threshold"])
                correct.append(int(predicted) == int(round(target)))
                log_terms.append(math.log(score if round(target) else 1.0 - score))
            exact.append(all(correct))
            desired_logs.append(sum(log_terms) / len(log_terms))
        result[condition] = {
            "direction": {axis: float(value) for axis, value in zip(axes, direction_key)},
            "generation_count": len(current),
            "nonempty_generation_rate": float(np.mean([
                bool(record["response"].strip()) for record in current
            ])),
            "mean_axis_scores": {
                axis: float(np.mean([record["axis_scores"][axis] for record in current]))
                for axis in axes
            },
            "binary_exact_match_accuracy": float(np.mean(exact)) if exact else None,
            "mean_desired_label_log_probability": (
                float(np.mean(desired_logs)) if desired_logs else None
            ),
        }
    return result


def summarize_generation_scores(records: Sequence[dict], classifier_metadata: Mapping[str, Mapping],
                                bootstrap_repetitions: int = 0, seed: int = 42):
    groups = defaultdict(list)
    for record in records:
        groups[(record["benchmark"], record["family"], record["method"])].append(record)
    summaries = {}
    for (benchmark, family, method), current in groups.items():
        key = f"{benchmark}::{method}"
        axes = list(classifier_metadata)
        by_axis = {}
        for axis in axes:
            levels = np.asarray([item["direction"].get(axis, 0.0) for item in current])
            scores = np.asarray([item["axis_scores"][axis] for item in current])
            endpoint = np.isclose(levels, 0.0) | np.isclose(levels, 1.0)
            threshold = float(classifier_metadata[axis]["threshold"])
            endpoint_metrics = binary_metrics(
                np.rint(levels[endpoint]).astype(int), scores[endpoint], threshold,
            ) if endpoint.any() else None
            means = {
                f"{level:g}": float(scores[np.isclose(levels, level)].mean())
                for level in sorted(set(levels.tolist()))
            }
            by_axis[axis] = {
                "endpoint": endpoint_metrics,
                "endpoint_bootstrap_95_ci": generation_bootstrap_ci(
                    current, axis, threshold, bootstrap_repetitions,
                    seed + 1009 * axes.index(axis),
                ),
                "continuous": correlation_metrics(levels, scores),
                "mean_score_by_level": means,
                "monotonicity": monotonicity_report(current, axis, axes),
            }

        corner_records = [item for item in current if all(
            np.isclose(value, 0.0) or np.isclose(value, 1.0)
            for value in item["direction"].values()
        )]
        exact = []
        desired_log_score = []
        for item in corner_records:
            correct = []
            log_terms = []
            for axis in axes:
                target = int(round(item["direction"].get(axis, 0.0)))
                score = min(max(float(item["axis_scores"][axis]), 1e-7), 1 - 1e-7)
                predicted = score >= float(classifier_metadata[axis]["threshold"])
                correct.append(int(predicted) == target)
                log_terms.append(math.log(score if target else 1.0 - score))
            exact.append(all(correct))
            desired_log_score.append(sum(log_terms) / len(log_terms))
        summaries[key] = {
            "benchmark": benchmark,
            "family": family,
            "method": method,
            "generation_count": len(current),
            "nonempty_generation_count": sum(bool(item["response"].strip()) for item in current),
            "nonempty_generation_rate": float(np.mean([
                bool(item["response"].strip()) for item in current
            ])),
            "prompt_count": len({(item.get("source_index"), item.get("prompt_index"), item["prompt"])
                                 for item in current}),
            "by_axis": by_axis,
            "by_condition": condition_summaries(current, axes, classifier_metadata),
            "binary_combination": {
                "count": len(corner_records),
                "exact_match_accuracy": float(np.mean(exact)) if exact else None,
                "mean_desired_label_log_probability": (
                    float(np.mean(desired_log_score)) if desired_log_score else None
                ),
            },
        }
    return summaries


def score_generations(args):
    if args.update_existing and not args.include_method:
        raise ValueError("--update_existing requires at least one --include_method")
    records = load_manifest(args.manifest, args.generation_json_loading)
    include_methods = set(args.include_method)
    if include_methods:
        records = [record for record in records if record["method"] in include_methods]
        if not records:
            raise ValueError(
                f"No generation records matched --include_method={sorted(include_methods)}"
            )
        LOG.info(
            "Incremental scoring selected %d records for methods %s",
            len(records), sorted(include_methods),
        )
    classifier_root = args.classifier_root.resolve()
    index = json.loads((classifier_root / "classifier_index.json").read_text())
    metadata = {item["axis"]: item for item in index["classifiers"]}
    unique_texts = {}
    record_text_keys = []
    for record in records:
        text = format_generation(record["prompt"], record["response"])
        key = hashlib.sha256(text.encode()).hexdigest()
        unique_texts.setdefault(key, text)
        record_text_keys.append(key)
        record["axis_scores"] = {}
    keys = list(unique_texts)
    dummy_records = [LabeledRecord(unique_texts[key], 0, key) for key in keys]
    device = torch.device(args.device)
    max_lengths = {
        int(current_metadata["training"]["max_length"])
        for current_metadata in metadata.values()
    }
    if len(max_lengths) != 1:
        raise ValueError(f"Classifiers use incompatible max lengths: {sorted(max_lengths)}")
    first_axis = next(iter(metadata))
    tokenizer = AutoTokenizer.from_pretrained(
        classifier_root / first_axis, trust_remote_code=True,
    )
    tokenizer.truncation_side = str(metadata[first_axis].get("truncation_side", "left"))
    encoded_generations = PreTokenizedDataset(
        dummy_records, tokenizer, max_lengths.pop(), args.tokenization_batch_size,
        "unique generation records (shared by every axis)",
    )
    for axis, current_metadata in metadata.items():
        LOG.info("Scoring %d unique generations with axis %s", len(keys), axis)
        axis_dir = classifier_root / axis
        model = AutoModelForSequenceClassification.from_pretrained(
            axis_dir, trust_remote_code=True,
        ).to(device)
        if device.type == "cuda" and args.bf16:
            model.to(dtype=torch.bfloat16)
        scores = predict_tokenized(
            model, tokenizer, encoded_generations, device, args.batch_size,
            f"axis {axis}", args.inference_log_every,
        )
        lookup = dict(zip(keys, scores.tolist()))
        for record, text_key in zip(records, record_text_keys):
            record["axis_scores"][axis] = lookup[text_key]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scores_path = args.scored_generations_jsonl
    if scores_path is None:
        scores_path = args.output.with_name(args.output.stem + ".scored.jsonl")
    if args.update_existing and scores_path.is_file():
        retained = []
        with scores_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                previous = json.loads(line)
                if previous.get("method") not in include_methods:
                    retained.append(previous)
        LOG.info(
            "Preserving %d unchanged scored records and replacing %d methods",
            len(retained), len(include_methods),
        )
        records = retained + records
    summary = summarize_generation_scores(
        records, metadata, args.bootstrap_repetitions, args.seed,
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {
        "classifier_root": str(classifier_root),
        "manifest": str(args.manifest.resolve()),
        "classifiers": metadata,
        "summary": summary,
        "scored_generations_jsonl": str(scores_path.resolve()),
        "scored_generation_count": len(records),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="DeBERTa behavioral classifier evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--axis_dataset", action="append", required=True, metavar="AXIS=DATASET")
    train.add_argument("--output_dir", type=Path, required=True)
    train.add_argument("--validation_prompt_jsonl", type=Path, action="append", required=True,
                       help="Repeatable prompt JSONL excluded before classifier splitting.")
    train.add_argument("--model_name", default="microsoft/deberta-v3-base")
    train.add_argument("--dataset_split", default="train")
    train.add_argument("--max_pairs", type=int)
    train.add_argument("--validation_fraction", type=float, default=0.1)
    train.add_argument("--test_fraction", type=float, default=0.1)
    train.add_argument("--max_length", type=int, default=256)
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--batch_size", type=int, default=16)
    train.add_argument("--eval_batch_size", type=int, default=64)
    train.add_argument("--gradient_accumulation_steps", type=int, default=2)
    train.add_argument("--learning_rate", type=float, default=2e-5)
    train.add_argument("--weight_decay", type=float, default=0.01)
    train.add_argument("--warmup_ratio", type=float, default=0.1)
    train.add_argument("--num_workers", type=int, default=0,
                       help="Deprecated compatibility flag; tokenization is batched in-process.")
    train.add_argument("--tokenization_batch_size", type=int, default=2048)
    train.add_argument("--bootstrap_repetitions", type=int, default=500)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--log_every", type=int, default=50)
    train.add_argument("--hf_token")
    train.add_argument("--bf16", action="store_true", default=False,
                       help="Experimental for DeBERTa; FP32+TF32 is the stable default.")
    train.add_argument("--no_bf16", action="store_false", dest="bf16",
                       help=argparse.SUPPRESS)
    train.add_argument("--overwrite", action="store_true")

    score = commands.add_parser("score-generations")
    score.add_argument("--classifier_root", type=Path, required=True)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument(
        "--include_method", action="append", default=[],
        help="Score only this post-alias method; repeat for multiple methods.",
    )
    score.add_argument(
        "--update_existing", action="store_true",
        help=("Replace --include_method records in an existing scored JSONL and "
              "recompute summaries while preserving every other method."),
    )
    score.add_argument("--generation_json_loading", choices=("auto", "eager", "stream"),
                       default="auto",
                       help="auto streams large files; eager parses one whole source in RAM; "
                            "stream forces bounded-memory parsing.")
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--scored_generations_jsonl", type=Path)
    score.add_argument("--batch_size", type=int, default=128)
    score.add_argument("--num_workers", type=int, default=0,
                       help="Deprecated compatibility flag; tokenization is batched in-process.")
    score.add_argument("--tokenization_batch_size", type=int, default=2048)
    score.add_argument("--inference_log_every", type=int, default=25)
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--bootstrap_repetitions", type=int, default=200)
    score.add_argument("--seed", type=int, default=42)
    score.add_argument("--hf_token")
    score.add_argument("--no_bf16", action="store_false", dest="bf16", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if args.command == "train":
        specs = parse_axis_specs(args.axis_dataset)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        excluded = validation_prompt_set(args.validation_prompt_jsonl)
        LOG.info("Loaded %d normalized generation-validation prompts for exclusion", len(excluded))
        set_seed(args.seed)
        device = torch.device(args.device)
        results = []
        for spec in specs:
            existing_metadata = args.output_dir / spec.axis / "classifier_metadata.json"
            if existing_metadata.exists() and not args.overwrite:
                LOG.info("Reusing completed classifier for axis %s from %s",
                         spec.axis, existing_metadata.parent)
                results.append(json.loads(existing_metadata.read_text()))
                continue
            LOG.info("Training classifier for axis %s from %s", spec.axis, spec.dataset)
            results.append(train_axis_classifier(spec, args, excluded, device))
        index = {
            "format": "deberta_v3_axis_classifiers",
            "validation_prompt_jsonl": [str(path) for path in args.validation_prompt_jsonl],
            "excluded_prompt_count": len(excluded),
            "classifiers": results,
        }
        (args.output_dir / "classifier_index.json").write_text(json.dumps(index, indent=2) + "\n")
        LOG.info("Saved classifier index to %s", args.output_dir / "classifier_index.json")
    else:
        payload = score_generations(args)
        json_dump(args.output, payload)
        LOG.info("Saved generation classifier report to %s", args.output)


if __name__ == "__main__":
    main()
