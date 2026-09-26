"""Dataset loading, validation, tokenization, and batching for steering training.

Every configured Hugging Face dataset owns one steering axis. A source row has
``original_messages`` at s=0 and ``messages`` at s=1. Failed generations are
excluded. This module intentionally does not read axis names or severities from
source datasets.
"""
from __future__ import annotations

import json
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import PreTrainedTokenizerBase

from config import AxisDataset, TrainingConfig

logger = logging.getLogger(__name__)


def _json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _is_failed(row: Dict[str, Any]) -> bool:
    """Accept both historical Hub status conventions."""
    if str(row.get("_status", "")).strip().lower() == "failed":
        return True
    value = row.get("generation_failed", False)
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _turn_terminator_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """Resolve the terminator actually emitted by a completed assistant turn.

    Gemma, Qwen, and Llama expose named end-of-turn tokens. Mistral Instruct
    instead closes assistant turns with the ordinary EOS token. The fallback is
    accepted only when the tokenizer's own chat template demonstrably emits
    that EOS in a completed assistant target.
    """
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    for token in ("<turn|>", "<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unknown_id:
            return token_id
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = ([eos] if isinstance(eos, int)
               else list(eos) if isinstance(eos, (list, tuple)) else [])
    if eos_ids and _has_chat_template(tokenizer):
        history = [{"role": "user", "content": "x"}]
        completed_chat = history + [{"role": "assistant", "content": "y"}]
        try:
            prefix = tokenizer.apply_chat_template(
                history, tokenize=True, add_generation_prompt=True
            )
            completed = tokenizer.apply_chat_template(
                completed_chat, tokenize=True, add_generation_prompt=False
            )
            if hasattr(prefix, "keys") and "input_ids" in prefix:
                prefix = prefix["input_ids"]
            if hasattr(completed, "keys") and "input_ids" in completed:
                completed = completed["input_ids"]
            if isinstance(prefix, torch.Tensor):
                prefix = prefix.tolist()
            if isinstance(completed, torch.Tensor):
                completed = completed.tolist()
            prefix, completed = list(prefix), list(completed)
            target = completed[len(prefix):] if completed[:len(prefix)] == prefix else completed
            for token_id in eos_ids:
                if token_id in target:
                    return int(token_id)
        except (TypeError, ValueError, KeyError, IndexError):
            pass
    raise ValueError("Chat tokenizer has no recognized assistant turn terminator")


def _template_name(tokenizer: PreTrainedTokenizerBase) -> str:
    return str(getattr(tokenizer, "name_or_path", "")).lower()


def _has_chat_template(tokenizer: PreTrainedTokenizerBase) -> bool:
    return bool(getattr(tokenizer, "chat_template", None) or
                getattr(tokenizer, "default_chat_template", None))


def _needs_prefix_spans(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Use rendered-prefix spans for templates whose mask support is unreliable.

    Gemma 4 has no generation blocks, and Qwen VL's assistant mask behavior
    varies across Transformers releases. Prefix boundaries are deterministic
    and, importantly, let us explicitly retain the Qwen ``<|im_end|>`` token.
    """
    name = _template_name(tokenizer)
    return "gemma-4" in name or "qwen" in name


def _template_message(message: Dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """Normalize only the templates that require content blocks."""
    if "gemma-4" not in _template_name(tokenizer):
        return dict(message)
    content = message.get("content", "")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    return {**message, "content": content}


def _encode_conversation(messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizerBase,
                         max_seq_len: int) -> Tuple[List[int], List[int], List[int]]:
    """Tokenize a chat and supervise every assistant response.

    Prefer the tokenizer's assistant-token mask.  Inferring assistant spans by
    comparing repeatedly rendered chat prefixes is brittle: Gemma 4's template
    has generation-control tokens whose prefix rendering is not byte-identical
    to a completed conversation.
    """
    if not messages:
        return [], [], []

    def template_ids(chat: List[Dict[str, Any]], *, add_generation_prompt: bool) -> List[int]:
        """Normalize Transformers 4's list and Transformers 5's dict result."""
        rendered = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=add_generation_prompt
        )
        if hasattr(rendered, "keys") and "input_ids" in rendered:
            rendered = rendered["input_ids"]
        if isinstance(rendered, torch.Tensor):
            rendered = rendered.tolist()
        return list(rendered)

    # Gemma 4's multimodal chat template expects each textual message as a
    # content block. The ChapAF datasets use the conventional string form.
    # Without this conversion Gemma silently renders an almost empty prompt.
    template_messages = messages
    if "gemma-4" in _template_name(tokenizer):
        template_messages = [
            {
                **message,
                "content": ([{"type": "text", "text": message.get("content", "")}]
                            if isinstance(message.get("content"), str) else message.get("content", [])),
            }
            for message in messages
        ]
    has_template = _has_chat_template(tokenizer)
    if has_template:
        is_gemma4 = "gemma-4" in _template_name(tokenizer)
        try:
            if _needs_prefix_spans(tokenizer):
                # These templates either have no reliable generation blocks or
                # differ across Transformers versions. Prefix spans below are
                # also what lets us include the real assistant terminator.
                raise ValueError("Use deterministic prefix-based assistant spans")
            rendered = tokenizer.apply_chat_template(
                template_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            ids = rendered["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            ids = list(ids)
            assistant_mask = rendered.get("assistant_masks")
            if assistant_mask is None:
                raise ValueError("chat template did not return assistant_masks")
            if any(message.get("role") == "assistant" for message in messages) and not any(assistant_mask):
                raise ValueError("chat template returned an empty assistant mask")
            labels = [token_id if is_assistant else -100
                      for token_id, is_assistant in zip(ids, assistant_mask)]
        except (TypeError, ValueError, KeyError):
            # Compatibility fallback for templates without mask support.
            ids = template_ids(template_messages, add_generation_prompt=False)
            labels = [-100] * len(ids)
            if is_gemma4:
                # Gemma 4's current canonical template has no generation
                # blocks, but its completed assistant turns *are* exact token
                # prefixes of the final transcript. This labels the entire
                # assistant span, including its <turn|> terminator, without
                # fragile standalone-text token matching.
                for index, message in enumerate(template_messages):
                    if message.get("role") != "assistant":
                        continue
                    prefix = template_ids(template_messages[:index], add_generation_prompt=True)
                    ending = template_ids(template_messages[:index + 1], add_generation_prompt=False)
                    if ids[:len(prefix)] != prefix or ids[:len(ending)] != ending:
                        raise ValueError("Gemma chat-template turn boundary did not align")
                    labels[len(prefix):len(ending)] = ids[len(prefix):len(ending)]
            else:
                for index, message in enumerate(template_messages):
                    if message.get("role") != "assistant":
                        continue
                    prefix = template_ids(template_messages[:index], add_generation_prompt=True)
                    ending = template_ids(template_messages[:index + 1], add_generation_prompt=False)
                    start = len(prefix) if ids[:len(prefix)] == prefix else len(prefix) + 1
                    end = min(len(ending) if ids[:len(ending)] == ending else len(ending) + 1, len(ids))
                    for token_index in range(start, end):
                        labels[token_index] = ids[token_index]
    else:
        ids, labels = [], []
        for index, message in enumerate(messages):
            text = f"{message.get('role', '')}: {message.get('content', '')}\n\n"
            token_ids = tokenizer.encode(text, add_special_tokens=index == 0)
            ids.extend(token_ids)
            labels.extend(token_ids if message.get("role") == "assistant" else [-100] * len(token_ids))
    ids, labels = ids[:max_seq_len], labels[:max_seq_len]
    if not any(label != -100 for label in labels):
        return [], [], []
    return ids, [1] * len(ids), labels


def _encode_assistant_turns(
    messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizerBase, max_seq_len: int,
) -> Dict[int, Tuple[List[int], List[int], List[int]]]:
    """Build one Gemma training sequence per completed assistant turn.

    Standard chat SFT keeps the completion and its real turn terminator intact,
    dropping old prompt context first.  Truncating a complete multi-turn chat
    from the right can otherwise cut an assistant response before ``<turn|>``
    and teaches no turn-closing behaviour.
    """
    has_template = _has_chat_template(tokenizer)
    if not has_template:
        encoded = _encode_conversation(messages, tokenizer, max_seq_len)
        return {0: encoded} if encoded[0] else {}

    template_messages: List[Dict[str, Any]] = [
        _template_message(message, tokenizer) for message in messages
    ]

    def template_ids(chat: List[Dict[str, Any]], *, add_generation_prompt: bool) -> List[int]:
        rendered = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=add_generation_prompt
        )
        if hasattr(rendered, "keys") and "input_ids" in rendered:
            rendered = rendered["input_ids"]
        if isinstance(rendered, torch.Tensor):
            rendered = rendered.tolist()
        return list(rendered)

    encoded_turns = {}
    for index, message in enumerate(template_messages):
        if message.get("role") != "assistant":
            continue

        prefix = template_ids(template_messages[:index], add_generation_prompt=True)
        completed = template_ids(template_messages[:index + 1], add_generation_prompt=False)
        if completed[:len(prefix)] != prefix:
            raise ValueError("Gemma chat-template assistant boundary did not align")

        target = completed[len(prefix):]
        # Keep the model's actual assistant turn terminator in the target.
        # This is essential for Qwen's <|im_end|> as well as Gemma's turn
        # token; never fabricate one after truncation.
        if _needs_prefix_spans(tokenizer):
            try:
                terminator_id = _turn_terminator_id(tokenizer)
                target = target[:target.index(terminator_id) + 1]
            except (ValueError, IndexError):
                continue
        # Do not fabricate an EOS after a clipped response.  A response that
        # cannot fit with its real <turn|> is excluded from this SFT dataset.
        if not target or len(target) > max_seq_len:
            continue

        context_budget = max_seq_len - len(target)
        context = prefix[-context_budget:] if context_budget else []
        input_ids = context + target
        labels = [-100] * len(context) + target
        encoded_turns[index] = (input_ids, [1] * len(input_ids), labels)
    return encoded_turns


def _encode_paired_assistant_turns(
    original_messages: List[Dict[str, str]],
    steered_messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    terminator_id: Optional[int] = None,
) -> Dict[int, Tuple[Tuple[List[int], List[int], List[int]],
                      Tuple[List[int], List[int], List[int]]]]:
    """Encode each endpoint against its rollout-consistent conversation history.

    Later transformed replies were authored after earlier transformed replies,
    not after the original assistant history.  Conditioning both endpoints on
    neutral history creates semantically mismatched training examples and a
    train/inference discrepancy.  Each side therefore keeps its own history
    and its own maximum available context budget.
    """
    has_template = _has_chat_template(tokenizer)
    if not has_template:
        pos = _encode_assistant_turns(original_messages, tokenizer, max_seq_len)
        neg = _encode_assistant_turns(steered_messages, tokenizer, max_seq_len)
        return {index: (pos[index], neg[index]) for index in sorted(pos.keys() & neg.keys())}

    if len(original_messages) != len(steered_messages):
        return {}
    if any(a.get("role") != b.get("role")
           for a, b in zip(original_messages, steered_messages)):
        return {}

    def template_ids(chat: List[Dict[str, Any]], *, add_generation_prompt: bool) -> List[int]:
        rendered = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=add_generation_prompt
        )
        if hasattr(rendered, "keys") and "input_ids" in rendered:
            rendered = rendered["input_ids"]
        if isinstance(rendered, torch.Tensor):
            rendered = rendered.tolist()
        return list(rendered)

    if terminator_id is None:
        terminator_id = _turn_terminator_id(tokenizer)

    original = [_template_message(message, tokenizer) for message in original_messages]
    steered = [_template_message(message, tokenizer) for message in steered_messages]
    encoded_pairs = {}
    for index, (pos_message, neg_message) in enumerate(zip(original, steered)):
        if pos_message.get("role") != "assistant":
            continue

        encoded = []
        for history, target_message in (
            (original[:index], pos_message),
            (steered[:index], neg_message),
        ):
            context_without_header = template_ids(history, add_generation_prompt=False)
            prefix = template_ids(history, add_generation_prompt=True)
            if prefix[:len(context_without_header)] != context_without_header:
                raise ValueError("Generation header did not extend the rendered history")
            generation_header_len = len(prefix) - len(context_without_header)
            completed = template_ids(history + [target_message], add_generation_prompt=False)
            if completed[:len(prefix)] != prefix:
                raise ValueError("Chat-template assistant boundary did not align")
            target = completed[len(prefix):]
            try:
                # Supervise the real terminator, but not template whitespace
                # emitted after it.
                target = target[:target.index(terminator_id) + 1]
            except ValueError as exc:
                raise ValueError("Assistant target has no turn terminator") from exc
            context_budget = max_seq_len - len(target)
            if not target or context_budget < generation_header_len:
                encoded = []
                break
            context = prefix[-context_budget:] if context_budget else []
            input_ids = context + target
            encoded.append((input_ids, [1] * len(input_ids), [-100] * len(context) + target))
        if len(encoded) == 2:
            encoded_pairs[index] = (encoded[0], encoded[1])
    return encoded_pairs


def _map_axis_rows(batch: Dict[str, List[Any]], *, tokenizer: PreTrainedTokenizerBase,
                   max_seq_len: int, axis_index: int,
                   tokenization_schema_version: int = 4) -> Dict[str, List[Any]]:
    # Included in Dataset.map's fingerprint so corrected tokenization can never
    # silently reuse a cache produced by the pre-fix pairing implementation.
    del tokenization_schema_version
    output = {key: [] for key in ("pos_input_ids", "pos_attention_mask", "pos_labels",
                                  "neg_input_ids", "neg_attention_mask", "neg_labels",
                                  "dim_idx")}
    count = len(batch["original_messages"])
    statuses = batch.get("_status", [None] * count)
    generation_failed = batch.get("generation_failed", [False] * count)
    has_template = _has_chat_template(tokenizer)
    terminator_id = _turn_terminator_id(tokenizer) if has_template else None
    for original, steered, status, failed in zip(
        batch["original_messages"], batch["messages"], statuses, generation_failed
    ):
        if _is_failed({"_status": status, "generation_failed": failed}):
            continue
        zero, one = _json(original), _json(steered)
        zero = zero if isinstance(zero, list) else []
        one = one if isinstance(one, list) else []
        paired_turns = _encode_paired_assistant_turns(
            zero, one, tokenizer, max_seq_len, terminator_id=terminator_id
        )
        if not paired_turns:
            continue
        for turn_index in sorted(paired_turns):
            pos, neg = paired_turns[turn_index]
            for prefix, encoded in (("pos", pos), ("neg", neg)):
                output[f"{prefix}_input_ids"].append(encoded[0])
                output[f"{prefix}_attention_mask"].append(encoded[1])
                output[f"{prefix}_labels"].append(encoded[2])
            output["dim_idx"].append(axis_index)
    return output


class SteeringDataset(Dataset):
    def __init__(self, hf_dataset: Any, tokenizer: PreTrainedTokenizerBase, cfg: TrainingConfig,
                 axis_index: int):
        workers = max(1, min(cfg.tokenization_num_proc, os.cpu_count() or 1))
        logger.info("Tokenizing axis %d with %d dataset-map workers", axis_index, workers)
        self.mapped_ds = hf_dataset.map(
            _map_axis_rows, batched=True, remove_columns=hf_dataset.column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_seq_len": cfg.max_seq_len,
                "axis_index": axis_index,
                "tokenization_schema_version": 3,
            },
            num_proc=workers, desc="Tokenizing steering pairs",
        ).with_format("torch")

    def __len__(self) -> int:
        return len(self.mapped_ds)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.mapped_ds[index]
        return {key: item[key].to(torch.long) for key in item}


class CombinedSteeringDataset(Dataset):
    def __init__(self, datasets: List[SteeringDataset]):
        self.datasets, self.offsets = datasets, []
        total = 0
        for dataset in datasets:
            self.offsets.append(total)
            total += len(dataset)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        for dataset, offset in reversed(list(zip(self.datasets, self.offsets))):
            if index >= offset:
                return dataset[index - offset]
        raise IndexError(index)


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    def pad(values: List[torch.Tensor], value: int) -> torch.Tensor:
        result = torch.full((len(values), max(v.size(0) for v in values)), value, dtype=torch.long)
        for i, tensor in enumerate(values):
            result[i, :tensor.size(0)] = tensor
        return result
    return {
        "pos_input_ids": pad([b["pos_input_ids"] for b in batch], pad_token_id),
        "pos_attention_mask": pad([b["pos_attention_mask"] for b in batch], 0),
        "pos_labels": pad([b["pos_labels"] for b in batch], -100),
        "neg_input_ids": pad([b["neg_input_ids"] for b in batch], pad_token_id),
        "neg_attention_mask": pad([b["neg_attention_mask"] for b in batch], 0),
        "neg_labels": pad([b["neg_labels"] for b in batch], -100),
        "dim_idx": torch.stack([b["dim_idx"] for b in batch]),
    }


def build_dataloader(cfg: TrainingConfig, tokenizer: PreTrainedTokenizerBase,
                     hf_token: Optional[str] = None
                     ) -> Tuple[DataLoader, Optional[DataLoader], List[str]]:
    """Build axis-balanced training data and a deterministic held-out loader."""
    from datasets import load_dataset

    if not cfg.axis_datasets:
        raise ValueError("At least one --axis_dataset AXIS=DATASET is required.")
    axis_names = [spec.axis for spec in cfg.axis_datasets]
    if len(set(axis_names)) != len(axis_names):
        raise ValueError(f"Axis names must be unique: {axis_names}")
    loaded = []
    validation_loaded = []
    for index, spec in enumerate(cfg.axis_datasets):
        kwargs = {"token": hf_token} if hf_token else {}
        logger.info("Loading axis %r from dataset %r", spec.axis, spec.dataset)
        raw = load_dataset(spec.dataset, split=cfg.dataset_split, **kwargs)
        required_columns = {"original_messages", "messages"}
        missing_columns = required_columns - set(raw.column_names)
        if missing_columns:
            raise ValueError(f"{spec.dataset} is missing required columns: {sorted(missing_columns)}")
        if cfg.max_samples is not None:
            raw = raw.select(range(min(len(raw), cfg.max_samples)))
        validation_rows = min(cfg.validation_samples_per_axis, max(0, len(raw) - 1))
        if validation_rows:
            shuffled = raw.shuffle(seed=cfg.seed + index)
            validation_raw = shuffled.select(range(validation_rows))
            training_raw = shuffled.select(range(validation_rows, len(shuffled)))
        else:
            validation_raw = None
            training_raw = raw
        dataset = SteeringDataset(training_raw, tokenizer, cfg, index)
        if len(dataset) == 0:
            raise ValueError(
                f"Axis {spec.axis!r} produced zero valid pairs from "
                "original_messages (s=0) and messages (s=1)."
            )
        if len(dataset) < cfg.min_samples_per_type:
            raise ValueError(f"Axis {spec.axis!r} has only {len(dataset)} valid s=0/s=1 pairs; "
                             f"need {cfg.min_samples_per_type}.")
        logger.info("Axis %r: %d valid pairs", spec.axis, len(dataset))
        loaded.append(dataset)
        if validation_raw is not None:
            validation_dataset = SteeringDataset(validation_raw, tokenizer, cfg, index)
            if len(validation_dataset):
                # A single conversation row may expand into several assistant
                # turns.  Bound validation by paired examples as well as raw
                # rows so the periodic pass has predictable cost.
                if len(validation_dataset) > cfg.validation_samples_per_axis:
                    validation_dataset.mapped_ds = validation_dataset.mapped_ds.select(
                        range(cfg.validation_samples_per_axis)
                    )
                validation_loaded.append(validation_dataset)
    cfg.signal_dim = len(axis_names)
    combined = CombinedSteeringDataset(loaded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # Sample axes uniformly, rather than letting a larger source dataset
    # dominate the sum-over-axes objective in THEORY.md.
    weights = torch.cat([torch.full((len(dataset),), 1.0 / len(dataset)) for dataset in loaded])
    sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)
    loader = DataLoader(combined, batch_size=cfg.batch_size, sampler=sampler,
                        collate_fn=partial(collate_fn, pad_token_id=pad_id),
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    logger.info("DataLoader ready: %d pairs across %d axes", len(combined), cfg.signal_dim)
    validation_loader = None
    if validation_loaded:
        validation = CombinedSteeringDataset(validation_loaded)
        validation_loader = DataLoader(
            validation, batch_size=1, shuffle=False,
            collate_fn=partial(collate_fn, pad_token_id=pad_id),
            num_workers=0, pin_memory=True, drop_last=False,
        )
        logger.info("Validation DataLoader ready: %d held-out pairs", len(validation))
    return loader, validation_loader, axis_names
