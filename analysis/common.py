from __future__ import annotations

import contextlib
import json
import random
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
FINETUNER = ROOT / "steered_finetuner"
if str(FINETUNER) not in sys.path:
    sys.path.insert(0, str(FINETUNER))

from config import AxisDataset, TrainingConfig  # noqa: E402
from model import AnchoredNeuralMergerWeightAdapter, _decoder_layers, build_model  # noqa: E402


@dataclass
class AnalysisContext:
    checkpoint: Path
    checkpoint_step: int
    axes: List[str]
    config: TrainingConfig
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype

    @property
    def adapters(self) -> List[Tuple[str, AnchoredNeuralMergerWeightAdapter]]:
        return [(key, adapter) for key, adapter in self.model.lora.all_adapters()
                if isinstance(adapter, AnchoredNeuralMergerWeightAdapter)]

    @property
    def layers(self):
        return _decoder_layers(self.model.llama)


def find_checkpoint(root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        checkpoint = explicit
    else:
        candidates = [path for path in root.iterdir()
                      if path.is_dir() and path.name.startswith("checkpoint-")]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-N directory under {root}")
        checkpoint = max(candidates, key=lambda path: int(path.name.rsplit("-", 1)[1]))
    required = ("meta.json", "intervention_types.json", "lora_adapters.pt")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint {checkpoint} is missing: {', '.join(missing)}")
    return checkpoint.resolve()


def config_from_meta(meta: Mapping, signal_dim: int) -> TrainingConfig:
    valid = {item.name for item in fields(TrainingConfig)}
    raw = {key: value for key, value in meta.get("config", {}).items() if key in valid}
    raw["signal_dim"] = signal_dim
    raw["freeze_base_model"] = True
    return TrainingConfig(**raw)


def load_context(checkpoint_root: Path, checkpoint_dir: Path | None, device_name: str,
                 attn_implementation: str = "sdpa", derivative_fp32: bool = False,
                 max_prompt_tokens: int = 512) -> AnalysisContext:
    checkpoint = find_checkpoint(checkpoint_root, checkpoint_dir)
    meta = json.loads((checkpoint / "meta.json").read_text())
    axes = json.loads((checkpoint / "intervention_types.json").read_text())
    if meta.get("config", {}).get("adapter_architecture") != "anchored_neural_merger":
        raise ValueError("The analysis library requires adapter_architecture=anchored_neural_merger.")
    cfg = config_from_meta(meta, len(axes))
    model = build_model(cfg, device=device_name, attn_implementation=attn_implementation)
    state = torch.load(checkpoint / "lora_adapters.pt", map_location="cpu", weights_only=True)
    model.lora.load_state_dict(state)
    if derivative_fp32:
        model.float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = max_prompt_tokens
    device = torch.device(device_name)
    dtype = next(model.lora.parameters()).dtype
    return AnalysisContext(checkpoint, int(meta.get("step", -1)), list(axes), cfg, model,
                           tokenizer, device, dtype)


def load_independent_context(independent_root: Path, axes: Sequence[str],
                             axis_datasets: Sequence[tuple[str, str]],
                             device_name: str, max_prompt_tokens: int = 512,
                             attn_implementation: str = "sdpa"):
    """Load a raw base model and independent adapters without a joint checkpoint."""
    from transformers import AutoTokenizer
    from .independent import load_independent_loras

    baseline = load_independent_loras(independent_root, axes)
    datasets = [AxisDataset(axis, dataset) for axis, dataset in axis_datasets]
    if [item.axis for item in datasets] != list(axes):
        raise ValueError("Axis dataset order must match the independent adapter order")
    cfg = TrainingConfig(
        base_model=baseline.base_model,
        signal_dim=len(axes),
        axis_datasets=datasets,
        adapter_architecture="anchored_neural_merger",
        lora_rank=1,
        adapter_hidden_dim=1,
        freeze_base_model=True,
        bf16=True,
    )
    model = build_model(cfg, device=device_name, attn_implementation=attn_implementation)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    baseline.validate_uniform_subset(sorted(model._adapted_linears))
    tokenizer = AutoTokenizer.from_pretrained(baseline.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = max_prompt_tokens
    ctx = AnalysisContext(
        independent_root.resolve(), -1, list(axes), cfg, model, tokenizer,
        torch.device(device_name), next(model.llama.parameters()).dtype,
    )
    return ctx, baseline


def read_prompts(path: Path, maximum: int | None = None, seed: int = 42) -> List[dict]:
    records = []
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text())
        values = payload.get("prompts", payload) if isinstance(payload, dict) else payload
    for index, item in enumerate(values):
        if isinstance(item, str):
            item = {"prompt": item}
        if isinstance(item, dict) and isinstance(item.get("prompt"), str) and item["prompt"].strip():
            records.append({**item, "source_index": item.get("source_index", index)})
    if not records:
        raise ValueError(f"No prompts found in {path}")
    random.Random(seed).shuffle(records)
    return records if maximum is None else records[:maximum]


def render_prompts(tokenizer, prompts: Sequence[str]) -> List[str]:
    if not getattr(tokenizer, "chat_template", None):
        return list(prompts)
    return [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                          add_generation_prompt=True) for prompt in prompts]


def encode(tokenizer, prompts: Sequence[str], device: torch.device,
           max_length: int | None = None) -> Dict[str, torch.Tensor]:
    rendered = render_prompts(tokenizer, prompts)
    return encode_rendered(tokenizer, rendered, device, max_length)


def encode_rendered(tokenizer, rendered: Sequence[str], device: torch.device,
                    max_length: int | None = None) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
    return {key: value.to(device) for key, value in encoded.items()}


def encode_messages(tokenizer, conversations: Sequence[Sequence[Mapping[str, str]]], device: torch.device,
                    max_length: int | None = None,
                    add_generation_prompt: bool = True) -> Dict[str, torch.Tensor]:
    if not getattr(tokenizer, "chat_template", None):
        rendered = ["\n".join(message["content"] for message in conversation)
                    for conversation in conversations]
    else:
        rendered = [tokenizer.apply_chat_template(
            conversation, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
                    for conversation in conversations]
    return encode_rendered(tokenizer, rendered, device, max_length)


def pad_token_sequences(tokenizer, sequences: Sequence[torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    features = [{"input_ids": sequence.tolist()} for sequence in sequences]
    encoded = tokenizer.pad(features, padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def signal(values: Sequence[float], device: torch.device, dtype: torch.dtype,
           batch_size: int = 1) -> torch.Tensor:
    result = torch.tensor(values, device=device, dtype=dtype).reshape(1, -1)
    return result.expand(batch_size, -1).clone()


def signal_grid(axis_count: int, levels: Sequence[float]) -> Iterator[Tuple[float, ...]]:
    import itertools
    return itertools.product(levels, repeat=axis_count)


def endpoint_combinations(axis_count: int) -> Iterator[Tuple[int, ...]]:
    import itertools
    return itertools.product((0, 1), repeat=axis_count)


@contextlib.contextmanager
def active_factors(model, factors):
    model._ctx["active"] = True
    model._ctx["precomputed"] = factors
    try:
        yield
    finally:
        model._ctx["active"] = False
        model._ctx["precomputed"] = {}


def chunked(values: Sequence, size: int) -> Iterator[Sequence]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start:start + size]


def layer_from_key(key: str) -> int:
    return int(key.split(".", 1)[0].removeprefix("layer"))


def exact_factor_frobenius_sq(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    """Exact squared Frobenius norm of scale * B @ A using small Gram matrices."""
    A, B = A.float(), B.float()
    return scale * scale * ((A @ A.T) * (B.T @ B)).sum()


def json_dump(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
