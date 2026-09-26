"""Train one conventional PEFT LoRA adapter per behavioral axis.

This trainer is deliberately limited to conventional independent LoRA adapters.  It has
no steering vector, neutral adapter, nonlinear gates, or cross-axis loss.  A
single process trains axes sequentially to keep GPU memory bounded, but each
axis has its own freshly loaded frozen base model, optimizer, scheduler,
dataset iterator, checkpoints, and Hugging Face-ready adapter directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

from config import AxisDataset, DEFAULT_AXIS_DATASETS
from dataset import SteeringDataset, collate_fn
from model import _decoder_layers, _load_base_model


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_SAFE_AXIS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DECODER_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass
class IndependentLoRAConfig:
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    axis_datasets: List[AxisDataset] = None  # populated by argparse
    dataset_split: str = "train"
    output_dir: str = "/app/output"
    max_steps: int = 500  # per axis, not shared across axes
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_seq_len: int = 512
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    save_every: int = 250
    log_every: int = 10
    num_workers: int = 4
    tokenization_num_proc: int = 16
    max_samples: Optional[int] = None
    bf16: bool = True
    seed: int = 42
    resume: bool = False
    overwrite: bool = False
    use_wandb: bool = False
    wandb_project: str = "herd-merging"
    run_name: str = "independent-lora"
    layers_to_transform: Optional[List[int]] = None


def normalize_layer_indices(values: Optional[List[int]]) -> Optional[List[int]]:
    if values is None:
        return None
    if not values:
        raise ValueError("layers_to_transform cannot be empty")
    if any(value < 0 for value in values):
        raise ValueError("layers_to_transform must contain nonnegative decoder-layer indices")
    if len(set(values)) != len(values):
        raise ValueError("layers_to_transform contains duplicate indices")
    return sorted(values)


def text_projection_module_names(model, layers: Optional[List[int]] = None) -> List[str]:
    """Return exact PEFT targets from the text decoder, never the vision tower.

    Suffix-only PEFT targets such as ``q_proj`` also match Gemma 4's vision
    projections.  Some of those are ``Gemma4ClippableLinear`` wrappers rather
    than supported ``nn.Linear`` modules.  Resolve the intended decoder
    projections by object identity and pass their complete module names to
    PEFT so matching is architecture-safe.
    """
    decoder_layers = _decoder_layers(model)
    selected = set(range(len(decoder_layers))) if layers is None else set(layers)
    invalid = sorted(selected - set(range(len(decoder_layers))))
    if invalid:
        raise ValueError(
            f"Decoder layers {invalid} do not exist; available indices are "
            f"0..{len(decoder_layers) - 1}"
        )
    names_by_id = {id(module): name for name, module in model.named_modules()}
    targets: List[str] = []
    for layer_index, decoder_layer in enumerate(decoder_layers):
        if layer_index not in selected:
            continue
        for parent, projection_names in (
            (decoder_layer.self_attn, TARGET_MODULES[:4]),
            (decoder_layer.mlp, TARGET_MODULES[4:]),
        ):
            for projection_name in projection_names:
                if not hasattr(parent, projection_name):
                    continue
                module = getattr(parent, projection_name)
                if not isinstance(module, torch.nn.Linear):
                    raise TypeError(
                        f"Text projection layer {layer_index}.{projection_name} is "
                        f"{type(module).__name__}, not torch.nn.Linear"
                    )
                name = names_by_id.get(id(module))
                if name is None:
                    raise RuntimeError(
                        f"Cannot resolve module path for layer {layer_index}.{projection_name}"
                    )
                targets.append(name)
    if not targets:
        raise RuntimeError("No supported text-decoder projections were found for PEFT")
    return targets


def setup_logger(axis: str) -> logging.Logger:
    logger = logging.getLogger(f"independent_lora.{axis}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def axis_path(root: Path, axis: str) -> Path:
    if not _SAFE_AXIS.fullmatch(axis):
        raise ValueError(
            f"Axis {axis!r} cannot be used as an output-folder name. "
            "Use letters, digits, '.', '_', or '-'."
        )
    return root / axis


def load_axis_loader(
    spec: AxisDataset,
    axis_index: int,
    tokenizer,
    cfg: IndependentLoRAConfig,
    hf_token: Optional[str],
) -> DataLoader:
    """Load one axis and retain only its transformed (s=1) training target.

    ``SteeringDataset`` is intentionally reused for its audited chat-template,
    multi-turn, truncation, and real-terminator behavior.  This ablation ignores
    the paired neutral tensors after encoding and trains only ``neg_*``.
    """
    from datasets import load_dataset

    kwargs = {"token": hf_token} if hf_token else {}
    raw = load_dataset(spec.dataset, split=cfg.dataset_split, **kwargs)
    required = {"original_messages", "messages"}
    missing = required - set(raw.column_names)
    if missing:
        raise ValueError(f"{spec.dataset} is missing required columns: {sorted(missing)}")
    if cfg.max_samples is not None:
        raw = raw.select(range(min(len(raw), cfg.max_samples)))

    # The shared formatter needs only these small configuration attributes.
    encoding_cfg = SimpleNamespace(
        max_seq_len=cfg.max_seq_len,
        tokenization_num_proc=cfg.tokenization_num_proc,
    )
    dataset = SteeringDataset(raw, tokenizer, encoding_cfg, axis_index)
    if not len(dataset):
        raise ValueError(f"Axis {spec.axis!r} produced no valid transformed assistant targets")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    generator = torch.Generator().manual_seed(cfg.seed + axis_index)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=partial(collate_fn, pad_token_id=pad_id),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def endless(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def learning_rate_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup followed by cosine decay to five percent of the LR."""
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint_dirs(axis_dir: Path) -> List[Path]:
    valid = []
    for path in axis_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        required = ("adapter_config.json", "optimizer.pt", "scheduler.pt", "training_state.json")
        if all((path / name).is_file() for name in required):
            valid.append(path)
    return sorted(valid, key=lambda path: int(path.name.removeprefix("checkpoint-")))


def save_adapter_artifact(
    model, tokenizer, target: Path, optimizer, scheduler, step: int,
    cfg: IndependentLoRAConfig, spec: AxisDataset,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    torch.save(optimizer.state_dict(), target / "optimizer.pt")
    torch.save(scheduler.state_dict(), target / "scheduler.pt")
    (target / "training_state.json").write_text(json.dumps({"step": step}, indent=2))
    peft_config = next(iter(model.peft_config.values()))
    (target / "training_config.json").write_text(json.dumps({
        **asdict(cfg),
        "axis_datasets": [{"axis": item.axis, "dataset": item.dataset} for item in cfg.axis_datasets],
        "trained_axis": spec.axis,
        "trained_dataset": spec.dataset,
        "target_modules": sorted(peft_config.target_modules),
        "layers_to_transform": cfg.layers_to_transform,
        "semantics": "independent standard LoRA trained only on transformed messages",
    }, indent=2))
    (target / "README.md").write_text(
        f"# {spec.axis} independent LoRA ablation\n\n"
        f"Base model: `{cfg.base_model}`\n\n"
        f"Dataset: `{spec.dataset}`\n\n"
        f"Decoder layers: `{cfg.layers_to_transform if cfg.layers_to_transform is not None else 'all'}`\n\n"
        "This is a conventional independently trained PEFT LoRA adapter. It has no "
        "steering vector, neutral adapter, joint residual, or cross-axis training.\n"
    )


def build_peft_model(cfg: IndependentLoRAConfig, device: torch.device, checkpoint: Optional[Path]):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    base = _load_base_model(cfg.base_model, dtype=dtype, device=str(device))
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    if checkpoint is not None:
        training_config = checkpoint / "training_config.json"
        saved = json.loads(training_config.read_text()) if training_config.is_file() else json.loads(
            (checkpoint / "adapter_config.json").read_text()
        )
        saved_layers = normalize_layer_indices(saved.get("layers_to_transform"))
        if saved_layers != cfg.layers_to_transform:
            raise ValueError(
                f"Resume layer selection mismatch: checkpoint={saved_layers}, "
                f"requested={cfg.layers_to_transform}"
            )
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    else:
        target_modules = text_projection_module_names(base, cfg.layers_to_transform)
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(base, peft_cfg)
    return model


def validate_trainable_layers(model, expected: Optional[List[int]]) -> List[int]:
    """Fail closed if PEFT attached trainable factors outside requested layers."""
    names = [name for name, parameter in model.named_parameters()
             if parameter.requires_grad and "lora_" in name]
    if not names:
        raise RuntimeError("PEFT model exposes no trainable LoRA parameters")
    actual = set()
    unscoped = []
    for name in names:
        match = _DECODER_LAYER.search(name)
        if match is None:
            unscoped.append(name)
        else:
            actual.add(int(match.group(1)))
    if expected is not None and (actual != set(expected) or unscoped):
        raise RuntimeError(
            "PEFT layer filtering failed: "
            f"expected={expected}, actual={sorted(actual)}, unscoped={unscoped[:5]}"
        )
    return sorted(actual)


def train_axis(
    cfg: IndependentLoRAConfig,
    spec: AxisDataset,
    axis_index: int,
    tokenizer,
    device: torch.device,
    hf_token: Optional[str],
) -> None:
    log = setup_logger(spec.axis)
    root = Path(cfg.output_dir)
    output = axis_path(root, spec.axis)
    existing = checkpoint_dirs(output) if output.exists() else []
    if output.exists() and any(output.iterdir()) and not (cfg.resume or cfg.overwrite):
        raise FileExistsError(
            f"{output} already contains an adapter or checkpoint. Use a fresh --output_dir, "
            "or pass --resume to continue its latest checkpoint."
        )
    if cfg.overwrite and output.exists():
        shutil.rmtree(output)
        existing = []

    checkpoint = existing[-1] if (cfg.resume and existing) else None
    if cfg.resume and output.exists() and not checkpoint:
        raise FileNotFoundError(f"--resume requested but {output} has no complete checkpoint")

    loader = load_axis_loader(spec, axis_index, tokenizer, cfg, hf_token)
    total_steps = cfg.max_steps
    model = build_peft_model(cfg, device, checkpoint)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    actual_layers = validate_trainable_layers(model, cfg.layers_to_transform)
    optimizer = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_lambda(step, cfg.warmup_steps, total_steps),
    )
    step = 0
    if checkpoint is not None:
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu"))
        scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu"))
        training_state = json.loads((checkpoint / "training_state.json").read_text())
        step = int(training_state["step"])
        log.info("Resuming axis=%s from %s at step %d", spec.axis, checkpoint, step)

    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=f"{cfg.run_name}-{spec.axis}",
                config={**asdict(cfg), "axis": spec.axis, "dataset": spec.dataset},
                reinit="finish_previous",
            )
        except ImportError:
            log.warning("wandb is not installed; continuing without it")

    log.info(
        "Independent LoRA axis=%s dataset=%s pairs=%d batches/epoch=%d "
        "step=%d/%d layers=%s trainable=%d",
        spec.axis, spec.dataset, len(loader.dataset), len(loader),
        step, total_steps, actual_layers,
        sum(parameter.numel() for parameter in trainable),
    )
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    autocast_enabled = device.type == "cuda" and cfg.bf16
    iterator = endless(loader)
    started = time.time()
    running_loss = 0.0
    running_microbatches = 0
    optimizer.zero_grad(set_to_none=True)

    while step < total_steps:
        for _ in range(cfg.gradient_accumulation_steps):
            batch = next(iterator)
            input_ids = batch["neg_input_ids"].to(device)
            attention_mask = batch["neg_attention_mask"].to(device)
            labels = batch["neg_labels"].to(device)
            with torch.amp.autocast(device.type, dtype=dtype, enabled=autocast_enabled):
                output_model = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = output_model.loss
                if loss is None:
                    raise RuntimeError("Model did not return a causal-LM loss")
                (loss / cfg.gradient_accumulation_steps).backward()
            running_loss += float(loss.detach().item())
            running_microbatches += 1

        clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0:
            average_loss = running_loss / running_microbatches
            log.info(
                "axis=%s step %d/%d loss=%.4f lr=%.3e elapsed=%ds",
                spec.axis, step, total_steps, average_loss,
                scheduler.get_last_lr()[0], time.time() - started,
            )
            if wandb_run is not None:
                wandb_run.log({"loss/train": average_loss, "lr": scheduler.get_last_lr()[0]}, step=step)
            running_loss = 0.0
            running_microbatches = 0

        # Persist the completed update before doing any optional reporting.
        if step % cfg.save_every == 0:
            save_adapter_artifact(
                model, tokenizer, output / f"checkpoint-{step}", optimizer, scheduler,
                step, cfg, spec,
            )
            log.info("Checkpoint saved → %s", output / f"checkpoint-{step}")

    # The axis root is the standalone, directly pushable PEFT adapter.
    save_adapter_artifact(
        model, tokenizer, output, optimizer, scheduler, step, cfg, spec,
    )
    log.info("Independent adapter saved → %s", output)
    if wandb_run is not None:
        wandb_run.finish()
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> IndependentLoRAConfig:
    parser = argparse.ArgumentParser(
        description="Train independent conventional PEFT LoRA adapters, one per axis."
    )
    parser.add_argument("--base_model", "--base-model", dest="base_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--axis_dataset", "--axis-dataset", dest="axis_dataset", action="append", metavar="AXIS=DATASET")
    parser.add_argument("--dataset_split", "--dataset-split", dest="dataset_split", default="train")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="/app/output")
    parser.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int, default=500,
                        help="Optimizer updates for each axis independently.")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", "--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_seq_len", "--max-seq-len", dest="max_seq_len", type=int, default=512)
    parser.add_argument("--lora_rank", "--lora-rank", dest="lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", "--lora-alpha", dest="lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", "--lora-dropout", dest="lora_dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", "--warmup-steps", dest="warmup_steps", type=int, default=50)
    parser.add_argument("--save_every", "--save-every", dest="save_every", type=int, default=250)
    parser.add_argument("--log_every", "--log-every", dest="log_every", type=int, default=10)
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--tokenization_num_proc", "--tokenization-num-proc", dest="tokenization_num_proc", type=int, default=16)
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_bf16", "--no-bf16", dest="no_bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_wandb", "--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wandb_project", "--wandb-project", dest="wandb_project", default="herd-merging")
    parser.add_argument("--run_name", "--run-name", dest="run_name", default="independent-lora")
    parser.add_argument(
        "--layers_to_transform", "--layers-to-transform", dest="layers_to_transform",
        type=int, nargs="+",
        help=("Zero-indexed decoder layers receiving LoRA factors. Omit to adapt all "
              "decoder layers."),
    )
    args = parser.parse_args()
    try:
        axis_datasets = (
            [AxisDataset(*value.split("=", 1)) for value in args.axis_dataset]
            if args.axis_dataset else list(DEFAULT_AXIS_DATASETS)
        )
    except (TypeError, ValueError) as exc:
        parser.error(f"--axis_dataset must be AXIS=DATASET: {exc}")
    if not axis_datasets or any(not item.axis or not item.dataset for item in axis_datasets):
        parser.error("At least one non-empty --axis_dataset AXIS=DATASET is required")
    if len({item.axis for item in axis_datasets}) != len(axis_datasets):
        parser.error("Axis names must be unique")
    if args.max_steps < 1 or args.gradient_accumulation_steps < 1 or args.save_every < 1:
        parser.error("max_steps, gradient_accumulation_steps, and save_every must be positive")
    if args.warmup_steps < 0:
        parser.error("warmup_steps cannot be negative")
    try:
        layers_to_transform = normalize_layer_indices(args.layers_to_transform)
    except ValueError as exc:
        parser.error(str(exc))
    return IndependentLoRAConfig(
        base_model=args.base_model,
        axis_datasets=axis_datasets,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_len=args.max_seq_len,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        num_workers=args.num_workers,
        tokenization_num_proc=args.tokenization_num_proc,
        max_samples=args.max_samples,
        bf16=not args.no_bf16,
        seed=args.seed,
        resume=args.resume,
        overwrite=args.overwrite,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        layers_to_transform=layers_to_transform,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Independent LoRA ablations require CUDA in the supplied container")
    device = torch.device("cuda")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for index, spec in enumerate(cfg.axis_datasets):
        train_axis(cfg, spec, index, tokenizer, device, hf_token)


if __name__ == "__main__":
    main()
