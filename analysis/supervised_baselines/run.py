"""Train supervised steering baselines, generate matched outputs, and score them.

LoReFT follows https://arxiv.org/abs/2404.03592 (orthogonal LoReFT);
BiPO follows algorithm 1 of https://arxiv.org/abs/2406.00045; LoFiT follows
the two-stage head selection and bias tuning in https://arxiv.org/abs/2406.01563.
The data and model paths and all experiment hyperparameters are explicit in the
JSON configuration. No LoRA weights are used by these baselines.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import logging
import math
import random
import subprocess
import sys
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from analysis.common import encode, json_dump, read_prompts
from config import AxisDataset
from dataset import _encode_paired_assistant_turns, _is_failed, _json, collate_fn
from model import _decoder_layers, _load_base_model, _text_config

LOG = logging.getLogger("supervised_baselines")


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


class LoReFT(nn.Module):
    """h + R^T(Wh+b-Rh), with R's rows constrained orthonormal."""
    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.rotation = nn.Linear(rank, hidden_size, bias=False)
        nn.init.orthogonal_(self.rotation.weight)
        nn.utils.parametrizations.orthogonal(self.rotation, "weight")
        self.source = nn.Linear(hidden_size, rank)
        with torch.no_grad():
            self.source.weight.copy_(self.rotation.weight.T)
            self.source.bias.zero_()

    def delta(self, hidden: torch.Tensor) -> torch.Tensor:
        value = hidden.float()
        rotation = self.rotation.weight
        source = self.source(value) - F.linear(value, rotation.T)
        return F.linear(source, rotation).to(hidden.dtype)


class BiPOVector(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.vector = nn.Parameter(torch.zeros(hidden_size))

    def delta(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.vector.to(dtype=hidden.dtype)


class LoFiT(nn.Module):
    """Headwise scaling for localization, then sparse attention-head offsets."""
    def __init__(self, dimensions: list[tuple[int, int]]):
        super().__init__()
        self.scales = nn.ParameterList()
        self.biases = nn.ParameterList()
        for index, (heads, width) in enumerate(dimensions):
            self.scales.append(nn.Parameter(torch.randn(heads, width) * 0.001))
            self.biases.append(nn.Parameter(torch.randn(heads, width) * 0.001))
            self.register_buffer(f"mask_{index}", torch.zeros(heads, 1))
        self.stage = "selection"

    def delta(self, layer: int, value: torch.Tensor) -> torch.Tensor:
        heads, width = self.scales[layer].shape
        if value.shape[-1] != heads * width:
            raise ValueError(f"LoFiT head shape mismatch at layer {layer}: {value.shape[-1]} != {heads}*{width}")
        if self.stage == "selection":
            shaped = value.reshape(*value.shape[:-1], heads, width)
            return (shaped * self.scales[layer].to(value.dtype)).reshape_as(value)
        offset = self.biases[layer] * getattr(self, f"mask_{layer}")
        return offset.reshape(-1).to(value.dtype)

    def select_heads(self, fraction: float) -> list[tuple[int, int]]:
        scores = torch.cat([value.detach().norm(dim=-1) for value in self.scales])
        count = max(1, math.ceil(len(scores) * fraction))
        chosen = set(torch.topk(scores, count).indices.cpu().tolist())
        selected = []
        offset = 0
        for layer, scale in enumerate(self.scales):
            mask = getattr(self, f"mask_{layer}")
            for head in range(scale.shape[0]):
                if offset + head in chosen:
                    mask[head, 0] = 1
                    selected.append((layer, head))
            offset += scale.shape[0]
        self.stage = "bias"
        return selected


class Hooks:
    def __init__(self, layers, method: str, interventions: list[nn.Module], layer: int):
        self.handles = []
        self.interventions = interventions
        self.direction = [0.0] * len(interventions)
        self.method = method
        if method == "lofit":
            for index, block in enumerate(layers):
                self.handles.append(block.self_attn.o_proj.register_forward_pre_hook(self._head_hook(index)))
        else:
            self.handles.append(layers[layer].register_forward_hook(self._residual_hook))

    def _residual_hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden
        for coefficient, intervention in zip(self.direction, self.interventions):
            if coefficient:
                edited = edited + coefficient * intervention.delta(hidden)
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    def _head_hook(self, layer: int):
        def apply(_module, args):
            value = args[0]
            edited = value
            for coefficient, intervention in zip(self.direction, self.interventions):
                if coefficient:
                    edited = edited + coefficient * intervention.delta(layer, value)
            return (edited, *args[1:])
        return apply

    def close(self):
        for handle in self.handles:
            handle.remove()


def load_pairs(dataset_name: str, tokenizer, excluded: set[str], max_length: int,
               validation_fraction: float, seed: int, limit: int | None = None):
    """Keep only changed assistant answers paired with exactly the same history."""
    from datasets import load_dataset

    raw = load_dataset(dataset_name, split="train")
    pairs, dropped = [], {"failed": 0, "test_overlap": 0, "invalid_or_long": 0}
    for row_number, row in enumerate(raw):
        if _is_failed(row) or (row.get("_status") and str(row["_status"]).lower() != "success"):
            dropped["failed"] += 1
            continue
        original, positive = _json(row.get("original_messages")), _json(row.get("messages"))
        if not isinstance(original, list) or not isinstance(positive, list) or len(original) != len(positive):
            dropped["invalid_or_long"] += 1
            continue
        candidates = {normalize(str(row.get("prompt", "")))}
        candidates.update(normalize(str(message.get("content", "")))
                          for message in original + positive
                          if message.get("role") in {"user", "system"})
        if candidates & excluded:
            dropped["test_overlap"] += 1
            continue
        turn = next((i for i, (a, b) in enumerate(zip(original, positive))
                     if a.get("role") == "assistant" and b.get("role") == "assistant"
                     and a.get("content") != b.get("content") and original[:i] == positive[:i]), None)
        if turn is None:
            dropped["invalid_or_long"] += 1
            continue
        try:
            encoded = _encode_paired_assistant_turns(original[:turn + 1], positive[:turn + 1],
                                                       tokenizer, max_length).get(turn)
        except (ValueError, TypeError, KeyError):
            encoded = None
        if encoded is None:
            dropped["invalid_or_long"] += 1
            continue
        negative, target = encoded
        item = {}
        for prefix, values in (("pos", negative), ("neg", target)):
            for suffix, tensor in zip(("input_ids", "attention_mask", "labels"), values):
                item[f"{prefix}_{suffix}"] = torch.tensor(tensor, dtype=torch.long)
        item["dim_idx"] = torch.tensor(0, dtype=torch.long)
        group = hashlib.sha256(json.dumps(original[:turn], sort_keys=True).encode()).hexdigest()
        pairs.append((group, row_number, item))
        if limit and len(pairs) >= limit:
            break
        if (row_number + 1) % 2000 == 0:
            LOG.info("%s: prepared %d/%d source rows", dataset_name, row_number + 1, len(raw))
    groups = sorted({group for group, _, _ in pairs})
    random.Random(seed).shuffle(groups)
    if len(groups) < 2:
        raise ValueError(f"Not enough valid distinct training prompts in {dataset_name}: {len(groups)}")
    val_groups = set(groups[:max(1, round(len(groups) * validation_fraction))])
    train = [item for group, _, item in pairs if group not in val_groups]
    validation = [item for group, _, item in pairs if group in val_groups]
    metadata = {"dataset": dataset_name, "training_pairs": len(train),
                "validation_pairs": len(validation), "dropped": dropped,
                "validation_prompt_sha256": sorted(val_groups),
                "pair_definition": "first differing assistant response with identical prior conversation"}
    if not train or not validation:
        raise ValueError(f"Empty train/validation split for {dataset_name}")
    return train, validation, metadata


def sequence_log_prob(model, batch, prefix: str):
    ids = batch[f"{prefix}_input_ids"].cuda()
    mask = batch[f"{prefix}_attention_mask"].cuda()
    labels = batch[f"{prefix}_labels"].cuda()
    outputs = model(input_ids=ids, attention_mask=mask, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    safe = targets.masked_fill(~valid, 0)
    chosen = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    token_log_prob = chosen - torch.logsumexp(logits, dim=-1)
    return (token_log_prob * valid).sum(-1), valid.sum(-1)


def loss_for_batch(model, hooks: Hooks, method: str, batch: dict, beta: float,
                   direction: int = 1):
    if method == "bipo":
        hooks.direction = [0.0]
        with torch.no_grad():
            reference_target, _ = sequence_log_prob(model, batch, "neg")
            reference_opposite, _ = sequence_log_prob(model, batch, "pos")
        hooks.direction = [float(direction)]
        target, _ = sequence_log_prob(model, batch, "neg")
        opposite, _ = sequence_log_prob(model, batch, "pos")
        margin = direction * beta * ((target - reference_target) - (opposite - reference_opposite))
        return -F.logsigmoid(margin).mean()
    hooks.direction = [1.0]
    target, token_count = sequence_log_prob(model, batch, "neg")
    return -target.sum() / token_count.sum().clamp_min(1)


def train_axis(model, layers, method: str, layer: int, intervention: nn.Module,
               train, validation, settings: dict, steps: int, checkpoint: Path, axis: str):
    device = next(model.parameters()).device
    intervention.to(device)
    hooks = Hooks(layers, method, [intervention], layer)
    pad_id = settings["pad_token_id"]
    collator = lambda batch: collate_fn(batch, pad_token_id=pad_id)
    generator = torch.Generator().manual_seed(settings["seed"])
    loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True,
                        collate_fn=collator, generator=generator)
    val_loader = DataLoader(validation[:settings["validation_examples"]],
                            batch_size=settings["batch_size"], shuffle=False, collate_fn=collator)
    phases = ("selection", "bias") if method == "lofit" else ("train",)
    history = []
    try:
        for phase in phases:
            # Optional total-budget interpretation: split LoFiT's optimizer
            # steps between its selection and bias phases.
            phase_steps = steps
            if method == 'lofit' and settings.get('lofit_steps_are_total', False):
                phase_steps = steps // 2 if phase == 'selection' else steps - steps // 2
                if phase_steps < 1:
                    raise ValueError('LoFiT needs at least two total optimizer steps')
            if method == "lofit":
                intervention.stage = phase
                params = intervention.scales if phase == "selection" else intervention.biases
                learning_rate = settings[f"lofit_{'selection' if phase == 'selection' else 'bias'}_learning_rate"]
            else:
                params = intervention.parameters()
                learning_rate = settings[f"{method}_learning_rate"]
            params = list(params)
            optimizer = torch.optim.AdamW(params, lr=learning_rate,
                                          weight_decay=settings.get("bipo_weight_decay", 0.0) if method == "bipo" else 0.0)
            iterator = iter(loader)
            best_loss, best_state, best_step = float("inf"), None, None
            for step in range(1, phase_steps + 1):
                intervention.zero_grad(set_to_none=True)
                total = 0.0
                for _ in range(settings["gradient_accumulation_steps"]):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        batch = next(iterator)
                    sign = random.choice((-1, 1)) if method == "bipo" else 1
                    loss = loss_for_batch(model, hooks, method, batch, settings["bipo_beta"], sign)
                    if method == "lofit" and phase == "selection":
                        loss = loss + settings["lofit_l1_weight"] * sum(x.abs().sum() for x in intervention.scales)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {method}/{axis}/{phase} loss at step {step}")
                    total += float(loss.detach())
                    (loss / settings["gradient_accumulation_steps"]).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                if step == 1 or step % 10 == 0:
                    LOG.info("%s axis=%s phase=%s step=%d/%d loss=%.5f", method, axis, phase,
                             step, phase_steps, total / settings["gradient_accumulation_steps"])
                if step % settings["validation_every"] == 0 or step == phase_steps:
                    losses = []
                    with torch.no_grad():
                        for batch in val_loader:
                            if method == "bipo":
                                val = (loss_for_batch(model, hooks, method, batch, settings["bipo_beta"], 1)
                                       + loss_for_batch(model, hooks, method, batch, settings["bipo_beta"], -1)) / 2
                            else:
                                val = loss_for_batch(model, hooks, method, batch, settings["bipo_beta"])
                            losses.append(float(val))
                    val_loss = sum(losses) / len(losses)
                    LOG.info("%s axis=%s phase=%s step=%d validation_loss=%.5f", method, axis, phase, step, val_loss)
                    history.append({"phase": phase, "step": step, "validation_loss": val_loss})
                    if math.isfinite(val_loss) and val_loss < best_loss:
                        best_loss, best_step = val_loss, step
                        best_state = {k: v.detach().cpu().clone() for k, v in intervention.state_dict().items()}
                        torch.save({"state_dict": best_state, "phase": phase, "step": step}, checkpoint)
            if best_state is None:
                raise RuntimeError(f"No valid {method}/{axis}/{phase} checkpoint")
            intervention.load_state_dict(best_state)
            LOG.info("%s axis=%s phase=%s selected_step=%d validation_loss=%.5f", method, axis, phase, best_step, best_loss)
            if method == "lofit" and phase == "selection":
                selected = intervention.select_heads(settings["lofit_head_fraction"])
                LOG.info("lofit axis=%s selected %d heads across %d layers", axis, len(selected), len(layers))
                (checkpoint.parent / "selected_heads.json").write_text(json.dumps(selected, indent=2))
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in intervention.state_dict().items()},
                    "method": method, "axis": axis, "history": history, "complete": True}, checkpoint)
        return history
    finally:
        hooks.close()


def generate(model, tokenizer, layers, method, interventions, layer, settings, prompts, output):
    hooks = Hooks(layers, method, interventions, layer)
    axes = [x["name"] for x in settings["axes"]]
    partial = output.with_name("generations.partial.json")
    directions = json.loads(partial.read_text())["analysis"]["directions"] if partial.exists() else {}
    try:
        for values in itertools.product((0, 1), repeat=len(axes)):
            name = "+".join(axis for axis, active in zip(axes, values) if active) or "neutral"
            if name in directions:
                examples = directions[name]["examples"]
                if (directions[name]["direction"] != list(values) or len(examples) != len(prompts)
                    or any((example["source_index"], example["prompt"]) != (prompt["source_index"], prompt["prompt"])
                           for example, prompt in zip(examples, prompts))):
                    raise ValueError(f"Partial generations do not match the evaluation protocol for {name}")
                LOG.info("%s keeping completed direction %s", method, name)
                continue
            hooks.direction = list(values)
            LOG.info("%s generating %s for %d prompts", method, name, len(prompts))
            examples = []
            for start in range(0, len(prompts), settings["generation_batch_size"]):
                chunk = prompts[start:start + settings["generation_batch_size"]]
                encoded = encode(tokenizer, [item["prompt"] for item in chunk],
                                 next(model.parameters()).device, settings["max_prompt_tokens"])
                with torch.no_grad():
                    tokens = model.generate(**encoded, max_new_tokens=settings["max_new_tokens"],
                                            do_sample=False, use_cache=True)
                responses = tokenizer.batch_decode(tokens[:, encoded["input_ids"].shape[1]:],
                                                   skip_special_tokens=True)
                examples.extend({"prompt_index": start + j, "source_index": item["source_index"],
                                 "prompt": item["prompt"], f"{method}_generation": response}
                                for j, (item, response) in enumerate(zip(chunk, responses)))
            directions[name] = {"direction": list(values), "examples": examples}
            json_dump(partial,
                      {"analysis": {"axes": axes, "directions": directions}})
        json_dump(output, {"analysis": {"axes": axes, "directions": directions,
                                        "layer": layer, "method": method,
                                        "composition": "sum of independently trained intervention deltas",
                                        "prompt_count": len(prompts)}})
    finally:
        hooks.close()


def score_and_report(model_dir: Path, classifier_root: str, method: str, seed: int):
    manifest = model_dir / "manifest.json"
    manifest.write_text(json.dumps({"sources": [{
        "path": str(model_dir / "generations.json"), "benchmark": "steering",
        "family": "activation_steering", "schema": "directions", "binary_corners_only": True,
        "include_methods": [f"{method}_generation"], "method_aliases": {f"{method}_generation": method},
    }]}, indent=2))
    scores = model_dir / "generation_scores.json"
    scored = model_dir / "generation_scores.scored.jsonl"
    subprocess.run([sys.executable, "-m", "analysis.behavior_classifier", "score-generations",
                    "--classifier_root", classifier_root, "--manifest", str(manifest),
                    "--output", str(scores), "--scored_generations_jsonl", str(scored),
                    "--generation_json_loading", "eager", "--batch_size", "256",
                    "--bootstrap_repetitions", "200", "--seed", str(seed), "--device", "cuda:0"], check=True)
    subprocess.run([sys.executable, "-m", "analysis.behavior_classifier_paper_report",
                    "--scores_json", str(scores), "--scored_generations_jsonl", str(scored),
                    "--output_dir", str(model_dir / "paper"), "--bootstrap_repetitions", "2000",
                    "--seed", str(seed)], check=True)


def run_model(method: str, spec: dict, settings: dict, output_root: Path,
              smoke: bool = False):
    model_dir = output_root / method / spec["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(settings)
    if smoke:
        settings.update({"batch_size": 1, "gradient_accumulation_steps": 1,
                         "validation_examples": 2, "validation_every": 1})
    LOG.info("Starting %s/%s: base=%s layer=%s", method, spec["name"], spec["base_model"], spec["layer"])
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(spec["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    settings["pad_token_id"] = tokenizer.pad_token_id
    model = _load_base_model(spec["base_model"], torch.bfloat16, "cuda:0", attn_implementation="sdpa")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = _decoder_layers(model)
    text_config = _text_config(model)
    hidden_size = int(text_config.hidden_size)
    heads = int(text_config.num_attention_heads)
    dimensions = []
    if method == "lofit":
        for block in layers:
            width = int(block.self_attn.o_proj.in_features)
            if width % heads:
                raise ValueError(f"Attention projection width {width} is not divisible by {heads} heads")
            dimensions.append((heads, width // heads))
    excluded = {normalize(x["prompt"]) for x in read_prompts(Path(settings["test_prompts"]))}
    model_record = {"method": method, "model": spec, "settings": settings,
                    "intervention_positions": "all tokens at selected residual layer" if method != "lofit" else "all tokens at selected attention heads across layers",
                    "smoke": smoke, "axes": {}}
    json_dump(model_dir / "run_config.json", model_record)
    interventions = []
    for axis_info in (settings["axes"][:1] if smoke else settings["axes"]):
        axis = axis_info["name"]
        axis_dir = model_dir / axis
        axis_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = axis_dir / "intervention.pt"
        if method == "reft":
            intervention = LoReFT(hidden_size, settings["rank"])
        elif method == "bipo":
            intervention = BiPOVector(hidden_size)
        else:
            intervention = LoFiT(dimensions)
        train, validation, metadata = load_pairs(axis_info["dataset"], tokenizer, excluded,
                                                   settings["max_seq_len"], settings["validation_fraction"],
                                                   settings["seed"], limit=24 if smoke else None)
        model_record["axes"][axis] = metadata
        json_dump(model_dir / "run_config.json", model_record)
        LOG.info("%s/%s/%s data: %d training, %d validation; excluded test overlaps=%d",
                 method, spec["name"], axis, len(train), len(validation), metadata["dropped"]["test_overlap"])
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not saved.get("complete") or saved.get("method") != method or saved.get("axis") != axis:
                raise FileExistsError(f"Incomplete or incompatible checkpoint, refusing to overwrite: {checkpoint}")
            intervention.load_state_dict(saved["state_dict"])
            intervention.to(next(model.parameters()).device)
            if method == "lofit":
                intervention.stage = "bias"
            LOG.info("Reusing completed intervention %s", checkpoint)
        else:
            train_axis(model, layers, method, int(spec["layer"]), intervention, train, validation,
                       settings, 1 if smoke else spec["steps"], checkpoint, axis)
        interventions.append(intervention)
        del train, validation
    if smoke:
        smoke_settings = {**settings, "axes": settings["axes"][:1], "generation_batch_size": 1,
                          "max_new_tokens": 4}
        smoke_prompts = read_prompts(Path(settings["test_prompts"]),
                                     settings["fit_prompt_count"] + 1, settings["seed"])[-1:]
        generate(model, tokenizer, layers, method, interventions, int(spec["layer"]), smoke_settings,
                 smoke_prompts, model_dir / "generations.json")
    else:
        all_prompts = read_prompts(Path(settings["test_prompts"]),
                                   settings["fit_prompt_count"] + spec["evaluation_prompts"], settings["seed"])
        prompts = all_prompts[settings["fit_prompt_count"]:]
        if len(prompts) != spec["evaluation_prompts"]:
            raise ValueError("Not enough disjoint evaluation prompts")
        if not (model_dir / "generations.json").exists():
            generate(model, tokenizer, layers, method, interventions, int(spec["layer"]), settings,
                     prompts, model_dir / "generations.json")
        else:
            LOG.info("Reusing completed generations %s", model_dir / "generations.json")
    del interventions, model, layers
    gc.collect()
    torch.cuda.empty_cache()
    if not smoke and not (model_dir / "paper" / "RESULTS.md").exists():
        score_and_report(model_dir, spec["classifier_root"], method, settings["seed"])
    LOG.info("Completed %s/%s", method, spec["name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("reft", "bipo", "lofit"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", help="Restrict this run to named models in the config.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = json.loads(args.config.read_text())
    selected = [item for item in settings["models"] if not args.models or item["name"] in args.models]
    if args.models and set(args.models) != {item["name"] for item in selected}:
        raise ValueError("One or more requested models are absent from the configuration")
    for model_spec in (selected[:1] if args.smoke else selected):
        run_model(args.method, model_spec, settings, args.output_root, args.smoke)


if __name__ == "__main__":
    main()
