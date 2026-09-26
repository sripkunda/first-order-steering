"""Generation helper for dense independent-LoRA and HeRD weight updates."""
from __future__ import annotations

import gc
from typing import Mapping, Sequence

import torch

from .common import AnalysisContext, chunked, encode
from .merging import DenseDeltaHooks


def _decode_batch(tokenizer, output: torch.Tensor, prefix: int) -> list[str]:
    return [tokenizer.decode(row[prefix:], skip_special_tokens=True) for row in output]


def _generate_merged(
    ctx: AnalysisContext, prompts: Sequence[str], updates: Mapping[str, torch.Tensor],
    batch_size: int, max_prompt_tokens: int, max_new_tokens: int, temperature: float,
    anchored_neutral: bool = False,
) -> list[str]:
    """Generate under fixed dense task-arithmetic, baseline, or HeRD updates.

    The legacy neutral-adapter flag is retained only for call-site compatibility
    and is rejected because this repository has no anchored-merger path.
    """
    if anchored_neutral:
        raise ValueError("HeRD generation has no anchored-merger neutral adapter")
    device_updates = {key: value.to(ctx.device, non_blocking=True) for key, value in updates.items()}
    generations: list[str] = []
    try:
        with DenseDeltaHooks(ctx.model, device_updates):
            for prompt_batch in chunked(list(prompts), batch_size):
                encoded = encode(ctx.tokenizer, prompt_batch, ctx.device, max_prompt_tokens)
                kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "use_cache": True}
                if temperature > 0:
                    kwargs["temperature"] = temperature
                with torch.no_grad():
                    output = ctx.model.llama.generate(**encoded, **kwargs)
                generations.extend(_decode_batch(ctx.tokenizer, output, encoded["input_ids"].shape[1]))
    finally:
        del device_updates
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return generations
