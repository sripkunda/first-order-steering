"""Select and save per-axis CAA-optimal residual layers from independent LoRAs."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .common import json_dump, load_independent_context, read_prompts
from .steering_baselines import standardized_mean_separation
from .steering_comparison import _endpoint_activations


def axis_dataset(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected AXIS=DATASET")
    return tuple(value.split("=", 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent_root", type=Path, required=True)
    parser.add_argument("--axis_dataset", type=axis_dataset, action="append", required=True)
    parser.add_argument("--prompts_file", type=Path, required=True)
    parser.add_argument("--fit_prompts", type=int, default=80)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    axes = [axis for axis, _dataset in args.axis_dataset]
    ctx, baseline = load_independent_context(
        args.independent_root, axes, args.axis_dataset, args.device,
        args.max_prompt_tokens, "sdpa",
    )
    prompts = read_prompts(args.prompts_file, args.fit_prompts, args.seed)
    neutral, positive = _endpoint_activations(
        ctx, baseline, prompts, args.batch_size, args.max_prompt_tokens,
    )
    count = len(next(iter(neutral.values())))
    validation_count = max(1, round(args.validation_fraction * count))
    training_count = count - validation_count
    if training_count < 2:
        raise ValueError("CAA calibration needs at least two training prompts")
    by_axis = {}
    for axis_index, axis in enumerate(axes):
        scores = {
            layer: standardized_mean_separation(
                positive[axis_index][layer][:training_count],
                neutral[layer][:training_count],
            )
            for layer in neutral
        }
        selected = max(scores, key=scores.get)
        by_axis[axis] = {
            "selected_layer": selected,
            "training_prompts": training_count,
            "validation_prompts": validation_count,
            "layer_selection_standardized_mean_separation": {
                str(layer): float(score) for layer, score in sorted(scores.items())
            },
        }
    selected_layers = {
        axis: report["selected_layer"] for axis, report in by_axis.items()
    }
    json_dump(args.output, {
        "protocol": "calibration-only CAA standardized mean separation",
        "base_model": baseline.base_model,
        "independent_root": str(baseline.root),
        "axes": axes,
        "prompt_file": str(args.prompts_file.resolve()),
        "fit_prompts": len(prompts),
        "seed": args.seed,
        "by_axis": by_axis,
        "selected_layers_by_axis": selected_layers,
        "selected_layer_union": sorted({report["selected_layer"] for report in by_axis.values()}),
    })
    json_dump(args.output.with_name("caa_layers_by_axis.json"), selected_layers)


if __name__ == "__main__":
    main()
