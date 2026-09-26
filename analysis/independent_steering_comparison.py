"""Run classical steering baselines from independent LoRAs, with no joint checkpoint."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .caa_layer_calibration import axis_dataset
from .common import json_dump, load_independent_context, read_prompts
from .steering_comparison import CLASSICAL_METHODS, steering_method_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent_root", type=Path, required=True)
    parser.add_argument("--axis_dataset", type=axis_dataset, action="append", required=True)
    parser.add_argument("--caa_calibration", type=Path, required=True)
    parser.add_argument("--prompts_file", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(CLASSICAL_METHODS))
    parser.add_argument("--levels", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--fit_prompts", type=int, default=80)
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--controller_validation_fraction", type=float, default=0.2)
    parser.add_argument("--activation_batch_size", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--direction_batch_size", type=int, default=8)
    parser.add_argument("--method_batch_size", type=int, default=5)
    parser.add_argument("--max_prompt_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--recontrol_hidden", type=int, default=256)
    parser.add_argument("--recontrol_epochs", type=int, default=20)
    parser.add_argument("--recontrol_batch_size", type=int, default=512)
    parser.add_argument("--recontrol_learning_rate", type=float, default=1e-3)
    parser.add_argument("--recontrol_step_size", type=float, default=0.1)
    parser.add_argument("--recontrol_iterations", type=int, default=3)
    parser.add_argument("--odesteer_features", type=int, default=8000)
    parser.add_argument("--odesteer_gamma", type=float, default=0.1)
    parser.add_argument("--odesteer_c0", type=float, default=1.0)
    parser.add_argument("--odesteer_time", type=float, default=14.0)
    parser.add_argument("--odesteer_steps", type=int, default=10)
    parser.add_argument("--odesteer_logistic_c", type=float, default=1.0)
    parser.add_argument("--odesteer_logistic_steps", type=int, default=1000)
    parser.add_argument("--odesteer_feature_batch_size", type=int, default=256)
    parser.add_argument("--odesteer_training_samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    axes = [axis for axis, _dataset in args.axis_dataset]
    ctx, _baseline = load_independent_context(
        args.independent_root, axes, args.axis_dataset, args.device,
        args.max_prompt_tokens, "sdpa",
    )
    calibration = json.loads(args.caa_calibration.read_text())
    if calibration.get("axes") != axes:
        raise ValueError("CAA calibration axis order does not match independent adapters")
    layers = calibration["selected_layers_by_axis"]
    prompts = read_prompts(
        args.prompts_file, args.fit_prompts + args.num_prompts, args.seed,
    )
    methods = [item for item in args.methods.split(",") if item]
    unsupported = sorted(set(methods) - set(CLASSICAL_METHODS))
    if unsupported:
        raise ValueError(
            "Independent steering comparison accepts classical methods only; "
            f"use compositional static generation for ours: {unsupported}"
        )
    analysis = steering_method_comparison(
        ctx=ctx, prompts=prompts, true_baseline=args.independent_root,
        methods=methods, levels=[float(item) for item in args.levels.split(",")],
        fit_prompt_count=args.fit_prompts,
        validation_fraction=args.controller_validation_fraction,
        activation_batch_size=args.activation_batch_size,
        generation_batch_size=args.generation_batch_size,
        direction_batch_size=args.direction_batch_size,
        method_batch_size=args.method_batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        profile_repetitions=0, profile_warmup=0,
        recontrol_hidden=args.recontrol_hidden,
        recontrol_epochs=args.recontrol_epochs,
        recontrol_batch_size=args.recontrol_batch_size,
        recontrol_learning_rate=args.recontrol_learning_rate,
        recontrol_step_size=args.recontrol_step_size,
        recontrol_iterations=args.recontrol_iterations,
        odesteer_features=args.odesteer_features,
        odesteer_gamma=args.odesteer_gamma,
        odesteer_c0=args.odesteer_c0,
        odesteer_time=args.odesteer_time,
        odesteer_steps=args.odesteer_steps,
        odesteer_logistic_c=args.odesteer_logistic_c,
        odesteer_logistic_steps=args.odesteer_logistic_steps,
        odesteer_feature_batch_size=args.odesteer_feature_batch_size,
        odesteer_training_samples=args.odesteer_training_samples,
        odesteer_layer="caa", caa_layers_by_axis=layers, seed=args.seed,
    )
    json_dump(args.output, {
        "independent_root": str(args.independent_root.resolve()),
        "caa_calibration": str(args.caa_calibration.resolve()),
        "analysis": analysis,
    })


if __name__ == "__main__":
    main()
