"""Build paper tables from DeBERTa-scored binary-corner generations.

The unit of resampling is a prompt.  Every included method must contain one
complete 2^k block for every prompt retained in its family's common prompt
intersection.  Classifier thresholds are fixed by the classifier validation
sets and are never tuned on generated responses.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orjson

from .behavior_classifier_report import display_name


LOG = logging.getLogger("behavior_classifier_paper_report")


def _prompt_key(record: dict) -> tuple:
    """Cross-file prompt identity retained by every generation schema."""
    source_index = record.get("source_index")
    if source_index is not None:
        return int(source_index), " ".join(record["prompt"].split())
    return -1, " ".join(record["prompt"].split())


def _direction(record: dict, axes: list[str]) -> tuple[int, ...] | None:
    values = []
    for axis in axes:
        value = float(record["direction"].get(axis, 0.0))
        if not (math.isclose(value, 0.0) or math.isclose(value, 1.0)):
            return None
        values.append(int(round(value)))
    return tuple(values)


def load_records(path: Path, axes: list[str]):
    groups = defaultdict(lambda: defaultdict(dict))
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = orjson.loads(line)
            direction = _direction(record, axes)
            if direction is None:
                continue
            family = str(record["family"])
            method = str(record["method"])
            prompt = _prompt_key(record)
            previous = groups[(family, method)][prompt].get(direction)
            if previous is not None:
                raise ValueError(
                    f"Duplicate generation for {family}/{method}, prompt {prompt[0]}, "
                    f"direction {direction}"
                )
            groups[(family, method)][prompt][direction] = record
    return groups


def complete_prompt_sets(groups, family: str, axes: list[str]):
    expected = {
        tuple((mask >> index) & 1 for index in range(len(axes)))
        for mask in range(1 << len(axes))
    }
    methods = sorted(method for current_family, method in groups if current_family == family)
    if not methods:
        return [], {}, set(), expected, True
    complete = {}
    for method in methods:
        complete[method] = {
            prompt for prompt, directions in groups[(family, method)].items()
            if set(directions) == expected
        }
        LOG.info(
            "%s/%s: %d prompts with a complete %d-corner block",
            family, method, len(complete[method]), len(expected),
        )
    common = set.intersection(*(complete[method] for method in methods))
    paired = bool(common)
    if paired:
        selected = {method: common for method in methods}
        LOG.info("%s: %d prompts in the complete common intersection", family, len(common))
    else:
        # Preserve already-computed judgments when a historical method used a
        # disjoint prompt slice.  The output marks the comparison unpaired;
        # it must not silently claim a paired paper comparison.
        selected = complete
        LOG.warning(
            "%s has no global prompt intersection; reporting each method on its "
            "own complete prompts as an explicitly unpaired comparison",
            family,
        )
    return methods, selected, common, expected, paired


def _bootstrap(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    if repetitions <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.shape[0], size=(repetitions, values.shape[0]))
    estimates = values[draws].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def method_metrics(records_by_prompt: dict, prompts: set[tuple], directions: set[tuple],
                   axes: list[str], thresholds: dict[str, float], bootstrap: int,
                   seed: int) -> tuple[dict, list[dict]]:
    prompt_rows = []
    combination_values = defaultdict(list)
    ordered_directions = sorted(directions)
    for prompt in sorted(prompts):
        records = records_by_prompt[prompt]
        active_by_axis = [[] for _ in axes]
        inactive_by_axis = [[] for _ in axes]
        joint, exact, nonempty = [], [], []
        all_active = all_inactive = None
        for direction in ordered_directions:
            record = records[direction]
            predicted = tuple(
                float(record["axis_scores"][axis]) >= thresholds[axis]
                for axis in axes
            )
            exact_value = float(all(int(value) == target
                                    for value, target in zip(predicted, direction)))
            exact.append(exact_value)
            combination_values[direction].append(exact_value)
            nonempty.append(float(bool(record["response"].strip())))
            for index, target in enumerate(direction):
                (active_by_axis if target else inactive_by_axis)[index].append(
                    float(predicted[index] if target else not predicted[index])
                )
            if any(direction):
                joint.append(float(all(predicted[index]
                                       for index, target in enumerate(direction) if target)))
            if all(direction):
                all_active = float(all(predicted))
            if not any(direction):
                all_inactive = float(not any(predicted))
        row = {
            **{f"axis_{index}_active": float(np.mean(values))
               for index, values in enumerate(active_by_axis)},
            **{f"axis_{index}_specificity": float(np.mean(values))
               for index, values in enumerate(inactive_by_axis)},
            "macro_active": float(np.mean([np.mean(values) for values in active_by_axis])),
            "macro_specificity": float(np.mean([np.mean(values) for values in inactive_by_axis])),
            "joint_requested": float(np.mean(joint)),
            "all_three": float(all_active),
            "all_inactive": float(all_inactive),
            "exact_combination": float(np.mean(exact)),
            "nonempty": float(np.mean(nonempty)),
        }
        prompt_rows.append(row)

    keys = list(prompt_rows[0])
    arrays = {key: np.asarray([row[key] for row in prompt_rows], dtype=np.float64) for key in keys}
    metrics = {}
    for offset, (key, values) in enumerate(arrays.items()):
        low, high = _bootstrap(values, bootstrap, seed + 1009 * offset)
        metrics[key] = {"mean": float(values.mean()), "ci95_low": low, "ci95_high": high}
    combinations = [{
        "direction": {axis: target for axis, target in zip(axes, direction)},
        "exact_combination_accuracy": float(np.mean(values)),
    } for direction, values in sorted(combination_values.items())]
    return metrics, combinations


def _flatten_row(method: str, prompts: int, metrics: dict) -> dict:
    row = {"method": display_name(method), "method_key": method, "prompts": prompts}
    for metric, values in metrics.items():
        row[metric] = values["mean"]
        row[f"{metric}_ci95_low"] = values["ci95_low"]
        row[f"{metric}_ci95_high"] = values["ci95_high"]
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _estimate(row: dict, key: str) -> str:
    mean = 100.0 * float(row[key])
    low = 100.0 * float(row[f"{key}_ci95_low"])
    high = 100.0 * float(row[f"{key}_ci95_high"])
    return f"{mean:.1f} [{low:.1f}, {high:.1f}]"


def _markdown_table(rows: list[dict], axes: list[str]) -> str:
    columns = [
        ("method", "Method"),
        *[(f"axis_{index}_active", axis.replace("_", " ").title())
          for index, axis in enumerate(axes)],
        ("macro_active", "Macro active"),
        ("macro_specificity", "Inactive specificity"),
        ("joint_requested", "Joint requested"),
        ("all_three", "All three"),
        ("exact_combination", "Exact combination"),
    ]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            str(row[key]) if key == "method" else _estimate(row, key)
            for key, _ in columns
        ) + " |")
    return "\n".join(lines)


def _latex_table(rows: list[dict], axes: list[str], caption: str, label: str) -> str:
    axis_headers = [axis.replace("_", " ").title() for axis in axes]
    headers = ["Method", *axis_headers, "Macro", "Specificity", "Joint", "All three", "Exact"]
    keys = [
        *[f"axis_{index}_active" for index in range(len(axes))],
        "macro_active", "macro_specificity", "joint_requested", "all_three",
        "exact_combination",
    ]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{l" + "c" * len(keys) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = str(row["method"]).replace("_", "\\_")
        cells = [method, *[_estimate(row, key) for key in keys]]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}. Values are percentages with prompt-bootstrap 95\\% confidence intervals.}}",
        f"\\label{{{label}}}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def _plot(rows: list[dict], output: Path, title: str) -> None:
    if not rows:
        return
    keys = ["macro_active", "macro_specificity", "joint_requested", "all_three",
            "exact_combination"]
    labels = ["Macro active", "Inactive specificity", "Joint requested", "All three", "Exact"]
    values = np.asarray([[row[key] for key in keys] for row in rows])
    figure, panel = plt.subplots(figsize=(9.5, max(3.5, 0.55 * len(rows) + 1.6)))
    image = panel.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    panel.set_xticks(range(len(labels)), labels)
    panel.set_yticks(range(len(rows)), [row["method"] for row in rows])
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            panel.text(column_index, row_index, f"{100 * value:.1f}", ha="center", va="center",
                       color="white" if value >= 0.58 else "black", fontsize=8.5)
    panel.set_title(title)
    panel.set_xlabel("Percentage; higher is better")
    figure.colorbar(image, ax=panel, fraction=0.025, pad=0.025, label="Rate")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores_json", type=Path, required=True)
    parser.add_argument("--scored_generations_jsonl", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    payload = json.loads(args.scores_json.read_text())
    axes = list(payload["classifiers"])
    thresholds = {
        axis: float(metadata["threshold"])
        for axis, metadata in payload["classifiers"].items()
    }
    predictions = args.scored_generations_jsonl or args.scores_json.with_name(
        args.scores_json.stem + ".scored.jsonl"
    )
    groups = load_records(predictions, axes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "protocol": {
            "directions": f"all {1 << len(axes)} binary corners",
            "prompt_selection": "complete within-family prompt intersection",
            "judgment": "axis-specific DeBERTa probability thresholded at validation threshold",
            "uncertainty": "nonparametric prompt-cluster bootstrap 95% interval",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "axes": axes,
            "metric_definitions": {
                "axis_i_active": "P(classifier i passes | requested s_i=1)",
                "macro_active": "unweighted mean of per-axis active pass rates",
                "macro_specificity": "unweighted mean P(classifier i absent | requested s_i=0)",
                "joint_requested": "P(all requested active axes pass), averaged over nonzero corners",
                "all_three": "P(all three axes pass | s=(1,1,1))",
                "all_inactive": "P(no axis activates | s=(0,0,0))",
                "exact_combination": "P(all active axes pass and all inactive axes remain absent)",
            },
        },
        "families": {},
    }
    latex = []
    markdown = ["# Paper behavioral evaluation", ""]
    specifications = [
        ("activation_steering", "steering", "Activation-steering behavioral adherence",
         "tab:steering-adherence"),
        ("weight_merging", "merging", "Weight-merging behavioral adherence",
         "tab:merging-adherence"),
    ]
    for family_index, (family, stem, title, label) in enumerate(specifications):
        methods, prompt_sets, common_prompts, directions, paired = complete_prompt_sets(
            groups, family, axes,
        )
        rows, family_payload = [], {}
        combinations = []
        for method_index, method in enumerate(methods):
            prompts = prompt_sets[method]
            metrics, by_combination = method_metrics(
                groups[(family, method)], prompts, directions, axes, thresholds,
                args.bootstrap_repetitions,
                args.seed + 100_003 * family_index + 1009 * method_index,
            )
            rows.append(_flatten_row(method, len(prompts), metrics))
            family_payload[method] = {"metrics": metrics, "by_combination": by_combination}
            for item in by_combination:
                combinations.append({"method": display_name(method), "method_key": method, **item})
        rows.sort(key=lambda row: row["macro_active"], reverse=True)
        results["families"][family] = {
            "paired_comparison": paired,
            "common_complete_prompts": len(common_prompts),
            "prompts_by_method": {
                method: len(prompt_sets[method]) for method in methods
            },
            "methods": family_payload,
        }
        _write_csv(args.output_dir / f"{stem}_paper_table.csv", rows)
        _write_csv(args.output_dir / f"{stem}_by_combination.csv", combinations)
        _plot(rows, args.output_dir / f"{stem}_paper_table", title)
        latex.append(_latex_table(rows, axes, title, label))
        comparison_note = (
            f"Paired comparison on {len(common_prompts)} shared prompts."
            if paired else
            "Unpaired historical comparison: methods used disjoint prompt slices."
        )
        markdown.extend([
            f"## {title}", "", comparison_note, "", _markdown_table(rows, axes), "",
        ])
    (args.output_dir / "paper_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output_dir / "paper_tables.tex").write_text("\n\n".join(latex) + "\n")
    markdown.extend([
        "All entries are percentages with prompt-bootstrap 95% confidence intervals.",
        "Every estimate uses complete binary-corner blocks. Comparisons use the shared prompt intersection where one exists; historical disjoint-prompt results are explicitly marked unpaired.",
    ])
    (args.output_dir / "RESULTS.md").write_text("\n".join(markdown) + "\n")
    LOG.info("Saved standardized paper evaluation to %s", args.output_dir)


if __name__ == "__main__":
    main()
