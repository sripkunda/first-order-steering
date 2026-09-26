"""Create plots and comparison tables from behavioral-classifier scores."""
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
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)


LOG = logging.getLogger("behavior_classifier_report")

DISPLAY_NAMES = {
    "caa": "CAA",
    "repe": "RepE",
    "mimic": "MiMiC",
    "linear_act": "Linear-AcT",
    "re_control": "RE-Control",
    "odesteer": "ODESteer",
    "ours_task_arithmetic_static": "Ours static (task arithmetic, all layers)",
    "ours_task_arithmetic_static_single_layer": (
        "Ours static (task arithmetic, shared calibrated layer)"
    ),
    "ours_task_arithmetic_static_layer13": "Ours static (task arithmetic, layer 13)",
    "ours_task_arithmetic_static_layer15": "Ours static (task arithmetic, layer 15)",
    "ours_task_arithmetic_static_selected_layers": "Ours static (task arithmetic, matched layers)",
    "ours_fitted_interaction_static": "Ours static (HeRD merging, all layers)",
    "ours_fitted_interaction_static_single_layer": (
        "Ours static (HeRD merging, shared calibrated layer)"
    ),
    "ours_fitted_interaction_static_single_layer_static10x": (
        "Ours static (HeRD merging, 10x context regularization)"
    ),
    "ours_fitted_interaction_static_layer13": "Ours static (HeRD merging, layer 13)",
    "ours_fitted_interaction_static_layer15": "Ours static (HeRD merging, layer 15)",
    "ours_fitted_interaction_static_selected_layers": "Ours static (HeRD merging, matched layers)",
    "ours_fitted_interaction_static_caa_layer34_500": (
        "Ours static (HeRD merging, layer 34)"
    ),
    "ours_task_arithmetic_static_caa_layer34_500": (
        "Ours static (task arithmetic, layer 34)"
    ),
    "ours_fitted_interaction_static_no_marginal_layer13": (
        "Ours static (no marginal; layer 13)"
    ),
    "ours_fitted_interaction_static_no_marginal_layer15": (
        "Ours static (no marginal; layer 15)"
    ),
    "ours_fitted_interaction_static_exact_local_losses_layer13": (
        "Ours static (exact local losses; layer 13)"
    ),
    "ours_fitted_interaction_static_exact_local_losses_layer15": (
        "Ours static (exact local losses; layer 15)"
    ),
    "independent_task_arithmetic": "Task arithmetic",
    "independent_ties": "TIES",
    "independent_dare_task_arithmetic": "DARE task arithmetic",
    "independent_dare_ties": "DARE-TIES",
    "independent_knots_ties": "KNoTs-TIES",
    "ours_fitted_interaction": "Ours HeRD merging",
    "ours_fitted_interaction_eta_0p1": "Ours HeRD merging (eta = 0.1)",
    "ours_fitted_interaction_eta_0p05": "Ours HeRD merging (eta = 0.05)",
    "ours_fitted_interaction_eta_0p05_layer34": (
        "Ours HeRD merging (eta = 0.05; layer 34)"
    ),
    "ours_fitted_interaction_eta_0p05_full": (
        "Ours HeRD merging (eta = 0.05)"
    ),
    "ours_fitted_interaction_no_marginal_layer13": (
        "Ours HeRD merging (no marginal; layer 13)"
    ),
    "ours_fitted_interaction_no_marginal_layer15": (
        "Ours HeRD merging (no marginal; layer 15)"
    ),
    "ours_fitted_interaction_exact_local_losses_layer13": (
        "Ours HeRD merging (exact local losses; layer 13)"
    ),
    "ours_fitted_interaction_exact_local_losses_layer15": (
        "Ours HeRD merging (exact local losses; layer 15)"
    ),
}


def display_name(method: str) -> str:
    return DISPLAY_NAMES.get(method, method.replace("_", " ").title())


def axis_name(axis: str) -> str:
    return axis.replace("_", " ").title()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join([header, rule, *body])


def classifier_rows(classifiers: dict) -> list[dict]:
    rows = []
    for axis, metadata in classifiers.items():
        test = metadata["test"]
        auroc_ci = test.get("bootstrap_95_ci", {}).get("auroc", {})
        auprc_ci = test.get("bootstrap_95_ci", {}).get("auprc", {})
        rows.append({
            "axis": axis_name(axis),
            "n": test["count"],
            "auroc": test["auroc"],
            "auroc_95_low": auroc_ci.get("lower_2.5"),
            "auroc_95_high": auroc_ci.get("upper_97.5"),
            "auprc": test["auprc"],
            "auprc_95_low": auprc_ci.get("lower_2.5"),
            "auprc_95_high": auprc_ci.get("upper_97.5"),
            "balanced_accuracy": test["balanced_accuracy"],
            "f1": test["f1"],
            "brier": test["brier"],
            "ece_15": test["ece_15"],
            "paired_accuracy": test.get("paired_accuracy"),
            "threshold": metadata["threshold"],
        })
    return rows


def method_rows(summary: dict, family: str, axes: list[str]) -> tuple[list[dict], list[dict]]:
    aggregate = []
    detailed = []
    for item in summary.values():
        if item["family"] != family:
            continue
        metrics = []
        for axis in axes:
            endpoint = item["by_axis"][axis]["endpoint"]
            bootstrap = item["by_axis"][axis].get("endpoint_bootstrap_95_ci", {})
            metrics.append(endpoint)
            detailed.append({
                "method": display_name(item["method"]),
                "method_key": item["method"],
                "axis": axis_name(axis),
                "n": endpoint["count"],
                "auroc": endpoint["auroc"],
                "auroc_95_low": bootstrap.get("auroc", {}).get("lower_2.5"),
                "auroc_95_high": bootstrap.get("auroc", {}).get("upper_97.5"),
                "auprc": endpoint["auprc"],
                "auprc_95_low": bootstrap.get("auprc", {}).get("lower_2.5"),
                "auprc_95_high": bootstrap.get("auprc", {}).get("upper_97.5"),
                "balanced_accuracy": endpoint["balanced_accuracy"],
                "f1": endpoint["f1"],
                "brier": endpoint["brier"],
                "pearson": item["by_axis"][axis]["continuous"]["pearson"],
                "spearman": item["by_axis"][axis]["continuous"]["spearman"],
                "monotonicity_violation_rate": item["by_axis"][axis]["monotonicity"]["violation_rate"],
            })
        aggregate.append({
            "method": display_name(item["method"]),
            "method_key": item["method"],
            "prompts": item["prompt_count"],
            "generations": item["generation_count"],
            "macro_auroc": float(np.mean([value["auroc"] for value in metrics])),
            "macro_auprc": float(np.mean([value["auprc"] for value in metrics])),
            "macro_balanced_accuracy": float(np.mean([value["balanced_accuracy"] for value in metrics])),
            "macro_f1": float(np.mean([value["f1"] for value in metrics])),
            "macro_task_adherence": float(np.mean([value["recall"] for value in metrics])),
            "macro_false_activation_rate": float(np.mean([
                value["false_positive_rate"] for value in metrics
            ])),
            "macro_brier": float(np.mean([value["brier"] for value in metrics])),
            "macro_spearman": float(np.mean([
                item["by_axis"][axis]["continuous"]["spearman"] for axis in axes
            ])),
            "macro_monotonicity_violation_rate": float(np.mean([
                item["by_axis"][axis]["monotonicity"]["violation_rate"] for axis in axes
            ])),
            "exact_combination_accuracy": item["binary_combination"]["exact_match_accuracy"],
            "desired_label_geometric_mean": math.exp(
                item["binary_combination"]["mean_desired_label_log_probability"]
            ),
            "nonempty_generation_rate": item["nonempty_generation_rate"],
        })
    aggregate.sort(key=lambda row: row["macro_task_adherence"], reverse=True)
    detailed.sort(key=lambda row: (row["axis"], -row["auroc"]))
    return aggregate, detailed


def load_endpoint_predictions(path: Path, axes: list[str], classifiers: dict):
    values = defaultdict(lambda: defaultdict(lambda: {axis: [[], [], [], []] for axis in axes}))
    corners = defaultdict(lambda: defaultdict(list))
    prompt_sets = defaultdict(set)
    total = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = orjson.loads(line)
            family, method = record["family"], record["method"]
            # Match the paper report: source identity plus whitespace-normalized
            # text. Historical generation files sometimes differ only in newline
            # formatting, which must not turn the same evaluation prompt into an
            # apparently unpaired example.
            prompt_key = (
                int(record["source_index"]) if record.get("source_index") is not None else -1,
                " ".join(record["prompt"].split()),
            )
            prompt_sets[(family, method)].add(prompt_key)
            binary_corner = all(
                math.isclose(float(record["direction"].get(name, 0.0)), 0.0) or
                math.isclose(float(record["direction"].get(name, 0.0)), 1.0)
                for name in axes
            )
            for axis in axes:
                level = float(record["direction"].get(axis, 0.0))
                if math.isclose(level, 0.0) or math.isclose(level, 1.0):
                    values[family][method][axis][0].append(int(round(level)))
                    values[family][method][axis][1].append(float(record["axis_scores"][axis]))
                    values[family][method][axis][2].append(prompt_key)
                    values[family][method][axis][3].append(binary_corner)
            if binary_corner:
                correct = []
                log_terms = []
                for axis in axes:
                    target = int(round(float(record["direction"].get(axis, 0.0))))
                    score = min(max(float(record["axis_scores"][axis]), 1e-7), 1 - 1e-7)
                    predicted = score >= float(classifiers[axis]["threshold"])
                    correct.append(int(predicted) == target)
                    log_terms.append(math.log(score if target else 1.0 - score))
                corners[family][method].append((
                    prompt_key, all(correct), sum(log_terms) / len(log_terms),
                    (all(
                        float(record["axis_scores"][axis]) >= float(classifiers[axis]["threshold"])
                        for axis in axes
                        if int(round(float(record["direction"].get(axis, 0.0)))) == 1
                    ) if any(int(round(float(record["direction"].get(axis, 0.0)))) == 1
                             for axis in axes) else None),
                ))
            total += 1
            if total % 100_000 == 0:
                LOG.info("Read %d scored generation records", total)
    LOG.info("Read %d scored generation records total", total)
    return values, corners, prompt_sets


def common_prompt_sets(values, prompt_sets):
    result = {}
    for family, methods in values.items():
        sets = [prompt_sets[(family, method)] for method in methods]
        result[family] = set.intersection(*sets) if sets else set()
    return result


def common_method_rows(values, corners, family: str, axes: list[str], classifiers: dict,
                       selected_prompts: set[tuple]) -> list[dict]:
    rows = []
    for method in values[family]:
        per_axis = []
        for axis in axes:
            labels, scores, prompts, corners_only = values[family][method][axis]
            selected = np.asarray([
                prompt in selected_prompts and corner
                for prompt, corner in zip(prompts, corners_only)
            ])
            labels = np.asarray(labels)[selected]
            scores = np.asarray(scores)[selected]
            predicted = scores >= float(classifiers[axis]["threshold"])
            per_axis.append({
                "auroc": roc_auc_score(labels, scores),
                "auprc": average_precision_score(labels, scores),
                "balanced_accuracy": balanced_accuracy_score(labels, predicted),
                "f1": f1_score(labels, predicted, zero_division=0),
                "task_adherence": float(predicted[labels == 1].mean()),
                "false_activation_rate": float(predicted[labels == 0].mean()),
            })
        corner = [item for item in corners[family][method] if item[0] in selected_prompts]
        active_corner = [item[3] for item in corner if item[3] is not None]
        axis_adherence = {
            f"{axis}_task_adherence": per_axis[index]["task_adherence"]
            for index, axis in enumerate(axes)
        }
        rows.append({
            "method": display_name(method),
            "method_key": method,
            "prompts": len(selected_prompts),
            "macro_auroc": float(np.mean([item["auroc"] for item in per_axis])),
            "macro_auprc": float(np.mean([item["auprc"] for item in per_axis])),
            "macro_balanced_accuracy": float(np.mean([
                item["balanced_accuracy"] for item in per_axis
            ])),
            "macro_f1": float(np.mean([item["f1"] for item in per_axis])),
            **axis_adherence,
            "macro_task_adherence": float(np.mean([
                item["task_adherence"] for item in per_axis
            ])),
            "macro_false_activation_rate": float(np.mean([
                item["false_activation_rate"] for item in per_axis
            ])),
            "joint_active_task_adherence": float(np.mean(active_corner)),
            "exact_combination_accuracy": float(np.mean([item[1] for item in corner])),
            "desired_label_geometric_mean": math.exp(float(np.mean([item[2] for item in corner]))),
        })
    rows.sort(key=lambda row: row["macro_task_adherence"], reverse=True)
    return rows


def plot_task_adherence(rows: list[dict], axes: list[str], output_dir: Path) -> None:
    """Plot the primary fixed-threshold behavioral judgments for matched steering runs."""
    if not rows:
        return
    columns = [
        *[(f"{axis}_task_adherence", f"{axis_name(axis).replace(' ', chr(10))}\npass")
          for axis in axes],
        ("macro_task_adherence", "Macro\npass"),
        ("joint_active_task_adherence", "Joint active\npass"),
        ("exact_combination_accuracy", "Exact\ncombination"),
        ("macro_false_activation_rate", "False\nactivation"),
    ]
    values = np.asarray([
        [float(row[key]) for key, _ in columns]
        for row in rows
    ])
    height = max(4.8, 0.55 * len(rows) + 1.8)
    fig, panel = plt.subplots(figsize=(11.5, height))
    image = panel.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    panel.set_xticks(range(len(columns)), [label for _, label in columns])
    panel.set_yticks(range(len(rows)), [row["method"] for row in rows])
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            panel.text(column_index, row_index, f"{value:.3f}", ha="center", va="center",
                       color="white" if value >= 0.58 else "black", fontsize=8.5)
    panel.set_title("Activation steering: task adherence on matched prompts and binary corners")
    panel.set_xlabel("Higher is better except false activation (lower is better)")
    fig.colorbar(image, ax=panel, fraction=0.025, pad=0.025, label="Rate")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"steering_task_adherence.{suffix}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def plot_curves(values, family: str, axes: list[str], output_dir: Path,
                curve_type: str, selected_prompts: set[int] | None = None) -> None:
    methods = sorted(values[family], key=lambda method: display_name(method))
    colors = plt.get_cmap("tab10")
    fig, panels = plt.subplots(1, len(axes), figsize=(6.2 * len(axes), 5.4), squeeze=False)
    for axis_index, axis in enumerate(axes):
        panel = panels[0, axis_index]
        for method_index, method in enumerate(methods):
            labels, scores, prompts, corners_only = values[family][method][axis]
            if selected_prompts is not None:
                selected = np.asarray([
                    prompt in selected_prompts and corner
                    for prompt, corner in zip(prompts, corners_only)
                ])
                labels = np.asarray(labels)[selected]
                scores = np.asarray(scores)[selected]
            else:
                labels = np.asarray(labels)
                scores = np.asarray(scores)
            if curve_type == "roc":
                x, y, _ = roc_curve(labels, scores, drop_intermediate=True)
                metric = roc_auc_score(labels, scores)
                panel.plot(x, y, linewidth=1.8, color=colors(method_index % 10),
                           label=f"{display_name(method)} ({metric:.3f})")
            else:
                precision, recall, _ = precision_recall_curve(labels, scores)
                metric = average_precision_score(labels, scores)
                panel.plot(recall, precision, linewidth=1.8, color=colors(method_index % 10),
                           label=f"{display_name(method)} ({metric:.3f})")
        if curve_type == "roc":
            panel.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
            panel.set_xlabel("False-positive rate")
            panel.set_ylabel("True-positive rate")
            title_metric = "AUROC"
        else:
            panel.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
            panel.set_xlabel("Recall")
            panel.set_ylabel("Precision")
            title_metric = "AUPRC"
        panel.set_title(axis_name(axis))
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1.01)
        panel.grid(alpha=0.2)
        panel.legend(title=title_metric, fontsize=7, title_fontsize=8, loc="best")
    family_title = "Activation steering" if family == "activation_steering" else "Weight merging"
    fig.suptitle(f"{family_title}: endpoint {curve_type.upper()} curves", fontsize=15)
    fig.tight_layout()
    stem = "steering" if family == "activation_steering" else "merging"
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}_{curve_type}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt(value, digits=3):
    return "--" if value is None else f"{float(value):.{digits}f}"


def make_markdown(classifiers, family_tables, common_tables, common_sets, output_dir: Path) -> str:
    classifier_display = [{
        **row,
        "auroc_display": fmt(row["auroc"], 6),
        "auprc_display": fmt(row["auprc"], 6),
        "balanced_display": fmt(row["balanced_accuracy"], 4),
        "f1_display": fmt(row["f1"], 4),
        "brier_display": fmt(row["brier"], 6),
        "ece_display": fmt(row["ece_15"], 6),
    } for row in classifiers]
    sections = [
        "# Behavioral-classifier evaluation\n",
        "## Held-out classifier quality\n",
        _markdown_table(classifier_display, [
            ("axis", "Axis"), ("n", "N"), ("auroc_display", "AUROC"),
            ("auprc_display", "AUPRC"), ("balanced_display", "Balanced acc."),
            ("f1_display", "F1"), ("brier_display", "Brier"), ("ece_display", "ECE"),
        ]),
    ]
    if common_tables.get("activation_steering"):
        rows = [{
            **row,
            "auroc_display": fmt(row["macro_auroc"]),
            "auprc_display": fmt(row["macro_auprc"]),
            "balanced_display": fmt(row["macro_balanced_accuracy"]),
            "f1_display": fmt(row["macro_f1"]),
            "exact_display": fmt(row["exact_combination_accuracy"]),
            "desired_display": fmt(row["desired_label_geometric_mean"]),
            "analogy_display": fmt(row.get("analogy_task_adherence")),
            "bullet_display": fmt(row.get("bulletpointer_task_adherence")),
            "sophisticated_display": fmt(row.get("sophisticated_language_task_adherence")),
            "adherence_display": fmt(row["macro_task_adherence"]),
            "false_activation_display": fmt(row["macro_false_activation_rate"]),
            "joint_active_display": fmt(row["joint_active_task_adherence"]),
        } for row in common_tables["activation_steering"]]
        sections.extend([
            "\n## Activation-steering judgments: matched prompts and binary corners\n",
            _markdown_table(rows, [
                ("method", "Method"), ("prompts", "Prompts"),
                ("analogy_display", "Analogy pass"),
                ("bullet_display", "Bullet pass"),
                ("sophisticated_display", "Sophisticated pass"),
                ("adherence_display", "Macro pass"),
                ("false_activation_display", "False activation"),
                ("joint_active_display", "Joint active pass"),
                ("exact_display", "Exact combination acc."),
                ("auroc_display", "AUROC"), ("auprc_display", "AUPRC"),
            ]),
        ])
    for family, (aggregate, _) in family_tables.items():
        intersection = len(common_sets.get(family, []))
        fully_paired = bool(intersection) and all(
            int(row["prompts"]) == intersection for row in aggregate
        )
        if family == "activation_steering":
            title = "Activation-steering judgments: all available prompts"
        else:
            title = (
                "Weight-merging judgments: matched prompts and binary corners"
                if fully_paired else "Weight-merging judgments: unpaired prompt sets"
            )
        display = [{
            **row,
            "auroc_display": fmt(row["macro_auroc"]),
            "auprc_display": fmt(row["macro_auprc"]),
            "balanced_display": fmt(row["macro_balanced_accuracy"]),
            "f1_display": fmt(row["macro_f1"]),
            "spearman_display": fmt(row["macro_spearman"]),
            "exact_display": fmt(row["exact_combination_accuracy"]),
            "desired_display": fmt(row["desired_label_geometric_mean"]),
            "adherence_display": fmt(row["macro_task_adherence"]),
            "false_activation_display": fmt(row["macro_false_activation_rate"]),
        } for row in aggregate]
        sections.extend([
            f"\n## {title}\n",
            _markdown_table(display, [
                ("method", "Method"), ("prompts", "Prompts"),
                ("adherence_display", "Macro task adherence"),
                ("false_activation_display", "False activation"),
                ("exact_display", "Exact combination acc."),
                ("auroc_display", "Macro AUROC"), ("auprc_display", "Macro AUPRC"),
                ("spearman_display", "Macro Spearman"),
            ]),
        ])
    sections.extend([
        "\n## Interpretation notes\n",
        "- Task adherence is the fixed-threshold pass rate $\\Pr(\\hat y_i=1\\mid s_i=1)$; false activation is $\\Pr(\\hat y_i=1\\mid s_i=0)$. These are the primary behavioral metrics.\n",
        "- Joint active-task adherence requires every requested active axis to pass, while exact combination accuracy additionally requires all inactive axes to remain absent.\n",
        "- Generation AUROC/AUPRC is a supplementary threshold-free separation diagnostic between $s_i=0$ and $s_i=1$, not an adherence percentage.\n",
        "- Exact combination accuracy requires all axis judgments to match a binary steering corner simultaneously.\n",
        "- Steering methods were not all evaluated on the same number of prompts; consult the prompt-count column and avoid treating unpaired differences as significance tests.\n",
        f"- The primary steering curves/table use the {len(common_sets.get('activation_steering', []))}-prompt intersection and the same $2^3$ binary corners shared by every steering method.\n",
        f"- The common-prompt intersection between every merging method is {len(common_sets.get('weight_merging', []))}; "
        + (
            "the weight-merging comparison is paired on that shared set.\n"
            if common_sets.get("weight_merging") and all(
                int(row["prompts"]) == len(common_sets["weight_merging"])
                for row in family_tables.get("weight_merging", ([], []))[0]
            )
            else "the weight-merging comparison remains unpaired.\n"
        ),
        "- Full per-axis metrics and confidence intervals are in the detailed CSV tables.\n",
    ])
    report = "\n".join(sections)
    (output_dir / "REPORT.md").write_text(report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores_json", type=Path, required=True)
    parser.add_argument("--scored_generations_jsonl", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.scores_json.read_text())
    axes = list(payload["classifiers"])
    predictions_path = args.scored_generations_jsonl
    if predictions_path is None:
        predictions_path = args.scores_json.with_name(args.scores_json.stem + ".scored.jsonl")

    classifiers = classifier_rows(payload["classifiers"])
    _write_csv(args.output_dir / "classifier_test_metrics.csv", classifiers)
    family_tables = {}
    for family, stem in (("activation_steering", "steering"), ("weight_merging", "merging")):
        aggregate, detailed = method_rows(payload["summary"], family, axes)
        family_tables[family] = (aggregate, detailed)
        _write_csv(args.output_dir / f"{stem}_method_comparison.csv", aggregate)
        _write_csv(args.output_dir / f"{stem}_per_axis_metrics.csv", detailed)

    values, corners, prompt_sets = load_endpoint_predictions(
        predictions_path, axes, payload["classifiers"],
    )
    common_sets = common_prompt_sets(values, prompt_sets)
    common_tables = {}
    for family, selected_prompts in common_sets.items():
        if selected_prompts:
            common_tables[family] = common_method_rows(
                values, corners, family, axes, payload["classifiers"], selected_prompts,
            )
            stem = "steering" if family == "activation_steering" else "merging"
            _write_csv(args.output_dir / f"{stem}_matched_prompt_comparison.csv",
                       common_tables[family])
    for family in ("activation_steering", "weight_merging"):
        selected = common_sets[family] or None
        plot_curves(values, family, axes, args.output_dir, "roc", selected)
        plot_curves(values, family, axes, args.output_dir, "prc", selected)
    plot_task_adherence(common_tables.get("activation_steering", []), axes, args.output_dir)

    overlaps = {}
    for family in values:
        methods = sorted(values[family])
        overlaps[family] = {
            f"{left}::{right}": len(prompt_sets[(family, left)] & prompt_sets[(family, right)])
            for left_index, left in enumerate(methods)
            for right in methods[left_index + 1:]
        }
    (args.output_dir / "prompt_overlaps.json").write_text(json.dumps(overlaps, indent=2) + "\n")
    report = make_markdown(classifiers, family_tables, common_tables, common_sets,
                           args.output_dir)
    print(report)


if __name__ == "__main__":
    main()
