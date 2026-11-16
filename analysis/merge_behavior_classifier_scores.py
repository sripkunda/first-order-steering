"""Merge newly scored methods into an existing DeBERTa score artifact.

This is a streaming file operation.  It never runs a classifier: unchanged
judgments are copied and newly scored records are appended. By default,
same-named methods are updated; --append_only rejects any such collision.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import orjson


LOG = logging.getLogger("merge_behavior_classifier_scores")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_scores_json", type=Path, required=True)
    parser.add_argument("--base_scored_jsonl", type=Path, required=True)
    parser.add_argument("--replacement_scores_json", type=Path, required=True)
    parser.add_argument("--replacement_scored_jsonl", type=Path, required=True)
    parser.add_argument("--output_scores_json", type=Path, required=True)
    parser.add_argument("--output_scored_jsonl", type=Path, required=True)
    parser.add_argument(
        "--append_only", action="store_true",
        help="Refuse to remove or replace any method already present in the base artifact.",
    )
    parser.add_argument(
        "--drop_method", action="append", default=[], metavar="FAMILY=METHOD",
        help="Repeatable stale method to remove while merging the replacement artifact.",
    )
    parser.add_argument(
        "--include_method", action="append", default=[], metavar="FAMILY=METHOD",
        help=("Optional repeatable allowlist for methods imported from the replacement "
              "artifact. This permits reusing already-scored rows without rerunning "
              "classifier inference."),
    )
    return parser.parse_args()


def _classifier_signature(payload: dict) -> dict:
    return {
        axis: {
            "threshold": metadata["threshold"],
            "dataset": metadata.get("dataset"),
            "model_name": metadata.get("model_name"),
        }
        for axis, metadata in payload["classifiers"].items()
    }


def _method_specs(values: list[str], argument: str) -> set[tuple[str, str]]:
    result = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"{argument} must be FAMILY=METHOD")
        family, method = value.split("=", 1)
        if not family or not method:
            raise ValueError(f"{argument} must contain nonempty FAMILY and METHOD")
        result.add((family, method))
    return result


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    base = json.loads(args.base_scores_json.read_text())
    replacement = json.loads(args.replacement_scores_json.read_text())
    if _classifier_signature(base) != _classifier_signature(replacement):
        raise ValueError("Base and replacement artifacts use different classifiers or thresholds")
    available = {
        (item["family"], item["method"])
        for item in replacement["summary"].values()
    }
    included = _method_specs(args.include_method, "--include_method")
    missing = included - available
    if missing:
        raise ValueError(f"Replacement artifact does not contain requested methods: {sorted(missing)}")
    replaced = available if not included else included
    dropped = _method_specs(args.drop_method, "--drop_method")
    removal = replaced | dropped
    if not replaced:
        raise ValueError("Replacement score artifact contains no methods")
    base_methods = {
        (item["family"], item["method"])
        for item in base["summary"].values()
    }
    collisions = replaced & base_methods
    if args.append_only and collisions:
        raise ValueError(
            "Append-only merge would overwrite existing methods: "
            f"{sorted(collisions)}. Give the new methods distinct aliases."
        )
    LOG.info(
        "%s methods: %s",
        "Appending" if args.append_only else "Upserting",
        sorted(replaced),
    )

    args.output_scored_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_scored_jsonl.with_suffix(args.output_scored_jsonl.suffix + ".tmp")
    kept = added = removed = 0
    with temporary.open("wb") as output:
        with args.base_scored_jsonl.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = orjson.loads(line)
                if not args.append_only and (record["family"], record["method"]) in removal:
                    removed += 1
                    continue
                output.write(line if line.endswith(b"\n") else line + b"\n")
                kept += 1
        with args.replacement_scored_jsonl.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = orjson.loads(line)
                if (record["family"], record["method"]) not in replaced:
                    continue
                output.write(line if line.endswith(b"\n") else line + b"\n")
                added += 1
    temporary.replace(args.output_scored_jsonl)

    summary = {
        key: item for key, item in base["summary"].items()
        if args.append_only or (item["family"], item["method"]) not in removal
    }
    summary.update({
        key: item for key, item in replacement["summary"].items()
        if (item["family"], item["method"]) in replaced
    })
    merged = {
        **base,
        "manifest": {
            "base": str(args.base_scores_json.resolve()),
            "replacement": str(args.replacement_scores_json.resolve()),
        },
        "summary": summary,
        "scored_generations_jsonl": str(args.output_scored_jsonl.resolve()),
        "scored_generation_count": kept + added,
        "merge": {
            "input_methods": [list(item) for item in sorted(replaced)],
            "replaced_methods": [list(item) for item in sorted(collisions)]
            if not args.append_only else [],
            "dropped_methods": [list(item) for item in sorted(dropped)],
            "mode": "append_only" if args.append_only else "upsert",
            "kept_records": kept,
            "removed_stale_records": removed,
            "added_replacement_records": added,
            "classifier_inference_rerun_for_baselines": False,
        },
    }
    args.output_scores_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_scores_json.write_text(json.dumps(merged, indent=2) + "\n")
    LOG.info("Kept %d, removed %d, added %d scored records", kept, removed, added)
    LOG.info("Saved merged scores to %s", args.output_scores_json)


if __name__ == "__main__":
    main()
