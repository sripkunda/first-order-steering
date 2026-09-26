import json

import numpy as np

from analysis.behavior_classifier import (
    LabeledRecord,
    PairRecord,
    extract_generation_records,
    grouped_split,
    monotonicity_report,
    normalize_prompt,
    select_threshold,
)


def test_grouped_split_never_separates_a_pair():
    pairs = [PairRecord(str(i), str(i), f"negative {i}", f"positive {i}") for i in range(30)]
    splits = grouped_split(pairs, 0.2, 0.2, 17)
    ids = {name: {pair.pair_id for pair in values} for name, values in splits.items()}
    assert not (ids["train"] & ids["validation"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["validation"] & ids["test"])
    assert set.union(*ids.values()) == {str(i) for i in range(30)}


def test_grouped_split_keeps_duplicate_prompts_together():
    pairs = [
        PairRecord("one", "same prompt", "n1", "p1"),
        PairRecord("two", "same prompt", "n2", "p2"),
    ] + [PairRecord(str(i), f"prompt {i}", f"n{i}", f"p{i}") for i in range(3, 30)]
    splits = grouped_split(pairs, 0.2, 0.2, 17)
    containing = [name for name, values in splits.items()
                  if any(pair.pair_id == "one" for pair in values)]
    assert len(containing) == 1
    assert any(pair.pair_id == "two" for pair in splits[containing[0]])


def test_prompt_normalization_is_case_and_space_insensitive():
    assert normalize_prompt("  A  Prompt\nHere ") == normalize_prompt("a prompt here")


def test_manifest_reads_direction_schema_and_filters_methods(tmp_path):
    path = tmp_path / "generations.json"
    path.write_text(json.dumps({
        "analysis": {
            "axes": ["a", "b"],
            "directions": {
                "both": {
                    "direction": [1, 1],
                    "examples": [{
                        "prompt": "question", "prompt_index": 0,
                        "keep": "answer", "drop": "other",
                    }],
                },
            },
        },
    }))
    records = extract_generation_records({
        "path": str(path), "include_methods": ["keep"],
        "method_aliases": {"keep": "renamed"},
    }, tmp_path)
    assert len(records) == 1
    assert records[0]["method"] == "renamed"
    assert records[0]["direction"] == {"a": 1.0, "b": 1.0}


def test_threshold_and_monotonicity():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert 0.2 < select_threshold(labels, scores) <= 0.8
    records = []
    for level, score in zip((0.0, 0.5, 1.0), (0.1, 0.6, 0.9)):
        records.append({
            "source_index": 1,
            "prompt_index": 1,
            "prompt": "p",
            "direction": {"a": level, "b": 0.0},
            "axis_scores": {"a": score},
        })
    report = monotonicity_report(records, "a", ["a", "b"])
    assert report["adjacent_comparisons"] == 2
    assert report["violation_rate"] == 0.0
