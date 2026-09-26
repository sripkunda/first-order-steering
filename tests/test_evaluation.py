from __future__ import annotations

import numpy as np

from analysis.behavior_classifier_paper_report import method_metrics


def test_exact_combination_metrics_cover_one_two_and_three_behavior_requests():
    axes = ["a", "b", "c"]
    thresholds = {axis: 0.5 for axis in axes}
    directions = {
        tuple((mask >> index) & 1 for index in range(3)) for mask in range(8)
    }
    records = {}
    prompt = (0, "test prompt")
    records[prompt] = {}
    for direction in directions:
        # Perfect classifier predictions reproduce every requested binary corner.
        records[prompt][direction] = {
            "axis_scores": {axis: float(target) for axis, target in zip(axes, direction)},
            "response": "output",
        }
    metrics, by_combination = method_metrics(
        records, {prompt}, directions, axes, thresholds, bootstrap=20, seed=0,
    )
    assert metrics["exact_combination"]["mean"] == 1.0
    assert metrics["all_three"]["mean"] == 1.0
    assert len(by_combination) == 8
    assert np.allclose([row["exact_combination_accuracy"] for row in by_combination], 1.0)
