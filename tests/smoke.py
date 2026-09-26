"""Dependency-light CPU test runner for environments without pytest."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_evaluation import test_exact_combination_metrics_cover_one_two_and_three_behavior_requests
from test_herd_core import (
    test_opinion_pool_has_exact_origin_and_pure_axis_boundaries,
    test_static_least_squares_vector_is_the_prompt_mean,
    test_task_arithmetic_and_herd_interactions_preserve_pure_axes,
)

def main():
    test_opinion_pool_has_exact_origin_and_pure_axis_boundaries()
    test_static_least_squares_vector_is_the_prompt_mean()
    test_exact_combination_metrics_cover_one_two_and_three_behavior_requests()
    with TemporaryDirectory() as directory:
        test_task_arithmetic_and_herd_interactions_preserve_pure_axes(Path(directory))
    print("4 CPU smoke tests passed")

if __name__ == "__main__":
    main()
