"""Small command dispatcher for the retained HeRD workflow."""
from __future__ import annotations

import sys


COMMANDS = {
    "train-lora": "steered_finetuner.train_independent_lora",
    "fit-herd": "analysis.herd_merging",
    "fit-gains": "analysis.herd_gain_fit",
    "generate": "analysis.compositional_generation",
    "compare-steering": "analysis.independent_steering_comparison",
    "train-classifier": "analysis.behavior_classifier",
    "report": "analysis.behavior_classifier_paper_report",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        choices = ", ".join(COMMANDS)
        raise SystemExit(f"Usage: python -m analysis <{choices}> [arguments]")
    command = sys.argv.pop(1)
    module = __import__(COMMANDS[command], fromlist=["main"])
    module.main()


if __name__ == "__main__":
    main()
