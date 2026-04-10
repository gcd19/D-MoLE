#!/usr/bin/env python3
"""Generate a single-task D-MoLE architecture JSON from one zero-cost score CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from internvl.scorer.score_policy import load_score_frame, select_enable_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--score-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--budget-portion", type=float, default=0.5)
    parser.add_argument(
        "--allow-non-authoritative-score",
        action="store_true",
        help="Permit legacy score CSVs that are not LoneStarPhysics invariance weighted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_authoritative = not args.allow_non_authoritative_score
    scores = load_score_frame(
        args.score_path,
        require_authoritative=require_authoritative,
    )
    enable_layers = select_enable_layers(
        scores,
        budget_portion=args.budget_portion,
        require_authoritative=require_authoritative,
    )
    architecture = {layer: [args.task_id] for layer in sorted(enable_layers)}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(architecture, indent=4, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote single-task architecture to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
