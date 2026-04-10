#!/usr/bin/env python3
"""Generate a single-task D-MoLE architecture JSON from one zero-cost score CSV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def get_enable_layers(scores: pd.DataFrame, budget_portion: float) -> list[str]:
    llm_scores = (
        scores[scores["layer"].str.contains("language_model")]
        .sort_values(by="score", ascending=False)
        .reset_index(drop=True)
    )
    vit_scores = (
        scores[scores["layer"].str.contains("vision_model")]
        .sort_values(by="score", ascending=False)
        .reset_index(drop=True)
    )

    total_score = llm_scores["score"].sum() + vit_scores["score"].sum()
    if not math.isfinite(total_score) or total_score <= 0:
        raise ValueError(
            f"non-finite or non-positive total score encountered: {total_score!r}"
        )
    llm_budget_portion = llm_scores["score"].sum() / total_score
    vit_budget_portion = vit_scores["score"].sum() / total_score
    total_budget = int(len(scores) * budget_portion + 0.5)

    vit_min_budget = max(
        int(total_budget * 0.25), int(total_budget * vit_budget_portion + 0.5)
    )
    llm_enable_layers_count = total_budget - vit_min_budget

    enable_layers: list[str] = []

    if llm_enable_layers_count > 0:
        llm_threshold = llm_scores.iloc[llm_enable_layers_count - 1]["score"]
        llm_enable_layers = list(
            llm_scores[llm_scores["score"] > llm_threshold]["layer"]
        )
        if len(llm_scores[llm_scores["score"] == llm_threshold]) > 1:
            remaining_layers = llm_scores[llm_scores["score"] == llm_threshold][
                "layer"
            ].sample(
                n=llm_enable_layers_count - len(llm_enable_layers), random_state=42
            )
            llm_enable_layers.extend(remaining_layers.tolist())
        else:
            llm_enable_layers = list(
                llm_scores[llm_scores["score"] >= llm_threshold]["layer"]
            )
        enable_layers.extend(llm_enable_layers)

    if vit_min_budget > 0:
        vit_threshold = vit_scores.iloc[vit_min_budget - 1]["score"]
        vit_enable_layers = list(
            vit_scores[vit_scores["score"] > vit_threshold]["layer"]
        )
        if len(vit_scores[vit_scores["score"] == vit_threshold]) > 1:
            remaining_layers = vit_scores[vit_scores["score"] == vit_threshold][
                "layer"
            ].sample(n=vit_min_budget - len(vit_enable_layers), random_state=42)
            vit_enable_layers.extend(remaining_layers.tolist())
        else:
            vit_enable_layers = list(
                vit_scores[vit_scores["score"] >= vit_threshold]["layer"]
            )
        enable_layers.extend(vit_enable_layers)

    assert len(enable_layers) == total_budget, "unexpected enabled-layer count"

    if enable_layers and "base_model" in enable_layers[0]:
        return [".".join(layer.split(".")[2:]) for layer in enable_layers]
    return [".".join(layer.split(".")[0:]) for layer in enable_layers]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--score-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--budget-portion", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.score_path)
    score_values = scores["score"].astype(float)
    if not score_values.map(math.isfinite).all():
        raise ValueError(f"score CSV contains non-finite values: {args.score_path}")
    scores = scores.assign(score=score_values)
    enable_layers = get_enable_layers(scores, budget_portion=args.budget_portion)
    architecture = {layer: [args.task_id] for layer in sorted(enable_layers)}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(architecture, indent=4, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote single-task architecture to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
