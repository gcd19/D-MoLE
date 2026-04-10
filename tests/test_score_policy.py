from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from internvl.scorer.score_policy import (
    STRICT_SCORE_PROVENANCE,
    load_score_frame,
    select_enable_layers,
)


def _authoritative_score_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": "language_model.layers.0.attn.qkv",
                "score": 0.90,
                "raw_score": 1.20,
                "share_mean": 0.60,
                "share_variance": 0.01,
                "stability_penalty": 0.0277,
                "modality": "language_model",
                "modality_trust": 0.95,
                "modality_frechet_variance": 0.02,
                "sanitized_nonfinite": False,
                "score_batches": 4,
                "score_provenance": STRICT_SCORE_PROVENANCE,
            },
            {
                "layer": "language_model.layers.1.attn.qkv",
                "score": 0.70,
                "raw_score": 1.05,
                "share_mean": 0.40,
                "share_variance": 0.01,
                "stability_penalty": 0.0625,
                "modality": "language_model",
                "modality_trust": 0.95,
                "modality_frechet_variance": 0.02,
                "sanitized_nonfinite": False,
                "score_batches": 4,
                "score_provenance": STRICT_SCORE_PROVENANCE,
            },
            {
                "layer": "vision_model.layers.0.attn.proj",
                "score": 0.80,
                "raw_score": 0.90,
                "share_mean": 0.55,
                "share_variance": 0.005,
                "stability_penalty": 0.0165,
                "modality": "vision_model",
                "modality_trust": 0.90,
                "modality_frechet_variance": 0.03,
                "sanitized_nonfinite": False,
                "score_batches": 4,
                "score_provenance": STRICT_SCORE_PROVENANCE,
            },
            {
                "layer": "vision_model.layers.1.attn.proj",
                "score": 0.50,
                "raw_score": 0.60,
                "share_mean": 0.45,
                "share_variance": 0.005,
                "stability_penalty": 0.0246,
                "modality": "vision_model",
                "modality_trust": 0.90,
                "modality_frechet_variance": 0.03,
                "sanitized_nonfinite": False,
                "score_batches": 4,
                "score_provenance": STRICT_SCORE_PROVENANCE,
            },
        ]
    )


def test_load_score_frame_rejects_sanitized_authoritative_scores(tmp_path: Path) -> None:
    frame = _authoritative_score_frame()
    frame.loc[0, "sanitized_nonfinite"] = True
    path = tmp_path / "score.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="sanitized_nonfinite"):
        load_score_frame(path, require_authoritative=True)


def test_select_enable_layers_prefers_deterministic_authoritative_order() -> None:
    enable_layers = select_enable_layers(
        _authoritative_score_frame(),
        budget_portion=0.5,
        require_authoritative=True,
    )

    assert enable_layers == [
        "language_model.layers.0.attn.qkv",
        "vision_model.layers.0.attn.proj",
    ]


def test_select_enable_layers_rejects_zero_authoritative_surface() -> None:
    frame = _authoritative_score_frame()
    frame["score"] = 0.0

    with pytest.raises(ValueError, match="collapsed to zero"):
        select_enable_layers(
            frame,
            budget_portion=0.5,
            require_authoritative=True,
        )


def test_select_enable_layers_allows_deterministic_legacy_zero_fallback() -> None:
    frame = pd.DataFrame(
        [
            {"layer": "language_model.layers.1.attn.qkv", "score": 0.0},
            {"layer": "language_model.layers.0.attn.qkv", "score": 0.0},
            {"layer": "vision_model.layers.1.attn.proj", "score": 0.0},
            {"layer": "vision_model.layers.0.attn.proj", "score": 0.0},
        ]
    )

    enable_layers = select_enable_layers(
        frame,
        budget_portion=0.5,
        require_authoritative=False,
    )

    assert enable_layers == [
        "language_model.layers.0.attn.qkv",
        "vision_model.layers.0.attn.proj",
    ]
