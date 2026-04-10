from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

STRICT_SCORE_PROVENANCE = "LONESTAR_PHYSICS_INVARIANCE_WEIGHTED"
FALLBACK_SCORE_PROVENANCE = "GRADIENT_NORM_NON_AUTHORITATIVE"
REQUIRED_AUTHORITATIVE_COLUMNS = frozenset(
    {
        "layer",
        "score",
        "raw_score",
        "share_mean",
        "share_variance",
        "stability_penalty",
        "modality",
        "modality_trust",
        "modality_frechet_variance",
        "sanitized_nonfinite",
        "score_batches",
        "score_provenance",
    }
)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    coerced = lowered.map(mapping)
    if coerced.isna().any():
        raise ValueError(
            "sanitized_nonfinite contains values that cannot be coerced to bool"
        )
    return coerced.astype(bool)


def _eligible_scores(scores: pd.DataFrame) -> pd.DataFrame:
    eligible = scores[
        scores["layer"].astype(str).str.contains("language_model|vision_model")
    ].copy()
    if eligible.empty:
        raise ValueError(
            "score surface does not contain any language_model or vision_model rows"
        )
    return eligible


def load_score_frame(
    path: Path | str,
    *,
    require_authoritative: bool,
) -> pd.DataFrame:
    score_path = Path(path)
    scores = pd.read_csv(score_path)
    missing_core = {"layer", "score"} - set(scores.columns)
    if missing_core:
        raise ValueError(
            f"score CSV is missing required columns {sorted(missing_core)}: {score_path}"
        )

    score_values = pd.to_numeric(scores["score"], errors="coerce")
    if score_values.isna().any() or not score_values.map(math.isfinite).all():
        raise ValueError(f"score CSV contains non-finite score values: {score_path}")
    if (score_values < 0.0).any():
        raise ValueError(f"score CSV contains negative score values: {score_path}")
    scores = scores.assign(score=score_values.astype(float))

    if not require_authoritative:
        return _eligible_scores(scores)

    missing = REQUIRED_AUTHORITATIVE_COLUMNS - set(scores.columns)
    if missing:
        raise ValueError(
            "authoritative score CSV is missing required columns "
            f"{sorted(missing)}: {score_path}"
        )

    raw_score = pd.to_numeric(scores["raw_score"], errors="coerce")
    share_mean = pd.to_numeric(scores["share_mean"], errors="coerce")
    share_variance = pd.to_numeric(scores["share_variance"], errors="coerce")
    stability_penalty = pd.to_numeric(scores["stability_penalty"], errors="coerce")
    modality_trust = pd.to_numeric(scores["modality_trust"], errors="coerce")
    modality_frechet_variance = pd.to_numeric(
        scores["modality_frechet_variance"], errors="coerce"
    )
    score_batches = pd.to_numeric(scores["score_batches"], errors="coerce")
    sanitized = _coerce_bool_series(scores["sanitized_nonfinite"])
    provenance = scores["score_provenance"].astype(str)

    numeric_columns = {
        "raw_score": raw_score,
        "share_mean": share_mean,
        "share_variance": share_variance,
        "stability_penalty": stability_penalty,
        "modality_trust": modality_trust,
        "modality_frechet_variance": modality_frechet_variance,
        "score_batches": score_batches,
    }
    for column_name, column_values in numeric_columns.items():
        if column_values.isna().any() or not column_values.map(math.isfinite).all():
            raise ValueError(
                f"authoritative score CSV has non-finite {column_name} values: {score_path}"
            )

    if (raw_score < 0.0).any():
        raise ValueError(f"authoritative score CSV has negative raw_score values: {score_path}")
    if (share_mean < 0.0).any():
        raise ValueError(f"authoritative score CSV has negative share_mean values: {score_path}")
    if (share_variance < 0.0).any():
        raise ValueError(
            f"authoritative score CSV has negative share_variance values: {score_path}"
        )
    if (stability_penalty < 0.0).any():
        raise ValueError(
            f"authoritative score CSV has negative stability_penalty values: {score_path}"
        )
    if ((modality_trust <= 0.0) | (modality_trust > 1.0)).any():
        raise ValueError(
            f"authoritative score CSV has modality_trust outside (0, 1]: {score_path}"
        )
    if (score_batches < 1).any():
        raise ValueError(
            f"authoritative score CSV has score_batches < 1: {score_path}"
        )
    if sanitized.any():
        raise ValueError(
            "authoritative score CSV contains sanitized_nonfinite layers; "
            f"rerun compute_zc_score in strict mode: {score_path}"
        )
    if not provenance.eq(STRICT_SCORE_PROVENANCE).all():
        raise ValueError(
            "authoritative score CSV does not carry strict LoneStarPhysics provenance: "
            f"{score_path}"
        )

    scores = scores.assign(
        raw_score=raw_score.astype(float),
        share_mean=share_mean.astype(float),
        share_variance=share_variance.astype(float),
        stability_penalty=stability_penalty.astype(float),
        modality_trust=modality_trust.astype(float),
        modality_frechet_variance=modality_frechet_variance.astype(float),
        score_batches=score_batches.astype(int),
        sanitized_nonfinite=sanitized,
    )
    return _eligible_scores(scores)


def _select_group_layers(scores: pd.DataFrame, count: int) -> list[str]:
    if count <= 0 or scores.empty:
        return []
    ordered = scores.sort_values(
        by=["score", "layer"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ordered.head(count)["layer"].tolist()


def select_enable_layers(
    scores: pd.DataFrame,
    *,
    budget_portion: float,
    require_authoritative: bool,
) -> list[str]:
    if not 0.0 < budget_portion <= 1.0:
        raise ValueError(f"budget_portion must be in (0, 1], got {budget_portion!r}")

    eligible = _eligible_scores(scores)
    llm_scores = (
        eligible[eligible["layer"].astype(str).str.contains("language_model")]
        .sort_values(by=["score", "layer"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    vit_scores = (
        eligible[eligible["layer"].astype(str).str.contains("vision_model")]
        .sort_values(by=["score", "layer"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )

    total_score = float(llm_scores["score"].sum() + vit_scores["score"].sum())
    total_budget = int(len(eligible) * budget_portion + 0.5)
    if total_budget <= 0:
        return []

    if total_score == 0.0:
        if require_authoritative:
            raise ValueError(
                "authoritative score surface collapsed to zero; rerun compute_zc_score"
            )
        if llm_scores.empty or vit_scores.empty:
            raise ValueError(
                "zero-score fallback requires both language_model and vision_model rows"
            )
        vit_budget = min(max(int(total_budget * 0.25), 1), len(vit_scores), total_budget)
        llm_budget = min(total_budget - vit_budget, len(llm_scores))
        remaining_budget = total_budget - (vit_budget + llm_budget)
        if remaining_budget > 0:
            extra_vit = min(remaining_budget, len(vit_scores) - vit_budget)
            vit_budget += extra_vit
            remaining_budget -= extra_vit
        if remaining_budget > 0:
            extra_llm = min(remaining_budget, len(llm_scores) - llm_budget)
            llm_budget += extra_llm
            remaining_budget -= extra_llm
        if remaining_budget != 0:
            raise ValueError("unable to satisfy zero-score fallback budget")

        enable_layers = []
        enable_layers.extend(_select_group_layers(llm_scores, llm_budget))
        enable_layers.extend(_select_group_layers(vit_scores, vit_budget))
    else:
        llm_budget_portion = float(llm_scores["score"].sum()) / total_score
        vit_budget_portion = float(vit_scores["score"].sum()) / total_score
        vit_budget = max(
            int(total_budget * 0.25),
            int(total_budget * vit_budget_portion + 0.5),
        )
        vit_budget = min(vit_budget, len(vit_scores), total_budget)
        llm_budget = min(total_budget - vit_budget, len(llm_scores))
        remaining_budget = total_budget - (vit_budget + llm_budget)
        if remaining_budget > 0:
            if llm_budget_portion >= vit_budget_portion:
                extra_llm = min(remaining_budget, len(llm_scores) - llm_budget)
                llm_budget += extra_llm
                remaining_budget -= extra_llm
            if remaining_budget > 0:
                extra_vit = min(remaining_budget, len(vit_scores) - vit_budget)
                vit_budget += extra_vit
                remaining_budget -= extra_vit
        if remaining_budget != 0:
            raise ValueError(
                "unable to allocate the full D-MoLE layer budget across modalities"
            )

        enable_layers = []
        enable_layers.extend(_select_group_layers(llm_scores, llm_budget))
        enable_layers.extend(_select_group_layers(vit_scores, vit_budget))

    if len(enable_layers) != total_budget:
        raise AssertionError(
            f"expected {total_budget} enabled layers, got {len(enable_layers)}"
        )

    if enable_layers and "base_model" in enable_layers[0]:
        return [".".join(layer.split(".")[2:]) for layer in enable_layers]
    return [".".join(layer.split(".")[0:]) for layer in enable_layers]
