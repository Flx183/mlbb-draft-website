from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
from xgboost import XGBRanker

from backend.services.common.file_utils import load_json

ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "backend/data/modeling/models"
PICK_SIGNAL_PROFILE_PATH = MODEL_DIR / "pick_signal_profile.json"
PICK_RANKER_REPORT_PATH = MODEL_DIR / "pick_ranker_report.json"
PICK_GLOBAL_RANKER_PATH = MODEL_DIR / "pick_xgb_ranker_global.json"

POSITIVE_SIGNAL_COLUMNS: tuple[str, ...] = (
    "secure_power_signal",
    "flexibility_signal",
    "ally_pick_synergy_signal",
    "counter_vs_enemy_picks_signal",
    "ally_role_completion_signal",
)
PENALTY_SIGNAL_COLUMN = "redundancy_penalty"
ALL_SIGNAL_COLUMNS: tuple[str, ...] = (*POSITIVE_SIGNAL_COLUMNS, PENALTY_SIGNAL_COLUMN)

SIGNAL_DIRECTION: dict[str, Literal["positive", "penalty"]] = {
    "secure_power_signal": "positive",
    "flexibility_signal": "positive",
    "ally_pick_synergy_signal": "positive",
    "counter_vs_enemy_picks_signal": "positive",
    "ally_role_completion_signal": "positive",
    "redundancy_penalty": "penalty",
}

SECURE_POWER_FEATURES: tuple[str, ...] = (
    "candidate_pick_rate",
    "candidate_ban_rate",
    "candidate_raw_win_rate",
    "candidate_adjusted_win_rate",
)
FLEXIBILITY_FEATURES: tuple[str, ...] = (
    "candidate_hero_flexibility",
    "candidate_flexibility_roles",
)
ROLE_OVERLAP_FEATURES: tuple[str, ...] = (
    "candidate_role_overlap_to_our_picks_mean",
    "candidate_role_overlap_to_our_picks_max",
)


class SignalFeature(TypedDict):
    feature: str
    importance: float
    weight: float


class PickSignalGroup(TypedDict):
    direction: Literal["positive", "penalty"]
    total_importance: float
    features: list[SignalFeature]


class PickSignalProfile(TypedDict):
    source: str
    model: str | None
    feature_count: int
    signal_groups: dict[str, PickSignalGroup]
    positive_signal_weights: dict[str, float]
    redundancy_penalty_weight: float


def _pick_signal_feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    available = set(feature_names)
    ordered_names = list(feature_names)
    return {
        "secure_power_signal": [feature for feature in SECURE_POWER_FEATURES if feature in available],
        "flexibility_signal": [feature for feature in FLEXIBILITY_FEATURES if feature in available],
        "ally_pick_synergy_signal": [
            feature for feature in ordered_names if feature.startswith("ally_pick_synergy_")
        ],
        "counter_vs_enemy_picks_signal": [
            feature for feature in ordered_names if feature.startswith("counter_vs_enemy_picks_")
        ],
        "ally_role_completion_signal": [
            feature for feature in ordered_names if feature.startswith("ally_role_")
        ],
        "redundancy_penalty": [feature for feature in ROLE_OVERLAP_FEATURES if feature in available],
    }


def _normalize_group_weights(weights_by_feature: dict[str, float]) -> dict[str, float]:
    positive_weights = {
        feature_name: max(0.0, float(importance))
        for feature_name, importance in weights_by_feature.items()
    }
    total = sum(positive_weights.values())
    if total <= 0 and positive_weights:
        uniform_weight = 1.0 / len(positive_weights)
        return {feature_name: uniform_weight for feature_name in positive_weights}
    if total <= 0:
        return {}
    return {
        feature_name: importance / total
        for feature_name, importance in positive_weights.items()
    }


def _normalize_positive_signal_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {
        signal_name: max(0.0, float(weights.get(signal_name, 0.0)))
        for signal_name in POSITIVE_SIGNAL_COLUMNS
    }
    total = sum(clipped.values())
    if total <= 0:
        equal_weight = 1.0 / len(POSITIVE_SIGNAL_COLUMNS)
        return {signal_name: equal_weight for signal_name in POSITIVE_SIGNAL_COLUMNS}
    return {
        signal_name: weight / total
        for signal_name, weight in clipped.items()
    }


def build_pick_signal_profile(
    feature_names: list[str],
    feature_importances: list[float],
    model_name: str | None = "pick_xgb_ranker_global",
) -> PickSignalProfile:
    if len(feature_names) != len(feature_importances):
        raise ValueError("Feature names and feature importances must have the same length.")

    importances_by_feature = {
        feature_name: max(0.0, float(feature_importance))
        for feature_name, feature_importance in zip(feature_names, feature_importances, strict=False)
    }
    feature_groups = _pick_signal_feature_groups(feature_names)

    signal_groups: dict[str, PickSignalGroup] = {}
    for signal_name in ALL_SIGNAL_COLUMNS:
        group_features = feature_groups.get(signal_name, [])
        group_importances = {
            feature_name: importances_by_feature.get(feature_name, 0.0)
            for feature_name in group_features
        }
        feature_weights = _normalize_group_weights(group_importances)
        total_importance = float(sum(group_importances.values()))
        signal_groups[signal_name] = {
            "direction": SIGNAL_DIRECTION[signal_name],
            "total_importance": total_importance,
            "features": [
                {
                    "feature": feature_name,
                    "importance": group_importances.get(feature_name, 0.0),
                    "weight": feature_weights.get(feature_name, 0.0),
                }
                for feature_name in group_features
            ],
        }

    positive_signal_weights = _normalize_positive_signal_weights(
        {
            signal_name: signal_groups[signal_name]["total_importance"]
            for signal_name in POSITIVE_SIGNAL_COLUMNS
        }
    )
    positive_total_importance = sum(
        signal_groups[signal_name]["total_importance"]
        for signal_name in POSITIVE_SIGNAL_COLUMNS
    )
    penalty_total_importance = signal_groups[PENALTY_SIGNAL_COLUMN]["total_importance"]
    redundancy_penalty_weight = (
        float(penalty_total_importance / positive_total_importance)
        if positive_total_importance > 0
        else 0.0
    )

    return {
        "source": "trained-xgbranker-feature-importances",
        "model": model_name,
        "feature_count": len(feature_names),
        "signal_groups": signal_groups,
        "positive_signal_weights": positive_signal_weights,
        "redundancy_penalty_weight": redundancy_penalty_weight,
    }


def _validate_pick_signal_profile(payload: Any) -> PickSignalProfile | None:
    if not isinstance(payload, dict):
        return None
    signal_groups = payload.get("signal_groups")
    positive_signal_weights = payload.get("positive_signal_weights")
    if not isinstance(signal_groups, dict) or not isinstance(positive_signal_weights, dict):
        return None

    normalized_positive_signal_weights = _normalize_positive_signal_weights(
        {
            signal_name: float(positive_signal_weights.get(signal_name, 0.0))
            for signal_name in POSITIVE_SIGNAL_COLUMNS
        }
    )
    validated_signal_groups: dict[str, PickSignalGroup] = {}
    for signal_name in ALL_SIGNAL_COLUMNS:
        group_payload = signal_groups.get(signal_name, {})
        if not isinstance(group_payload, dict):
            group_payload = {}
        raw_features = group_payload.get("features", [])
        if not isinstance(raw_features, list):
            raw_features = []

        weights_by_feature: dict[str, float] = {}
        importances_by_feature: dict[str, float] = {}
        feature_names: list[str] = []
        for item in raw_features:
            if not isinstance(item, dict) or not item.get("feature"):
                continue
            feature_name = str(item["feature"])
            feature_names.append(feature_name)
            importances_by_feature[feature_name] = max(0.0, float(item.get("importance", 0.0)))
            weights_by_feature[feature_name] = max(0.0, float(item.get("weight", 0.0)))
        normalized_feature_weights = _normalize_group_weights(weights_by_feature)
        validated_signal_groups[signal_name] = {
            "direction": SIGNAL_DIRECTION[signal_name],
            "total_importance": float(group_payload.get("total_importance", sum(importances_by_feature.values()))),
            "features": [
                {
                    "feature": feature_name,
                    "importance": importances_by_feature.get(feature_name, 0.0),
                    "weight": normalized_feature_weights.get(feature_name, 0.0),
                }
                for feature_name in feature_names
            ],
        }

    return {
        "source": str(payload.get("source", "trained-xgbranker-feature-importances")),
        "model": str(payload.get("model")) if payload.get("model") else None,
        "feature_count": int(payload.get("feature_count", 0) or 0),
        "signal_groups": validated_signal_groups,
        "positive_signal_weights": normalized_positive_signal_weights,
        "redundancy_penalty_weight": max(0.0, float(payload.get("redundancy_penalty_weight", 0.0))),
    }


def _load_ranker_feature_importances_from_artifacts() -> tuple[list[str], list[float]]:
    payload = load_json(PICK_RANKER_REPORT_PATH)
    if not isinstance(payload, dict):
        raise FileNotFoundError(f"Pick ranker report not found at {PICK_RANKER_REPORT_PATH}")
    feature_names = payload.get("features")
    if not isinstance(feature_names, list) or not all(isinstance(item, str) for item in feature_names):
        raise ValueError(f"Pick ranker report at {PICK_RANKER_REPORT_PATH} is missing feature names.")

    ranker = XGBRanker()
    ranker.load_model(PICK_GLOBAL_RANKER_PATH)
    feature_importances = ranker.feature_importances_.tolist()
    if len(feature_names) != len(feature_importances):
        raise ValueError("Pick ranker feature importances do not match the report feature names.")
    return feature_names, [float(value) for value in feature_importances]


@lru_cache(maxsize=1)
def load_pick_signal_profile(path: Path = PICK_SIGNAL_PROFILE_PATH) -> PickSignalProfile:
    if path.exists():
        validated_payload = _validate_pick_signal_profile(load_json(path))
        if validated_payload is not None:
            return validated_payload

    try:
        feature_names, feature_importances = _load_ranker_feature_importances_from_artifacts()
    except (FileNotFoundError, ValueError):
        raise FileNotFoundError(
            f"Pick signal profile not found at {path} and could not be rebuilt from trained pick ranker artifacts."
        )

    return build_pick_signal_profile(
        feature_names=feature_names,
        feature_importances=feature_importances,
        model_name="pick_xgb_ranker_global",
    )


def _percentile_series(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(numeric_series) <= 1 or numeric_series.nunique(dropna=False) <= 1:
        return pd.Series(0.0, index=series.index, dtype=float)
    ranks = numeric_series.rank(method="average")
    return ((ranks - 1.0) / max(len(numeric_series) - 1, 1)).astype(float)


def _build_trained_signal_group_frame(
    frame: pd.DataFrame,
    signal_profile: PickSignalProfile,
) -> pd.DataFrame:
    signal_frame = pd.DataFrame(index=frame.index)
    our_picks_gate = (
        (pd.to_numeric(frame.get("our_picks_count", 0.0), errors="coerce").fillna(0.0) > 0).astype(float)
        if "our_picks_count" in frame.columns
        else pd.Series(1.0, index=frame.index, dtype=float)
    )

    for signal_name in ALL_SIGNAL_COLUMNS:
        signal_series = pd.Series(0.0, index=frame.index, dtype=float)
        group_payload = signal_profile["signal_groups"].get(signal_name)
        if group_payload:
            for feature_payload in group_payload["features"]:
                feature_name = feature_payload["feature"]
                if feature_name not in frame.columns:
                    continue
                signal_series += float(feature_payload["weight"]) * _percentile_series(frame[feature_name])
        if signal_name == PENALTY_SIGNAL_COLUMN:
            signal_series *= our_picks_gate
        signal_frame[signal_name] = signal_series.clip(lower=0.0, upper=1.0)

    return signal_frame

def build_pick_signal_frame(
    frame: pd.DataFrame,
    query_column: str | None = None,
    signal_profile: PickSignalProfile | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if all(column_name in frame.columns for column_name in ALL_SIGNAL_COLUMNS):
        return frame.copy()

    scored = frame.copy()
    resolved_profile = signal_profile or load_pick_signal_profile()
    resolved_query_column = query_column or ("query_id" if "query_id" in scored.columns else None)

    if resolved_query_column and resolved_query_column in scored.columns:
        signal_frame = (
            scored.groupby(resolved_query_column, sort=False, group_keys=False)
            .apply(lambda group: _build_trained_signal_group_frame(group, resolved_profile))
            .reindex(scored.index)
        )
    else:
        signal_frame = _build_trained_signal_group_frame(scored, resolved_profile)

    for column_name in ALL_SIGNAL_COLUMNS:
        scored[column_name] = signal_frame[column_name].astype(float)
    return scored


def weighted_signal_average(
    frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    total = sum(float(weights.get(signal_name, 0.0)) for signal_name in POSITIVE_SIGNAL_COLUMNS)
    if total <= 0:
        equal_weight = 1.0 / len(POSITIVE_SIGNAL_COLUMNS)
        total = 1.0
        weights = {signal_name: equal_weight for signal_name in POSITIVE_SIGNAL_COLUMNS}

    weighted_sum = sum(
        float(weights.get(signal_name, 0.0)) * frame[signal_name]
        for signal_name in POSITIVE_SIGNAL_COLUMNS
    )
    return weighted_sum / total


def pick_signal_prior_score(
    frame: pd.DataFrame,
    query_column: str | None = None,
    signal_profile: PickSignalProfile | None = None,
) -> pd.Series:
    resolved_profile = signal_profile or load_pick_signal_profile()
    scored = build_pick_signal_frame(
        frame,
        query_column=query_column,
        signal_profile=resolved_profile,
    )
    positive_signal_weights = (
        resolved_profile["positive_signal_weights"]
        if resolved_profile is not None
        else {signal_name: 1.0 / len(POSITIVE_SIGNAL_COLUMNS) for signal_name in POSITIVE_SIGNAL_COLUMNS}
    )
    redundancy_penalty_weight = (
        float(resolved_profile["redundancy_penalty_weight"])
        if resolved_profile is not None
        else 0.0
    )
    return weighted_signal_average(scored, positive_signal_weights) - (
        redundancy_penalty_weight * scored[PENALTY_SIGNAL_COLUMN]
    )
