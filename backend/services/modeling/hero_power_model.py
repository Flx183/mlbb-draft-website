from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from xgboost import XGBRanker

from backend.services.common.file_utils import load_json

ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "backend/data/modeling/models"
HERO_POWER_PROFILE_PATH = MODEL_DIR / "hero_power_profile.json"

PICK_RANKER_REPORT_PATH = MODEL_DIR / "pick_ranker_report.json"
PICK_GLOBAL_RANKER_PATH = MODEL_DIR / "pick_xgb_ranker_global.json"
BAN_RANKER_REPORT_PATH = MODEL_DIR / "ban_ranker_report.json"
BAN_GLOBAL_RANKER_PATH = MODEL_DIR / "ban_xgb_ranker_global.json"

HERO_POWER_FEATURE_NAMES: tuple[str, ...] = (
    "pick_rate",
    "ban_rate",
    "adjusted_win_rate",
)


class HeroPowerSource(TypedDict):
    source: str
    total_importance: float
    feature_importances: dict[str, float]


class HeroPowerProfile(TypedDict):
    source: str
    model_sources: list[HeroPowerSource]
    feature_weights: dict[str, float]
    total_importance: float


def bootstrap_hero_power_profile() -> HeroPowerProfile:
    equal_weight = 1.0 / len(HERO_POWER_FEATURE_NAMES)
    return {
        "source": "bootstrap-equal-weights",
        "model_sources": [],
        "feature_weights": {
            feature_name: equal_weight
            for feature_name in HERO_POWER_FEATURE_NAMES
        },
        "total_importance": 0.0,
    }


def _normalize_feature_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {
        feature_name: max(0.0, float(weights.get(feature_name, 0.0)))
        for feature_name in HERO_POWER_FEATURE_NAMES
    }
    total = sum(clipped.values())
    if total <= 0:
        equal_weight = 1.0 / len(HERO_POWER_FEATURE_NAMES)
        return {feature_name: equal_weight for feature_name in HERO_POWER_FEATURE_NAMES}
    return {
        feature_name: weight / total
        for feature_name, weight in clipped.items()
    }


def _ranker_feature_importances(
    source_name: str,
    report_path: Path,
    model_path: Path,
) -> HeroPowerSource | None:
    if not report_path.exists() or not model_path.exists():
        return None

    payload = load_json(report_path)
    if not isinstance(payload, dict):
        return None

    feature_names = payload.get("features")
    if not isinstance(feature_names, list) or not all(isinstance(item, str) for item in feature_names):
        return None

    ranker = XGBRanker()
    ranker.load_model(model_path)
    raw_importances = ranker.feature_importances_.tolist()
    if len(feature_names) != len(raw_importances):
        return None

    feature_importances = {
        "pick_rate": 0.0,
        "ban_rate": 0.0,
        "adjusted_win_rate": 0.0,
    }
    for feature_name, importance in zip(feature_names, raw_importances, strict=False):
        if feature_name == "candidate_pick_rate":
            feature_importances["pick_rate"] += float(importance)
        elif feature_name == "candidate_ban_rate":
            feature_importances["ban_rate"] += float(importance)
        elif feature_name == "candidate_adjusted_win_rate":
            feature_importances["adjusted_win_rate"] += float(importance)

    total_importance = float(sum(feature_importances.values()))
    return {
        "source": source_name,
        "total_importance": total_importance,
        "feature_importances": feature_importances,
    }


def build_hero_power_profile(model_sources: list[HeroPowerSource]) -> HeroPowerProfile:
    if not model_sources:
        raise FileNotFoundError("No trained ranker artifacts are available to build hero power weights.")

    combined_importances = {
        feature_name: 0.0
        for feature_name in HERO_POWER_FEATURE_NAMES
    }
    for source in model_sources:
        source_total = max(0.0, float(source["total_importance"]))
        if source_total <= 0:
            continue
        normalized_source = _normalize_feature_weights(source["feature_importances"])
        for feature_name in HERO_POWER_FEATURE_NAMES:
            combined_importances[feature_name] += source_total * normalized_source[feature_name]

    return {
        "source": "trained-ranker-importances",
        "model_sources": model_sources,
        "feature_weights": _normalize_feature_weights(combined_importances),
        "total_importance": float(sum(combined_importances.values())),
    }


def _validate_hero_power_profile(payload: Any) -> HeroPowerProfile | None:
    if not isinstance(payload, dict):
        return None
    feature_weights = payload.get("feature_weights")
    model_sources = payload.get("model_sources", [])
    if not isinstance(feature_weights, dict) or not isinstance(model_sources, list):
        return None

    validated_sources: list[HeroPowerSource] = []
    for source in model_sources:
        if not isinstance(source, dict):
            continue
        feature_importances = source.get("feature_importances", {})
        if not isinstance(feature_importances, dict):
            feature_importances = {}
        validated_sources.append(
            {
                "source": str(source.get("source", "unknown")),
                "total_importance": float(source.get("total_importance", 0.0)),
                "feature_importances": {
                    feature_name: float(feature_importances.get(feature_name, 0.0))
                    for feature_name in HERO_POWER_FEATURE_NAMES
                },
            }
        )

    return {
        "source": str(payload.get("source", "trained-ranker-importances")),
        "model_sources": validated_sources,
        "feature_weights": _normalize_feature_weights(
            {
                feature_name: float(feature_weights.get(feature_name, 0.0))
                for feature_name in HERO_POWER_FEATURE_NAMES
            }
        ),
        "total_importance": float(payload.get("total_importance", 0.0)),
    }


def build_current_hero_power_profile() -> HeroPowerProfile:
    model_sources = [
        source
        for source in [
            _ranker_feature_importances(
                "pick_xgb_ranker_global",
                PICK_RANKER_REPORT_PATH,
                PICK_GLOBAL_RANKER_PATH,
            ),
            _ranker_feature_importances(
                "ban_xgb_ranker_global",
                BAN_RANKER_REPORT_PATH,
                BAN_GLOBAL_RANKER_PATH,
            ),
        ]
        if source is not None
    ]
    if not model_sources:
        return bootstrap_hero_power_profile()
    return build_hero_power_profile(model_sources)


@lru_cache(maxsize=1)
def load_hero_power_profile(path: Path = HERO_POWER_PROFILE_PATH) -> HeroPowerProfile:
    if path.exists():
        validated_payload = _validate_hero_power_profile(load_json(path))
        if validated_payload is not None:
            return validated_payload
    return build_current_hero_power_profile()


def compute_hero_power(
    pick_rate_rank: float,
    ban_rate_rank: float,
    adjusted_win_rate_rank: float,
    profile: HeroPowerProfile | None = None,
) -> float:
    resolved_profile = profile or load_hero_power_profile()
    weights = resolved_profile["feature_weights"]
    return (
        float(weights["pick_rate"]) * float(pick_rate_rank)
        + float(weights["ban_rate"]) * float(ban_rate_rank)
        + float(weights["adjusted_win_rate"]) * float(adjusted_win_rate_rank)
    )
