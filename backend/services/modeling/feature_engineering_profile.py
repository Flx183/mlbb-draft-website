from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from backend.services.common.file_utils import load_json

ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "backend/data/modeling/models"
FEATURE_ENGINEERING_PROFILE_PATH = MODEL_DIR / "feature_engineering_profile.json"


class FeatureEngineeringProfile(TypedDict):
    source: str
    objective: str
    adjusted_win_rate_smoothing_games: int
    flexibility_role_threshold: float
    pair_prior_games: int
    pick_validation_metrics: dict[str, float]
    ban_validation_metrics: dict[str, float]
    search_candidates: dict[str, list[float]]


def bootstrap_feature_engineering_profile() -> FeatureEngineeringProfile:
    return {
        "source": "bootstrap-default-feature-engineering-profile",
        "objective": "pending-feature-profile-training",
        "adjusted_win_rate_smoothing_games": 8,
        "flexibility_role_threshold": 0.15,
        "pair_prior_games": 4,
        "pick_validation_metrics": {},
        "ban_validation_metrics": {},
        "search_candidates": {},
    }


def _validated_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in payload.items()
        if isinstance(key, str)
    }


def _validated_candidate_lists(payload: Any) -> dict[str, list[float]]:
    if not isinstance(payload, dict):
        return {}
    validated: dict[str, list[float]] = {}
    for key, values in payload.items():
        if not isinstance(key, str) or not isinstance(values, list):
            continue
        validated[key] = [float(value) for value in values]
    return validated


def _validate_feature_engineering_profile(payload: Any) -> FeatureEngineeringProfile | None:
    if not isinstance(payload, dict):
        return None
    if not all(key in payload for key in ("adjusted_win_rate_smoothing_games", "flexibility_role_threshold", "pair_prior_games")):
        return None

    return {
        "source": str(payload.get("source", "trained-feature-engineering-profile")),
        "objective": str(payload.get("objective", "pick-then-ban-ranking")),
        "adjusted_win_rate_smoothing_games": int(payload["adjusted_win_rate_smoothing_games"]),
        "flexibility_role_threshold": float(payload["flexibility_role_threshold"]),
        "pair_prior_games": int(payload["pair_prior_games"]),
        "pick_validation_metrics": _validated_metrics(payload.get("pick_validation_metrics")),
        "ban_validation_metrics": _validated_metrics(payload.get("ban_validation_metrics")),
        "search_candidates": _validated_candidate_lists(payload.get("search_candidates")),
    }


@lru_cache(maxsize=1)
def load_feature_engineering_profile(
    path: Path = FEATURE_ENGINEERING_PROFILE_PATH,
) -> FeatureEngineeringProfile:
    validated_payload = _validate_feature_engineering_profile(load_json(path))
    if validated_payload is not None:
        return validated_payload
    return bootstrap_feature_engineering_profile()


def _quantile_candidates(
    values: list[float],
    quantiles: tuple[float, ...],
    *,
    integer: bool,
    minimum: float,
    maximum: float | None = None,
    baseline: float,
) -> list[float]:
    series = pd.Series([float(value) for value in values if float(value) > 0.0], dtype=float)
    candidates: set[float] = {float(baseline)}
    if not series.empty:
        for quantile in quantiles:
            raw_value = float(series.quantile(quantile))
            bounded_value = max(minimum, raw_value)
            if maximum is not None:
                bounded_value = min(maximum, bounded_value)
            if integer:
                candidates.add(float(int(round(bounded_value))))
            else:
                candidates.add(round(bounded_value, 4))
    return sorted(candidate for candidate in candidates if candidate >= minimum)


def derive_feature_engineering_candidates(processed_stats: dict[str, Any]) -> dict[str, list[float]]:
    heroes = processed_stats.get("heroes", {})
    if not isinstance(heroes, dict):
        raise ValueError("Processed stats payload is missing the hero table.")

    hero_pick_counts: list[float] = []
    role_probabilities: list[float] = []
    pair_game_counts: list[float] = []

    for payload in heroes.values():
        if not isinstance(payload, dict):
            continue
        stats = payload.get("stats", {})
        if not isinstance(stats, dict):
            continue

        picked = int(stats.get("picked", 0) or 0)
        if picked > 0:
            hero_pick_counts.append(float(picked))
            roles = stats.get("roles", {})
            if isinstance(roles, dict):
                for role_stats in roles.values():
                    if not isinstance(role_stats, dict):
                        continue
                    role_picks = int(role_stats.get("picked", 0) or 0)
                    if role_picks > 0:
                        role_probabilities.append(role_picks / picked)

        for matrix_name in ("synergy_matrix", "counter_matrix"):
            matrix = payload.get(matrix_name, {})
            if not isinstance(matrix, dict):
                continue
            for record in matrix.values():
                if not isinstance(record, dict):
                    continue
                games = int(record.get("games", 0) or 0)
                if games > 0:
                    pair_game_counts.append(float(games))

    return {
        "adjusted_win_rate_smoothing_games": _quantile_candidates(
            hero_pick_counts,
            (0.05, 0.15, 0.3, 0.5, 0.7),
            integer=True,
            minimum=1.0,
            baseline=8.0,
        ),
        "flexibility_role_threshold": _quantile_candidates(
            role_probabilities,
            (0.01, 0.05, 0.1, 0.15, 0.2),
            integer=False,
            minimum=0.01,
            maximum=0.5,
            baseline=0.15,
        ),
        "pair_prior_games": _quantile_candidates(
            pair_game_counts,
            (0.05, 0.15, 0.3, 0.5, 0.7),
            integer=True,
            minimum=1.0,
            baseline=4.0,
        ),
    }


def _profile_signature(profile: FeatureEngineeringProfile) -> tuple[int, float, int]:
    return (
        int(profile["adjusted_win_rate_smoothing_games"]),
        round(float(profile["flexibility_role_threshold"]), 4),
        int(profile["pair_prior_games"]),
    )


def _profile_comparison_key(
    pick_metrics: dict[str, float],
    ban_metrics: dict[str, float],
) -> tuple[float, float, float, float, float, float]:
    return (
        float(pick_metrics.get("ndcg_at_3", 0.0)),
        float(pick_metrics.get("top3_hit_rate", 0.0)),
        float(pick_metrics.get("top1_hit_rate", 0.0)),
        float(ban_metrics.get("ndcg_at_3", 0.0)),
        float(ban_metrics.get("top3_hit_rate", 0.0)),
        float(ban_metrics.get("top1_hit_rate", 0.0)),
    )


def _middle_candidate(values: list[float], *, integer: bool) -> float:
    ordered = sorted(values)
    midpoint = ordered[len(ordered) // 2]
    if integer:
        return float(int(round(midpoint)))
    return float(midpoint)


def _candidate_profile(
    smoothing_games: float,
    flexibility_role_threshold: float,
    pair_prior_games: float,
) -> FeatureEngineeringProfile:
    return {
        "source": "feature-engineering-tuning-candidate",
        "objective": "pick-then-ban-ranking",
        "adjusted_win_rate_smoothing_games": int(round(smoothing_games)),
        "flexibility_role_threshold": float(flexibility_role_threshold),
        "pair_prior_games": int(round(pair_prior_games)),
        "pick_validation_metrics": {},
        "ban_validation_metrics": {},
        "search_candidates": {},
    }


def _evaluate_feature_engineering_profile(
    profile: FeatureEngineeringProfile,
    processed_stats_path: Path,
    raw_dir: Path,
    pick_candidate_params: list[dict[str, Any]],
    ban_candidate_params: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    from backend.services.modeling.ban_training import build_ban_feature_columns
    from backend.services.modeling.dataset_builder import build_ban_dataset, build_pick_fit_dataset
    from backend.services.modeling.pick_training import build_pick_feature_columns
    from backend.services.modeling.training import chronological_split, tune_xgb_ranker_params

    pick_dataset = build_pick_fit_dataset(
        processed_stats_path=processed_stats_path,
        raw_dir=raw_dir,
        signals_only=False,
        feature_profile=profile,
    )
    pick_df = pd.DataFrame(pick_dataset["rows"])
    if pick_df.empty:
        raise ValueError("Pick dataset is empty during feature profile tuning.")
    pick_train_df, _ = chronological_split(pick_df, entity_column="query_id")
    pick_columns = build_pick_feature_columns(
        pick_df,
        {
            "query_id",
            "game_id",
            "date",
            "patch",
            "tournament",
            "source_file",
            "team",
            "slot_index",
            "actual_pick",
            "candidate_hero",
            "label_is_pick_fit",
        },
    )
    _, pick_metrics = tune_xgb_ranker_params(
        train_df=pick_train_df,
        columns=pick_columns,
        label_column="label_is_pick_fit",
        query_column="query_id",
        candidate_params=pick_candidate_params,
    )

    ban_metrics: dict[str, float] = {}
    if ban_candidate_params:
        ban_dataset = build_ban_dataset(
            processed_stats_path=processed_stats_path,
            raw_dir=raw_dir,
            feature_profile=profile,
        )
        ban_df = pd.DataFrame(ban_dataset["rows"])
        if ban_df.empty:
            raise ValueError("Ban dataset is empty during feature profile tuning.")
        ban_train_df, _ = chronological_split(ban_df, entity_column="query_id")
        ban_columns = build_ban_feature_columns(
            ban_df,
            {
                "query_id",
                "game_id",
                "date",
                "patch",
                "tournament",
                "source_file",
                "team",
                "actual_ban",
                "candidate_hero",
                "label_is_ban",
            },
        )
        _, ban_metrics = tune_xgb_ranker_params(
            train_df=ban_train_df,
            columns=ban_columns,
            label_column="label_is_ban",
            query_column="query_id",
            candidate_params=ban_candidate_params,
        )
    return pick_metrics, ban_metrics


def tune_feature_engineering_profile(
    processed_stats: dict[str, Any],
    processed_stats_path: Path,
    raw_dir: Path,
    pick_candidate_params: list[dict[str, Any]],
    ban_candidate_params: list[dict[str, Any]],
    rounds: int = 2,
) -> FeatureEngineeringProfile:
    candidate_values = derive_feature_engineering_candidates(processed_stats)
    current_profile = _candidate_profile(
        smoothing_games=_middle_candidate(candidate_values["adjusted_win_rate_smoothing_games"], integer=True),
        flexibility_role_threshold=_middle_candidate(
            candidate_values["flexibility_role_threshold"],
            integer=False,
        ),
        pair_prior_games=_middle_candidate(candidate_values["pair_prior_games"], integer=True),
    )

    evaluation_cache: dict[tuple[int, float, int], tuple[dict[str, float], dict[str, float]]] = {}

    def evaluate(profile: FeatureEngineeringProfile) -> tuple[dict[str, float], dict[str, float]]:
        signature = _profile_signature(profile)
        if signature not in evaluation_cache:
            evaluation_cache[signature] = _evaluate_feature_engineering_profile(
                profile=profile,
                processed_stats_path=processed_stats_path,
                raw_dir=raw_dir,
                pick_candidate_params=pick_candidate_params,
                ban_candidate_params=ban_candidate_params,
            )
        return evaluation_cache[signature]

    best_pick_metrics, best_ban_metrics = evaluate(current_profile)

    ordered_fields = (
        "adjusted_win_rate_smoothing_games",
        "flexibility_role_threshold",
        "pair_prior_games",
    )
    for _ in range(max(1, rounds)):
        improved = False
        for field_name in ordered_fields:
            field_best_profile = current_profile
            field_best_pick_metrics = best_pick_metrics
            field_best_ban_metrics = best_ban_metrics
            for candidate_value in candidate_values[field_name]:
                trial_profile = {
                    **current_profile,
                    field_name: (
                        int(round(candidate_value))
                        if field_name != "flexibility_role_threshold"
                        else float(candidate_value)
                    ),
                }
                trial_pick_metrics, trial_ban_metrics = evaluate(trial_profile)
                if _profile_comparison_key(trial_pick_metrics, trial_ban_metrics) > _profile_comparison_key(
                    field_best_pick_metrics,
                    field_best_ban_metrics,
                ):
                    field_best_profile = trial_profile
                    field_best_pick_metrics = trial_pick_metrics
                    field_best_ban_metrics = trial_ban_metrics
            if _profile_signature(field_best_profile) != _profile_signature(current_profile):
                current_profile = field_best_profile
                best_pick_metrics = field_best_pick_metrics
                best_ban_metrics = field_best_ban_metrics
                improved = True
        if not improved:
            break

    return {
        "source": "trained-feature-engineering-coordinate-search",
        "objective": (
            "maximize pick ndcg@3, then pick top-3, then ban ndcg@3, then ban top-3"
            if ban_candidate_params
            else "maximize pick ndcg@3, then pick top-3"
        ),
        "adjusted_win_rate_smoothing_games": int(current_profile["adjusted_win_rate_smoothing_games"]),
        "flexibility_role_threshold": float(current_profile["flexibility_role_threshold"]),
        "pair_prior_games": int(current_profile["pair_prior_games"]),
        "pick_validation_metrics": best_pick_metrics,
        "ban_validation_metrics": best_ban_metrics,
        "search_candidates": candidate_values,
    }
