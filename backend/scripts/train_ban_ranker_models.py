from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xgboost import XGBRanker

from backend.services.common.file_utils import save_json
from backend.services.modeling.dataset_builder import build_ban_dataset
from backend.services.modeling.feature_engineering_profile import (
    FeatureEngineeringProfile,
    load_feature_engineering_profile,
)
from backend.services.modeling.hero_power_model import (
    HERO_POWER_PROFILE_PATH,
    build_current_hero_power_profile,
)
from backend.services.modeling.ban_training import (
    attach_scores,
    build_ban_feature_columns,
    evaluate_prediction_frame,
    refresh_processed_stats,
)
from backend.services.modeling.training import (
    chronological_split,
    load_dataset_frame,
    query_group_sizes,
    sort_for_grouped_ranking,
    tune_xgb_ranker_params,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
    
DATASET_PATH = Path("backend/data/modeling/ban_dataset.json")
OUTPUT_DIR = Path("backend/data/modeling/models")
RAW_TOURNAMENTS_DIR = Path("backend/data/raw/tournaments")
PROCESSED_DIR = Path("backend/data/processed")

BAN_RANKER_CANDIDATE_PARAMS = [
    {
        "n_estimators": 220,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
    },
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
    },
    {
        "n_estimators": 360,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
    },
    {
        "n_estimators": 320,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
    },
    {
        "n_estimators": 420,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.95,
        "colsample_bytree": 0.85,
    },
]


def _feature_profile_signature(profile: FeatureEngineeringProfile) -> dict[str, float]:
    return {
        "adjusted_win_rate_smoothing_games": float(profile["adjusted_win_rate_smoothing_games"]),
        "flexibility_role_threshold": round(float(profile["flexibility_role_threshold"]), 4),
        "pair_prior_games": float(profile["pair_prior_games"]),
    }


def _dataset_matches_feature_profile(metadata: dict[str, object], profile: FeatureEngineeringProfile) -> bool:
    payload = metadata.get("feature_engineering_profile", {})
    if not isinstance(payload, dict):
        return False
    return {
        "adjusted_win_rate_smoothing_games": float(payload.get("adjusted_win_rate_smoothing_games", -1)),
        "flexibility_role_threshold": round(float(payload.get("flexibility_role_threshold", -1.0)), 4),
        "pair_prior_games": float(payload.get("pair_prior_games", -1)),
    } == _feature_profile_signature(profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the ban ranker from the latest raw tournament data."
    )
    parser.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="Reuse the existing ban_dataset.json instead of rebuilding it from raw tournament data.",
    )
    parser.add_argument(
        "--skip-processed-refresh",
        action="store_true",
        help="Skip rebuilding processed stats from raw tournament files before training.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_processed_refresh:
        raw_file_count = refresh_processed_stats(RAW_TOURNAMENTS_DIR, PROCESSED_DIR)
        print(
            "Refreshed processed stats from "
            f"{raw_file_count} raw tournament file{'s' if raw_file_count != 1 else ''}."
        )
    feature_profile = load_feature_engineering_profile()

    if args.reuse_dataset and DATASET_PATH.exists():
        metadata, _ = load_dataset_frame(DATASET_PATH)
        if _dataset_matches_feature_profile(metadata, feature_profile):
            print(f"Reusing existing ban dataset at {DATASET_PATH}")
        else:
            save_json(DATASET_PATH, build_ban_dataset(feature_profile=feature_profile))
            print(f"Rebuilt ban dataset from raw tournaments at {DATASET_PATH} to match the tuned feature profile.")
    else:
        save_json(DATASET_PATH, build_ban_dataset(feature_profile=feature_profile))
        print(f"Rebuilt ban dataset from raw tournaments at {DATASET_PATH}")

    metadata, df = load_dataset_frame(DATASET_PATH)
    if df.empty:
        raise ValueError(
            "Ban dataset is empty. Fetch tournament data and rebuild processed stats before training."
        )
    excluded_columns = {
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
    }
    columns = build_ban_feature_columns(df, excluded_columns)

    train_df, test_df = chronological_split(df, entity_column="query_id")
    train_sorted = sort_for_grouped_ranking(train_df, "query_id")
    test_sorted = sort_for_grouped_ranking(test_df, "query_id")

    tuned_params, validation_metrics = tune_xgb_ranker_params(
        train_df=train_df,
        columns=columns,
        label_column="label_is_ban",
        query_column="query_id",
        candidate_params=BAN_RANKER_CANDIDATE_PARAMS,
    )
    xgb_ranker = XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@3",
        random_state=42,
        tree_method="hist",
        **tuned_params,
    )
    xgb_ranker.fit(
        train_sorted[columns].fillna(0.0),
        train_sorted["label_is_ban"],
        group=query_group_sizes(train_sorted, "query_id"),
    )

    prediction_frames = {}
    prediction_name, prediction_frame = attach_scores(
        test_sorted,
        xgb_ranker.predict(test_sorted[columns].fillna(0.0)).tolist(),
        "xgb_ranker_global",
    )
    prediction_frames[prediction_name] = prediction_frame

    heuristic_specs = {
        "heuristic_ban_rate": "candidate_ban_rate",
        "heuristic_current_slot_share": "candidate_current_slot_share",
        "heuristic_phase_fit_share": "candidate_phase_fit_share",
        "heuristic_hero_power": "candidate_hero_power",
    }
    for heuristic_name, column_name in heuristic_specs.items():
        name, frame = attach_scores(test_df, test_df[column_name].tolist(), heuristic_name)
        prediction_frames[name] = frame

    report = {
        "dataset_metadata": metadata,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(columns),
        "features": columns,
        "tuned_ranker_params": tuned_params,
        "validation_metrics": validation_metrics,
        "top_global_ranker_features": [
            {
                "feature": feature_name,
                "importance": float(importance),
            }
            for feature_name, importance in sorted(
                zip(columns, xgb_ranker.feature_importances_.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        ],
        "models": {
            model_name: evaluate_prediction_frame(frame)
            for model_name, frame in prediction_frames.items()
        },
    }

    xgb_ranker.save_model(OUTPUT_DIR / "ban_xgb_ranker_global.json")
    save_json(OUTPUT_DIR / "ban_ranker_report.json", report)
    save_json(HERO_POWER_PROFILE_PATH, build_current_hero_power_profile())

    best_global = report["models"]["xgb_ranker_global"]["ranking"]["top3_hit_rate"]
    print(f"Saved ban ranker report to {OUTPUT_DIR / 'ban_ranker_report.json'}")
    print(f"Saved hero power profile to {HERO_POWER_PROFILE_PATH}")
    print(f"Global ranker top-3 hit rate: {best_global:.4f}")
