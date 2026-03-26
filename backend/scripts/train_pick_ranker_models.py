from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
from xgboost import XGBRanker

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.scripts.train_ban_ranker_models import BAN_RANKER_CANDIDATE_PARAMS
from backend.services.common.file_utils import load_json, save_json
from backend.services.modeling.dataset_builder import build_pick_fit_dataset
from backend.services.modeling.feature_engineering_profile import (
    FEATURE_ENGINEERING_PROFILE_PATH,
    FeatureEngineeringProfile,
    load_feature_engineering_profile,
    tune_feature_engineering_profile,
)
from backend.services.modeling.features import PROCESSED_STATS_PATH
from backend.services.modeling.hero_power_model import (
    HERO_POWER_PROFILE_PATH,
    build_current_hero_power_profile,
)
from backend.services.modeling.pick_signal_model import (
    PICK_SIGNAL_PROFILE_PATH,
    build_pick_signal_frame,
    build_pick_signal_profile,
)
from backend.services.modeling.pick_order_profiles import (
    PICK_ORDER_PROFILE_PATH,
    score_pick_order_profiles_frame,
    train_pick_order_profiles,
)
from backend.services.modeling.pick_training import (
    attach_scores,
    build_pick_feature_columns,
    evaluate_prediction_frame,
)
from backend.services.modeling.training import (
    chronological_split,
    load_dataset_frame,
    query_group_sizes,
    sort_for_grouped_ranking,
    tune_xgb_ranker_params,
)

DATASET_PATH = Path("backend/data/modeling/pick_fit_ranker_dataset.json")
OUTPUT_DIR = Path("backend/data/modeling/models")
MODEL_PATH = OUTPUT_DIR / "pick_xgb_ranker_global.json"
REPORT_PATH = OUTPUT_DIR / "pick_ranker_report.json"
RAW_TOURNAMENTS_DIR = Path("backend/data/raw/tournaments")

PICK_RANKER_CANDIDATE_PARAMS = [
    {
        "n_estimators": 180,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
    },
    {
        "n_estimators": 240,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
    },
    {
        "n_estimators": 320,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
    },
    {
        "n_estimators": 260,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
    },
    {
        "n_estimators": 360,
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
        description="Train the pick-fit XGBoost ranker from the latest raw tournament data."
    )
    parser.add_argument(
        "--retune-feature-profile",
        action="store_true",
        help="Retune the feature engineering profile before training instead of reusing the saved profile.",
    )
    parser.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="Reuse the existing pick_fit_ranker_dataset.json instead of rebuilding it from raw tournament data.",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="Persist the rebuilt pick-fit dataset for reuse on later runs.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed_stats = load_json(PROCESSED_STATS_PATH)
    if not isinstance(processed_stats, dict):
        raise ValueError(f"Expected processed hero stats dict at {PROCESSED_STATS_PATH}")

    feature_profile_started_at = perf_counter()
    if args.retune_feature_profile or not FEATURE_ENGINEERING_PROFILE_PATH.exists():
        feature_profile = tune_feature_engineering_profile(
            processed_stats=processed_stats,
            processed_stats_path=PROCESSED_STATS_PATH,
            raw_dir=RAW_TOURNAMENTS_DIR,
            pick_candidate_params=PICK_RANKER_CANDIDATE_PARAMS,
            ban_candidate_params=BAN_RANKER_CANDIDATE_PARAMS,
        )
        save_json(FEATURE_ENGINEERING_PROFILE_PATH, feature_profile)
        print(
            f"Saved feature engineering profile to {FEATURE_ENGINEERING_PROFILE_PATH} "
            f"in {perf_counter() - feature_profile_started_at:.1f}s"
        )
    else:
        feature_profile = load_feature_engineering_profile()
        print(
            f"Reusing feature engineering profile from {FEATURE_ENGINEERING_PROFILE_PATH} "
            f"in {perf_counter() - feature_profile_started_at:.1f}s"
        )

    if args.reuse_dataset and DATASET_PATH.exists():
        metadata, df = load_dataset_frame(DATASET_PATH)
        if _dataset_matches_feature_profile(metadata, feature_profile):
            print(f"Reusing existing pick-fit dataset at {DATASET_PATH}")
        else:
            dataset = build_pick_fit_dataset(signals_only=False, feature_profile=feature_profile)
            metadata = dataset["metadata"]
            df = pd.DataFrame(dataset["rows"])
            save_json(DATASET_PATH, dataset)
            print(f"Rebuilt pick-fit dataset at {DATASET_PATH} to match the tuned feature profile.")
    else:
        dataset = build_pick_fit_dataset(signals_only=False, feature_profile=feature_profile)
        metadata = dataset["metadata"]
        df = pd.DataFrame(dataset["rows"])
        if args.save_dataset or args.reuse_dataset:
            save_json(DATASET_PATH, dataset)
            print(f"Saved pick-fit dataset to {DATASET_PATH}")
        else:
            print("Rebuilt pick-fit dataset in memory from raw tournaments.")

    if df.empty:
        raise ValueError("Pick-fit dataset is empty. Rebuild the dataset before training the pick ranker.")

    excluded_columns = {
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
    }
    columns = build_pick_feature_columns(df, excluded_columns)

    train_df, test_df = chronological_split(df, entity_column="query_id")
    train_sorted = sort_for_grouped_ranking(train_df, "query_id")
    test_sorted = sort_for_grouped_ranking(test_df, "query_id")

    tuned_params, validation_metrics = tune_xgb_ranker_params(
        train_df=train_df,
        columns=columns,
        label_column="label_is_pick_fit",
        query_column="query_id",
        candidate_params=PICK_RANKER_CANDIDATE_PARAMS,
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
        train_sorted["label_is_pick_fit"],
        group=query_group_sizes(train_sorted, "query_id"),
    )

    prediction_frames = {}
    prediction_name, prediction_frame = attach_scores(
        test_sorted,
        xgb_ranker.predict(test_sorted[columns].fillna(0.0)).tolist(),
        "xgb_ranker_global",
    )
    prediction_frames[prediction_name] = prediction_frame

    signal_profile = build_pick_signal_profile(
        feature_names=columns,
        feature_importances=xgb_ranker.feature_importances_.tolist(),
        model_name="pick_xgb_ranker_global",
    )
    train_signal_df = build_pick_signal_frame(
        train_df,
        query_column="query_id",
        signal_profile=signal_profile,
    )
    train_signal_df["prior_score"] = xgb_ranker.predict(train_df[columns].fillna(0.0)).tolist()
    order_profiles = train_pick_order_profiles(train_signal_df)

    signal_test_df = build_pick_signal_frame(
        test_df,
        query_column="query_id",
        signal_profile=signal_profile,
    )
    signal_test_df["prior_score"] = xgb_ranker.predict(test_df[columns].fillna(0.0)).tolist()
    signal_specs = {
        "signal_secure_power": "secure_power_signal",
        "signal_ally_role_completion": "ally_role_completion_signal",
        "signal_ally_pick_synergy": "ally_pick_synergy_signal",
        "signal_counter_vs_enemy_picks": "counter_vs_enemy_picks_signal",
        "signal_flexibility": "flexibility_signal",
        "signal_redundancy_penalty": "redundancy_penalty",
    }
    for signal_name, column_name in signal_specs.items():
        name, frame = attach_scores(test_df, signal_test_df[column_name].tolist(), signal_name)
        prediction_frames[name] = frame
    blended_name, blended_frame = attach_scores(
        test_df,
        score_pick_order_profiles_frame(signal_test_df, order_profiles).tolist(),
        "trained_slot_blend",
    )
    prediction_frames[blended_name] = blended_frame

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

    xgb_ranker.save_model(MODEL_PATH)
    save_json(REPORT_PATH, report)
    save_json(PICK_SIGNAL_PROFILE_PATH, signal_profile)
    save_json(PICK_ORDER_PROFILE_PATH, order_profiles)
    save_json(HERO_POWER_PROFILE_PATH, build_current_hero_power_profile())

    best_global = report["models"]["xgb_ranker_global"]["ranking"]["top3_hit_rate"]
    print(f"Saved pick ranker report to {REPORT_PATH}")
    print(f"Saved pick signal profile to {PICK_SIGNAL_PROFILE_PATH}")
    print(f"Saved pick order profiles to {PICK_ORDER_PROFILE_PATH}")
    print(f"Saved hero power profile to {HERO_POWER_PROFILE_PATH}")
    print(f"Pick ranker top-3 hit rate: {best_global:.4f}")
