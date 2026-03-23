from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBRanker

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.services.common.file_utils import save_json
from backend.services.modeling.dataset_builder import build_pick_fit_dataset
from backend.services.modeling.pick_signal_model import build_pick_signal_frame
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
)

DATASET_PATH = Path("backend/data/modeling/pick_fit_ranker_dataset.json")
OUTPUT_DIR = Path("backend/data/modeling/models")
MODEL_PATH = OUTPUT_DIR / "pick_xgb_ranker_global.json"
REPORT_PATH = OUTPUT_DIR / "pick_ranker_report.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the pick-fit XGBoost ranker from the latest raw tournament data."
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

    if args.reuse_dataset and DATASET_PATH.exists():
        print(f"Reusing existing pick-fit dataset at {DATASET_PATH}")
        metadata, df = load_dataset_frame(DATASET_PATH)
    else:
        dataset = build_pick_fit_dataset(signals_only=False)
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

    xgb_ranker = XGBRanker(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="rank:pairwise",
        eval_metric="ndcg@3",
        random_state=42,
        tree_method="hist",
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

    signal_test_df = build_pick_signal_frame(test_df)
    heuristic_specs = {
        "heuristic_secure_power_signal": "secure_power_signal",
        "heuristic_ally_role_completion_signal": "ally_role_completion_signal",
        "heuristic_hero_power": "candidate_hero_power",
    }
    for heuristic_name, column_name in heuristic_specs.items():
        source_df = signal_test_df if column_name in signal_test_df.columns else test_df
        name, frame = attach_scores(test_df, source_df[column_name].tolist(), heuristic_name)
        prediction_frames[name] = frame

    report = {
        "dataset_metadata": metadata,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(columns),
        "features": columns,
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

    best_global = report["models"]["xgb_ranker_global"]["ranking"]["top3_hit_rate"]
    print(f"Saved pick ranker report to {REPORT_PATH}")
    print(f"Pick ranker top-3 hit rate: {best_global:.4f}")
