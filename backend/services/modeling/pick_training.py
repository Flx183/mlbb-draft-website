from __future__ import annotations

from typing import Sequence, TypeAlias

import pandas as pd

from backend.services.modeling.training import feature_columns, rank_metrics

MetricsReport: TypeAlias = dict[str, float]
GroupedMetricsReport: TypeAlias = dict[str, MetricsReport]
PredictionReport: TypeAlias = dict[str, MetricsReport | GroupedMetricsReport]


def build_pick_feature_columns(df: pd.DataFrame, excluded_columns: set[str]) -> list[str]:
    return feature_columns(df, excluded_columns)


def evaluate_prediction_frame(frame: pd.DataFrame) -> PredictionReport:
    return {
        "ranking": rank_metrics(frame, "query_id", "label_is_pick_fit", "score"),
        "by_slot_index": {
            str(int(slot_index)): rank_metrics(
                subset,
                "query_id",
                "label_is_pick_fit",
                "score",
            )
            for slot_index, subset in frame.groupby("slot_index", sort=True)
        },
        "by_team": {
            team_name: rank_metrics(
                subset,
                "query_id",
                "label_is_pick_fit",
                "score",
            )
            for team_name, subset in frame.groupby("team", sort=True)
        },
    }


def attach_scores(
    df: pd.DataFrame,
    scores: Sequence[float],
    score_name: str,
) -> tuple[str, pd.DataFrame]:
    frame = df[
        [
            "query_id",
            "team",
            "slot_index",
            "candidate_hero",
            "label_is_pick_fit",
        ]
    ].copy()
    frame["score"] = list(scores)
    return score_name, frame
