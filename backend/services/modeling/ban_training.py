from __future__ import annotations

from pathlib import Path
from typing import Sequence, TypeAlias

import pandas as pd

from backend.services.common.file_utils import load_json, save_json
from backend.services.liquipedia.counter_stats import (
    build_counter_matrix_from_tournament,
    finalize_counter_stats,
    merge_counter_matrices,
)
from backend.services.liquipedia.hero_stats import (
    build_hero_stats_from_grouped_tournament,
    calculate_win_rates,
    combine_all_hero_stats,
    merge_hero_stats,
)
from backend.services.liquipedia.synergy_stats import (
    build_synergy_matrix_from_tournament,
    finalize_synergy_stats,
    merge_synergy_matrices,
)
from backend.services.modeling.training import feature_columns, rank_metrics

MetricsReport: TypeAlias = dict[str, float]
GroupedMetricsReport: TypeAlias = dict[str, MetricsReport]
PredictionReport: TypeAlias = dict[str, MetricsReport | GroupedMetricsReport]


def build_ban_feature_columns(df: pd.DataFrame, excluded_columns: set[str]) -> list[str]:
    columns = feature_columns(df, excluded_columns)
    blocked_tokens = (
        "ban_priority",
        "average_ban_order_priority",
    )
    return [
        column_name
        for column_name in columns
        if not any(blocked_token in column_name for blocked_token in blocked_tokens)
    ]


def evaluate_prediction_frame(frame: pd.DataFrame) -> PredictionReport:
    return {
        "ranking": rank_metrics(frame, "query_id", "label_is_ban", "score"),
        "by_ban_order": {
            str(int(ban_order)): rank_metrics(
                subset,
                "query_id",
                "label_is_ban",
                "score",
            )
            for ban_order, subset in frame.groupby("ban_order", sort=True)
        },
        "by_phase": {
            str(int(phase_index)): rank_metrics(
                subset,
                "query_id",
                "label_is_ban",
                "score",
            )
            for phase_index, subset in frame.groupby("phase_index", sort=True)
        },
        "by_team": {
            team_name: rank_metrics(
                subset,
                "query_id",
                "label_is_ban",
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
            "ban_order",
            "phase_index",
            "candidate_hero",
            "label_is_ban",
        ]
    ].copy()
    frame["score"] = list(scores)
    return score_name, frame


def refresh_processed_stats(raw_dir: Path, processed_dir: Path) -> int:
    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        raise FileNotFoundError(f"No tournament files found in {raw_dir}")

    combined_stats: dict[str, object] = {}
    combined_counters: dict[str, object] = {}
    combined_synergy: dict[str, object] = {}

    for file_path in raw_files:
        tournament_data = load_json(file_path)
        if not isinstance(tournament_data, dict):
            continue

        hero_stats = build_hero_stats_from_grouped_tournament(tournament_data)
        counter_matrix = build_counter_matrix_from_tournament(tournament_data)
        synergy_matrix = build_synergy_matrix_from_tournament(tournament_data)

        combined_stats = merge_hero_stats(combined_stats, hero_stats)
        combined_counters = merge_counter_matrices(combined_counters, counter_matrix)
        combined_synergy = merge_synergy_matrices(combined_synergy, synergy_matrix)

    final_hero_stats = calculate_win_rates(combined_stats)
    final_counter = finalize_counter_stats(combined_counters)
    final_synergy = finalize_synergy_stats(combined_synergy)
    combined_complete_stats = combine_all_hero_stats(final_hero_stats, final_counter, final_synergy)

    save_json(processed_dir / "all_hero_stats.json", final_hero_stats)
    save_json(processed_dir / "counter_matrices.json", final_counter)
    save_json(processed_dir / "synergy_matrices.json", final_synergy)
    save_json(processed_dir / "complete_hero_stats.json", combined_complete_stats)

    return len(raw_files)
