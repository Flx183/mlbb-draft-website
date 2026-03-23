from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from backend.services.common.file_utils import load_json


def load_dataset_frame(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dataset dict at {path}")

    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected dataset rows list at {path}")

    return payload.get("metadata", {}), pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame, excluded_columns: set[str]) -> list[str]:
    columns: list[str] = []
    for column_name in df.columns:
        if column_name in excluded_columns:
            continue
        if is_numeric_dtype(df[column_name]):
            columns.append(column_name)
    return sorted(columns)


def chronological_split(
    df: pd.DataFrame,
    entity_column: str,
    date_column: str = "date",
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entity_dates = (
        df[[entity_column, date_column]]
        .drop_duplicates(subset=[entity_column])
        .sort_values(by=[date_column, entity_column], na_position="last")
        .reset_index(drop=True)
    )
    split_index = max(1, math.floor(len(entity_dates) * train_fraction))
    split_entities = set(entity_dates.iloc[:split_index][entity_column].tolist())

    train_df = df[df[entity_column].isin(split_entities)].reset_index(drop=True)
    test_df = df[~df[entity_column].isin(split_entities)].reset_index(drop=True)

    return train_df, test_df
def rank_metrics(
    df: pd.DataFrame,
    query_column: str,
    label_column: str,
    score_column: str,
) -> dict[str, float]:
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_ranks = 0.0
    query_count = 0
    actual_ranks: list[int] = []
    ndcg_at_3 = 0.0
    ndcg_at_5 = 0.0

    for _, group in df.groupby(query_column, sort=False):
        query_count += 1
        ranked = group.sort_values(score_column, ascending=False).reset_index(drop=True)
        labels = ranked[label_column].tolist()

        if labels and labels[0] == 1:
            top1_hits += 1
        if 1 in labels[:3]:
            top3_hits += 1
        if 1 in labels[:5]:
            top5_hits += 1
        if 1 in labels:
            actual_rank = labels.index(1) + 1
            actual_ranks.append(actual_rank)
            reciprocal_ranks += 1.0 / actual_rank

            if actual_rank <= 3:
                ndcg_at_3 += 1.0 / math.log2(actual_rank + 1)
            if actual_rank <= 5:
                ndcg_at_5 += 1.0 / math.log2(actual_rank + 1)

    if query_count == 0:
        return {
            "top1_hit_rate": 0.0,
            "top3_hit_rate": 0.0,
            "top5_hit_rate": 0.0,
            "mean_reciprocal_rank": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "ndcg_at_3": 0.0,
            "ndcg_at_5": 0.0,
            "query_count": 0.0,
        }

    return {
        "top1_hit_rate": top1_hits / query_count,
        "top3_hit_rate": top3_hits / query_count,
        "top5_hit_rate": top5_hits / query_count,
        "mean_reciprocal_rank": reciprocal_ranks / query_count,
        "mean_rank": float(sum(actual_ranks) / len(actual_ranks)) if actual_ranks else 0.0,
        "median_rank": float(statistics.median(actual_ranks)) if actual_ranks else 0.0,
        "ndcg_at_3": ndcg_at_3 / query_count,
        "ndcg_at_5": ndcg_at_5 / query_count,
        "query_count": float(query_count),
    }


def sort_for_grouped_ranking(df: pd.DataFrame, query_column: str) -> pd.DataFrame:
    return df.sort_values(by=[query_column]).reset_index(drop=True)


def query_group_sizes(df: pd.DataFrame, query_column: str) -> list[int]:
    return df.groupby(query_column, sort=False).size().tolist()
