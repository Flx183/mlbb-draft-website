from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from backend.services.common.file_utils import load_json

ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "backend/data/modeling/models"
PICK_SIGNAL_WEIGHTS_PATH = MODEL_DIR / "pick_signal_weights.json"

POSITIVE_SIGNAL_COLUMNS: tuple[str, ...] = (
    "secure_power_signal",
    "flexibility_signal",
    "ally_pick_synergy_signal",
    "counter_vs_enemy_picks_signal",
    "ally_role_completion_signal",
)
PENALTY_SIGNAL_COLUMN = "redundancy_penalty"
ALL_SIGNAL_COLUMNS: tuple[str, ...] = (*POSITIVE_SIGNAL_COLUMNS, PENALTY_SIGNAL_COLUMN)

DEFAULT_POSITIVE_SIGNAL_WEIGHTS: dict[str, float] = {
    "secure_power_signal": 0.34,
    "flexibility_signal": 0.08,
    "ally_pick_synergy_signal": 0.22,
    "counter_vs_enemy_picks_signal": 0.20,
    "ally_role_completion_signal": 0.16,
}
DEFAULT_REDUNDANCY_PENALTY_WEIGHT = 0.10


class PickSignalWeights(TypedDict):
    source: str
    model: str | None
    positive_signal_weights: dict[str, float]
    redundancy_penalty_weight: float
    intercept: float


def _neutral_pair_signal(value: float) -> float:
    return max(0.0, min(1.0, (float(value) - 0.5) * 2.0))


def _clip_scalar(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clipped(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def build_pick_signal_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        "secure_power_signal": _clip_scalar(
            0.50 * float(row.get("candidate_hero_power", 0.0))
            + 0.20 * float(row.get("candidate_adjusted_win_rate", 0.0))
            + 0.15 * float(row.get("candidate_pick_rate", 0.0))
            + 0.15 * float(row.get("candidate_ban_rate", 0.0))
        ),
        "flexibility_signal": _clip_scalar(float(row.get("candidate_hero_flexibility", 0.0))),
        "ally_pick_synergy_signal": _clip_scalar(
            0.65 * _neutral_pair_signal(float(row.get("ally_pick_synergy_synergy_score_max", 0.0)))
            + 0.35 * _neutral_pair_signal(float(row.get("ally_pick_synergy_synergy_score_mean", 0.0)))
        ),
        "counter_vs_enemy_picks_signal": _clip_scalar(
            0.65 * _neutral_pair_signal(float(row.get("counter_vs_enemy_picks_counter_score_max", 0.0)))
            + 0.35 * _neutral_pair_signal(float(row.get("counter_vs_enemy_picks_counter_score_mean", 0.0)))
        ),
        "ally_role_completion_signal": _clip_scalar(
            0.70 * float(row.get("ally_role_role_completion_max", 0.0))
            + 0.30 * float(row.get("ally_role_role_completion_mean", 0.0))
        ),
        "redundancy_penalty": float(row.get("candidate_role_overlap_to_our_picks_max", 0.0))
        * (1.0 if float(row.get("our_picks_count", 0.0)) > 0 else 0.0),
    }


def build_pick_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if all(column_name in frame.columns for column_name in ALL_SIGNAL_COLUMNS):
        return frame.copy()

    scored = frame.copy()
    signal_rows = [build_pick_signal_row(row) for row in scored.to_dict(orient="records")]
    signal_frame = pd.DataFrame(signal_rows, index=scored.index)
    for column_name in ALL_SIGNAL_COLUMNS:
        scored[column_name] = signal_frame[column_name]
    return scored


def _normalize_positive_signal_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {
        signal_name: max(0.0, float(weights.get(signal_name, 0.0)))
        for signal_name in POSITIVE_SIGNAL_COLUMNS
    }
    total = sum(clipped.values())
    if total <= 0:
        return DEFAULT_POSITIVE_SIGNAL_WEIGHTS.copy()
    return {
        signal_name: weight / total
        for signal_name, weight in clipped.items()
    }


def default_pick_signal_weights() -> PickSignalWeights:
    return {
        "source": "default",
        "model": None,
        "positive_signal_weights": DEFAULT_POSITIVE_SIGNAL_WEIGHTS.copy(),
        "redundancy_penalty_weight": DEFAULT_REDUNDANCY_PENALTY_WEIGHT,
        "intercept": 0.0,
    }


@lru_cache(maxsize=1)
def load_pick_signal_weights(path: Path = PICK_SIGNAL_WEIGHTS_PATH) -> PickSignalWeights:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return default_pick_signal_weights()

    signal_weights = payload.get("positive_signal_weights", {})
    if not isinstance(signal_weights, dict):
        return default_pick_signal_weights()

    try:
        redundancy_penalty_weight = max(
            0.0,
            float(payload.get("redundancy_penalty_weight", DEFAULT_REDUNDANCY_PENALTY_WEIGHT)),
        )
        intercept = float(payload.get("intercept", 0.0))
    except (TypeError, ValueError):
        return default_pick_signal_weights()

    return {
        "source": str(payload.get("source", "trained")),
        "model": str(payload.get("model")) if payload.get("model") else None,
        "positive_signal_weights": _normalize_positive_signal_weights(
            {signal_name: float(signal_weights.get(signal_name, 0.0)) for signal_name in POSITIVE_SIGNAL_COLUMNS}
        ),
        "redundancy_penalty_weight": (
            redundancy_penalty_weight if redundancy_penalty_weight > 0 else DEFAULT_REDUNDANCY_PENALTY_WEIGHT
        ),
        "intercept": intercept,
    }


def weighted_signal_average(
    frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    total = sum(float(weights.get(signal_name, 0.0)) for signal_name in POSITIVE_SIGNAL_COLUMNS)
    if total <= 0:
        total = sum(DEFAULT_POSITIVE_SIGNAL_WEIGHTS.values())
        weights = DEFAULT_POSITIVE_SIGNAL_WEIGHTS

    weighted_sum = sum(
        float(weights.get(signal_name, 0.0)) * frame[signal_name]
        for signal_name in POSITIVE_SIGNAL_COLUMNS
    )
    return weighted_sum / total
