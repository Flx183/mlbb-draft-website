from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from xgboost import XGBRanker

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.services.common.file_utils import load_json
from backend.services.modeling.ban_constants import BAN_SEQUENCE
from backend.services.modeling.features import (
    PROCESSED_STATS_PATH,
    build_ban_candidate_feature_row,
    build_hero_feature_table,
    infer_missing_roles,
    summarize_candidate_counter,
    summarize_candidate_role_completion,
    summarize_candidate_synergy,
)

MODEL_DIR = ROOT_DIR / "backend/data/modeling/models"
RANKER_REPORT_PATH = MODEL_DIR / "ban_ranker_report.json"
GLOBAL_RANKER_PATH = MODEL_DIR / "ban_xgb_ranker_global.json"
PROCESSED_STATS_ABS_PATH = ROOT_DIR / PROCESSED_STATS_PATH


def _normalize_team(team: str) -> str:
    team_name = team.strip().lower()
    if team_name not in {"blue", "red"}:
        raise ValueError("Team must be 'blue' or 'red'")
    return team_name


def _as_unique_heroes(hero_names: list[str] | None) -> list[str]:
    unique_names: list[str] = []
    seen: set[str] = set()
    for hero_name in hero_names or []:
        if not hero_name or hero_name in seen:
            continue
        seen.add(hero_name)
        unique_names.append(hero_name)
    return unique_names


@lru_cache(maxsize=1)
def _load_ranker_features() -> list[str]:
    payload = load_json(RANKER_REPORT_PATH)
    if not isinstance(payload, dict):
        raise FileNotFoundError(
            f"Ranker report not found at {RANKER_REPORT_PATH}. Run train_ban_ranker_models.py first."
        )

    feature_names = payload.get("features", [])
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(f"Expected feature list in {RANKER_REPORT_PATH}")

    return [str(feature_name) for feature_name in feature_names]


def _load_ranker(path: Path) -> XGBRanker:
    if not path.exists():
        raise FileNotFoundError(f"Expected trained ranker at {path}")

    model = XGBRanker()
    model.load_model(path)
    return model


@lru_cache(maxsize=1)
def _load_global_ranker() -> XGBRanker:
    return _load_ranker(GLOBAL_RANKER_PATH)


@lru_cache(maxsize=1)
def _load_hero_table() -> dict[str, Any]:
    return build_hero_feature_table(PROCESSED_STATS_ABS_PATH)


@lru_cache(maxsize=1)
def _load_complete_stats() -> dict[str, Any]:
    payload = load_json(PROCESSED_STATS_ABS_PATH)
    if not isinstance(payload, dict):
        raise FileNotFoundError(f"Processed hero stats not found at {PROCESSED_STATS_ABS_PATH}")
    return payload


def _current_turn_index(blue_bans: list[str], red_bans: list[str]) -> int:
    return len(blue_bans) + len(red_bans)


def resolve_next_ban_turn(
    blue_bans: list[str] | None = None,
    red_bans: list[str] | None = None,
    team: str | None = None,
    strict_turn: bool = True,
) -> dict[str, Any]:
    blue_bans = _as_unique_heroes(blue_bans)
    red_bans = _as_unique_heroes(red_bans)
    turn_index = _current_turn_index(blue_bans, red_bans)

    if turn_index >= len(BAN_SEQUENCE):
        raise ValueError("No remaining ban turns in the current draft state")

    current_team, ban_order, draft_turn = BAN_SEQUENCE[turn_index]
    if team is not None:
        requested_team = _normalize_team(team)
        if strict_turn and requested_team != current_team:
            raise ValueError(
                f"It is currently {current_team}'s ban turn, not {requested_team}'s."
            )
        current_team = requested_team
        if current_team == "blue":
            ban_order = len(blue_bans) + 1
        else:
            ban_order = len(red_bans) + 1

    return {
        "team": current_team,
        "ban_order": int(ban_order),
        "turn_index": int(draft_turn),
        "phase_index": 1 if int(ban_order) <= 3 else 2,
    }


def _legal_ban_candidates(
    hero_table: dict[str, Any],
    blue_picks: list[str],
    red_picks: list[str],
    blue_bans: list[str],
    red_bans: list[str],
) -> list[str]:
    unavailable = set(blue_picks) | set(red_picks) | set(blue_bans) | set(red_bans)
    return [hero_name for hero_name in sorted(hero_table["heroes"].keys()) if hero_name not in unavailable]


def _score_reason_flags(row: dict[str, Any], pool_frame: pd.DataFrame) -> list[str]:
    reasons: list[str] = []

    def is_top_quantile(column_name: str, quantile: float = 0.9) -> bool:
        if column_name not in pool_frame or pool_frame.empty:
            return False
        threshold = float(pool_frame[column_name].quantile(quantile))
        return float(row.get(column_name, 0.0)) >= threshold and float(row.get(column_name, 0.0)) > 0

    if is_top_quantile("candidate_current_slot_share"):
        reasons.append("strong fit for this exact ban slot")
    if is_top_quantile("candidate_phase_fit_share"):
        reasons.append("historically prioritized in this ban phase")
    if is_top_quantile("candidate_ban_rate"):
        reasons.append("high global ban contest rate")
    if is_top_quantile("candidate_hero_power"):
        reasons.append("high overall hero power")
    if is_top_quantile("enemy_pick_synergy_synergy_score_max"):
        reasons.append("fits strongly with revealed enemy picks")
    if is_top_quantile("counter_vs_our_picks_counter_score_max"):
        reasons.append("threatens our revealed picks if left open")
    if is_top_quantile("enemy_role_role_completion_max"):
        reasons.append("cleanly completes an enemy missing role")

    if not reasons:
        reasons.append("best overall ranker score for the current ban state")

    return reasons[:3]


def _feature_frame(rows: list[dict[str, Any]], feature_names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for feature_name in feature_names:
        if feature_name not in frame.columns:
            frame[feature_name] = 0.0
    return frame


def _context_sort_frame(
    frame: pd.DataFrame,
    acting_team: str,
    blue_picks: list[str],
    red_picks: list[str],
    hero_table: dict[str, Any],
    complete_stats: dict[str, Any],
    top_pool_size: int,
) -> pd.DataFrame:
    if not blue_picks and not red_picks:
        frame["context_peak"] = 0.0
        frame["context_support"] = 0.0
        frame["final_score"] = frame["prior_score"]
        return frame.sort_values(by=["final_score", "candidate_ban_rate"], ascending=False).reset_index(drop=True)

    enemy_picks = red_picks if acting_team == "blue" else blue_picks
    our_picks = blue_picks if acting_team == "blue" else red_picks
    enemy_missing_roles = infer_missing_roles(enemy_picks, hero_table)

    top_pool = frame.head(top_pool_size).copy()
    for row_index, row in top_pool.iterrows():
        candidate_hero = str(row["candidate_hero"])

        synergy_summary = summarize_candidate_synergy(
            candidate_hero,
            enemy_picks,
            complete_stats,
            "enemy_pick_synergy",
        )
        counter_summary = summarize_candidate_counter(
            candidate_hero,
            our_picks,
            complete_stats,
            "counter_vs_our_picks",
        )
        role_summary = summarize_candidate_role_completion(
            candidate_hero,
            enemy_missing_roles,
            hero_table,
            "enemy_role",
        )

        for feature_name, feature_value in {
            **synergy_summary,
            **counter_summary,
            **role_summary,
        }.items():
            top_pool.loc[row_index, feature_name] = float(feature_value)

    context_columns = [
        "enemy_pick_synergy_synergy_score_max",
        "enemy_pick_synergy_synergy_score_mean",
        "counter_vs_our_picks_counter_score_max",
        "counter_vs_our_picks_counter_score_mean",
        "enemy_role_role_completion_max",
        "enemy_role_role_completion_mean",
    ]
    for column_name in context_columns:
        if column_name not in top_pool.columns:
            top_pool[column_name] = 0.0

    top_pool["context_peak"] = top_pool[
        [
            "enemy_pick_synergy_synergy_score_max",
            "counter_vs_our_picks_counter_score_max",
            "enemy_role_role_completion_max",
        ]
    ].max(axis=1)
    top_pool["context_support"] = top_pool[
        [
            "enemy_pick_synergy_synergy_score_mean",
            "counter_vs_our_picks_counter_score_mean",
            "enemy_role_role_completion_mean",
        ]
    ].mean(axis=1)
    top_pool["final_score"] = top_pool["prior_score"]

    remaining_pool = frame.iloc[top_pool_size:].copy()
    if not remaining_pool.empty:
        remaining_pool["context_peak"] = 0.0
        remaining_pool["context_support"] = 0.0
        remaining_pool["final_score"] = remaining_pool["prior_score"]

    reranked_top = top_pool.sort_values(
        by=[
            "context_peak",
            "context_support",
            "prior_score",
            "candidate_ban_rate",
        ],
        ascending=False,
    )
    combined = pd.concat([reranked_top, remaining_pool], axis=0, ignore_index=True)

    return combined.reset_index(drop=True)


def recommend_next_bans(
    blue_picks: list[str] | None = None,
    red_picks: list[str] | None = None,
    blue_bans: list[str] | None = None,
    red_bans: list[str] | None = None,
    team: str | None = None,
    top_k: int = 3,
    strict_turn: bool = True,
    rerank_pool_size: int | None = None,
) -> dict[str, Any]:
    blue_picks = _as_unique_heroes(blue_picks)
    red_picks = _as_unique_heroes(red_picks)
    blue_bans = _as_unique_heroes(blue_bans)
    red_bans = _as_unique_heroes(red_bans)

    turn = resolve_next_ban_turn(blue_bans=blue_bans, red_bans=red_bans, team=team, strict_turn=strict_turn)
    hero_table = _load_hero_table()
    feature_names = _load_ranker_features()
    candidate_heroes = _legal_ban_candidates(hero_table, blue_picks, red_picks, blue_bans, red_bans)

    if not candidate_heroes:
        return {
            **turn,
            "recommendations": [],
        }

    rows: list[dict[str, Any]] = []
    for candidate_hero in candidate_heroes:
        feature_row = build_ban_candidate_feature_row(
            candidate_hero=candidate_hero,
            acting_team=turn["team"],
            ban_order=turn["ban_order"],
            prior_blue_bans=blue_bans,
            prior_red_bans=red_bans,
            hero_table=hero_table,
        )
        rows.append(
            {
                "candidate_hero": candidate_hero,
                **feature_row,
            }
        )

    feature_frame = _feature_frame(rows, feature_names)
    ranker = _load_global_ranker()
    scores = ranker.predict(feature_frame[feature_names].fillna(0.0)).tolist()
    feature_frame["prior_score"] = scores
    sorted_prior_frame = feature_frame.sort_values(
        by=["prior_score", "candidate_ban_rate"],
        ascending=False,
    ).reset_index(drop=True)
    sorted_frame = _context_sort_frame(
        frame=sorted_prior_frame,
        acting_team=turn["team"],
        blue_picks=blue_picks,
        red_picks=red_picks,
        hero_table=hero_table,
        complete_stats=_load_complete_stats(),
        top_pool_size=max(top_k * 4, 12) if rerank_pool_size is None else max(1, rerank_pool_size),
    )

    recommendations: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted_frame.head(max(1, top_k)).to_dict(orient="records"), start=1):
        recommendations.append(
            {
                "hero": row["candidate_hero"],
                "rank": rank,
                "score": float(row.get("final_score", row.get("prior_score", 0.0))),
                "score_components": {
                    "prior_score": float(row.get("prior_score", 0.0)),
                    "context_peak": float(row.get("context_peak", 0.0)),
                    "context_support": float(row.get("context_support", 0.0)),
                    "current_slot_share": float(row.get("candidate_current_slot_share", 0.0)),
                    "phase_fit_share": float(row.get("candidate_phase_fit_share", 0.0)),
                    "ban_rate": float(row.get("candidate_ban_rate", 0.0)),
                    "hero_power": float(row.get("candidate_hero_power", 0.0)),
                    "enemy_pick_synergy_max": float(row.get("enemy_pick_synergy_synergy_score_max", 0.0)),
                    "counter_vs_our_picks_max": float(row.get("counter_vs_our_picks_counter_score_max", 0.0)),
                    "enemy_role_completion_max": float(row.get("enemy_role_role_completion_max", 0.0)),
                },
                "reasons": _score_reason_flags(row, sorted_frame),
            }
        )

    return {
        **turn,
        "candidate_count": int(len(candidate_heroes)),
        "rerank_pool_size": int(
            min(
                len(candidate_heroes),
                max(top_k * 4, 12) if rerank_pool_size is None else max(1, rerank_pool_size),
            )
        ),
        "recommendations": recommendations,
    }


def simulate_ban_sequence(
    blue_picks: list[str] | None = None,
    red_picks: list[str] | None = None,
    blue_bans: list[str] | None = None,
    red_bans: list[str] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    state = {
        "blue_picks": _as_unique_heroes(blue_picks),
        "red_picks": _as_unique_heroes(red_picks),
        "blue_bans": _as_unique_heroes(blue_bans),
        "red_bans": _as_unique_heroes(red_bans),
    }
    steps: list[dict[str, Any]] = []

    while _current_turn_index(state["blue_bans"], state["red_bans"]) < len(BAN_SEQUENCE):
        step = recommend_next_bans(
            blue_picks=state["blue_picks"],
            red_picks=state["red_picks"],
            blue_bans=state["blue_bans"],
            red_bans=state["red_bans"],
            team=None,
            top_k=top_k,
            strict_turn=True,
        )
        steps.append(step)

        if not step["recommendations"]:
            break

        selected_hero = step["recommendations"][0]["hero"]
        if step["team"] == "blue":
            state["blue_bans"].append(selected_hero)
        else:
            state["red_bans"].append(selected_hero)

    return {
        "steps": steps,
        "blue_bans": state["blue_bans"],
        "red_bans": state["red_bans"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend ordered bans from the trained XGBoost ranker.")
    parser.add_argument("--team", default="blue", choices=["blue", "red"])
    parser.add_argument("--blue-picks", default="")
    parser.add_argument("--red-picks", default="")
    parser.add_argument("--blue-bans", default="")
    parser.add_argument("--red-bans", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    def parse_csv(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    if args.simulate:
        payload = simulate_ban_sequence(
            blue_picks=parse_csv(args.blue_picks),
            red_picks=parse_csv(args.red_picks),
            blue_bans=parse_csv(args.blue_bans),
            red_bans=parse_csv(args.red_bans),
            top_k=args.top_k,
        )
    else:
        payload = recommend_next_bans(
            blue_picks=parse_csv(args.blue_picks),
            red_picks=parse_csv(args.red_picks),
            blue_bans=parse_csv(args.blue_bans),
            red_bans=parse_csv(args.red_bans),
            team=args.team,
            top_k=args.top_k,
        )

    print(json.dumps(payload, indent=2))
