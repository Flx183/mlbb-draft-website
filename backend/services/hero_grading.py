from dataclasses import dataclass
from typing import Optional

from backend.services.common.hero_grade_utils import (
    GRADE_ORDER,
    base_grade,
    build_notes,
    confidence_label,
    grade_cap,
    infer_total_games,
    lower_grade,
    parse_percent,
    parse_pick_input,
    percentile_ranks,
    weighted_priority_score,
)
from backend.services.liquipedia.page_scraper import (
    DEFAULT_STAGE,
    DEFAULT_TOURNAMENT_NAME,
    get_liquipedia_hero_data,
)
from backend.services.modeling.hero_power_model import load_hero_power_profile


@dataclass
class HeroGradeRow:
    hero: str
    picks: int
    wins: float
    raw_win_rate: Optional[float]
    adjusted_win_rate: float
    pick_rate: float
    ban_rate: float
    presence: float
    ban_share: float
    priority_score: float
    hero_grade: str
    confidence: str
    notes: str


def build_raw_rows(
    hero_data: dict[str, dict[str, str]],
    total_games: Optional[int],
) -> list[dict[str, float | int | str | None]]:
    raw_rows: list[dict[str, float | int | str | None]] = []
    for hero, stats in hero_data.items():
        picks, scraped_pick_rate = parse_pick_input(stats, total_games)
        raw_win_rate = parse_percent(stats["win_rate"])
        ban_rate = parse_percent(stats["ban_rate"]) or 0.0
        wins = 0.0 if raw_win_rate is None else picks * (raw_win_rate / 100)
        raw_rows.append(
            {
                "hero": hero,
                "picks": picks,
                "wins": wins,
                "raw_win_rate": raw_win_rate,
                "scraped_pick_rate": scraped_pick_rate,
                "ban_rate": ban_rate,
            }
        )
    return raw_rows


def resolve_total_games_from_rows(
    raw_rows: list[dict[str, float | int | str | None]],
    total_games: Optional[int],
) -> int:
    if total_games is not None:
        return total_games

    inferred_total_games = infer_total_games([float(row["ban_rate"]) for row in raw_rows])
    if inferred_total_games is None:
        raise ValueError(
            "Could not infer total games from ban-rate increments. Pass --games explicitly."
        )
    return inferred_total_games


def calculate_global_win_rate(raw_rows: list[dict[str, float | int | str | None]]) -> float:
    total_picks = sum(int(row["picks"]) for row in raw_rows)
    total_wins = sum(float(row["wins"]) for row in raw_rows)
    return 0.5 if total_picks == 0 else total_wins / total_picks


def enrich_raw_rows(
    raw_rows: list[dict[str, float | int | str | None]],
    total_games: int,
    smoothing_games: int,
) -> None:
    global_win_rate = calculate_global_win_rate(raw_rows)

    for row in raw_rows:
        picks = int(row["picks"])
        wins = float(row["wins"])
        scraped_pick_rate = row["scraped_pick_rate"]
        ban_rate = float(row["ban_rate"])

        pick_rate = (
            float(scraped_pick_rate)
            if scraped_pick_rate is not None
            else (picks / total_games) * 100
        )
        presence = pick_rate + ban_rate
        ban_share = 0.0 if presence == 0 else ban_rate / presence
        adjusted_win_rate = (
            ((wins + (smoothing_games * global_win_rate)) / (picks + smoothing_games)) * 100
            if picks > 0
            else global_win_rate * 100
        )

        row["pick_rate"] = pick_rate
        row["presence"] = presence
        row["ban_share"] = ban_share
        row["adjusted_win_rate"] = adjusted_win_rate


def calculate_priority_scores(
    raw_rows: list[dict[str, float | int | str | None]],
) -> tuple[list[float], dict[str, float]]:
    profile = load_hero_power_profile()
    weights = {
        "pick_rate": float(profile["feature_weights"]["pick_rate"]),
        "ban_rate": float(profile["feature_weights"]["ban_rate"]),
        "adjusted_win_rate": float(profile["feature_weights"]["adjusted_win_rate"]),
    }

    pick_rate_ranks = percentile_ranks([float(row["pick_rate"]) for row in raw_rows])
    ban_rate_ranks = percentile_ranks([float(row["ban_rate"]) for row in raw_rows])
    adjusted_win_rate_ranks = percentile_ranks(
        [float(row["adjusted_win_rate"]) for row in raw_rows]
    )
    priority_scores = weighted_priority_score(
        pick_rate_ranks,
        ban_rate_ranks,
        adjusted_win_rate_ranks,
        weights,
    )
    return priority_scores, weights


def build_graded_rows(
    raw_rows: list[dict[str, float | int | str | None]],
    priority_scores: list[float],
    total_games: int,
) -> list[HeroGradeRow]:
    graded_rows: list[HeroGradeRow] = []
    for row, priority_score in zip(raw_rows, priority_scores):
        picks = int(row["picks"])
        ban_rate = float(row["ban_rate"])
        presence = float(row["presence"])
        ban_share = float(row["ban_share"])
        raw_win_rate = row["raw_win_rate"]
        capped_grade = grade_cap(picks, ban_rate, presence, total_games)
        computed_grade = lower_grade(base_grade(priority_score), capped_grade)
        graded_rows.append(
            HeroGradeRow(
                hero=str(row["hero"]),
                picks=picks,
                wins=round(float(row["wins"]), 2),
                raw_win_rate=raw_win_rate if raw_win_rate is None else float(raw_win_rate),
                adjusted_win_rate=round(float(row["adjusted_win_rate"]), 2),
                pick_rate=round(float(row["pick_rate"]), 2),
                ban_rate=round(ban_rate, 2),
                presence=round(presence, 2),
                ban_share=round(ban_share, 4),
                priority_score=round(priority_score, 4),
                hero_grade=computed_grade,
                confidence=confidence_label(picks, presence, total_games),
                notes=build_notes(picks, presence, ban_share, raw_win_rate),
            )
        )
    return graded_rows


def sort_graded_rows(rows: list[HeroGradeRow]) -> list[HeroGradeRow]:
    return sorted(
        rows,
        key=lambda row: (
            GRADE_ORDER.index(row.hero_grade),
            row.priority_score,
            row.presence,
            row.adjusted_win_rate,
        ),
        reverse=True,
    )


def build_hero_grades(
    tournament_name: Optional[str] = None,
    stage: Optional[str] = None,
    total_games: Optional[int] = None,
    smoothing_games: int = 8,
) -> tuple[list[HeroGradeRow], int, dict[str, float]]:
    tournament_name = tournament_name or DEFAULT_TOURNAMENT_NAME
    stage = DEFAULT_STAGE if stage is None else stage

    hero_data = get_liquipedia_hero_data(tournament_name, stage)
    if not hero_data:
        raise ValueError("No hero data returned from Liquipedia page scraper.")

    raw_rows = build_raw_rows(hero_data, total_games)
    resolved_total_games = resolve_total_games_from_rows(raw_rows, total_games)
    enrich_raw_rows(raw_rows, resolved_total_games, smoothing_games)
    priority_scores, weights = calculate_priority_scores(raw_rows)
    graded_rows = build_graded_rows(raw_rows, priority_scores, resolved_total_games)

    return sort_graded_rows(graded_rows), resolved_total_games, weights
