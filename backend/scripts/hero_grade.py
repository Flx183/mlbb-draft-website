import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.hero_grading import HeroGradeRow, build_hero_grades
from backend.services.liquipedia.page_scraper import (
    DEFAULT_STAGE,
    DEFAULT_TOURNAMENT_NAME,
)


def print_report(
    rows: list[HeroGradeRow],
    tournament_name: str,
    stage: str,
    total_games: int,
    weights: dict[str, float],
    weighting_method: str,
) -> None:
    label = tournament_name if not stage else f"{tournament_name} / {stage}"
    print(f"Hero Grade report for {label}")
    print(f"Inferred total games: {total_games}")
    print(
        "Weights "
        f"({weighting_method}): "
        f"pick_rate={weights['pick_rate']:.3f}, "
        f"ban_rate={weights['ban_rate']:.3f}, "
        f"adjusted_win_rate={weights['adjusted_win_rate']:.3f}"
    )
    print(
        f"{'Hero':20} {'Grade':5} {'Score':6} {'Picks':5} {'Pick%':6} {'Ban%':6} "
        f"{'Presence':8} {'AdjWR%':7} {'Conf':6} Notes"
    )
    for row in rows:
        print(
            f"{row.hero:20} {row.hero_grade:5} {row.priority_score:6.3f} {row.picks:5d} "
            f"{row.pick_rate:6.2f} {row.ban_rate:6.2f} {row.presence:8.2f} "
            f"{row.adjusted_win_rate:7.2f} {row.confidence:6} {row.notes}"
        )


def write_csv(rows: list[HeroGradeRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate tournament hero grades from Liquipedia statistics."
    )
    parser.add_argument(
        "tournament_name",
        nargs="?",
        default=None,
        help=(
            "Optional Liquipedia tournament name override. If omitted, uses "
            f"the default tournament ({DEFAULT_TOURNAMENT_NAME})."
        ),
    )
    parser.add_argument(
        "--stage",
        default=None,
        help=(
            "Optional Liquipedia stage override. If omitted, uses "
            f"the default stage ({DEFAULT_STAGE})."
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Override total games if ban-rate inference is not possible.",
    )
    parser.add_argument(
        "--smoothing-games",
        type=int,
        default=8,
        help="Pseudo-games used to smooth win rate for small samples.",
    )
    parser.add_argument(
        "--weighting",
        choices=["critic", "entropy", "manual"],
        default="critic",
        help="How to determine feature weights for pick rate, ban rate, and adjusted win rate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tournament_name = args.tournament_name or DEFAULT_TOURNAMENT_NAME
    stage = DEFAULT_STAGE if args.stage is None else args.stage
    rows, total_games, weights = build_hero_grades(
        tournament_name=tournament_name,
        stage=stage,
        total_games=args.games,
        smoothing_games=args.smoothing_games,
        weighting_method=args.weighting,
    )
    print_report(rows, tournament_name, stage, total_games, weights, args.weighting)
    if args.output is not None:
        write_csv(rows, args.output)
        print(f"\nSaved CSV to {args.output}")


if __name__ == "__main__":
    main()
