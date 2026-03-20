from pathlib import Path

from backend.services.common.file_utils import load_json, save_json
from backend.services.liquipedia.hero_stats import (
    build_hero_stats_from_grouped_tournament,
    merge_hero_stats,
    calculate_win_rates,
)


if __name__ == "__main__":
    input_dir = Path("backend/data/raw/tournaments")
    output_path = Path("backend/data/processed/all_hero_stats.json")

    combined_stats = {}

    for file_path in input_dir.glob("*.json"):
        print(f"Processing {file_path.name}...")

        tournament_data = load_json(file_path)
        hero_stats = build_hero_stats_from_grouped_tournament(tournament_data)

        combined_stats = merge_hero_stats(combined_stats, hero_stats)

    final_stats = calculate_win_rates(combined_stats)

    save_json(output_path, final_stats)

    print(f"\nSaved combined hero stats to {output_path}")