
from pathlib import Path

from backend.services.common.file_utils import load_json, save_json
from backend.services.liquipedia.hero_stats import combine_all_hero_stats
if __name__ == "__main__":
    processed_dir = Path("backend/data/processed")

    hero_stats = load_json(processed_dir / "all_hero_stats.json")
    synergy_matrix = load_json(processed_dir / "synergy_matrices.json")
    counter_matrix = load_json(processed_dir / "counter_matrices.json")

    combined_stats = combine_all_hero_stats(hero_stats, counter_matrix, synergy_matrix)

    output_path = processed_dir / "complete_hero_stats.json"
    save_json(output_path, combined_stats)

    print(f"Combined hero stats saved to {output_path}")