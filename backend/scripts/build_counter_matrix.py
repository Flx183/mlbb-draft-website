
import json
from pathlib import Path

from backend.services.common.file_utils import load_json, save_json
from backend.services.liquipedia.counter_stats import build_counter_matrix_from_tournament, finalize_counter_stats, merge_counter_matrices


if __name__ == "__main__":
    input_dir = Path("backend/data/raw/tournaments")
    output_dir = Path("backend/data/processed/counter_matrices.json")

    combined_counters = {}
    for file in input_dir.glob("*.json"):
        print(f"Processing {file.name}...")

        tournament_data = load_json(file)
        counter_matrix = build_counter_matrix_from_tournament(tournament_data)
        combined_counters = merge_counter_matrices(combined_counters, counter_matrix)

    final_counter = finalize_counter_stats(combined_counters)
    save_json(output_dir, final_counter)
    print(f"Saved counter matrix to {output_dir}")