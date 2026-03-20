from pathlib import Path
from backend.services.common.file_utils import load_json, save_json
from backend.services.liquipedia.synergy_stats import build_synergy_matrix_from_tournament, finalize_synergy_stats, merge_synergy_matrices

if __name__ == "__main__":
    input_dir = Path("backend/data/raw/tournaments")
    output_dir = Path("backend/data/processed/synergy_matrices.json")

    combined_synergy = {}
    for file in input_dir.glob("*.json"):
        print(f"Processing {file.name}...")

        tournament_data = load_json(file)
        synergy_matrix = build_synergy_matrix_from_tournament(tournament_data)
        combined_synergy = merge_synergy_matrices(combined_synergy, synergy_matrix)
    
    final_synergy = finalize_synergy_stats(combined_synergy)
    save_json(output_dir, final_synergy)
    print(f"Saved synergy matrix to {output_dir}")
