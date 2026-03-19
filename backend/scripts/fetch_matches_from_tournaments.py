import os
from pathlib import Path
from dotenv import load_dotenv

from backend.services.common.file_utils import load_json, save_json
from backend.services.liquipedia.match_finder import get_matches_from_tournament
from backend.services.common.parser import pagename_to_filename

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("LIQUIPEDIA_API_KEY")
    if not api_key:
        raise ValueError("LIQUIPEDIA_API_KEY not set")
    
    tournaments = load_json(Path("backend/data/tournaments.json"))
    output_dir = Path("backend/data/raw/tournaments")
    output_dir.mkdir(parents=True,exist_ok=True)
    for tournament in tournaments:
        if not tournament.get("active"):
            continue
            
        grouped_tournament = get_matches_from_tournament(api_key, tournament)
        if len(grouped_tournament.get('series', [])) == 0:
            print(f"No matches found for {tournament['display_name']}")
            continue
        output_path = output_dir / f"{pagename_to_filename(tournament['pagename'])}_games.json"
        if output_path.exists():
            print(f"Skipping {tournament['display_name']} (already exists)")
            continue
        save_json(output_path, grouped_tournament)

        print(f"Saved {len(grouped_tournament.get('series', []))} series to {output_path}")
        