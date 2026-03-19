from backend.services.liquipedia.liquipedia_api import fetch_table
from backend.services.liquipedia.tournament_finder import get_tournaments_by_date, update_active_flags,  merge_tournaments
from backend.services.common.file_utils import load_json, save_json

from pathlib import Path
from datetime import date, timedelta
import os
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("LIQUIPEDIA_API_KEY")
    today = date.today()
    months = 1 # Change this for different ranges

    tournaments = get_tournaments_by_date(
        api_key=api_key,
        wiki="mobilelegends",
        start_date=(today - timedelta(days=months * 30)).strftime("%Y-%m-%d"), # Look back 3 months for tournaments
        end_date=today.strftime("%Y-%m-%d"),
        allowed_tiers={"1", "2"}, # S and A Tier tournaments only
    )

    existing = load_json(Path("backend/data/tournaments.json"))
    
    merged = merge_tournaments(existing, tournaments)

    updated = update_active_flags(merged, months=months) 

    save_json(Path("backend/data/tournaments.json"), updated)

    print(f"Saved {len(updated)} tournaments")