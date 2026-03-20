import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.liquipedia.page_scraper import (
    DEFAULT_STAGE,
    DEFAULT_TOURNAMENT_NAME,
    build_statistics_url,
    get_liquipedia_hero_data,
)


def main() -> None:
    tournament_name = DEFAULT_TOURNAMENT_NAME
    url = build_statistics_url(tournament_name, DEFAULT_STAGE)
    print(f"Scraping data from: {url}")
    hero_data = get_liquipedia_hero_data(tournament_name, DEFAULT_STAGE)
    if not hero_data:
        print(f"Failed to retrieve data from {url}")
        return
    for hero, stats in hero_data.items():
        print(
            f"{hero}: Picks: {stats['picks']}, Pick Rate: {stats['pick_rate']}, "
            f"Win Rate: {stats['win_rate']}, Ban Rate: {stats['ban_rate']}"
        )


if __name__ == "__main__":
    main()
