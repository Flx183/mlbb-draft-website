import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.official_mlbb.hero_stats import fetch_rank_page, parse_main_heroes


def main() -> None:
    AUTH = "j6D946jBOrKKJ3Pr1bJHMu6wxAo="  # your captured value (may expire)
    data = fetch_rank_page(AUTH, bigrank=8, page_index=1)
    rows = parse_main_heroes(data)
    print(rows)


if __name__ == "__main__":
    main()
