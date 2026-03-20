from backend.services.common.parser import slugify
from datetime import datetime, timedelta

from backend.services.liquipedia.liquipedia_api import fetch_table

def build_tournament_conditions(start_date: str, end_date: str, allowed_tiers: set[str]) -> str:
    tier_clause = " OR ".join(f"[[liquipediatier::{tier}]]" for tier in sorted(allowed_tiers))
    return (
        f"[[date::>{start_date}]] AND "
        f"[[date::<{end_date}]] AND "
        f"({tier_clause})"
    )

def normalize_tournament_row(row: dict, wiki: str) -> dict:
    pagename = row.get("pagename") or row.get("parent") or row.get("name")
    display_name = row.get("name") or row.get("displayname") or pagename
    return {
        "name": slugify(display_name),
        "wiki": wiki,
        "pagename": pagename,
        "display_name": display_name,
        "liquipediatier": row.get("liquipediatier"),
        "startdate": row.get("startdate"),
        "enddate": row.get("enddate"),
        "conditions": f"[[pagename::{pagename}]]",
        "active": True,
    }

def get_tournaments_by_date(
    api_key: str,
    wiki: str,
    start_date: str,
    end_date: str,
    allowed_tiers: set[str],
) -> list[dict]:
    conditions = build_tournament_conditions(start_date, end_date, allowed_tiers)
    data = fetch_table(api_key, "match", wiki, conditions, limit=1000)

    seen = {}

    for row in data.get("result", []):
        pagename = row.get("pagename")
        if not pagename:
            continue

        if pagename not in seen:
            normalized = normalize_tournament_row(
                {
                    "pagename": pagename,
                    "name": row.get("tournament"),
                    "liquipediatier": row.get("liquipediatier"),
                    "startdate": row.get("date"),
                    "enddate": row.get("date"),
                },
                wiki,
            )
            seen[pagename] = normalized
        else:
            existing = seen[pagename]
            date_val = row.get("date")

            if date_val:
                if not existing["startdate"] or date_val < existing["startdate"]:
                    existing["startdate"] = date_val
                if not existing["enddate"] or date_val > existing["enddate"]:
                    existing["enddate"] = date_val

    return list(seen.values())

def merge_tournaments(existing: list[dict], new: list[dict]) -> list[dict]:
    tournaments = {}

    for tour in existing:
        pagename = tour.get("pagename")
        if pagename:
            tournaments[pagename] = tour

    for tour in new:
        pagename = tour.get("pagename")
        if not pagename:
            continue
        if pagename not in tournaments:
            tournaments[pagename] = tour
        else:
            existing_tour = tournaments[pagename]
            for key, value in tour.items():
                if key == "active":
                    continue

                if value and not existing_tour.get(key):
                    existing_tour[key] = value

    listTour = list(tournaments.values())

    return sorted(listTour, key=lambda x: x.get("startdate") or "", reverse=True)

def update_active_flags(tournaments: list[dict], months: int = 3) -> list[dict]:
    cutoff_date = datetime.now() - timedelta(days=months * 30)

    for tour in tournaments:
        startDate = tour.get("startdate")

        if not startDate:
            tour["active"] = False
            continue
        try:
            start_dt = start_dt = datetime.fromisoformat(startDate)
        except ValueError:
            tour["active"] = start_dt >= cutoff_date    
            continue
        
        tour["active"] = start_dt >= cutoff_date
    return tournaments
