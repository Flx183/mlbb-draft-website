from collections import defaultdict
from datetime import datetime


def make_hero_record() -> dict:
    return {
        "picked": 0,
        "banned": 0,
        "wins": 0,
        "win_rate": 0.0,
        "ban_orders": defaultdict(int),
        "roles": defaultdict(lambda: {
            "picked": 0,
            "wins": 0,
            "win_rate": 0.0,
        }),
    }

def update_team_picks(hero_stats: dict, picks: list[dict], did_win: bool):
    for pick in picks:
        hero = pick.get("hero")
        role = pick.get("role")

        if not hero:
            continue

        if hero not in hero_stats:
            hero_stats[hero] = make_hero_record()

        hero_stats[hero]["picked"] += 1
        if did_win:
            hero_stats[hero]["wins"] += 1

        if role:
            hero_stats[hero]["roles"][role]["picked"] += 1
            if did_win:
                hero_stats[hero]["roles"][role]["wins"] += 1

def update_team_bans(hero_stats: dict, bans: list[dict]):
    for ban in bans:
        hero = ban.get("hero")
        ban_order = ban.get("ban_order")
        if not hero:
            continue
        if hero not in hero_stats:
            hero_stats[hero] = make_hero_record()
        hero_stats[hero]["banned"] += 1
        if ban_order is not None:
            hero_stats[hero]["ban_orders"][str(ban_order)] += 1

def calculate_win_rates(hero_stats: dict):
    finalized_stats = {}
    for hero, stats in hero_stats.items():
        picked = stats["picked"]
        wins = stats["wins"]
        banned = stats["banned"]
        win_rate = round(wins / picked, 4) if picked > 0 else 0.0

        hero_record = {
            "picked": picked,
            "banned": banned,
            "wins": wins,
            "win_rate": win_rate,
            "ban_orders": dict(sorted(stats["ban_orders"].items(), key=lambda x: int(x[0]))),
            "roles": {},
        }

        for roles, role_stats in stats["roles"].items():
            role_picked = role_stats["picked"]
            role_wins = role_stats["wins"]
            role_win_rate = round(role_wins / role_picked, 4) if role_picked > 0 else 0.0

            hero_record["roles"][roles] = {
                "picked": role_picked,
                "wins": role_wins,
                "win_rate": role_win_rate,
            }
        
        finalized_stats[hero] = hero_record
    
    return dict(sorted(finalized_stats.items(), key=lambda x: x[0]))

def build_hero_stats_from_grouped_tournament(tournament_data: dict) -> dict:
    hero_stats = defaultdict(make_hero_record)

    for series in tournament_data.get("series", []):
        for game in series.get("games", []):
            blue_team = game.get("blue_team", [])
            red_team = game.get("red_team", [])
            blue_bans = game.get("blue_bans", [])
            red_bans = game.get("red_bans", [])
            winner = game.get("winner")

            update_team_picks(hero_stats, blue_team, did_win=(winner == "blue"))
            update_team_picks(hero_stats, red_team, did_win=(winner == "red"))

            update_team_bans(hero_stats, blue_bans)
            update_team_bans(hero_stats, red_bans)

    return calculate_win_rates(hero_stats)

def merge_hero_stats(base: dict, incoming: dict) -> dict:
    for hero, stats in incoming.items():
        if hero not in base:
            base[hero] = stats
            continue

        base[hero]["picked"] += stats["picked"]
        base[hero]["banned"] += stats["banned"]
        base[hero]["wins"] += stats["wins"]

        for ban_order, count in stats.get("ban_orders", {}).items():
            if "ban_orders" not in base[hero]:
                base[hero]["ban_orders"] = {}
            base[hero]["ban_orders"][ban_order] = base[hero]["ban_orders"].get(ban_order, 0) + count

        for role, role_stats in stats["roles"].items():
            if role not in base[hero]["roles"]:
                base[hero]["roles"][role] = role_stats
            else:
                base[hero]["roles"][role]["picked"] += role_stats["picked"]
                base[hero]["roles"][role]["wins"] += role_stats["wins"]

    return base

def combine_all_hero_stats(hero_stats: dict, counter_matrix: dict, synergy_matrix: dict) -> dict:
    all_heroes = set(hero_stats.keys()) | set(synergy_matrix.keys()) | set(counter_matrix.keys())

    heroes = {}

    for hero in all_heroes:
        heroes[hero] = {
            "stats": hero_stats.get(hero, {}),
            "counter_matrix": counter_matrix.get(hero, {}),
            "synergy_matrix": synergy_matrix.get(hero, {}),
        }

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "Liquipedia",
            "version": 1,
        },
        "heroes": heroes,
    }