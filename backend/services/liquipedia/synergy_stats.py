from itertools import combinations
from collections import defaultdict
import copy

def make_synergy_record() -> dict:
    return {
        "games": 0,
        "wins": 0,
        "win_rate": 0.0,
    }

def update_team_synergy(synergy_stats: dict, team_picks: list[dict], did_win: bool) -> None:
    heroes = [pick.get("hero") for pick in team_picks if pick.get("hero")]

    for hero_a, hero_b in combinations(heroes, 2):
        for h1, h2 in [(hero_a, hero_b), (hero_b, hero_a)]:
            if h1 not in synergy_stats:
                synergy_stats[h1] = {}
            
            if h2 not in synergy_stats[h1]:
                synergy_stats[h1][h2] = make_synergy_record()

            synergy_stats[h1][h2]["games"] += 1
            if did_win:
                synergy_stats[h1][h2]["wins"] += 1


def finalize_synergy_stats(synergy_stats: dict) -> dict:
    finalized = {}

    for hero, partners in synergy_stats.items():
        finalized[hero] = {}

        for partner, stats in partners.items():
            games = stats["games"]
            wins = stats["wins"]
            win_rate = round(wins / games, 4) if games > 0 else 0.0

            finalized[hero][partner] = {
                "games": games,
                "wins": wins,
                "win_rate": win_rate,
            }
    return finalized

def build_synergy_matrix_from_tournament(tournament: dict) -> dict:
    synergy_stats = defaultdict(make_synergy_record)

    for match in tournament.get("series", []):
        for game in match.get("games", []):
            blue_team = game.get("blue_team", [])
            red_team = game.get("red_team", [])
            winner = game.get("winner") 

            update_team_synergy(synergy_stats, blue_team, winner == "blue")
            update_team_synergy(synergy_stats, red_team, winner == "red")
        
    return finalize_synergy_stats(synergy_stats)

def merge_synergy_matrices(original: dict, incoming: dict) -> dict:
    for hero, partners in incoming.items():
        if hero not in original:
            original[hero] = copy.deepcopy(partners)
            continue

        for partner, stats in partners.items():
            if partner not in original[hero]:
                original[hero][partner] = copy.deepcopy(stats)
            else:
                original[hero][partner]["games"] += stats["games"]
                original[hero][partner]["wins"] += stats["wins"]

    return original