
from collections import defaultdict


def make_counter_record() -> dict:
    return {
        "games": 0,
        "wins": 0,
        "win_rate": 0.0,
    }

def update_team_counter(counter_stats: dict, team_a: list[dict], team_b: list[dict], did_team_a_win: bool) -> None:
    heroes_a = [pick.get("hero") for pick in team_a if pick.get("hero")]
    heroes_b = [pick.get("hero") for pick in team_b if pick.get("hero")]

    for hero_a in heroes_a:
        if hero_a not in counter_stats:
            counter_stats[hero_a] = {}

        for hero_b in heroes_b:
            if hero_b not in counter_stats[hero_a]:
                counter_stats[hero_a][hero_b] = make_counter_record()

            counter_stats[hero_a][hero_b]["games"] += 1
            if did_team_a_win:
                counter_stats[hero_a][hero_b]["wins"] += 1

def finalize_counter_stats(counter_stats: dict) -> dict:
    finalized = {}

    for hero, counters in counter_stats.items():
        finalized[hero] = {}

        for counter, stats in counters.items():
            games = stats["games"]
            wins = stats["wins"]
            win_rate = round(wins / games, 4) if games > 0 else 0.0

            finalized[hero][counter] = {
                "games": games,
                "wins": wins,
                "win_rate": win_rate,
            }
    return finalized

def build_counter_matrix_from_tournament(tournament: dict) -> dict:
    counter_stats = defaultdict(make_counter_record)

    for match in tournament.get("series", []):
        for game in match.get("games", []):
            blue_team = game.get("blue_team", [])
            red_team = game.get("red_team", [])
            winner = game.get("winner") 

            update_team_counter(counter_stats, blue_team, red_team, winner == "blue")
            update_team_counter(counter_stats, red_team, blue_team, winner == "red")
        
    return finalize_counter_stats(counter_stats)

def merge_counter_matrices(original: dict, incoming: dict) -> dict:
    if not incoming:
        return original
    
    for hero, counters in incoming.items():
        if hero not in original:
            original[hero] = {}

        for counter, stats in counters.items():
            if counter not in original[hero]:
                original[hero][counter] = make_counter_record()

            original[hero][counter]["games"] += stats["games"]
            original[hero][counter]["wins"] += stats["wins"]

    return original