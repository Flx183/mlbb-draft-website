from __future__ import annotations

import math
from itertools import permutations
from pathlib import Path
from typing import Any

from backend.services.common.file_utils import load_json
from backend.services.common.hero_grade_utils import MANUAL_WEIGHTS, percentile_ranks

PROCESSED_STATS_PATH = Path("backend/data/processed/complete_hero_stats.json")
STANDARD_ROLES = ("EXP", "Jungle", "Mid", "Gold", "Roam")
BAN_SLOTS = ("1", "2", "3", "4", "5")


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _smoothed_win_rate(
    wins: float,
    picks: int,
    global_win_rate: float,
    smoothing_games: int = 8,
) -> float:
    if picks <= 0:
        return global_win_rate
    return (wins + (smoothing_games * global_win_rate)) / (picks + smoothing_games)


def _normalized_entropy(probabilities: list[float], buckets: int) -> float:
    if not probabilities or buckets <= 1:
        return 0.0

    entropy = 0.0
    for probability in probabilities:
        if probability <= 0:
            continue
        entropy -= probability * math.log(probability)

    return entropy / math.log(buckets)


def _cosine_similarity(values_a: list[float], values_b: list[float]) -> float:
    if not values_a or not values_b or len(values_a) != len(values_b):
        return 0.0

    dot_product = sum(value_a * value_b for value_a, value_b in zip(values_a, values_b, strict=False))
    norm_a = math.sqrt(sum(value_a * value_a for value_a in values_a))
    norm_b = math.sqrt(sum(value_b * value_b for value_b in values_b))

    return _safe_divide(dot_product, norm_a * norm_b)


def _load_complete_stats(path: Path = PROCESSED_STATS_PATH) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict) or "heroes" not in data:
        raise ValueError(f"Expected processed hero stats at {path}")
    return data


def _global_game_count(heroes: dict[str, dict[str, Any]]) -> int:
    total_picks = sum(int(payload.get("stats", {}).get("picked", 0) or 0) for payload in heroes.values())
    return max(1, total_picks // 10)


def _global_win_rate(heroes: dict[str, dict[str, Any]]) -> float:
    total_picks = 0
    total_wins = 0.0

    for payload in heroes.values():
        stats = payload.get("stats", {})
        total_picks += int(stats.get("picked", 0) or 0)
        total_wins += float(stats.get("wins", 0) or 0.0)

    return _safe_divide(total_wins, total_picks, default=0.5)


def _ban_slot_features(stats: dict[str, Any], total_games: int) -> dict[str, float]:
    banned = int(stats.get("banned", 0) or 0)
    ban_orders = stats.get("ban_orders", {})

    if banned <= 0:
        return {
            "ban_priority": 0.0,
            "first_phase_ban_share": 0.0,
            "second_phase_ban_share": 0.0,
            "average_ban_order": 0.0,
            "average_ban_order_priority": 0.0,
            "ban_slot_mode": 0.0,
            "ban_slot_entropy": 0.0,
            **{f"ban_slot_{slot}_share": 0.0 for slot in BAN_SLOTS},
        }

    slot_counts = {
        slot: float(int(ban_orders.get(slot, 0) or 0))
        for slot in BAN_SLOTS
    }
    smoothed_denominator = float(banned + len(BAN_SLOTS))
    slot_shares = {
        slot: _safe_divide(slot_counts[slot] + 1.0, smoothed_denominator)
        for slot in BAN_SLOTS
    }

    first_phase_share = sum(slot_shares[slot] for slot in ("1", "2", "3"))
    second_phase_share = sum(slot_shares[slot] for slot in ("4", "5"))
    average_ban_order = sum(int(slot) * slot_shares[slot] for slot in BAN_SLOTS)
    ban_slot_mode = float(max(BAN_SLOTS, key=lambda slot: (slot_shares[slot], -int(slot))))
    ban_slot_entropy = _normalized_entropy(list(slot_shares.values()), len(BAN_SLOTS))

    return {
        # Legacy compatibility: keep ban_priority as a generic contestedness proxy
        # and let the ban models learn slot timing from raw slot shares instead.
        "ban_priority": _safe_divide(banned, total_games),
        "first_phase_ban_share": first_phase_share,
        "second_phase_ban_share": second_phase_share,
        "average_ban_order": average_ban_order,
        "average_ban_order_priority": average_ban_order,
        "ban_slot_mode": ban_slot_mode,
        "ban_slot_entropy": ban_slot_entropy,
        **{f"ban_slot_{slot}_share": slot_shares[slot] for slot in BAN_SLOTS},
    }


def build_hero_feature_table(path: Path = PROCESSED_STATS_PATH) -> dict[str, Any]:
    complete_stats = _load_complete_stats(path)
    heroes: dict[str, dict[str, Any]] = complete_stats["heroes"]

    total_games = _global_game_count(heroes)
    global_win_rate = _global_win_rate(heroes)

    hero_rows: list[dict[str, Any]] = []
    for hero_name, payload in heroes.items():
        stats = payload.get("stats", {})
        picked = int(stats.get("picked", 0) or 0)
        banned = int(stats.get("banned", 0) or 0)
        wins = float(stats.get("wins", 0) or 0.0)
        raw_win_rate = float(stats.get("win_rate", global_win_rate) or global_win_rate)

        role_counts = {
            role_name: int(role_stats.get("picked", 0) or 0)
            for role_name, role_stats in stats.get("roles", {}).items()
        }
        role_probabilities = [
            _safe_divide(role_counts.get(role_name, 0), picked)
            for role_name in STANDARD_ROLES
            if picked > 0
        ]
        flexibility_score = _normalized_entropy(role_probabilities, len(STANDARD_ROLES))
        flexibility_roles = sum(1 for probability in role_probabilities if probability >= 0.15)

        adjusted_win_rate = _smoothed_win_rate(
            wins=wins,
            picks=picked,
            global_win_rate=global_win_rate,
        )
        ban_slot_features = _ban_slot_features(stats, total_games)

        hero_rows.append(
            {
                "hero": hero_name,
                "picked": picked,
                "banned": banned,
                "wins": wins,
                "pick_rate": _safe_divide(picked, total_games),
                "ban_rate": _safe_divide(banned, total_games),
                "raw_win_rate": raw_win_rate,
                "adjusted_win_rate": adjusted_win_rate,
                "hero_flexibility": flexibility_score,
                "flexibility_roles": flexibility_roles,
                **ban_slot_features,
                "role_counts": role_counts,
            }
        )

    pick_rate_ranks = percentile_ranks([row["pick_rate"] for row in hero_rows])
    ban_rate_ranks = percentile_ranks([row["ban_rate"] for row in hero_rows])
    adjusted_win_rate_ranks = percentile_ranks([row["adjusted_win_rate"] for row in hero_rows])

    hero_table: dict[str, Any] = {}
    for row, pick_rank, ban_rank, win_rank in zip(
        hero_rows,
        pick_rate_ranks,
        ban_rate_ranks,
        adjusted_win_rate_ranks,
    ):
        hero_power = (
            MANUAL_WEIGHTS["pick_rate"] * pick_rank
            + MANUAL_WEIGHTS["ban_rate"] * ban_rank
            + MANUAL_WEIGHTS["adjusted_win_rate"] * win_rank
        )

        hero_table[row["hero"]] = {
            **row,
            "pick_rate_rank": pick_rank,
            "ban_rate_rank": ban_rank,
            "adjusted_win_rate_rank": win_rank,
            "hero_power": hero_power,
        }

    return {
        "metadata": complete_stats.get("metadata", {}),
        "global": {
            "total_games": total_games,
            "global_win_rate": global_win_rate,
            "hero_count": len(hero_table),
        },
        "heroes": hero_table,
    }


def _hero_features_or_default(hero_name: str, hero_table: dict[str, Any]) -> dict[str, Any]:
    return hero_table["heroes"].get(
        hero_name,
        {
            "hero": hero_name,
            "pick_rate": 0.0,
            "ban_rate": 0.0,
            "raw_win_rate": hero_table["global"]["global_win_rate"],
            "adjusted_win_rate": hero_table["global"]["global_win_rate"],
            "hero_flexibility": 0.0,
            "flexibility_roles": 0,
            "ban_priority": 0.0,
            "first_phase_ban_share": 0.0,
            "second_phase_ban_share": 0.0,
            "average_ban_order": 0.0,
            "average_ban_order_priority": 0.0,
            "ban_slot_mode": 0.0,
            "ban_slot_entropy": 0.0,
            **{f"ban_slot_{slot}_share": 0.0 for slot in BAN_SLOTS},
            "pick_rate_rank": 0.0,
            "ban_rate_rank": 0.0,
            "adjusted_win_rate_rank": 0.0,
            "hero_power": 0.0,
            "role_counts": {},
        },
    )


def _pair_score(
    matrix_name: str,
    hero_a: str,
    hero_b: str,
    hero_payloads: dict[str, Any],
    default_rate: float = 0.5,
    prior_games: int = 4,
) -> tuple[float, int]:
    record = (
        hero_payloads.get(hero_a, {}).get(matrix_name, {}).get(hero_b)
        or hero_payloads.get(hero_b, {}).get(matrix_name, {}).get(hero_a)
    )
    if not record:
        return default_rate, 0

    games = int(record.get("games", 0) or 0)
    observed = float(record.get("win_rate", default_rate) or default_rate)
    score = _safe_divide((observed * games) + (default_rate * prior_games), games + prior_games, default_rate)

    return score, games


def _summarize_values(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_sum": 0.0,
        }

    return {
        f"{prefix}_mean": sum(values) / len(values),
        f"{prefix}_max": max(values),
        f"{prefix}_min": min(values),
        f"{prefix}_sum": sum(values),
    }


def summarize_hero_list(hero_names: list[str], hero_table: dict[str, Any], prefix: str) -> dict[str, float]:
    rows = [_hero_features_or_default(hero_name, hero_table) for hero_name in hero_names]
    powers = [row["hero_power"] for row in rows]
    flexibilities = [row["hero_flexibility"] for row in rows]
    ban_priorities = [row["ban_priority"] for row in rows]
    adjusted_win_rates = [row["adjusted_win_rate"] for row in rows]
    average_ban_orders = [row["average_ban_order"] for row in rows]
    ban_slot_modes = [row["ban_slot_mode"] for row in rows]
    ban_slot_entropies = [row["ban_slot_entropy"] for row in rows]

    summary = {
        f"{prefix}_hero_count": float(len(rows)),
        f"{prefix}_flexibility_roles_sum": float(sum(row["flexibility_roles"] for row in rows)),
    }
    summary.update(_summarize_values(powers, f"{prefix}_hero_power"))
    summary.update(_summarize_values(flexibilities, f"{prefix}_hero_flexibility"))
    summary.update(_summarize_values(ban_priorities, f"{prefix}_ban_priority"))
    summary.update(_summarize_values(adjusted_win_rates, f"{prefix}_adjusted_win_rate"))
    summary.update(_summarize_values(average_ban_orders, f"{prefix}_average_ban_order"))
    summary.update(_summarize_values(ban_slot_modes, f"{prefix}_ban_slot_mode"))
    summary.update(_summarize_values(ban_slot_entropies, f"{prefix}_ban_slot_entropy"))
    for slot in BAN_SLOTS:
        summary.update(
            _summarize_values(
                [row[f"ban_slot_{slot}_share"] for row in rows],
                f"{prefix}_ban_slot_{slot}_share",
            )
        )

    return summary
def role_distribution_for_hero(hero_name: str, hero_table: dict[str, Any]) -> dict[str, float]:
    hero_row = _hero_features_or_default(hero_name, hero_table)
    role_counts = hero_row.get("role_counts", {})
    total_role_games = sum(int(role_counts.get(role_name, 0) or 0) for role_name in STANDARD_ROLES)

    if total_role_games <= 0:
        return {role_name: 0.0 for role_name in STANDARD_ROLES}

    return {
        role_name: _safe_divide(float(role_counts.get(role_name, 0) or 0), float(total_role_games))
        for role_name in STANDARD_ROLES
    }


def role_entropy_for_heroes(hero_names: list[str], hero_table: dict[str, Any]) -> float:
    aggregate_counts = {role_name: 0.0 for role_name in STANDARD_ROLES}
    for hero_name in hero_names:
        hero_row = _hero_features_or_default(hero_name, hero_table)
        role_counts = hero_row.get("role_counts", {})
        for role_name in STANDARD_ROLES:
            aggregate_counts[role_name] += float(role_counts.get(role_name, 0) or 0.0)

    total_role_games = sum(aggregate_counts.values())
    if total_role_games <= 0:
        return 0.0

    probabilities = [
        _safe_divide(aggregate_counts[role_name], total_role_games)
        for role_name in STANDARD_ROLES
    ]
    return _normalized_entropy(probabilities, len(STANDARD_ROLES))


def hero_similarity_vector(hero_name: str, hero_table: dict[str, Any]) -> list[float]:
    hero_row = _hero_features_or_default(hero_name, hero_table)
    return [
        float(hero_row.get("pick_rate_rank", 0.0)),
        float(hero_row.get("ban_rate_rank", 0.0)),
        float(hero_row.get("adjusted_win_rate_rank", 0.0)),
        float(hero_row.get("hero_power", 0.0)),
        float(hero_row.get("hero_flexibility", 0.0)),
        float(hero_row.get("first_phase_ban_share", 0.0)),
        float(hero_row.get("second_phase_ban_share", 0.0)),
        *[float(hero_row.get(f"ban_slot_{slot}_share", 0.0)) for slot in BAN_SLOTS],
    ]


def build_candidate_role_overlap_features(
    candidate_hero: str,
    reference_heroes: list[str],
    hero_table: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    if not reference_heroes:
        return {
            f"candidate_role_overlap_to_{prefix}_mean": 0.0,
            f"candidate_role_overlap_to_{prefix}_max": 0.0,
        }

    candidate_distribution = role_distribution_for_hero(candidate_hero, hero_table)
    overlaps: list[float] = []
    for reference_hero in reference_heroes:
        reference_distribution = role_distribution_for_hero(reference_hero, hero_table)
        overlap_score = sum(
            min(candidate_distribution.get(role_name, 0.0), reference_distribution.get(role_name, 0.0))
            for role_name in STANDARD_ROLES
        )
        overlaps.append(overlap_score)

    return {
        f"candidate_role_overlap_to_{prefix}_mean": sum(overlaps) / len(overlaps),
        f"candidate_role_overlap_to_{prefix}_max": max(overlaps),
    }


def build_candidate_gap_features(
    candidate_hero: str,
    reference_heroes: list[str],
    hero_table: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    candidate_row = _hero_features_or_default(candidate_hero, hero_table)
    reference_rows = [_hero_features_or_default(hero_name, hero_table) for hero_name in reference_heroes]
    if not reference_rows:
        return {
            f"candidate_minus_{prefix}_hero_power_mean": 0.0,
            f"candidate_minus_{prefix}_hero_flexibility_mean": 0.0,
            f"candidate_minus_{prefix}_adjusted_win_rate_mean": 0.0,
            f"candidate_minus_{prefix}_ban_rate_mean": 0.0,
        }

    return {
        f"candidate_minus_{prefix}_hero_power_mean": float(candidate_row.get("hero_power", 0.0))
        - sum(float(row.get("hero_power", 0.0)) for row in reference_rows) / len(reference_rows),
        f"candidate_minus_{prefix}_hero_flexibility_mean": float(candidate_row.get("hero_flexibility", 0.0))
        - sum(float(row.get("hero_flexibility", 0.0)) for row in reference_rows) / len(reference_rows),
        f"candidate_minus_{prefix}_adjusted_win_rate_mean": float(candidate_row.get("adjusted_win_rate", 0.0))
        - sum(float(row.get("adjusted_win_rate", 0.0)) for row in reference_rows) / len(reference_rows),
        f"candidate_minus_{prefix}_ban_rate_mean": float(candidate_row.get("ban_rate", 0.0))
        - sum(float(row.get("ban_rate", 0.0)) for row in reference_rows) / len(reference_rows),
    }


def build_candidate_similarity_features(
    candidate_hero: str,
    reference_heroes: list[str],
    hero_table: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    if not reference_heroes:
        return {
            f"candidate_similarity_to_{prefix}_mean": 0.0,
            f"candidate_similarity_to_{prefix}_max": 0.0,
        }

    candidate_vector = hero_similarity_vector(candidate_hero, hero_table)
    similarities = [
        _cosine_similarity(candidate_vector, hero_similarity_vector(reference_hero, hero_table))
        for reference_hero in reference_heroes
    ]

    return {
        f"candidate_similarity_to_{prefix}_mean": sum(similarities) / len(similarities),
        f"candidate_similarity_to_{prefix}_max": max(similarities),
    }


def infer_missing_roles(hero_names: list[str], hero_table: dict[str, Any]) -> list[str]:
    unique_heroes = list(dict.fromkeys(hero_names))
    if not unique_heroes:
        return list(STANDARD_ROLES)

    role_count = min(len(unique_heroes), len(STANDARD_ROLES))
    best_roles: tuple[str, ...] = ()
    best_score = float("-inf")

    for role_assignment in permutations(STANDARD_ROLES, role_count):
        assignment_score = 0.0
        for hero_name, role_name in zip(unique_heroes, role_assignment, strict=False):
            role_probability = role_distribution_for_hero(hero_name, hero_table).get(role_name, 0.0)
            assignment_score += math.log(role_probability + 1e-8)

        if assignment_score > best_score:
            best_score = assignment_score
            best_roles = role_assignment

    return [role_name for role_name in STANDARD_ROLES if role_name not in best_roles]


def summarize_candidate_synergy(
    candidate_hero: str,
    revealed_heroes: list[str],
    complete_stats: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    heroes = complete_stats["heroes"]
    scores: list[float] = []
    games: list[float] = []

    for revealed_hero in revealed_heroes:
        score, sample_games = _pair_score("synergy_matrix", candidate_hero, revealed_hero, heroes)
        scores.append(score)
        games.append(float(sample_games))

    summary = {
        f"{prefix}_synergy_pairs": float(len(scores)),
        f"{prefix}_synergy_games_total": float(sum(games)),
    }
    summary.update(_summarize_values(scores, f"{prefix}_synergy_score"))

    return summary


def summarize_candidate_counter(
    candidate_hero: str,
    revealed_heroes: list[str],
    complete_stats: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    heroes = complete_stats["heroes"]
    scores: list[float] = []
    games: list[float] = []

    for revealed_hero in revealed_heroes:
        score, sample_games = _pair_score("counter_matrix", candidate_hero, revealed_hero, heroes)
        scores.append(score)
        games.append(float(sample_games))

    summary = {
        f"{prefix}_counter_pairs": float(len(scores)),
        f"{prefix}_counter_games_total": float(sum(games)),
    }
    summary.update(_summarize_values(scores, f"{prefix}_counter_score"))

    return summary


def summarize_candidate_role_completion(
    candidate_hero: str,
    missing_roles: list[str],
    hero_table: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    role_distribution = role_distribution_for_hero(candidate_hero, hero_table)
    role_scores = [role_distribution.get(role_name, 0.0) for role_name in missing_roles]

    summary = {
        f"{prefix}_missing_role_count": float(len(missing_roles)),
    }
    summary.update(_summarize_values(role_scores, f"{prefix}_role_completion"))

    return summary
def build_ban_candidate_feature_row(
    candidate_hero: str,
    acting_team: str,
    ban_order: int,
    prior_blue_bans: list[str],
    prior_red_bans: list[str],
    hero_table: dict[str, Any],
) -> dict[str, float]:
    candidate = _hero_features_or_default(candidate_hero, hero_table)
    phase_index = 1 if ban_order <= 3 else 2
    current_slot_share = candidate.get(f"ban_slot_{ban_order}_share", 0.0)
    average_ban_order = candidate.get("average_ban_order", 0.0)
    ally_prior_bans = prior_blue_bans if acting_team == "blue" else prior_red_bans
    enemy_prior_bans = prior_red_bans if acting_team == "blue" else prior_blue_bans

    features = {
        "team_is_blue": 1.0 if acting_team == "blue" else 0.0,
        "team_is_red": 1.0 if acting_team == "red" else 0.0,
        "ban_order": float(ban_order),
        "phase_index": float(phase_index),
        "prior_blue_bans_count": float(len(prior_blue_bans)),
        "prior_red_bans_count": float(len(prior_red_bans)),
        "ally_prior_bans_count": float(len(ally_prior_bans)),
        "enemy_prior_bans_count": float(len(enemy_prior_bans)),
        "candidate_pick_rate": candidate["pick_rate"],
        "candidate_ban_rate": candidate["ban_rate"],
        "candidate_raw_win_rate": candidate["raw_win_rate"],
        "candidate_adjusted_win_rate": candidate["adjusted_win_rate"],
        "candidate_hero_power": candidate["hero_power"],
        "candidate_hero_flexibility": candidate["hero_flexibility"],
        "candidate_flexibility_roles": float(candidate["flexibility_roles"]),
        "candidate_ban_priority": candidate["ban_priority"],
        "candidate_first_phase_ban_share": candidate["first_phase_ban_share"],
        "candidate_second_phase_ban_share": candidate["second_phase_ban_share"],
        "candidate_average_ban_order": average_ban_order,
        "candidate_average_ban_order_priority": candidate["average_ban_order_priority"],
        "candidate_ban_slot_mode": candidate["ban_slot_mode"],
        "candidate_ban_slot_entropy": candidate["ban_slot_entropy"],
        "candidate_current_slot_share": current_slot_share,
        "candidate_slot_distance_from_mean": (
            abs(float(ban_order) - average_ban_order) if average_ban_order > 0 else 0.0
        ),
        "candidate_phase_fit_share": (
            candidate["first_phase_ban_share"]
            if phase_index == 1
            else candidate["second_phase_ban_share"]
        ),
        "ally_prior_bans_role_entropy": role_entropy_for_heroes(ally_prior_bans, hero_table),
        "enemy_prior_bans_role_entropy": role_entropy_for_heroes(enemy_prior_bans, hero_table),
    }
    for slot in BAN_SLOTS:
        features[f"candidate_ban_slot_{slot}_share"] = candidate[f"ban_slot_{slot}_share"]

    features.update(summarize_hero_list(prior_blue_bans, hero_table, "prior_blue_bans"))
    features.update(summarize_hero_list(prior_red_bans, hero_table, "prior_red_bans"))
    features.update(build_candidate_gap_features(candidate_hero, ally_prior_bans, hero_table, "ally_prior_bans"))
    features.update(build_candidate_gap_features(candidate_hero, enemy_prior_bans, hero_table, "enemy_prior_bans"))
    features.update(
        build_candidate_similarity_features(candidate_hero, ally_prior_bans, hero_table, "ally_prior_bans")
    )
    features.update(
        build_candidate_similarity_features(candidate_hero, enemy_prior_bans, hero_table, "enemy_prior_bans")
    )
    features.update(
        build_candidate_role_overlap_features(candidate_hero, ally_prior_bans, hero_table, "ally_prior_bans")
    )
    features.update(
        build_candidate_role_overlap_features(candidate_hero, enemy_prior_bans, hero_table, "enemy_prior_bans")
    )

    return features


def build_pick_candidate_feature_row(
    candidate_hero: str,
    acting_team: str,
    pick_order: int,
    phase_index: int,
    our_picks: list[str],
    enemy_picks: list[str],
    blue_bans: list[str],
    red_bans: list[str],
    hero_table: dict[str, Any],
    complete_stats: dict[str, Any],
) -> dict[str, float]:
    candidate = _hero_features_or_default(candidate_hero, hero_table)
    all_bans = list(dict.fromkeys([*blue_bans, *red_bans]))
    our_missing_roles = infer_missing_roles(our_picks, hero_table) if our_picks else []
    enemy_missing_roles = infer_missing_roles(enemy_picks, hero_table) if enemy_picks else []

    features = {
        "team_is_blue": 1.0 if acting_team == "blue" else 0.0,
        "team_is_red": 1.0 if acting_team == "red" else 0.0,
        "pick_order": float(pick_order),
        "phase_index": float(phase_index),
        "our_picks_count": float(len(our_picks)),
        "enemy_picks_count": float(len(enemy_picks)),
        "blue_bans_count": float(len(blue_bans)),
        "red_bans_count": float(len(red_bans)),
        "all_bans_count": float(len(all_bans)),
        "our_missing_role_count": float(len(our_missing_roles)),
        "enemy_missing_role_count": float(len(enemy_missing_roles)),
        "candidate_pick_rate": candidate["pick_rate"],
        "candidate_ban_rate": candidate["ban_rate"],
        "candidate_raw_win_rate": candidate["raw_win_rate"],
        "candidate_adjusted_win_rate": candidate["adjusted_win_rate"],
        "candidate_hero_power": candidate["hero_power"],
        "candidate_hero_flexibility": candidate["hero_flexibility"],
        "candidate_flexibility_roles": float(candidate["flexibility_roles"]),
        "candidate_ban_priority": candidate["ban_priority"],
        "candidate_first_phase_ban_share": candidate["first_phase_ban_share"],
        "candidate_second_phase_ban_share": candidate["second_phase_ban_share"],
        "candidate_average_ban_order": candidate["average_ban_order"],
        "candidate_ban_slot_mode": candidate["ban_slot_mode"],
        "candidate_ban_slot_entropy": candidate["ban_slot_entropy"],
        "our_picks_role_entropy": role_entropy_for_heroes(our_picks, hero_table),
        "enemy_picks_role_entropy": role_entropy_for_heroes(enemy_picks, hero_table),
        "all_bans_role_entropy": role_entropy_for_heroes(all_bans, hero_table),
    }

    features.update(summarize_hero_list(our_picks, hero_table, "our_picks"))
    features.update(summarize_hero_list(enemy_picks, hero_table, "enemy_picks"))
    features.update(summarize_hero_list(blue_bans, hero_table, "blue_bans"))
    features.update(summarize_hero_list(red_bans, hero_table, "red_bans"))
    features.update(summarize_hero_list(all_bans, hero_table, "all_bans"))
    features.update(
        summarize_candidate_synergy(
            candidate_hero,
            our_picks,
            complete_stats,
            "ally_pick_synergy",
        )
    )
    features.update(
        summarize_candidate_counter(
            candidate_hero,
            enemy_picks,
            complete_stats,
            "counter_vs_enemy_picks",
        )
    )
    features.update(
        summarize_candidate_role_completion(
            candidate_hero,
            our_missing_roles,
            hero_table,
            "ally_role",
        )
    )
    features.update(
        summarize_candidate_role_completion(
            candidate_hero,
            enemy_missing_roles,
            hero_table,
            "enemy_role",
        )
    )
    features.update(build_candidate_gap_features(candidate_hero, our_picks, hero_table, "our_picks"))
    features.update(build_candidate_gap_features(candidate_hero, enemy_picks, hero_table, "enemy_picks"))
    features.update(
        build_candidate_similarity_features(candidate_hero, our_picks, hero_table, "our_picks")
    )
    features.update(
        build_candidate_similarity_features(candidate_hero, enemy_picks, hero_table, "enemy_picks")
    )
    features.update(
        build_candidate_role_overlap_features(candidate_hero, our_picks, hero_table, "our_picks")
    )
    features.update(
        build_candidate_role_overlap_features(candidate_hero, enemy_picks, hero_table, "enemy_picks")
    )

    return features
