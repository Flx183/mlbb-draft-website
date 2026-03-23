from backend.services.modeling.ban_recommender import (
    recommend_next_bans,
    resolve_next_ban_turn,
    simulate_ban_sequence,
)


def test_resolve_next_ban_turn_empty_state():
    turn = resolve_next_ban_turn()

    assert turn["team"] == "blue"
    assert turn["ban_order"] == 1
    assert turn["phase_index"] == 1


def test_recommend_next_bans_excludes_unavailable_heroes():
    recommendation = recommend_next_bans(
        blue_picks=["Akai"],
        red_picks=["Claude"],
        blue_bans=["Fanny"],
        red_bans=["Zhuxin"],
        team="blue",
        top_k=3,
    )

    unavailable = {"Akai", "Claude", "Fanny", "Zhuxin"}
    heroes = [item["hero"] for item in recommendation["recommendations"]]

    assert recommendation["team"] == "blue"
    assert recommendation["ban_order"] == 2
    assert len(heroes) == 3
    assert not unavailable.intersection(heroes)


def test_recommend_next_bans_exposes_context_components_with_picks():
    recommendation = recommend_next_bans(
        blue_picks=["Akai"],
        red_picks=["Yve"],
        team="blue",
        top_k=2,
    )

    first_item = recommendation["recommendations"][0]
    score_components = first_item["score_components"]

    assert recommendation["rerank_pool_size"] >= 2
    assert "prior_score" in score_components
    assert "context_peak" in score_components
    assert "enemy_pick_synergy_max" in score_components
    assert "counter_vs_our_picks_max" in score_components
    assert "enemy_role_completion_max" in score_components


def test_simulate_ban_sequence_progresses_in_order():
    simulation = simulate_ban_sequence(top_k=2)

    steps = simulation["steps"]
    assert len(steps) == 10
    assert steps[0]["team"] == "blue"
    assert steps[0]["ban_order"] == 1
    assert steps[1]["team"] == "red"
    assert steps[1]["ban_order"] == 1
    assert steps[-1]["team"] == "blue"
    assert steps[-1]["ban_order"] == 5
    assert len(simulation["blue_bans"]) == 5
    assert len(simulation["red_bans"]) == 5
