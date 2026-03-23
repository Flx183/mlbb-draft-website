from backend.services.modeling.advisor_pipeline import advise_picks


def test_advise_picks_uses_local_semantic_advisor():
    payload = advise_picks(
        team="blue",
        blue_bans=["Fanny", "Zhuxin", "Hayabusa"],
        red_bans=["Joy", "Suyou", "Lukas"],
        top_k=3,
    )

    assert "recommendation" in payload
    assert "advisor" in payload
    assert payload["advisor"]["uses_llm"] is False
    assert payload["advisor"]["provider"] == "local-semantic"
    assert payload["advisor"]["model"]
    assert payload["advisor"]["advice"]
    assert "retrieved_principles" in payload["advisor"]
    assert payload["advisor"]["advice"].splitlines()[0].startswith("Pick ")
    assert "Next: " in payload["advisor"]["advice"]
    assert "(" in payload["advisor"]["advice"]
    assert "; " in payload["advisor"]["advice"]
