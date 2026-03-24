from pathlib import Path

from backend.services.modeling import hero_power_model


def test_load_hero_power_profile_bootstraps_without_trained_artifacts(tmp_path, monkeypatch):
    missing_report = tmp_path / "missing_report.json"
    missing_model = tmp_path / "missing_model.json"
    missing_profile = tmp_path / "missing_profile.json"

    monkeypatch.setattr(hero_power_model, "PICK_RANKER_REPORT_PATH", missing_report)
    monkeypatch.setattr(hero_power_model, "PICK_GLOBAL_RANKER_PATH", missing_model)
    monkeypatch.setattr(hero_power_model, "BAN_RANKER_REPORT_PATH", missing_report)
    monkeypatch.setattr(hero_power_model, "BAN_GLOBAL_RANKER_PATH", missing_model)

    hero_power_model.load_hero_power_profile.cache_clear()
    try:
        profile = hero_power_model.load_hero_power_profile(Path(missing_profile))
    finally:
        hero_power_model.load_hero_power_profile.cache_clear()

    assert profile["source"] == "bootstrap-equal-weights"
    assert profile["model_sources"] == []
    assert profile["feature_weights"] == {
        "pick_rate": 1 / 3,
        "ban_rate": 1 / 3,
        "adjusted_win_rate": 1 / 3,
    }
