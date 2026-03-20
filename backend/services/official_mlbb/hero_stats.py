import requests


OFFICIAL_MLBB_HERO_STATS_URL = "https://api.gms.moontontech.com/api/gms/source/2669606/2756568"


def build_rank_headers(auth_token: str) -> dict[str, str]:
    return {
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json;charset=UTF-8",
        "origin": "https://www.mobilelegends.com",
        "referer": "https://www.mobilelegends.com/",
        "authorization": auth_token,
        "x-appid": "2669606",
        "x-actid": "2669607",
        "x-lang": "en",
    }


def build_rank_payload(
    bigrank: int = 8,
    match_type: int = 0,
    page_index: int = 1,
    page_size: int = 20,
) -> dict:
    return {
        "pageSize": page_size,
        "pageIndex": page_index,
        "filters": [
            {"field": "bigrank", "operator": "eq", "value": str(bigrank)},
            {"field": "match_type", "operator": "eq", "value": match_type},
        ],
        "sorts": [
            {"data": {"field": "main_hero_win_rate", "order": "desc"}, "type": "sequence"},
            {"data": {"field": "main_heroid", "order": "desc"}, "type": "sequence"},
        ],
        "fields": [
            "main_hero",
            "main_hero_appearance_rate",
            "main_hero_ban_rate",
            "main_hero_channel",
            "main_hero_win_rate",
            "main_heroid",
            "data.sub_hero.hero",
            "data.sub_hero.hero_channel",
            "data.sub_hero.increase_win_rate",
            "data.sub_hero.heroid",
        ],
    }


def fetch_rank_page(
    auth_token: str,
    bigrank: int = 8,
    match_type: int = 0,
    page_index: int = 1,
    page_size: int = 20,
) -> dict:
    response = requests.post(
        OFFICIAL_MLBB_HERO_STATS_URL,
        headers=build_rank_headers(auth_token),
        json=build_rank_payload(
            bigrank=bigrank,
            match_type=match_type,
            page_index=page_index,
            page_size=page_size,
        ),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def parse_main_heroes(response_json: dict) -> list[dict]:
    rows = []
    records = response_json.get("data", {}).get("records", [])

    for record in records:
        data = record.get("data", {})
        main_hero_data = (data.get("main_hero", {}) or {}).get("data", {}) or {}
        rows.append(
            {
                "main_hero_name": main_hero_data.get("name"),
                "appearance_rate": data.get("main_hero_appearance_rate"),
                "ban_rate": data.get("main_hero_ban_rate"),
                "win_rate": data.get("main_hero_win_rate"),
            }
        )

    return rows
