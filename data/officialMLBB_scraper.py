import requests

URL = "https://api.gms.moontontech.com/api/gms/source/2669606/2756568"

def fetch_rank_page(
    auth_token: str,
    bigrank: int = 8,
    match_type: int = 0,
    page_index: int = 1,
    page_size: int = 20,
):
    headers = {
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json;charset=UTF-8",
        "origin": "https://www.mobilelegends.com",
        "referer": "https://www.mobilelegends.com/",
        "authorization": auth_token,

        # These two often matter for this API since they appear to be app routing IDs
        "x-appid": "2669606",
        "x-actid": "2669607",
        "x-lang": "en",
    }

    payload = {
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

    r = requests.post(URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_main_heroes(resp_json: dict) -> list[dict]:
    """
    Returns a list of dict rows, one per main hero record.
    """
    rows = []
    records = resp_json.get("data", {}).get("records", [])

    for rec in records:
        d = rec.get("data", {})

        main_hero_data = (d.get("main_hero", {}) or {}).get("data", {}) or {}
        row = {
            "main_hero_name": main_hero_data.get("name"),
            "appearance_rate": d.get("main_hero_appearance_rate"),
            "ban_rate": d.get("main_hero_ban_rate"),
            "win_rate": d.get("main_hero_win_rate"),
        }
        rows.append(row)

    return rows

if __name__ == "__main__":
    AUTH = "j6D946jBOrKKJ3Pr1bJHMu6wxAo="  # your captured value (may expire)
    data = fetch_rank_page(AUTH, bigrank=8, page_index=1)
    rows = parse_main_heroes(data)
    print(rows)