import requests

def fetch_table(api_key: str, endpoint: str, wiki: str, conditions: str, limit: int = 100, query: str | None = None) -> dict:
    base_url = f"https://api.liquipedia.net/api/v3/{endpoint}"

    params = {
        "wiki": wiki,
        "limit": limit,
        "conditions": conditions,
    }
    if query:
        params["query"] = query

    headers = {
        "Authorization": f"Apikey {api_key}",
        "Accept-Encoding": "gzip",
    }

    response = requests.get(base_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()