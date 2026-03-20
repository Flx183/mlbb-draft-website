import requests
from bs4 import BeautifulSoup


LIQUIPEDIA_BASE_URL = "https://liquipedia.net/mobilelegends/"
DEFAULT_TOURNAMENT_NAME = "M7_World_Championship"
DEFAULT_STAGE = "Knockout_Stage"


def build_statistics_url(tournament_name: str, stage: str = "") -> str:
    split_name = tournament_name.split()
    url_name = "/".join(split_name)
    return f"{LIQUIPEDIA_BASE_URL}{url_name}/Statistics{f'/{stage}' if stage else ''}"


def parse_liquipedia_hero_data(page_content: bytes) -> dict[str, dict[str, str]]:
    soup = BeautifulSoup(page_content, "html.parser")
    hero_data: dict[str, dict[str, str]] = {}

    for row in soup.find_all("tr", class_="character-stats-row"):
        columns = row.find_all("td")
        hero = columns[1].find_all("a")[1].text.strip()
        hero_data[hero] = {
            "picks": columns[2].text.strip(),
            "wins": columns[3].text.strip(),
            "losses": columns[4].text.strip(),
            "win_rate": columns[5].text.strip(),
            "pick_rate": columns[6].text.strip(),
            "bans": columns[15].text.strip(),
            "ban_rate": columns[16].text.strip(),
            "presence_count": columns[17].text.strip(),
            "presence_rate": columns[18].text.strip(),
        }

    return hero_data


def get_liquipedia_hero_data(tournament_name: str, stage: str = "") -> dict[str, dict[str, str]]:
    url = build_statistics_url(tournament_name, stage)
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return {}
    return parse_liquipedia_hero_data(response.content)
