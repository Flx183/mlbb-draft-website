import requests
from bs4 import BeautifulSoup

liquipedia_base_url = 'https://liquipedia.net/mobilelegends/'
mlbb_official_base_url = 'https://www.mobilelegends.com/en/hero-stats'

# Function to scrape hero data from Liquipedia by giving the tournament's name
def get_liquipedia_hero_data(tournament_name, stage=""):
    split_name = tournament_name.split()
    url_name = '/'.join(split_name)
    new_url = liquipedia_base_url + url_name + '/Statistics' + (f'/{stage}' if stage else '')      
    print(f"Scraping data from: {new_url}")
    res = requests.get(new_url)

    if res.status_code != 200:
        print(f"Failed to retrieve data from {new_url}")
        return {}
    
    soup = BeautifulSoup(res.content, 'html.parser')
    hero_data = {}

    s = soup.find_all('tr', class_='character-stats-row')
    for row in s:
        hero = row.find_all('td')[1].find_all('a')[1].text.strip()
        win_rate = row.find_all('td')[5].text.strip()
        pick_rate = row.find_all('td')[2].text.strip()
        ban_rate = row.find_all('td')[16].text.strip()
        hero_data[hero] = {
            'win_rate': win_rate,
            'pick_rate': pick_rate,
            'ban_rate': ban_rate
        }
    return hero_data
    

if __name__ == "__main__":
    tournament_name = "M7_World_Championship"
    hero_data = get_liquipedia_hero_data(tournament_name, "Knockout_Stage")
    for hero, stats in hero_data.items():
        print(f"{hero}: Win Rate: {stats['win_rate']}, Pick Rate: {stats['pick_rate']}, Ban Rate: {stats['ban_rate']}")

