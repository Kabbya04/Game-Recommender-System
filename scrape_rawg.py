import requests
import pandas as pd
import time

# Your RAWG API Key
API_KEY = "cfd608b77bc54660bb30187f178403ca"  # Replace with your actual RAWG API key

# Number of games to scrape
GAMES_LIMIT = 500

# RAWG API URL for top games
url = f"https://api.rawg.io/api/games?key={API_KEY}&page_size={GAMES_LIMIT}"

# Function to fetch game data
def fetch_games():
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            games = []
            for game in data["results"]:
                games.append({
                    "Game Name": game.get("name", "N/A"),
                    "Release Date": game.get("released", "N/A"),
                    "Rating": game.get("rating", "N/A"),
                    "Platforms": ", ".join([p["platform"]["name"] for p in game.get("platforms", [])]) if game.get("platforms") else "N/A",
                    "Genres": ", ".join([g["name"] for g in game.get("genres", [])]) if game.get("genres") else "N/A"
                })
            return games
    else:
        print(f"⚠️ ERROR {response.status_code}: Could not fetch data from RAWG.")
        return []

# Fetch game data
print("🔍 Fetching new game dataset from RAWG...")
scraped_games = fetch_games()

# Convert to DataFrame
df = pd.DataFrame(scraped_games)

# Save to CSV
output_file = "rawg_new_dataset.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Scraping complete! Data saved as '{output_file}'.")
