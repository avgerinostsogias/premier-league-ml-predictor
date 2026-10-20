from pathlib import Path
import io
import pandas as pd
import requests

SEASONS = {
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}
def load_data():

    all_seasons = []
    for season_name, season_code in SEASONS.items():
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
        print(f"loading {season_name}...")
        df = pd.read_csv(url)

        df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR","HS","AS", "HST", "AST"]].copy()
        df["Season"] = season_name
        all_seasons.append(df)

    combined_df = pd.concat(all_seasons, ignore_index=True)
    combined_df["Date"] = pd.to_datetime(combined_df["Date"],dayfirst=True, errors="coerce")
    combined_df = combined_df.sort_values("Date").reset_index(drop=True)

    return combined_df

if __name__ == "__main__":
    df = load_data()

    print(df.head())
    print(df.shape)









