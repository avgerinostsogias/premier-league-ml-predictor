from data_collection import load_data
import numpy as np
import pandas as pd

def get_points(result, team_type):
    if result == "D":
        return 1
    if result == "H":
        return 3 if team_type == "home" else 0
    if result == "A":
        return 3 if team_type == "away" else 0

def calculate_home_points(result):
    return get_points(result, "home")


def calculate_away_points(result):
    return get_points(result, "away")

def average_last5(history):
    last5 = history[-5:]
    if len(last5) == 0:
        return 0
    return sum(last5) / len(last5)

def build_season_table(season_df):

    home = season_df[
        ["HomeTeam", "HomePoints", "FTHG", "FTAG"]
    ].copy()

    home.columns = [
        "Team",
        "Points",
        "GoalsFor",
        "GoalsAgainst"
    ]

    away = season_df[
        ["AwayTeam", "AwayPoints", "FTAG", "FTHG"]
    ].copy()

    away.columns = [
        "Team",
        "Points",
        "GoalsFor",
        "GoalsAgainst"
    ]

    table = pd.concat(
        [home, away],
        ignore_index=True)

    table = table.groupby(
        "Team",
        as_index=False).agg(Points=("Points", "sum"),
        GoalsFor=("GoalsFor", "sum"),
        GoalsAgainst=("GoalsAgainst", "sum"))

    table["GoalDifference"] = (table["GoalsFor"]- table["GoalsAgainst"])

    table = table.sort_values(
        ["Points", "GoalDifference", "GoalsFor"],
        ascending=[False, False, False]).reset_index(drop=True)

    table["Position"] = table.index + 1

    return table

def calculate_historical_strength(
        team,
        current_season,
        season_order,
        season_tables
):

    current_index = season_order.index(current_season)

    previous_seasons = season_order[:current_index]

    strengths = []

    for season in previous_seasons:

        table = season_tables[season]

        team_row = table[
            table["Team"] == team
        ]

        if not team_row.empty:

            position = team_row.iloc[0]["Position"]

            strength = 21 - position

            strengths.append(strength)

    if len(strengths) == 0:
        return np.nan

    return sum(strengths) / len(strengths)

def reset_promoted_history(
        history,
        reset_done,
        team,
        season,
        promoted
):

    team_season = (season, team)

    if promoted == 1 and team_season not in reset_done:

        history[team] = []

        reset_done.add(team_season)

def calculate_elo_expected(
        rating_a,
        rating_b
):

    return 1 / (
        1 + 10 ** (
            (rating_b - rating_a) / 400
        )
    )

def create_features(df):

    df = df.copy()

    # Points needed for season tables

    df["HomePoints"] = df["FTR"].apply(
        calculate_home_points
    )

    df["AwayPoints"] = df["FTR"].apply(
        calculate_away_points
    )

    # ----------------------------------
    # Historical Strength
    # ----------------------------------

    season_order = (
        df.groupby("Season")["Date"]
        .min()
        .sort_values()
        .index
        .tolist()
    )

    print(
        "Season Order:",
        season_order
    )

    season_tables = {}

    for season in season_order:
        season_df = df[
            df["Season"] == season
            ]

        season_tables[season] = (
            build_season_table(
                season_df
            )
        )

    print(
        "Season tables:",
        season_tables.keys()
    )

    home_historical_strength = []
    away_historical_strength = []

    home_promoted = []
    away_promoted = []

    for index, row in df.iterrows():

        current_season = row["Season"]

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_strength = (
            calculate_historical_strength(
                home_team,
                current_season,
                season_order,
                season_tables
            )
        )

        away_strength = (
            calculate_historical_strength(
                away_team,
                current_season,
                season_order,
                season_tables
            )
        )

        home_historical_strength.append(
            home_strength
        )

        away_historical_strength.append(
            away_strength
        )

        current_index = season_order.index(
            current_season
        )

        if current_index == 0:

            home_promoted.append(0)
            away_promoted.append(0)

        else:

            previous_season = season_order[
                current_index - 1
                ]

            previous_table = season_tables[
                previous_season
            ]

            previous_teams = set(
                previous_table["Team"]
            )

            home_promoted.append(
                0
                if home_team in previous_teams
                else 1
            )

            away_promoted.append(
                0
                if away_team in previous_teams
                else 1
            )

    df["HomeHistoricalStrength"] = (
        home_historical_strength
    )

    df["AwayHistoricalStrength"] = (
        away_historical_strength
    )

    df["HomePromoted"] = home_promoted
    df["AwayPromoted"] = away_promoted

    # ----------------------------------
    # Teams with no previous PL history
    # ----------------------------------

    df["HomeNoPLHistory"] = (
        df["HomeHistoricalStrength"]
        .isna()
        .astype(int)
    )

    df["AwayNoPLHistory"] = (
        df["AwayHistoricalStrength"]
        .isna()
        .astype(int)
    )

    # ----------------------------------
    # Replace missing HistoricalStrength
    # with a neutral value
    # ----------------------------------

    for season in season_order:

        current_mask = df["Season"] == season

        known_strengths = pd.concat([
            df.loc[current_mask, "HomeHistoricalStrength"],
            df.loc[current_mask, "AwayHistoricalStrength"]
        ]).dropna()

        if len(known_strengths) > 0:
            neutral_strength = known_strengths.median()
        else:
            neutral_strength = 10.5

        df.loc[
            current_mask,
            "HomeHistoricalStrength"
        ] = (
            df.loc[
                current_mask,
                "HomeHistoricalStrength"
            ].fillna(neutral_strength)
        )

        df.loc[
            current_mask,
            "AwayHistoricalStrength"
        ] = (
            df.loc[
                current_mask,
                "AwayHistoricalStrength"
            ].fillna(neutral_strength)
        )

    # ----------------------------------
    # Elo Rating v2
    # ----------------------------------

    INITIAL_ELO = 1500

    K_FACTOR = 20

    HOME_ADVANTAGE = 50

    # 0.75 = κρατάμε το 75% της απόστασης
    # από το 1500 μεταξύ δύο seasons
    SEASON_REVERSION = 0.75

    # Promoted teams ξεκινούν λίγο κάτω
    # από τον μέσο όρο των returning teams
    PROMOTED_PENALTY = 75

    elo_ratings = {}

    home_elo_list = []
    away_elo_list = []
    elo_difference_list = []

    current_elo_season = None

    for index, row in df.iterrows():

        season = row["Season"]

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        # ----------------------------------
        # Beginning of a new season
        # ----------------------------------

        if current_elo_season != season:

            # First season in dataset
            if current_elo_season is None:

                current_elo_season = season

            else:

                previous_season = current_elo_season

                previous_teams = set(
                    season_tables[
                        previous_season
                    ]["Team"]
                )

                current_teams = set(
                    season_tables[
                        season
                    ]["Team"]
                )

                # Teams that stayed in Premier League

                returning_teams = (
                        current_teams
                        & previous_teams
                )

                # Newly promoted teams

                promoted_teams = (
                        current_teams
                        - previous_teams
                )

                # ----------------------------------
                # Mean reversion for returning teams
                # ----------------------------------

                for team in returning_teams:

                    if team in elo_ratings:
                        old_elo = elo_ratings[team]

                        elo_ratings[team] = (
                                INITIAL_ELO
                                + SEASON_REVERSION
                                * (
                                        old_elo
                                        - INITIAL_ELO
                                )
                        )

                # ----------------------------------
                # Average Elo of returning teams
                # AFTER mean reversion
                # ----------------------------------

                returning_elos = [
                    elo_ratings[team]
                    for team in returning_teams
                    if team in elo_ratings
                ]

                if len(returning_elos) > 0:

                    league_mean_elo = np.mean(
                        returning_elos
                    )

                else:

                    league_mean_elo = INITIAL_ELO

                # ----------------------------------
                # Initial Elo for promoted teams
                # ----------------------------------

                promoted_initial_elo = (
                        league_mean_elo
                        - PROMOTED_PENALTY
                )

                for team in promoted_teams:
                    elo_ratings[team] = (
                        promoted_initial_elo
                    )

                print(
                    season,
                    "Promoted:",
                    promoted_teams,
                    "Returning mean Elo:",
                    round(league_mean_elo, 2),
                    "Promoted starting Elo:",
                    round(promoted_initial_elo, 2)
                )

                current_elo_season = season

        # ----------------------------------
        # Initialize completely unseen teams
        # Mainly first dataset season
        # ----------------------------------

        if home_team not in elo_ratings:
            elo_ratings[home_team] = (
                INITIAL_ELO
            )

        if away_team not in elo_ratings:
            elo_ratings[away_team] = (
                INITIAL_ELO
            )

        # ----------------------------------
        # Elo BEFORE current match
        # ----------------------------------

        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]

        home_elo_list.append(
            home_elo
        )

        away_elo_list.append(
            away_elo
        )

        elo_difference_list.append(
            home_elo - away_elo
        )

        # ----------------------------------
        # Expected result
        # ----------------------------------

        expected_home = (
            calculate_elo_expected(
                home_elo + HOME_ADVANTAGE,
                away_elo
            )
        )

        expected_away = (
                1 - expected_home
        )

        # ----------------------------------
        # Actual result
        # ----------------------------------

        if row["FTR"] == "H":

            actual_home = 1
            actual_away = 0

        elif row["FTR"] == "A":

            actual_home = 0
            actual_away = 1

        else:

            actual_home = 0.5
            actual_away = 0.5

        # ----------------------------------
        # Elo update AFTER match
        # ----------------------------------

        elo_ratings[home_team] = (
                home_elo
                + K_FACTOR
                * (
                        actual_home
                        - expected_home
                )
        )

        elo_ratings[away_team] = (
                away_elo
                + K_FACTOR
                * (
                        actual_away
                        - expected_away
                )
        )

    df["HomeElo"] = home_elo_list

    df["AwayElo"] = away_elo_list

    df["EloDifference"] = (
        elo_difference_list
    )



    # ----------------------------------
    # Average points for last 5 matches
    # ----------------------------------

    team_points_history = {}

    points_reset_done = set()

    home_points_avg_last5 = []
    away_points_avg_last5 = []

    home_recent_matches_available = []
    away_recent_matches_available = []

    for index, row in df.iterrows():

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        current_season = row["Season"]

        if home_team not in team_points_history:
            team_points_history[home_team] = []

        if away_team not in team_points_history:
            team_points_history[away_team] = []

        reset_promoted_history(
            team_points_history,
            points_reset_done,
            home_team,
            current_season,
            row["HomePromoted"]
        )

        reset_promoted_history(
            team_points_history,
            points_reset_done,
            away_team,
            current_season,
            row["AwayPromoted"]
        )

        home_history = team_points_history[home_team]
        away_history = team_points_history[away_team]

        home_recent_matches_available.append(
            min(len(home_history), 5)
        )

        away_recent_matches_available.append(
            min(len(away_history), 5)
        )

        home_points_avg_last5.append(average_last5(home_history))
        away_points_avg_last5.append(average_last5(away_history))

        team_points_history[home_team].append(row["HomePoints"])
        team_points_history[away_team].append(row["AwayPoints"])

    df["HomePointsAvgLast5"] = home_points_avg_last5
    df["AwayPointsAvgLast5"] = away_points_avg_last5

    df["HomeRecentMatchesAvailable"] = (
        home_recent_matches_available
    )

    df["AwayRecentMatchesAvailable"] = (
        away_recent_matches_available
    )

    # ----------------------------------
    # Goals average scored last 5
    # ----------------------------------

    team_goals_history = {}

    goals_reset_done = set()

    home_goals_avg_last5 = []
    away_goals_avg_last5 = []

    for index, row in df.iterrows():

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        current_season = row["Season"]

        reset_promoted_history(
            team_goals_history,
            goals_reset_done,
            home_team,
            current_season,
            row["HomePromoted"]
        )

        reset_promoted_history(
            team_goals_history,
            goals_reset_done,
            away_team,
            current_season,
            row["AwayPromoted"]
        )

        if home_team not in team_goals_history:
            team_goals_history[home_team] = []

        if away_team not in team_goals_history:
            team_goals_history[away_team] = []

        home_history = team_goals_history[home_team]
        away_history = team_goals_history[away_team]

        home_goals_avg_last5.append(average_last5(home_history))
        away_goals_avg_last5.append(average_last5(away_history))

        team_goals_history[home_team].append(row["FTHG"])
        team_goals_history[away_team].append(row["FTAG"])

    df["HomeGoalsAvgLast5"] = home_goals_avg_last5
    df["AwayGoalsAvgLast5"] = away_goals_avg_last5

    # ----------------------------------
    # Goals average conceded last 5
    # ----------------------------------

    team_conceded_history = {}

    conceded_reset_done = set()

    home_conceded_avg_last5 = []
    away_conceded_avg_last5 = []


    for index, row in df.iterrows():

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        current_season = row["Season"]

        reset_promoted_history(
            team_conceded_history,
            conceded_reset_done,
            home_team,
            current_season,
            row["HomePromoted"]
        )

        reset_promoted_history(
            team_conceded_history,
            conceded_reset_done,
            away_team,
            current_season,
            row["AwayPromoted"]
        )

        if home_team not in team_conceded_history:
            team_conceded_history[home_team] = []

        if away_team not in team_conceded_history:
            team_conceded_history[away_team] = []

        home_history = team_conceded_history[home_team]
        away_history = team_conceded_history[away_team]

        home_conceded_avg_last5.append(average_last5(home_history))
        away_conceded_avg_last5.append(average_last5(away_history))

        team_conceded_history[home_team].append(row["FTAG"])
        team_conceded_history[away_team].append(row["FTHG"])

    df["HomeConcededAvgLast5"] = home_conceded_avg_last5
    df["AwayConcededAvgLast5"] = away_conceded_avg_last5

    # ----------------------------------
    # Home average points for last 5 matches at home
    #              AND
    # Away average points for last 5 matches away
    # ----------------------------------

    home_venue_history = {}
    away_venue_history = {}

    home_points_avg_home_last5 = []
    away_points_avg_away_last5 = []

    home_venue_reset_done = set()
    away_venue_reset_done = set()

    for index, row in df.iterrows():

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        current_season = row["Season"]

        reset_promoted_history(
            home_venue_history,
            home_venue_reset_done,
            home_team,
            current_season,
            row["HomePromoted"]
        )

        reset_promoted_history(
            away_venue_history,
            away_venue_reset_done,
            away_team,
            current_season,
            row["AwayPromoted"]
        )

        if home_team not in home_venue_history:
            home_venue_history[home_team] = []

        if away_team not in away_venue_history:
            away_venue_history[away_team] = []
        home_history = home_venue_history[home_team]
        away_history = away_venue_history[away_team]

        home_points_avg_home_last5.append(average_last5(home_history))
        away_points_avg_away_last5.append(average_last5(away_history))

        home_venue_history[home_team].append(row["HomePoints"])
        away_venue_history[away_team].append(row["AwayPoints"])

    df["HomePointsAvgHomeLast5"] = home_points_avg_home_last5
    df["AwayPointsAvgAwayLast5"] = away_points_avg_away_last5

    #--------------------------------------------
    # average team shots history the last 5 games
    #--------------------------------------------

    team_shots_history = {}

    home_shots_avg_last5 = []
    away_shots_avg_last5 = []
    shots_reset_done = set()

    for index, row in df.iterrows():

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        current_season = row["Season"]

        reset_promoted_history(
            team_shots_history,
            shots_reset_done,
            home_team,
            current_season,
            row["HomePromoted"]
        )

        reset_promoted_history(
            team_shots_history,
            shots_reset_done,
            away_team,
            current_season,
            row["AwayPromoted"]
        )

        if home_team not in team_shots_history:
            team_shots_history[home_team] = []

        if away_team not in team_shots_history:
            team_shots_history[away_team] = []

        home_history = team_shots_history[home_team]
        away_history = team_shots_history[away_team]

        home_shots_avg_last5.append(average_last5(home_history))
        away_shots_avg_last5.append(average_last5(away_history))

        team_shots_history[home_team].append(row["HS"])
        team_shots_history[away_team].append(row["AS"])

    df["HomeShotsAvgLast5"] = home_shots_avg_last5
    df["AwayShotsAvgLast5"] = away_shots_avg_last5

    # --------------------------------------------
    # average team shots on target history the last 5 games
    # --------------------------------------------

    team_shots_target_history = {}

    home_shots_target_avg_last5 = []
    away_shots_target_avg_last5 = []

    shots_target_reset_done = set()

    for index, row in df.iterrows():

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        current_season = row["Season"]

        reset_promoted_history(
            team_shots_target_history,
            shots_target_reset_done,
            home_team,
            current_season,
            row["HomePromoted"]
        )

        reset_promoted_history(
            team_shots_target_history,
            shots_target_reset_done,
            away_team,
            current_season,
            row["AwayPromoted"]
        )

        if home_team not in team_shots_target_history:
            team_shots_target_history[home_team] = []
        if away_team not in team_shots_target_history:
            team_shots_target_history[away_team] = []

        home_history = team_shots_target_history[home_team]
        away_history = team_shots_target_history[away_team]

        home_shots_target_avg_last5.append(average_last5(home_history))
        away_shots_target_avg_last5.append(average_last5(away_history))

        team_shots_target_history[home_team].append(row["HST"])
        team_shots_target_history[away_team].append(row["AST"])

    df["HomeShotsTargetAvgLast5"] = home_shots_target_avg_last5
    df["AwayShotsTargetAvgLast5"] = away_shots_target_avg_last5

    #--------------------------------------
    # recent team form
    #--------------------------------------

    df["HomeShotAccuracyLast5"] = np.where(
        df["HomeShotsAvgLast5"] > 0,
        df["HomeShotsTargetAvgLast5"] / df["HomeShotsAvgLast5"],
        0
    )

    df["AwayShotAccuracyLast5"] = np.where(
        df["AwayShotsAvgLast5"] > 0,
        df["AwayShotsTargetAvgLast5"] / df["AwayShotsAvgLast5"],
        0
    )

    check = df[
        (df["Season"] == "2025-26")
        & (
                (df["HomeTeam"] == "Tottenham")
                | (df["AwayTeam"] == "Tottenham")
                | (df["HomeTeam"] == "Burnley")
                | (df["AwayTeam"] == "Burnley")
        )
        ]

    return df

if __name__ == "__main__":

    df = load_data()

    print("Data loaded:", len(df))

    df = create_features(df)

    print("\nFeatures created successfully!")

