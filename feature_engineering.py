from config_importation import *
from data_loading import load_history, load_wc_matches
from data_cleaning import clean_matches


def k_factor(tournament):
    """How much one match can move an Elo rating (bigger tournaments matter more)."""
    if tournament == WC_TOURNAMENT_NAME:
        return 60
    if tournament in {
        "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
        "Gold Cup", "Confederations Cup", "Oceania Nations Cup"
    }:
        return 50
    if "qualification" in tournament or "Nations League" in tournament:
        return 40
    if tournament == "Friendly":
        return 20
    return 30


def compute_elo(hist, cutoff):
    """Elo rating of every team on the day before the tournament."""
    ratings = {}
    past = hist[hist["date"] < cutoff].sort_values("date", kind="stable")
    for home, away, hs, as_, tour, neutral in zip(
        past["home_team"], past["away_team"], past["home_score"],
        past["away_score"], past["tournament"], past["neutral"]
    ):
        r_home, r_away = ratings.get(home, ELO_START), ratings.get(away, ELO_START)
        advantage = 0 if neutral else HOME_ADVANTAGE
        expected_home = 1 / (1 + 10 ** (-(r_home + advantage - r_away) / 400))
        result = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        margin = abs(hs - as_)
        multiplier = 1.0 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8
        change = k_factor(tour) * multiplier * (result - expected_home)
        ratings[home], ratings[away] = r_home + change, r_away - change
    return ratings


def long_history(hist, cutoff):
    """One row per team per match (before the cutoff date)."""
    past = hist[hist["date"] < cutoff]
    home = pd.DataFrame({
        "date": past["date"], "team": past["home_team"], "opponent": past["away_team"],
        "gf": past["home_score"], "ga": past["away_score"]
    })
    away = pd.DataFrame({
        "date": past["date"], "team": past["away_team"], "opponent": past["home_team"],
        "gf": past["away_score"], "ga": past["home_score"]
    })
    long = pd.concat([home, away]).sort_values("date", kind="stable")
    long["points"] = np.where(long["gf"] > long["ga"], 3, np.where(long["gf"] == long["ga"], 1, 0))
    return long


def world_cup_appearances(hist, cutoff):
    """Number of previous World Cups each team played in."""
    wc = hist[(hist["tournament"] == WC_TOURNAMENT_NAME) & (hist["date"] < cutoff)]
    teams = pd.concat([
        pd.DataFrame({"team": wc["home_team"], "year": wc["date"].dt.year}),
        pd.DataFrame({"team": wc["away_team"], "year": wc["date"].dt.year}),
    ])
    teams["team"] = teams["team"].replace(PREDECESSORS)
    return teams.drop_duplicates().groupby("team").size()


def build_team_table(hist, teams, cutoff):
    """One row per World Cup team with its raw strength and form numbers."""
    elo = pd.Series(compute_elo(hist, cutoff))
    long = long_history(hist, cutoff)
    long["opp_elo"] = long["opponent"].map(elo)

    last10 = long.groupby("team").tail(LAST_N_MATCHES)
    summary = last10.groupby("team").agg(
        gf_L10=("gf", "mean"),
        ga_L10=("ga", "mean"),
        ppg_L10=("points", "mean"),
        sos_L10=("opp_elo", "mean"),
        matches_used=("gf", "size")
    )
    table = summary.reindex(teams)
    table["elo"] = elo.reindex(teams)
    table["wc_appearances"] = world_cup_appearances(hist, cutoff).reindex(teams).fillna(0)
    assert table.notna().all().all(), f"Missing team data:\n{table[table.isna().any(axis=1)]}"
    assert (table["matches_used"] == LAST_N_MATCHES).all(), "Some team has fewer than 10 earlier matches"
    return table


def build_features(matches, team_table):
    """Join team numbers onto each match and subtract: team A minus team B."""
    a = team_table.add_suffix("_a")
    b = team_table.add_suffix("_b")
    m = matches.join(a, on="team_a").join(b, on="team_b")

    m["diff_elo"] = m["elo_a"] - m["elo_b"]
    m["diff_gf_L10"] = m["gf_L10_a"] - m["gf_L10_b"]
    m["diff_ga_L10"] = m["ga_L10_a"] - m["ga_L10_b"]
    m["diff_ppg_L10"] = m["ppg_L10_a"] - m["ppg_L10_b"]
    m["diff_wc_appearances"] = m["wc_appearances_a"] - m["wc_appearances_b"]
    m["diff_sos_L10"] = m["sos_L10_a"] - m["sos_L10_b"]

    final = m[["match_id", "date", "stage", "team_a", "team_b"] + FEATURES + [TARGET]]
    assert len(FEATURES) == 8, "The brief requires exactly 8 explanatory variables"
    assert final[FEATURES + [TARGET]].notna().all().all(), "Missing values after feature engineering"
    return final


def save_dataset(final):
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(PROCESSED_CSV, index=False)
    print(f"Saved {len(final)} rows x {len(FEATURES)} variables to {PROCESSED_CSV}")

    # Copy for submission: only ID columns + exactly 8 explanatory variables + the target
    submission = final[["match_id", "team_a", "team_b"] + FEATURES + [TARGET]]
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"Saved submission copy ({submission.shape[1]} columns) to {SUBMISSION_CSV}")


if __name__ == "__main__":
    raw_matches = load_wc_matches()
    history = load_history()
    matches = clean_matches(raw_matches)

    # Cutoff date: start of the tournament
    cutoff_date = pd.Timestamp(WC_START)

    # Get unique teams competing in this World Cup
    participating_teams = sorted(list(set(matches["team_a"]).union(set(matches["team_b"]))))
    assert len(participating_teams) == N_TEAMS, f"Expected {N_TEAMS} teams, found {len(participating_teams)}"

    # Build team profiles, construct features, and save datasets
    team_table = build_team_table(history, participating_teams, cutoff_date)
    final_dataset = build_features(matches, team_table)
    save_dataset(final_dataset)