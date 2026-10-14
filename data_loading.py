from config_importation import *


def load_wc_matches():
    """FBref World Cup 2026 match list (raw; scores are repaired in the cleaning step if needed)."""
    raw = pd.read_csv(RAW_MATCHES_CSV, encoding="utf-8-sig")
    # FBref puts an empty separator row between rounds; remove those rows
    raw = raw.dropna(subset=["Round", "Home", "Away", "Score"]).reset_index(drop=True)
    assert len(raw) == N_MATCHES, f"Expected {N_MATCHES} matches, found {len(raw)}"
    print(f"Loaded {len(raw)} World Cup matches from {RAW_MATCHES_CSV.name}")
    return raw


def load_history():
    """All international matches ever played (used for Elo, form and World Cup history)."""
    hist = pd.read_csv(RAW_HISTORY_CSV, parse_dates=["date"])
    needed = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    assert needed <= set(hist.columns), f"results.csv is missing columns: {needed - set(hist.columns)}"
    hist = hist.dropna(subset=["home_score", "away_score"])
    print(f"Loaded {len(hist)} historical matches (latest: {hist['date'].max().date()})")
    return hist

if __name__ == "__main__":
    df_matches = load_wc_matches()
    df_history = load_history()