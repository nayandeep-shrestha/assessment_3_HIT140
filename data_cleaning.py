from config_importation import *
from data_loading import load_history, load_wc_matches

def clean_team_name(name, side):
    """Remove the 2-3 letter country code, then use the same spelling as results.csv."""
    if side == "home":
        name = re.sub(r"\s+[a-z]{2,3}$", "", name.strip())      # 'Mexico mx' -> 'Mexico'
    else:
        name = re.sub(r"^[a-z]{2,3}\s+", "", name.strip())      # 'za South Africa' -> 'South Africa'
    return NAME_MAP.get(name, name)


def parse_score(text):
    """Return (goals_home, goals_away, penalties_home, penalties_away)."""
    s = str(text).strip().replace("–", "-").replace("—", "-").replace(":", "-")   # unify dashes

    m = re.fullmatch(r"\((\d+)\)\s*(\d+)\s*-\s*(\d+)\s*\((\d+)\)", s)     # '(3) 1-1 (4)'
    if m:
        return int(m[2]), int(m[3]), int(m[1]), int(m[4])

    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)                        # '2-0' (already correct)
    if m:
        return int(m[1]), int(m[2]), None, None

    m = re.fullmatch(r"([A-Za-z]{3})-00", s)                         # 'Feb-00' was '2-0'
    if m:
        return MONTHS[m[1].title()], 0, None, None

    m = re.fullmatch(r"(\d+)-([A-Za-z]{3})", s)                      # '1-Jul' was '7-1' (month-day)
    if m:
        return MONTHS[m[2].title()], int(m[1]), None, None

    raise ValueError(f"Cannot read score: {text!r}")


def parse_dates(series):
    """Read the Date column. Tries month/day/year, day/month/year and year-month-day, and keeps
    the format whose dates ALL fall inside the tournament (11 June - 19 July 2026)."""
    start, end = pd.Timestamp(WC_START), pd.Timestamp(WC_END)
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().all() and parsed.between(start, end).all():
            return parsed
    raise ValueError("Could not read the Date column: dates must all be between 11 June and 19 July 2026.")


def clean_matches(raw):
    """One tidy row per match, with team_a / team_b in alphabetical order."""
    dates = parse_dates(raw["Date"])
    rows = []
    for i, r in raw.iterrows():
        home = clean_team_name(r["Home"], "home")
        away = clean_team_name(r["Away"], "away")
        gh, ga, ph, pa = parse_score(r["Score"])
        rows.append({
            "match_no": i + 1,
            "date": dates.iloc[i],
            "stage": r["Round"],
            "home": home, "away": away,
            "goals_home": gh, "goals_away": ga,
            "pens_home": ph, "pens_away": pa,
        })
    m = pd.DataFrame(rows)

    # Alphabetical order for team_a / team_b avoids any 'home team' bias
    swap = m["home"] > m["away"]
    m["team_a"] = np.where(swap, m["away"], m["home"])
    m["team_b"] = np.where(swap, m["home"], m["away"])
    goals_a = np.where(swap, m["goals_away"], m["goals_home"])
    goals_b = np.where(swap, m["goals_home"], m["goals_away"])
    m["goals_a"], m["goals_b"] = goals_a, goals_b
    m[TARGET] = goals_a - goals_b

    m["is_knockout"] = (m["stage"] != "Group stage").astype(int)
    m["host_advantage"] = (m["team_a"].isin(HOSTS).astype(int) - m["team_b"].isin(HOSTS).astype(int))
    m["match_id"] = "M" + m["match_no"].astype(str).str.zfill(3)
    return m


def check_against_history(matches, hist):
    """Independent check: the cleaned scores must equal the scores in results.csv."""
    wc = hist[(hist["tournament"] == WC_TOURNAMENT_NAME) & (hist["date"] >= matches["date"].min() - pd.Timedelta(days=1))]
    assert len(wc) == N_MATCHES, f"results.csv has {len(wc)} World Cup 2026 matches, expected {N_MATCHES}"

    a = np.where(wc["home_team"] < wc["away_team"], wc["home_team"], wc["away_team"])
    gd = np.where(wc["home_team"] < wc["away_team"],
                  wc["home_score"] - wc["away_score"], wc["away_score"] - wc["home_score"])
    reference = pd.DataFrame({"team_a": a,
                              "team_b": np.where(wc["home_team"] < wc["away_team"], wc["away_team"], wc["home_team"]),
                              "gd_reference": gd})
    merged = matches.merge(reference, on=["team_a", "team_b"], how="left")
    assert len(merged) == N_MATCHES and merged["gd_reference"].notna().all(), "Team names do not line up"
    wrong = merged[merged[TARGET] != merged["gd_reference"]]
    assert wrong.empty, f"Score mismatch in {len(wrong)} matches:\n{wrong[['match_id','team_a','team_b',TARGET,'gd_reference']]}"
    print("Check passed: all 104 cleaned scores agree with the independent results.csv file.")

if __name__ == "__main__":
    raw_matches = load_wc_matches()
    history = load_history()
    cleaned = clean_matches(raw_matches)
    print(f"Cleaned {len(cleaned)} matches successfully.")
    check_against_history(cleaned, history)