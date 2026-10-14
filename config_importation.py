"""File 1: all libraries and all settings used by the other files.

Every other file starts with:  from imports_config import *
"""
# ---------- Libraries ----------
import re
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split

# ---------- Paths ----------
# Folder layout (everything sits in the Assignment3_Data_Science_Report folder):
#   World_Cup_MatchData_Stats_Raw.csv  <- your raw FBref file (you already have it)
#   results.csv                        <- downloaded automatically on the first run
#   model1_dataset.csv                 <- created by the code (the 8 variables + goal difference)
#   outputs/figures, outputs/tables    <- created by the code
ROOT = Path(__file__).resolve().parent
RAW_MATCHES_CSV = ROOT / "World_Cup_MatchData_Stats_Raw.csv"      # FBref match list
RAW_HISTORY_CSV = ROOT / "results.csv"                            # all international matches
HISTORY_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
PROCESSED_CSV = ROOT / "World_Cup_MatchData_Stats_Updated.csv"
SUBMISSION_CSV = ROOT / "World_Cup_MatchData_Stats_Submission.csv"   # ID columns + 8 variables + target only
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

# ---------- Tournament facts ----------
N_MATCHES = 104
N_TEAMS = 48
WC_START = "2026-06-11"                  # first match of the tournament
WC_END = "2026-07-19"                    # final
WC_TOURNAMENT_NAME = "FIFA World Cup"      # label used in results.csv
HOSTS = {"United States", "Canada", "Mexico"}

# FBref team names -> names used in results.csv (so both files can be joined)
NAME_MAP = {
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Bosnia–Herz": "Bosnia and Herzegovina",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "USA": "United States",
}
# Older countries counted together for World Cup appearances
PREDECESSORS = {"Czechoslovakia": "Czech Republic"}

# Month abbreviations (needed to repair scores that Excel turned into dates)
MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

# ---------- Elo rating settings ----------
ELO_START = 1500
HOME_ADVANTAGE = 100
LAST_N_MATCHES = 10

# ---------- Model settings ----------
TARGET = "goal_difference"      # team_a goals - team_b goals (score before any penalty shoot-out)
FEATURES = [
    "diff_elo",                # strength rating (Elo) A - B
    "diff_gf_L10",             # avg goals scored in last 10 matches A - B
    "diff_ga_L10",             # avg goals conceded in last 10 matches A - B
    "diff_ppg_L10",            # form: points per game in last 10 matches A - B
    "diff_wc_appearances",     # previous World Cup appearances A - B
    "diff_sos_L10",            # strength of schedule: avg opponent Elo in last 10 matches A - B
    "host_advantage",          # +1 if A is a host, -1 if B is a host, else 0
    "is_knockout",             # 1 = knockout match, 0 = group stage
]
RANDOM_STATE = 42
TEST_SIZE = 0.2