"""
IPL Fantasy Prediction — Configuration & Constants
"""
import os
from pathlib import Path

# ──────────────────────────── Paths ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MATCHES_CSV = DATA_RAW / "matches.csv"
DELIVERIES_CSV = DATA_RAW / "deliveries.csv"
MERGED_CSV = DATA_PROCESSED / "merged.csv"
PLAYER_FEATURES_CSV = DATA_PROCESSED / "player_features.csv"
TEAM_FEATURES_CSV = DATA_PROCESSED / "team_features.csv"

PLAYER_MODEL_PATH = MODELS_DIR / "player_model.pkl"
TEAM_MODEL_PATH = MODELS_DIR / "team_model.pkl"

# ──────────────────────────── Gemini ───────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBNMwu8C-1LYGepgTRtMKcXtU5stejIKWI")

# ──────────────── Fantasy Points Scoring Rules ─────────────────
FANTASY_SCORING = {
    # Batting
    "run": 1,
    "boundary_4": 4,
    "boundary_6": 6,
    "half_century_bonus": 20,
    "century_bonus": 50,
    "duck_penalty": -5,           # 0 runs & dismissed
    "sr_below_80_penalty": -6,    # min 10 balls faced
    "sr_above_170_bonus": 6,      # min 10 balls faced
    # Bowling
    "wicket": 25,
    "lbw_bowled_bonus": 8,        # extra for bowled / lbw
    "three_wicket_bonus": 15,
    "five_wicket_bonus": 30,
    "economy_below_5_bonus": 6,   # min 2 overs
    "economy_above_10_penalty": -6,
    # Fielding
    "catch": 8,
    "stumping": 12,
    "run_out_direct": 12,
    "run_out_indirect": 6,
}

# ──────────────── Team Name Normalization Map ──────────────────
TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Rising Pune Supergiants": "Rising Pune Supergiants",
    "Pune Warriors": "Pune Warriors India",
}

# ──────────────── Current IPL Teams (2026) ─────────────────────
CURRENT_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

# ─────────────── Player Model Feature Columns ─────────────────
PLAYER_FEATURES = [
    "total_runs", "batting_avg", "strike_rate", "boundary_pct",
    "total_wickets", "bowling_avg", "economy_rate",
    "catches", "run_outs",
    "recent_form_5", "recent_form_10",
    "venue_batting_avg", "venue_bowling_avg",
    "vs_opponent_batting_avg", "vs_opponent_bowling_avg",
    "matches_played", "innings_batted", "innings_bowled",
]

# ──────────────── Team Model Feature Columns ──────────────────
TEAM_FEATURES = [
    "avg_total_score", "avg_score_at_venue", "win_pct",
    "avg_powerplay_score", "avg_death_overs_score",
    "win_pct_vs_opponent", "toss_win_bat_first_avg",
    "toss_win_field_first_avg", "recent_form_5",
    "avg_first_innings_score", "avg_second_innings_score",
]
