"""
IPL Fantasy Prediction — Data Loading & Cleaning
"""
import pandas as pd
from src.config import (
    MATCHES_CSV, DELIVERIES_CSV, MERGED_CSV,
    DATA_PROCESSED, TEAM_NAME_MAP,
)


def _normalize_team(name: str) -> str:
    """Map legacy / renamed team names to their current name."""
    if pd.isna(name):
        return name
    return TEAM_NAME_MAP.get(name.strip(), name.strip())


def load_matches() -> pd.DataFrame:
    """Load matches.csv, parse dates, and normalise team names."""
    df = pd.read_csv(MATCHES_CSV)

    # Standardise column names (some datasets use varying casing)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Rename 'id' to 'match_id' if needed
    if "id" in df.columns and "match_id" not in df.columns:
        df.rename(columns={"id": "match_id"}, inplace=True)

    # Normalise team names
    for col in ["team1", "team2", "winner", "toss_winner"]:
        if col in df.columns:
            df[col] = df[col].apply(_normalize_team)

    # Ensure season column exists
    if "season" not in df.columns and "date" in df.columns:
        df["season"] = df["date"].dt.year

    return df


def load_deliveries() -> pd.DataFrame:
    """Load deliveries.csv and normalise names."""
    df = pd.read_csv(DELIVERIES_CSV)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename 'id' to 'match_id' if present
    if "id" in df.columns and "match_id" not in df.columns:
        df.rename(columns={"id": "match_id"}, inplace=True)

    # Normalise team names
    for col in ["batting_team", "bowling_team"]:
        if col in df.columns:
            df[col] = df[col].apply(_normalize_team)

    return df


def merge_data(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Merge matches + deliveries on match_id and add useful columns."""
    merged = deliveries.merge(
        matches[["match_id", "season", "date", "venue",
                 "team1", "team2", "winner",
                 "toss_winner", "toss_decision", "player_of_match"]],
        on="match_id",
        how="left",
    )
    return merged


def save_processed(df: pd.DataFrame) -> None:
    """Save merged DataFrame to processed directory."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(MERGED_CSV, index=False)
    print(f"✅ Saved merged data → {MERGED_CSV}  ({len(df):,} rows)")


def load_processed() -> pd.DataFrame:
    """Load previously saved merged data."""
    return pd.read_csv(MERGED_CSV)


def get_data() -> pd.DataFrame:
    """
    High-level helper: return merged DataFrame.
    Uses cached version if available, else builds from raw CSVs.
    """
    if MERGED_CSV.exists():
        print("📂 Loading cached merged data …")
        return load_processed()

    print("🔄 Building merged dataset from raw CSVs …")
    matches = load_matches()
    deliveries = load_deliveries()
    merged = merge_data(matches, deliveries)
    save_processed(merged)
    return merged
