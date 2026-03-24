"""
IPL Fantasy Prediction — Feature Engineering

Builds per-player and per-team feature DataFrames from the merged
ball-by-ball data for use in ML models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    FANTASY_SCORING,
    DATA_PROCESSED,
    PLAYER_FEATURES_CSV,
    TEAM_FEATURES_CSV,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper: calculate fantasy points for a single player-match row
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _batting_fantasy(row: pd.Series) -> float:
    """Calculate batting fantasy points from aggregated match stats."""
    pts = 0.0
    runs = row.get("runs_scored", 0)
    balls = row.get("balls_faced", 0)
    fours = row.get("fours", 0)
    sixes = row.get("sixes", 0)
    dismissed = row.get("dismissed", False)

    pts += runs * FANTASY_SCORING["run"]
    pts += fours * FANTASY_SCORING["boundary_4"]
    pts += sixes * FANTASY_SCORING["boundary_6"]

    if runs >= 100:
        pts += FANTASY_SCORING["century_bonus"]
    elif runs >= 50:
        pts += FANTASY_SCORING["half_century_bonus"]

    if runs == 0 and dismissed:
        pts += FANTASY_SCORING["duck_penalty"]

    if balls >= 10:
        sr = (runs / balls) * 100
        if sr >= 170:
            pts += FANTASY_SCORING["sr_above_170_bonus"]
        elif sr < 80:
            pts += FANTASY_SCORING["sr_below_80_penalty"]

    return pts


def _bowling_fantasy(row: pd.Series) -> float:
    """Calculate bowling fantasy points from aggregated match stats."""
    pts = 0.0
    wickets = row.get("wickets_taken", 0)
    lbw_bowled = row.get("lbw_bowled", 0)
    balls_bowled = row.get("balls_bowled", 0)
    runs_conceded = row.get("runs_conceded", 0)

    pts += wickets * FANTASY_SCORING["wicket"]
    pts += lbw_bowled * FANTASY_SCORING["lbw_bowled_bonus"]

    if wickets >= 5:
        pts += FANTASY_SCORING["five_wicket_bonus"]
    elif wickets >= 3:
        pts += FANTASY_SCORING["three_wicket_bonus"]

    overs = balls_bowled / 6
    if overs >= 2:
        economy = runs_conceded / overs
        if economy < 5:
            pts += FANTASY_SCORING["economy_below_5_bonus"]
        elif economy > 10:
            pts += FANTASY_SCORING["economy_above_10_penalty"]

    return pts


def _fielding_fantasy(row: pd.Series) -> float:
    """Calculate fielding fantasy points."""
    pts = 0.0
    pts += row.get("catches", 0) * FANTASY_SCORING["catch"]
    pts += row.get("stumpings", 0) * FANTASY_SCORING["stumping"]
    pts += row.get("run_out_direct", 0) * FANTASY_SCORING["run_out_direct"]
    pts += row.get("run_out_indirect", 0) * FANTASY_SCORING["run_out_indirect"]
    return pts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 1: Aggregate ball-by-ball → per-player-per-match stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _player_match_batting(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate batting stats per batter per match."""
    # Determine runs column name
    runs_col = "batsman_runs" if "batsman_runs" in df.columns else "batter_runs"
    if runs_col not in df.columns:
        # Try to find any column with 'bats' and 'run'
        candidates = [c for c in df.columns if "bats" in c.lower() and "run" in c.lower()]
        runs_col = candidates[0] if candidates else "batsman_runs"

    batter_col = "batter" if "batter" in df.columns else "batsman"

    # Count balls faced (exclude wides)
    valid_balls = df.copy()
    if "extras_type" in valid_balls.columns:
        valid_balls = valid_balls[valid_balls["extras_type"] != "wides"]
    elif "wide_runs" in valid_balls.columns:
        valid_balls = valid_balls[valid_balls["wide_runs"] == 0]
    elif "wides" in valid_balls.columns:
        valid_balls = valid_balls[valid_balls["wides"] == 0]

    balls_faced = (
        valid_balls.groupby(["match_id", batter_col])
        .size()
        .reset_index(name="balls_faced")
    )

    batting = (
        df.groupby(["match_id", batter_col])
        .agg(
            runs_scored=(runs_col, "sum"),
            fours=(runs_col, lambda x: (x == 4).sum()),
            sixes=(runs_col, lambda x: (x == 6).sum()),
            batting_team=("batting_team", "first"),
            bowling_team=("bowling_team", "first"),
            season=("season", "first"),
            venue=("venue", "first"),
        )
        .reset_index()
        .rename(columns={batter_col: "player"})
    )

    batting = batting.merge(balls_faced.rename(columns={batter_col: "player"}),
                            on=["match_id", "player"], how="left")
    batting["balls_faced"] = batting["balls_faced"].fillna(0).astype(int)

    # Check if dismissed
    dismiss_col = "player_dismissed" if "player_dismissed" in df.columns else "dismissal_kind"
    if "player_dismissed" in df.columns:
        dismissed = (
            df[df["player_dismissed"].notna()]
            .groupby(["match_id", "player_dismissed"])
            .size()
            .reset_index(name="_d")
        )
        dismissed.rename(columns={"player_dismissed": "player"}, inplace=True)
        batting = batting.merge(dismissed[["match_id", "player"]],
                                on=["match_id", "player"], how="left", indicator=True)
        batting["dismissed"] = batting["_merge"] == "both"
        batting.drop(columns=["_merge"], inplace=True)
    else:
        batting["dismissed"] = False

    return batting


def _player_match_bowling(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bowling stats per bowler per match."""
    runs_col = "total_runs" if "total_runs" in df.columns else "batsman_runs"

    # Only count legal deliveries for balls bowled
    legal = df.copy()
    if "extras_type" in legal.columns:
        legal = legal[~legal["extras_type"].isin(["wides", "noballs"])]
    elif "wide_runs" in legal.columns and "noball_runs" in legal.columns:
        legal = legal[(legal["wide_runs"] == 0) & (legal["noball_runs"] == 0)]
    elif "wides" in legal.columns and "noballs" in legal.columns:
        legal = legal[(legal["wides"] == 0) & (legal["noballs"] == 0)]

    balls_bowled = (
        legal.groupby(["match_id", "bowler"])
        .size()
        .reset_index(name="balls_bowled")
    )

    # Wickets (exclude run-outs as they aren't bowler dismissals)
    wicket_df = df[
        df["dismissal_kind"].isin([
            "bowled", "caught", "caught and bowled",
            "lbw", "stumped", "hit wicket",
        ])
    ] if "dismissal_kind" in df.columns else df[0:0]

    wickets = (
        wicket_df.groupby(["match_id", "bowler"])
        .size()
        .reset_index(name="wickets_taken")
    )

    lbw_bowled_df = df[
        df["dismissal_kind"].isin(["bowled", "lbw"])
    ] if "dismissal_kind" in df.columns else df[0:0]

    lbw_bowled = (
        lbw_bowled_df.groupby(["match_id", "bowler"])
        .size()
        .reset_index(name="lbw_bowled")
    )

    bowling = (
        df.groupby(["match_id", "bowler"])
        .agg(
            runs_conceded=(runs_col, "sum"),
            batting_team=("batting_team", "first"),
            bowling_team=("bowling_team", "first"),
            season=("season", "first"),
            venue=("venue", "first"),
        )
        .reset_index()
        .rename(columns={"bowler": "player"})
    )

    bowling = bowling.merge(balls_bowled.rename(columns={"bowler": "player"}),
                            on=["match_id", "player"], how="left")
    bowling = bowling.merge(wickets.rename(columns={"bowler": "player"}),
                            on=["match_id", "player"], how="left")
    bowling = bowling.merge(lbw_bowled.rename(columns={"bowler": "player"}),
                            on=["match_id", "player"], how="left")

    bowling["balls_bowled"] = bowling["balls_bowled"].fillna(0).astype(int)
    bowling["wickets_taken"] = bowling["wickets_taken"].fillna(0).astype(int)
    bowling["lbw_bowled"] = bowling["lbw_bowled"].fillna(0).astype(int)

    return bowling


def _player_match_fielding(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fielding stats per fielder per match."""
    records = []

    if "dismissal_kind" not in df.columns:
        return pd.DataFrame(columns=["match_id", "player", "catches",
                                     "stumpings", "run_out_direct", "run_out_indirect"])

    fielder_col = "fielder" if "fielder" in df.columns else "fielders"

    dismissals = df[df["dismissal_kind"].notna() & df[fielder_col].notna()].copy()

    catches = (
        dismissals[dismissals["dismissal_kind"].isin(["caught", "caught and bowled"])]
        .groupby(["match_id", fielder_col]).size()
        .reset_index(name="catches")
        .rename(columns={fielder_col: "player"})
    )

    stumpings = (
        dismissals[dismissals["dismissal_kind"] == "stumped"]
        .groupby(["match_id", fielder_col]).size()
        .reset_index(name="stumpings")
        .rename(columns={fielder_col: "player"})
    )

    run_outs = (
        dismissals[dismissals["dismissal_kind"] == "run out"]
        .groupby(["match_id", fielder_col]).size()
        .reset_index(name="run_out_direct")
        .rename(columns={fielder_col: "player"})
    )

    # Combine
    all_fielders = set()
    for fdf in [catches, stumpings, run_outs]:
        if len(fdf):
            all_fielders.update(
                fdf[["match_id", "player"]].apply(tuple, axis=1).tolist()
            )

    if not all_fielders:
        return pd.DataFrame(columns=["match_id", "player", "catches",
                                     "stumpings", "run_out_direct", "run_out_indirect"])

    base = pd.DataFrame(list(all_fielders), columns=["match_id", "player"])
    base = base.merge(catches, on=["match_id", "player"], how="left")
    base = base.merge(stumpings, on=["match_id", "player"], how="left")
    base = base.merge(run_outs, on=["match_id", "player"], how="left")
    base = base.fillna(0)
    base["run_out_indirect"] = 0  # simplified — dataset doesn't distinguish
    return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 2: Build per-player career / rolling features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _add_rolling(group: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    """Add rolling average columns for given windows."""
    group = group.sort_values("match_id")
    for w in windows:
        group[f"recent_form_{w}"] = (
            group[col].rolling(window=w, min_periods=1).mean().shift(1)
        )
    return group


def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comprehensive player feature DataFrame.

    Each row = one player's stats up to (but not including) a given match,
    plus the target fantasy_points they scored in that match.
    """
    print("⚙️  Building player match-level stats …")
    bat = _player_match_batting(df)
    bowl = _player_match_bowling(df)
    field = _player_match_fielding(df)

    # Merge batting + bowling on (match_id, player)
    player_match = bat.merge(
        bowl.drop(columns=["batting_team", "bowling_team", "season", "venue"],
                  errors="ignore"),
        on=["match_id", "player"],
        how="outer",
    )
    player_match = player_match.merge(field, on=["match_id", "player"], how="left")

    # Fill NaN numerics with 0
    num_cols = player_match.select_dtypes(include="number").columns
    player_match[num_cols] = player_match[num_cols].fillna(0)

    # Forward-fill categorical columns
    for col in ["batting_team", "bowling_team", "season", "venue"]:
        if col in player_match.columns:
            player_match[col] = player_match[col].ffill()

    # Fantasy points for the match
    player_match["fantasy_points"] = (
        player_match.apply(_batting_fantasy, axis=1)
        + player_match.apply(_bowling_fantasy, axis=1)
        + player_match.apply(_fielding_fantasy, axis=1)
    )

    print("⚙️  Computing career & rolling aggregates …")

    # Sort by match_id (proxy for chronological order)
    player_match = player_match.sort_values(["player", "match_id"]).reset_index(drop=True)

    # Career cumulative stats (shifted so we use only past info)
    career = (
        player_match.groupby("player")
        .apply(lambda g: g.assign(
            total_runs=g["runs_scored"].cumsum().shift(1, fill_value=0),
            innings_batted=(g["balls_faced"] > 0).cumsum().shift(1, fill_value=0),
            total_wickets=g["wickets_taken"].cumsum().shift(1, fill_value=0),
            innings_bowled=(g["balls_bowled"] > 0).cumsum().shift(1, fill_value=0),
            matches_played=range(len(g)),
            catches_total=g["catches"].cumsum().shift(1, fill_value=0) if "catches" in g.columns else 0,
            run_outs_total=(g["run_out_direct"].cumsum().shift(1, fill_value=0)
                           if "run_out_direct" in g.columns else 0),
        ), include_groups=False)
        .reset_index(drop=True)
    )

    player_match["total_runs"] = career["total_runs"]
    player_match["innings_batted"] = career["innings_batted"]
    player_match["total_wickets"] = career["total_wickets"]
    player_match["innings_bowled"] = career["innings_bowled"]
    player_match["matches_played"] = career["matches_played"]
    player_match["catches"] = career["catches_total"] if "catches_total" in career.columns else 0
    player_match["run_outs"] = career["run_outs_total"] if "run_outs_total" in career.columns else 0

    # Derived averages
    player_match["batting_avg"] = np.where(
        player_match["innings_batted"] > 0,
        player_match["total_runs"] / player_match["innings_batted"],
        0,
    )
    player_match["strike_rate"] = np.where(
        player_match["innings_batted"] > 0,
        player_match["total_runs"] / player_match["innings_batted"] * 100 / 20,  # approx
        0,
    )
    fours_col = player_match["fours"] if "fours" in player_match.columns else 0
    sixes_col = player_match["sixes"] if "sixes" in player_match.columns else 0
    player_match["boundary_pct"] = np.where(
        player_match["total_runs"] > 0,
        (fours_col * 4 + sixes_col * 6) / player_match["total_runs"],
        0,
    )
    player_match["bowling_avg"] = np.where(
        player_match["total_wickets"] > 0,
        player_match["total_runs"] / player_match["total_wickets"],
        0,
    )
    runs_conc = player_match["runs_conceded"] if "runs_conceded" in player_match.columns else 0
    balls_bwl = player_match["balls_bowled"] if "balls_bowled" in player_match.columns else 0
    overs_bwl = np.where(balls_bwl > 0, balls_bwl / 6, 1)  # avoid div-by-zero
    player_match["economy_rate"] = np.where(
        player_match["innings_bowled"] > 0,
        runs_conc / overs_bwl,
        0,
    )

    # Rolling form
    player_match = (
        player_match.groupby("player", group_keys=False)
        .apply(lambda g: _add_rolling(g, "fantasy_points", [5, 10]))
    )
    player_match["recent_form_5"] = player_match["recent_form_5"].fillna(0)
    player_match["recent_form_10"] = player_match["recent_form_10"].fillna(0)

    # Venue & opponent averages (simplified: overall avg as proxy)
    player_match["venue_batting_avg"] = player_match["batting_avg"]
    player_match["venue_bowling_avg"] = player_match["bowling_avg"]
    player_match["vs_opponent_batting_avg"] = player_match["batting_avg"]
    player_match["vs_opponent_bowling_avg"] = player_match["bowling_avg"]

    # ── Sanitize: replace inf / NaN with 0 ──
    num_cols = player_match.select_dtypes(include="number").columns
    player_match[num_cols] = player_match[num_cols].replace([np.inf, -np.inf], 0).fillna(0)

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    player_match.to_csv(PLAYER_FEATURES_CSV, index=False)
    print(f"✅ Player features saved → {PLAYER_FEATURES_CSV}  ({len(player_match):,} rows)")

    return player_match


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 3: Build per-team features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build team-level features per innings.

    Each row = one team's stats for a given innings in a match,
    plus target total_score for that innings.
    """
    print("⚙️  Building team features …")

    # Total per innings
    runs_col = "total_runs" if "total_runs" in df.columns else "batsman_runs"

    innings = (
        df.groupby(["match_id", "inning", "batting_team"])
        .agg(
            total_score=(runs_col, "sum"),
            venue=("venue", "first"),
            season=("season", "first"),
            bowling_team=("bowling_team", "first"),
        )
        .reset_index()
        .rename(columns={"batting_team": "team"})
    )

    # Powerplay (overs 1-6) and death (overs 16-20)
    pp = df[df["over"].between(1, 6) if "over" in df.columns else df.index < 0]
    death = df[df["over"].between(16, 20) if "over" in df.columns else df.index < 0]

    if len(pp):
        pp_scores = (
            pp.groupby(["match_id", "inning", "batting_team"])[runs_col]
            .sum().reset_index(name="powerplay_score")
            .rename(columns={"batting_team": "team"})
        )
        innings = innings.merge(pp_scores, on=["match_id", "inning", "team"], how="left")
    else:
        innings["powerplay_score"] = 0

    if len(death):
        death_scores = (
            death.groupby(["match_id", "inning", "batting_team"])[runs_col]
            .sum().reset_index(name="death_score")
            .rename(columns={"batting_team": "team"})
        )
        innings = innings.merge(death_scores, on=["match_id", "inning", "team"], how="left")
    else:
        innings["death_score"] = 0

    innings["powerplay_score"] = innings["powerplay_score"].fillna(0)
    innings["death_score"] = innings["death_score"].fillna(0)

    # Sort chronologically
    innings = innings.sort_values(["team", "match_id", "inning"]).reset_index(drop=True)

    # Career / rolling aggregates
    innings = innings.groupby("team", group_keys=False).apply(_team_rolling_features)

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    innings.to_csv(TEAM_FEATURES_CSV, index=False)
    print(f"✅ Team features saved → {TEAM_FEATURES_CSV}  ({len(innings):,} rows)")

    return innings


def _team_rolling_features(g: pd.DataFrame) -> pd.DataFrame:
    """Add rolling / cumulative features per team."""
    g = g.sort_values("match_id").copy()

    g["avg_total_score"] = g["total_score"].expanding().mean().shift(1).fillna(0)
    g["avg_powerplay_score"] = g["powerplay_score"].expanding().mean().shift(1).fillna(0)
    g["avg_death_overs_score"] = g["death_score"].expanding().mean().shift(1).fillna(0)

    # Recent form
    g["recent_form_5"] = g["total_score"].rolling(5, min_periods=1).mean().shift(1).fillna(0)

    # 1st vs 2nd innings avg
    g["avg_first_innings_score"] = (
        g[g["inning"] == 1]["total_score"]
        .expanding().mean().shift(1)
        .reindex(g.index).ffill().fillna(0)
    )
    g["avg_second_innings_score"] = (
        g[g["inning"] == 2]["total_score"]
        .expanding().mean().shift(1)
        .reindex(g.index).ffill().fillna(0)
    )

    # Venue avg (simplified: same as overall)
    g["avg_score_at_venue"] = g["avg_total_score"]

    # Win pct placeholder (needs match result integration)
    g["win_pct"] = 0.5
    g["win_pct_vs_opponent"] = 0.5
    g["toss_win_bat_first_avg"] = g["avg_total_score"]
    g["toss_win_field_first_avg"] = g["avg_total_score"]

    return g
