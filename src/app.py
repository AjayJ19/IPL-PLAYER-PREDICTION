"""
IPL Fantasy Prediction — Streamlit Dashboard
=============================================
Multi-page app: Player Predictor · Team Score · Historical Stats · AI Insights
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    CURRENT_TEAMS, PLAYER_FEATURES, TEAM_FEATURES,
    PLAYER_MODEL_PATH, TEAM_MODEL_PATH,
    PLAYER_FEATURES_CSV, TEAM_FEATURES_CSV,
)
from src.data_loader import get_data
from src.feature_engineering import build_player_features, build_team_features
from src.player_model import PlayerPerformanceModel
from src.team_model import TeamScoreModel
from src.ai_insights import generate_player_insight, generate_match_preview, ask_cricket_question


# ━━━━━━━━━━━━━━━━━━━━━━ Page Config ━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="IPL Fantasy Predictor 🏏",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━ Custom CSS ━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero gradient header */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .hero-header h1 {
        background: linear-gradient(90deg, #e94560, #ff6b6b, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .hero-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Player card */
    .player-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease;
    }
    .player-card:hover {
        transform: translateX(4px);
    }
    .player-card.high { border-color: #10b981; }
    .player-card.medium { border-color: #f59e0b; }
    .player-card.low { border-color: #ef4444; }

    .player-card .name {
        font-weight: 700;
        font-size: 1.1rem;
        color: #f1f5f9;
    }
    .player-card .pts {
        font-weight: 600;
        font-size: 1.3rem;
    }
    .player-card.high .pts { color: #10b981; }
    .player-card.medium .pts { color: #f59e0b; }
    .player-card.low .pts { color: #ef4444; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }

    /* Win prob bar */
    .win-bar {
        display: flex;
        height: 40px;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .win-bar .team1 {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        display: flex; align-items: center; justify-content: center; color: white;
    }
    .win-bar .team2 {
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
        display: flex; align-items: center; justify-content: center; color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━ Data & Model Loading ━━━━━━━━━━━━━━━━
@st.cache_data(show_spinner="🏏 Loading IPL data …")
def load_all_data():
    """Load merged data and build features."""
    merged = get_data()
    player_features = build_player_features(merged)
    team_features = build_team_features(merged)
    return merged, player_features, team_features


@st.cache_resource(show_spinner="🧠 Training models …")
def train_models(_player_df, _team_df):
    """Train (or load cached) player and team models."""
    pm = PlayerPerformanceModel()
    tm = TeamScoreModel()

    if PLAYER_MODEL_PATH.exists():
        pm.load()
    else:
        pm.train(_player_df)
        pm.save()

    if TEAM_MODEL_PATH.exists():
        tm.load()
    else:
        tm.train(_team_df)
        tm.save()

    return pm, tm


# ━━━━━━━━━━━━━━━━━━━━━━━ Sidebar ━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 🏏 IPL Fantasy Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏅 Player Predictor",
            "📊 Team Score Predictor",
            "📈 Historical Stats",
            "🤖 AI Insights",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#64748b; font-size:0.75rem; text-align:center;'>"
        "Built with ❤️ for Fantasy Cricket<br>Data: IPL 2008–2024</p>",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━ Try loading data ━━━━━━━━━━━━━━━━━━━
try:
    merged, player_df, team_df = load_all_data()
    player_model, team_model = train_models(player_df, team_df)
    DATA_LOADED = True
except FileNotFoundError:
    DATA_LOADED = False
    st.markdown("""
    <div class="hero-header">
        <h1>🚨 Data Not Found</h1>
        <p>Place <code>matches.csv</code> and <code>deliveries.csv</code> in 
        <code>data/raw/</code> and reload.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("📥 Download from [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)")
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE 1: PLAYER PERFORMANCE PREDICTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🏅 Player Predictor":
    st.markdown("""
    <div class="hero-header">
        <h1>🏅 Player Performance Predictor</h1>
        <p>Select a match scenario and discover the top fantasy picks ranked by predicted points</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Match selectors ---
    col1, col2, col3 = st.columns(3)

    all_teams = sorted(player_df["batting_team"].dropna().unique().tolist())
    venues = sorted(merged["venue"].dropna().unique().tolist())

    with col1:
        team1 = st.selectbox("🏠 Team 1", all_teams, index=0, key="pp_t1")
    with col2:
        team2_options = [t for t in all_teams if t != team1]
        team2 = st.selectbox("✈️ Team 2", team2_options, index=0, key="pp_t2")
    with col3:
        venue = st.selectbox("🏟️ Venue", venues, index=0, key="pp_venue")

    st.markdown("---")

    if st.button("🔮 Predict Player Performance", type="primary", use_container_width=True):
        with st.spinner("Crunching numbers …"):
            # Get latest stats for players from both teams
            team_players = player_df[
                player_df["batting_team"].isin([team1, team2])
            ].copy()

            # Take most recent record per player
            latest = (
                team_players.sort_values("match_id", ascending=False)
                .groupby("player")
                .first()
                .reset_index()
            )

            if len(latest) == 0:
                st.warning("No player data found for this matchup.")
            else:
                ranked = player_model.rank_players(latest, top_n=20)

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="value">{len(latest)}</div>
                        <div class="label">Players Analysed</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    top_pts = ranked["predicted_fantasy_pts"].iloc[0]
                    st.markdown(f"""<div class="metric-card">
                        <div class="value">{top_pts:.1f}</div>
                        <div class="label">Top Predicted Pts</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    avg_pts = ranked["predicted_fantasy_pts"].mean()
                    st.markdown(f"""<div class="metric-card">
                        <div class="value">{avg_pts:.1f}</div>
                        <div class="label">Avg Top-20 Pts</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""<div class="metric-card">
                        <div class="value">{venue[:20]}</div>
                        <div class="label">Venue</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("### 🏆 Top Fantasy Picks")

                # Player cards
                for i, row in ranked.iterrows():
                    pts = row["predicted_fantasy_pts"]
                    tier = "high" if pts >= avg_pts * 1.2 else ("medium" if pts >= avg_pts * 0.8 else "low")
                    emoji = "🟢" if tier == "high" else ("🟡" if tier == "medium" else "🔴")

                    st.markdown(f"""
                    <div class="player-card {tier}">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <span class="name">{emoji} #{i+1} {row['player']}</span>
                                <span style="color:#64748b; margin-left:12px; font-size:0.85rem;">
                                    {row.get('batting_team', '')}
                                </span>
                            </div>
                            <div class="pts">{pts:.1f} pts</div>
                        </div>
                        <div style="color:#94a3b8; font-size:0.8rem; margin-top:6px;">
                            Bat Avg: {row.get('batting_avg', 0):.1f} · 
                            SR: {row.get('strike_rate', 0):.1f} · 
                            Wickets: {int(row.get('total_wickets', 0))} · 
                            Form₅: {row.get('recent_form_5', 0):.1f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Plotly chart
                st.markdown("### 📊 Fantasy Points Comparison")
                fig = px.bar(
                    ranked.head(15),
                    x="predicted_fantasy_pts",
                    y="player",
                    orientation="h",
                    color="predicted_fantasy_pts",
                    color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                    labels={"predicted_fantasy_pts": "Predicted Fantasy Pts", "player": ""},
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#e2e8f0"),
                    yaxis=dict(autorange="reversed"),
                    showlegend=False,
                    height=500,
                    margin=dict(l=0, r=20, t=10, b=10),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE 2: TEAM SCORE PREDICTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📊 Team Score Predictor":
    st.markdown("""
    <div class="hero-header">
        <h1>📊 Team Score Predictor</h1>
        <p>Predict which team will dominate and compare expected innings scores</p>
    </div>
    """, unsafe_allow_html=True)

    all_teams = sorted(team_df["team"].dropna().unique().tolist())
    venues = sorted(team_df["venue"].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        t1 = st.selectbox("🏠 Team 1", all_teams, index=0, key="ts_t1")
    with col2:
        t2_opts = [t for t in all_teams if t != t1]
        t2 = st.selectbox("✈️ Team 2", t2_opts, index=0, key="ts_t2")

    col3, col4 = st.columns(2)
    with col3:
        venue = st.selectbox("🏟️ Venue", venues, index=0, key="ts_venue")
    with col4:
        toss_decision = st.selectbox("🪙 Toss Winner Chooses", ["bat", "field"], key="ts_toss")

    if st.button("🔮 Predict Scores", type="primary", use_container_width=True):
        with st.spinner("Predicting …"):
            # Get latest team features
            t1_data = team_df[team_df["team"] == t1].sort_values("match_id", ascending=False).head(1)
            t2_data = team_df[team_df["team"] == t2].sort_values("match_id", ascending=False).head(1)

            if len(t1_data) == 0 or len(t2_data) == 0:
                st.warning("Insufficient data for one or both teams.")
            else:
                t1_pred = team_model.predict(t1_data)[0]
                t2_pred = team_model.predict(t2_data)[0]
                total = t1_pred + t2_pred
                t1_pct = (t1_pred / total * 100) if total > 0 else 50
                t2_pct = 100 - t1_pct

                # Score cards
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="value" style="background: linear-gradient(90deg, #3b82f6, #60a5fa); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {t1_pred:.0f}
                        </div>
                        <div class="label">{t1} — Predicted Score</div>
                    </div>""", unsafe_allow_html=True)
                with sc2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="value" style="background: linear-gradient(90deg, #f59e0b, #fbbf24); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {t2_pred:.0f}
                        </div>
                        <div class="label">{t2} — Predicted Score</div>
                    </div>""", unsafe_allow_html=True)

                # Win probability bar
                st.markdown("### 🎯 Win Probability")
                st.markdown(f"""
                <div class="win-bar">
                    <div class="team1" style="width:{t1_pct:.1f}%;">{t1[:3].upper()} {t1_pct:.1f}%</div>
                    <div class="team2" style="width:{t2_pct:.1f}%;">{t2[:3].upper()} {t2_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Donut chart
                fig = go.Figure(data=[go.Pie(
                    labels=[t1, t2],
                    values=[t1_pct, t2_pct],
                    hole=0.55,
                    marker=dict(colors=["#3b82f6", "#f59e0b"]),
                    textinfo="label+percent",
                    textfont=dict(size=14, family="Inter"),
                )])
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#e2e8f0"),
                    showlegend=False,
                    height=350,
                    margin=dict(l=0, r=0, t=0, b=0),
                    annotations=[dict(
                        text=f"<b>{'⚡ ' + (t1 if t1_pct > t2_pct else t2)}</b>",
                        x=0.5, y=0.5, font_size=16,
                        font_color="#e2e8f0", showarrow=False,
                    )],
                )
                st.plotly_chart(fig, use_container_width=True)

                # Head-to-head history
                st.markdown("### 🤝 Head-to-Head History")
                h2h = merged[
                    ((merged["team1"] == t1) & (merged["team2"] == t2)) |
                    ((merged["team1"] == t2) & (merged["team2"] == t1))
                ]
                if "winner" in h2h.columns:
                    h2h_matches = h2h.drop_duplicates(subset=["match_id"])
                    wins = h2h_matches["winner"].value_counts()
                    h2h_df = pd.DataFrame({
                        "Team": wins.index, "Wins": wins.values
                    })
                    fig2 = px.bar(
                        h2h_df, x="Team", y="Wins",
                        color="Team",
                        color_discrete_sequence=["#3b82f6", "#f59e0b"],
                    )
                    fig2.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="#e2e8f0"),
                        showlegend=False, height=300,
                    )
                    st.plotly_chart(fig2, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE 3: HISTORICAL STATS EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📈 Historical Stats":
    st.markdown("""
    <div class="hero-header">
        <h1>📈 Historical Stats Explorer</h1>
        <p>Dive deep into IPL history — player careers, team trends, and venue analysis</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏏 Player Stats", "🏆 Team Trends", "🏟️ Venue Analysis"])

    # --- Player Stats ---
    with tab1:
        all_players = sorted(player_df["player"].dropna().unique().tolist())
        selected = st.selectbox("Search Player", all_players, index=0, key="hist_player")

        p_data = player_df[player_df["player"] == selected].sort_values("match_id")

        if len(p_data) == 0:
            st.warning("No data for this player.")
        else:
            latest = p_data.iloc[-1]
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Matches", int(latest.get("matches_played", 0)))
            with m2:
                st.metric("Total Runs", int(latest.get("total_runs", 0)))
            with m3:
                st.metric("Batting Avg", f"{latest.get('batting_avg', 0):.1f}")
            with m4:
                st.metric("Wickets", int(latest.get("total_wickets", 0)))
            with m5:
                st.metric("Avg Fantasy Pts", f"{p_data['fantasy_points'].mean():.1f}")

            # Season-by-season fantasy points
            if "season" in p_data.columns:
                season_data = (
                    p_data.groupby("season")["fantasy_points"]
                    .agg(["mean", "sum", "count"])
                    .reset_index()
                    .rename(columns={"mean": "Avg Pts", "sum": "Total Pts", "count": "Matches"})
                )
                fig = px.line(
                    season_data, x="season", y="Avg Pts",
                    markers=True,
                    title="Average Fantasy Points per Season",
                    color_discrete_sequence=["#e94560"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#e2e8f0"),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Rolling form
            st.markdown("#### 📉 Form Tracker (Rolling 5-match avg)")
            fig2 = px.line(
                p_data.reset_index(), y="recent_form_5",
                title="",
                color_discrete_sequence=["#10b981"],
            )
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#e2e8f0"),
                xaxis_title="Match Sequence",
                yaxis_title="Fantasy Points (5-match avg)",
                height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # --- Team Trends ---
    with tab2:
        all_teams_hist = sorted(team_df["team"].dropna().unique().tolist())
        team_sel = st.selectbox("Select Team", all_teams_hist, key="hist_team")

        t_data = team_df[team_df["team"] == team_sel].sort_values("match_id")

        if len(t_data) and "season" in t_data.columns:
            season_scores = (
                t_data.groupby("season")["total_score"]
                .agg(["mean", "max", "min"])
                .reset_index()
                .rename(columns={"mean": "Average", "max": "Highest", "min": "Lowest"})
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=season_scores["season"], y=season_scores["Average"],
                mode="lines+markers", name="Average",
                line=dict(color="#3b82f6", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=season_scores["season"], y=season_scores["Highest"],
                mode="lines+markers", name="Highest",
                line=dict(color="#10b981", width=2, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=season_scores["season"], y=season_scores["Lowest"],
                mode="lines+markers", name="Lowest",
                line=dict(color="#ef4444", width=2, dash="dash"),
            ))
            fig.update_layout(
                title=f"{team_sel} — Score Trends",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#e2e8f0"),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Powerplay vs Death
            st.markdown("#### ⚡ Powerplay vs Death Overs")
            phase = t_data.groupby("season")[["powerplay_score", "death_score"]].mean().reset_index()
            fig3 = px.bar(
                phase, x="season",
                y=["powerplay_score", "death_score"],
                barmode="group",
                color_discrete_sequence=["#60a5fa", "#f87171"],
                labels={"value": "Avg Runs", "variable": "Phase"},
            )
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#e2e8f0"),
                height=350,
            )
            st.plotly_chart(fig3, use_container_width=True)

    # --- Venue Analysis ---
    with tab3:
        all_venues = sorted(merged["venue"].dropna().unique().tolist())
        venue_sel = st.selectbox("Select Venue", all_venues, key="hist_venue")

        runs_col = "total_runs" if "total_runs" in merged.columns else "batsman_runs"
        v_data = merged[merged["venue"] == venue_sel]

        if len(v_data):
            innings_scores = (
                v_data.groupby(["match_id", "inning"])[runs_col]
                .sum().reset_index(name="score")
            )

            st.metric("Avg Innings Score", f"{innings_scores['score'].mean():.0f}")
            st.metric("Highest Innings Score", f"{innings_scores['score'].max():.0f}")

            fig = px.histogram(
                innings_scores, x="score", nbins=25,
                color_discrete_sequence=["#e94560"],
                title="Score Distribution at Venue",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#e2e8f0"),
                xaxis_title="Innings Score",
                yaxis_title="Frequency",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE 4: AI INSIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🤖 AI Insights":
    st.markdown("""
    <div class="hero-header">
        <h1>🤖 AI Cricket Analyst</h1>
        <p>Powered by Google Gemini — ask anything about IPL strategy, player form, or fantasy tips</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick analysis buttons
    st.markdown("### ⚡ Quick Analysis")
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("🏏 Top Batsmen Analysis", use_container_width=True):
            with st.spinner("Analysing …"):
                resp = ask_cricket_question(
                    "Who are the top 5 IPL batsmen for fantasy cricket in 2026 and why?"
                )
                st.markdown(resp)
    with qc2:
        if st.button("🎯 Best Bowlers Analysis", use_container_width=True):
            with st.spinner("Analysing …"):
                resp = ask_cricket_question(
                    "Who are the top 5 IPL bowlers for fantasy cricket in 2026 and why?"
                )
                st.markdown(resp)
    with qc3:
        if st.button("🌟 Dark Horse Picks", use_container_width=True):
            with st.spinner("Analysing …"):
                resp = ask_cricket_question(
                    "Name 5 underrated IPL players who could be surprise fantasy picks in 2026."
                )
                st.markdown(resp)

    st.markdown("---")

    # Free-form question
    st.markdown("### 💬 Ask the AI Analyst")
    question = st.text_area(
        "Type your cricket question …",
        placeholder="e.g. Should I pick Virat Kohli or Rohit Sharma as captain for a match at Wankhede?",
        height=100,
    )
    if st.button("🚀 Get Insight", type="primary", use_container_width=True):
        if question.strip():
            with st.spinner("Thinking …"):
                resp = ask_cricket_question(question)
                st.markdown(resp)
        else:
            st.warning("Please enter a question.")

    st.markdown("---")

    # Player-specific insight
    st.markdown("### 🔍 Player Deep Dive")
    all_players = sorted(player_df["player"].dropna().unique().tolist())
    chosen = st.selectbox("Select a player for AI analysis", all_players, key="ai_player")
    if st.button("📊 Analyse Player", use_container_width=True):
        with st.spinner(f"Analysing {chosen} …"):
            p_stats = player_df[player_df["player"] == chosen].iloc[-1].to_dict()
            safe_stats = {
                k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                for k, v in p_stats.items()
                if k in ["total_runs", "batting_avg", "strike_rate", "total_wickets",
                         "bowling_avg", "economy_rate", "matches_played",
                         "recent_form_5", "recent_form_10", "fantasy_points"]
            }
            resp = generate_player_insight(chosen, safe_stats)
            st.markdown(resp)
