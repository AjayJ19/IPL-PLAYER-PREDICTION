"""
IPL Fantasy Prediction — AI Insights (Google Gemini)

Provides natural-language analysis powered by Google Generative AI.
Gracefully degrades if no API key is configured.
"""
from __future__ import annotations

import time
from src.config import GEMINI_API_KEY

_client = None


def _get_client():
    """Lazy-init the Gemini client using the new google-genai SDK."""
    global _client
    if _client is not None:
        return _client

    if not GEMINI_API_KEY:
        return None

    try:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
        return _client
    except Exception as e:
        print(f"⚠️  Gemini init failed: {e}")
        return None


def _call(prompt: str, max_retries: int = 3) -> str:
    """Send a prompt to Gemini with retry logic for rate limits."""
    client = _get_client()
    if client is None:
        return "🔒 AI insights unavailable — set the GEMINI_API_KEY environment variable."

    # Try models in order of preference (lite has higher free-tier quota)
    models_to_try = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]

    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "rate" in error_str or "429" in error_str or "resource" in error_str:
                    wait_time = (attempt + 1) * 10
                    print(f"⏳ Rate limited on {model_name}, retrying in {wait_time}s …")
                    time.sleep(wait_time)
                elif "not found" in error_str or "invalid" in error_str:
                    print(f"⚠️  Model {model_name} not available, trying next …")
                    break  # try next model
                else:
                    return f"⚠️  AI error: {e}"

    return "⏳ API rate limit reached on all models. Please wait a minute and try again."


# ──────────────── Public API ────────────────

def generate_player_insight(player_name: str, stats: dict) -> str:
    """Generate a natural-language analysis for a player."""
    prompt = f"""You are an expert IPL cricket analyst. Analyse the following player 
based on their career stats and provide:
1. A brief overview of their playing style
2. Current form assessment
3. Key strengths & weaknesses
4. Fantasy cricket recommendation (buy / hold / avoid)

Player: {player_name}
Stats: {stats}

Keep the response concise (150-200 words), insightful, and fan-friendly."""
    return _call(prompt)


def generate_match_preview(
    team1: str, team2: str, venue: str, predictions: dict
) -> str:
    """Generate a match preview with predicted scores."""
    prompt = f"""You are an expert IPL cricket analyst. Create a short, exciting 
match preview for fantasy cricket players.

Match: {team1} vs {team2}
Venue: {venue}

Predicted Scores:
- {team1}: {predictions.get('team1_predicted_score', 'N/A')}
- {team2}: {predictions.get('team2_predicted_score', 'N/A')}
- Win Probability: {team1} {predictions.get('team1_win_prob', 50)}% | {team2} {predictions.get('team2_win_prob', 50)}%

Include:
1. Key matchups to watch
2. Venue insights
3. Top fantasy picks from both teams
4. Bold prediction

Keep it engaging and under 200 words."""
    return _call(prompt)


def ask_cricket_question(question: str) -> str:
    """Answer any IPL / cricket question."""
    prompt = f"""You are an expert IPL cricket analyst and fantasy sports advisor.
Answer the following question with data-driven insights.

Question: {question}

Keep your answer concise (100-200 words), insightful, and backed by cricket knowledge."""
    return _call(prompt)
