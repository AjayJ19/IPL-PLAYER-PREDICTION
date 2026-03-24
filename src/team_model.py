"""
IPL Fantasy Prediction — Team Score Model

Predicts the total innings score a team will put up in a given match context.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from src.config import TEAM_MODEL_PATH, TEAM_FEATURES, MODELS_DIR


class TeamScoreModel:
    """Gradient Boosting model for team innings-score prediction."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.85,
            min_samples_split=8,
            random_state=42,
        )
        self.feature_cols: list[str] = []
        self.is_trained = False

    # ──────────────── Training ────────────────
    def train(self, df: pd.DataFrame, target: str = "total_score") -> dict:
        """
        Train on historical team features.

        Returns dict with training metrics.
        """
        self.feature_cols = [c for c in TEAM_FEATURES if c in df.columns]
        if not self.feature_cols:
            raise ValueError("No matching feature columns found in DataFrame.")

        X = df[self.feature_cols].fillna(0).values
        y = df[target].fillna(0).values

        self.model.fit(X, y)
        self.is_trained = True

        cv_scores = cross_val_score(self.model, X, y, cv=5,
                                    scoring="neg_mean_absolute_error")

        metrics = {
            "n_samples": len(X),
            "n_features": len(self.feature_cols),
            "cv_mae": -cv_scores.mean(),
            "cv_mae_std": cv_scores.std(),
            "feature_importances": dict(
                zip(self.feature_cols,
                    self.model.feature_importances_.tolist())
            ),
        }

        print(f"✅ Team model trained — CV MAE: {metrics['cv_mae']:.2f} "
              f"(±{metrics['cv_mae_std']:.2f})")
        return metrics

    # ──────────────── Prediction ────────────────
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict innings total for one or more team–match records."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        X = features[self.feature_cols].fillna(0).values
        return self.model.predict(X)

    def compare_teams(
        self, team1_features: pd.Series, team2_features: pd.Series
    ) -> dict:
        """
        Compare predicted scores for two teams.

        Returns dict with predicted scores and win probabilities.
        """
        t1 = self.predict(pd.DataFrame([team1_features]))[0]
        t2 = self.predict(pd.DataFrame([team2_features]))[0]

        total = t1 + t2
        return {
            "team1_predicted_score": round(t1, 1),
            "team2_predicted_score": round(t2, 1),
            "team1_win_prob": round(t1 / total * 100, 1) if total > 0 else 50.0,
            "team2_win_prob": round(t2 / total * 100, 1) if total > 0 else 50.0,
        }

    # ──────────────── Persistence ────────────────
    def save(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_cols": self.feature_cols},
            TEAM_MODEL_PATH,
        )
        print(f"💾 Team model saved → {TEAM_MODEL_PATH}")

    def load(self) -> None:
        data = joblib.load(TEAM_MODEL_PATH)
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self.is_trained = True
        print(f"📂 Team model loaded ← {TEAM_MODEL_PATH}")
