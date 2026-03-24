"""
IPL Fantasy Prediction — Player Performance Model

Predicts fantasy points a player will score in a given match context.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from src.config import PLAYER_MODEL_PATH, PLAYER_FEATURES, MODELS_DIR


class PlayerPerformanceModel:
    """Gradient Boosting model for player fantasy-point prediction."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_split=10,
            random_state=42,
        )
        self.feature_cols: list[str] = []
        self.is_trained = False

    # ──────────────── Training ────────────────
    def train(self, df: pd.DataFrame, target: str = "fantasy_points") -> dict:
        """
        Train on historical player features.

        Parameters
        ----------
        df : DataFrame with player features and target column.
        target : column name for the prediction target.

        Returns
        -------
        dict with training metrics.
        """
        # Select features that actually exist in the DataFrame
        self.feature_cols = [c for c in PLAYER_FEATURES if c in df.columns]
        if not self.feature_cols:
            raise ValueError("No matching feature columns found in DataFrame.")

        X = df[self.feature_cols].fillna(0).values
        y = df[target].fillna(0).values

        # Fit
        self.model.fit(X, y)
        self.is_trained = True

        # Cross-val score for diagnostics
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

        print(f"✅ Player model trained — CV MAE: {metrics['cv_mae']:.2f} "
              f"(±{metrics['cv_mae_std']:.2f})")
        return metrics

    # ──────────────── Prediction ────────────────
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict fantasy points for one or more records."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        X = features[self.feature_cols].fillna(0).values
        return self.model.predict(X)

    def rank_players(self, df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """
        Rank players by predicted fantasy points.

        Parameters
        ----------
        df : player feature DataFrame (one row per player).
        top_n : how many players to return.

        Returns
        -------
        DataFrame sorted by predicted fantasy points descending.
        """
        df = df.copy()
        df["predicted_fantasy_pts"] = self.predict(df)
        return (
            df.sort_values("predicted_fantasy_pts", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ──────────────── Persistence ────────────────
    def save(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_cols": self.feature_cols},
            PLAYER_MODEL_PATH,
        )
        print(f"💾 Player model saved → {PLAYER_MODEL_PATH}")

    def load(self) -> None:
        data = joblib.load(PLAYER_MODEL_PATH)
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self.is_trained = True
        print(f"📂 Player model loaded ← {PLAYER_MODEL_PATH}")
