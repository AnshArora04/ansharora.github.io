"""
ML model for predicting UFC fight winners.
Uses a Random Forest classifier trained on fighter stats.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

MODEL_PATH = Path(__file__).parent.parent / "models" / "fight_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)

FEATURE_COLS = [
    "f1_win_pct",
    "f2_win_pct",
    "f1_wins",
    "f2_wins",
    "f1_losses",
    "f2_losses",
    "f1_total_fights",
    "f2_total_fights",
    "f1_height_in",
    "f2_height_in",
    "f1_reach_in",
    "f2_reach_in",
    "reach_diff",
    "height_diff",
    "win_pct_diff",
    "experience_diff",
]


def build_features(fighter1_stats: dict, fighter2_stats: dict) -> np.ndarray:
    """
    Build feature vector for a matchup.
    fighter1 is the fighter we're predicting a WIN for (label=1).
    """
    def safe_get(d, key, default=0.0):
        val = d.get(key, default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    f1_height = safe_get(fighter1_stats, "height_in", 70)
    f2_height = safe_get(fighter2_stats, "height_in", 70)
    f1_reach = safe_get(fighter1_stats, "reach_in", 70)
    f2_reach = safe_get(fighter2_stats, "reach_in", 70)
    f1_wins = safe_get(fighter1_stats, "wins")
    f2_wins = safe_get(fighter2_stats, "wins")
    f1_losses = safe_get(fighter1_stats, "losses")
    f2_losses = safe_get(fighter2_stats, "losses")
    f1_total = safe_get(fighter1_stats, "total_fights", 1)
    f2_total = safe_get(fighter2_stats, "total_fights", 1)
    f1_win_pct = safe_get(fighter1_stats, "win_pct", 0.5)
    f2_win_pct = safe_get(fighter2_stats, "win_pct", 0.5)

    features = [
        f1_win_pct,
        f2_win_pct,
        f1_wins,
        f2_wins,
        f1_losses,
        f2_losses,
        f1_total,
        f2_total,
        f1_height,
        f2_height,
        f1_reach,
        f2_reach,
        f1_reach - f2_reach,       # reach_diff
        f1_height - f2_height,     # height_diff
        f1_win_pct - f2_win_pct,   # win_pct_diff
        f1_total - f2_total,       # experience_diff
    ]
    return np.array(features, dtype=float)


def build_training_data(fights_df: pd.DataFrame, fighters_df: pd.DataFrame) -> tuple:
    """
    Build X, y from fight history + fighter stats.
    For each fight, create two rows (both orderings) with label 1=fighter1 wins.
    """
    fighter_lookup = fighters_df.set_index("name").to_dict("index")

    X_rows, y_rows = [], []

    for _, fight in fights_df.iterrows():
        f1_name = fight["fighter1"]
        f2_name = fight["fighter2"]
        winner = fight["winner"]

        f1_stats = fighter_lookup.get(f1_name, {})
        f2_stats = fighter_lookup.get(f2_name, {})

        if not f1_stats and not f2_stats:
            continue

        # Row 1: f1 vs f2
        X_rows.append(build_features(f1_stats, f2_stats))
        y_rows.append(1 if winner == f1_name else 0)

        # Row 2: f2 vs f1 (mirror) — increases training data
        X_rows.append(build_features(f2_stats, f1_stats))
        y_rows.append(1 if winner == f2_name else 0)

    X = np.array(X_rows)
    y = np.array(y_rows)
    return X, y


def train(fights_df: pd.DataFrame, fighters_df: pd.DataFrame) -> Pipeline:
    """Train and save the model. Returns the fitted pipeline."""
    print("[model] Building training features...")
    X, y = build_training_data(fights_df, fighters_df)
    print(f"[model] Training on {len(X)} samples ({y.sum()} wins, {len(y)-y.sum()} losses)...")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"[model] Cross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[model] Model saved to {MODEL_PATH}")
    return pipeline


def load_model() -> Pipeline | None:
    """Load the trained model if it exists."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def predict_fight(
    fighter1_stats: dict,
    fighter2_stats: dict,
    model: Pipeline,
) -> dict:
    """
    Predict the outcome of a fight.
    Returns:
        {
            "fighter1_prob": float,   # probability fighter1 wins
            "fighter2_prob": float,
            "predicted_winner": str,
            "confidence": float,
        }
    """
    features = build_features(fighter1_stats, fighter2_stats).reshape(1, -1)
    proba = model.predict_proba(features)[0]

    # proba[1] = P(fighter1 wins)
    f1_prob = float(proba[1])
    f2_prob = float(proba[0])

    return {
        "fighter1_prob": f1_prob,
        "fighter2_prob": f2_prob,
        "predicted_winner": "fighter1" if f1_prob >= 0.5 else "fighter2",
        "confidence": max(f1_prob, f2_prob),
    }
