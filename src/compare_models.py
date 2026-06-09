"""
Compare multiple regression models for next-day PM2.5 forecasting.

Key design decisions:
  - TIME-BASED split (no random split) to avoid data leakage in time-series.
  - Scaling applied ONLY to linear models (RF / GB / XGB don't need it).
  - Metrics: MAE, RMSE, R2 on a held-out TEST set (future years).

Usage:
    python compare_models.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Optional: XGBoost (skipped gracefully if not installed)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  xgboost not installed — skipping it. (pip install xgboost)")


# ==================================================================
# CONFIG
# ==================================================================

DATA_FILE = Path("data/processed/final_air_quality.parquet")  # ← adjust if needed
TARGET    = "target_PM25"
POLLUTANT = "PM2.5"

# Time split boundaries (by year)
TRAIN_END = 2022   # train: years <= 2022
VAL_YEAR  = 2023   # validation: year == 2023
# test: years > 2023

# Models that need feature scaling
LINEAR_MODELS = {"Linear Regression", "Ridge"}


# ==================================================================
# FEATURE SELECTION
# ==================================================================

def get_feature_columns(df: pd.DataFrame, pollutant: str = POLLUTANT) -> list:
    """Pick weather + temporal + lag + rolling features for the pollutant."""
    prefix = pollutant.replace(".", "")  # PM2.5 -> PM25

    weather = ["Temperature", "Humidity", "Precipitation",
               "WindSpeed", "Wind_X", "Wind_Y"]
    temporal = ["month", "day_of_year", "day_of_week", "is_weekend", "season"]
    lag  = [c for c in df.columns if c.startswith(f"{prefix}_lag_")]
    roll = [c for c in df.columns if c.startswith(f"{prefix}_roll_mean_")]

    cols = weather + temporal + lag + roll
    return [c for c in cols if c in df.columns]


# ==================================================================
# DATA PREP
# ==================================================================

def load_and_split(df: pd.DataFrame, features: list, target: str):
    """Time-based train/val/test split. Drops rows with NaN in used cols."""
    needed = features + [target, "year"]
    data = df[needed].dropna().copy()

    train = data[data["year"] <= TRAIN_END]
    val   = data[data["year"] == VAL_YEAR]
    test  = data[data["year"] >  VAL_YEAR]

    def xy(part):
        return part[features], part[target]

    X_train, y_train = xy(train)
    X_val,   y_val   = xy(val)
    X_test,  y_test  = xy(test)

    print(f"  Train: {len(X_train):>6,} rows (≤{TRAIN_END})")
    print(f"  Val:   {len(X_val):>6,} rows ({VAL_YEAR})")
    print(f"  Test:  {len(X_test):>6,} rows (>{VAL_YEAR})")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ==================================================================
# MODELS
# ==================================================================

def build_models() -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0,
        )
    return models


# ==================================================================
# EVALUATION
# ==================================================================

def evaluate(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2":   r2_score(y_true, y_pred),
    }


def run_comparison(df: pd.DataFrame):
    features = get_feature_columns(df)
    print(f"\nUsing {len(features)} features for {POLLUTANT}:")
    print("  " + ", ".join(features))

    print("\nSplitting (time-based)...")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_and_split(
        df, features, TARGET
    )

    # Scale once — reused by linear models. Fit on TRAIN only.
    scaler = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    models  = build_models()
    results = []

    print("\nTraining models...")
    for name, model in models.items():
        if name in LINEAR_MODELS:
            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_te_s)
        else:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)

        scores = evaluate(y_te, pred)
        scores["Model"] = name
        results.append(scores)
        print(f"  ✅ {name:<20} "
              f"MAE={scores['MAE']:.2f}  "
              f"RMSE={scores['RMSE']:.2f}  "
              f"R²={scores['R2']:.3f}")

    res_df = (
        pd.DataFrame(results)
        [["Model", "MAE", "RMSE", "R2"]]
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 50)
    print("FINAL RANKING (best RMSE first)")
    print("=" * 50)
    print(res_df.to_string(index=False))

    best = res_df.iloc[0]["Model"]
    print(f"\n🏆 Best model: {best}")

    return res_df


# ==================================================================
# MAIN
# ==================================================================

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet(DATA_FILE)

    if "year" not in df.columns and "Date" in df.columns:
        df["year"] = pd.to_datetime(df["Date"]).dt.year

    results = run_comparison(df)

    # Save the comparison table
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    results.to_csv(out / "model_comparison.csv", index=False)
    print(f"\nSaved → {out / 'model_comparison.csv'}")