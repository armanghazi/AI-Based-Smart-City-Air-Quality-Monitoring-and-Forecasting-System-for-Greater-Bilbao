import pandas as pd
import numpy as np


LAG_DAYS     = [1, 3, 7, 30, 90, 365]
ROLL_WINDOWS = [7, 14, 30, 90, 365]
POLLUTANTS   = ["PM2.5", "PM10", "NO2", "SO2"]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]        = df["Date"].dt.year
    df["month"]       = df["Date"].dt.month
    df["day"]         = df["Date"].dt.day
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["week_of_year"]= df["Date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"]  = df["Date"].dt.weekday >= 5
    df["season"]      = df["month"].map(
        {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
    )
    return df


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """WindDirection → Wind_X, Wind_Y"""
    if "WindDirection" not in df.columns:
        return df
    df = df.copy()
    df["Wind_X"] = np.cos(np.radians(df["WindDirection"]))
    df["Wind_Y"] = np.sin(np.radians(df["WindDirection"]))
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for pollutant in POLLUTANTS:
        prefix = pollutant.replace(".", "")
        for d in LAG_DAYS:
            df[f"{prefix}_lag_{d}"] = (
                df.groupby("station")[pollutant].shift(d)
            )
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for pollutant in POLLUTANTS:
        prefix = pollutant.replace(".", "")
        for w in ROLL_WINDOWS:
            df[f"{prefix}_roll_mean_{w}"] = (
                df.groupby("station")[pollutant]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
    return df


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Next-day target per station"""
    df = df.copy()
    df = df.sort_values(["station", "Date"])
    for pollutant in POLLUTANTS:
        prefix = pollutant.replace(".", "")
        df[f"target_{prefix}"] = df.groupby("station")[pollutant].shift(-1)
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("Adding temporal features...")
    df = add_temporal_features(df)
    print("Adding wind components...")
    df = add_wind_components(df)
    print("Adding lag features...")
    df = add_lag_features(df)
    print("Adding rolling features...")
    df = add_rolling_features(df)
    print("Adding target columns...")
    df = add_target_columns(df)
    print("Feature engineering complete.")
    return df