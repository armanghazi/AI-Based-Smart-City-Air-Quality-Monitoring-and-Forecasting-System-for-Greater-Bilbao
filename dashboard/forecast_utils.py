"""Shared feature engineering used by forecast pages and the test suite.

Extracted from 5_Forecasting.py and 6_smart_city_decision_support.py which
held identical copies. Import from here; do not duplicate.
"""
import streamlit as st
import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]

PLOTLY_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": False,
    "doubleClick": False,
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "zoomIn2d", "zoomOut2d"],
}


def plotly_touch_config():
    """Apply touch-friendly CSS for all plotly charts."""
    st.markdown("""
        <style>
        .js-plotly-plot .plotly {
            touch-action: pan-x pan-y;
        }
        </style>
    """, unsafe_allow_html=True)


def prepare_features(
    df: pd.DataFrame,
    required_features: list,
    station_codes: dict | None = None,
) -> pd.DataFrame:
    """Build every feature the models need, matching training-time names and order.

    config.load_data() does not produce all required features; this fills the gaps.

    station_codes: global {station -> code} mapping built over the full dataset.
    Using a per-station-subset .cat.codes instead collapses all codes to 0,
    silently mismatching the codes the model saw during training.
    """
    df = df.copy()

    # config.load_data() renames wind_u/wind_v to Wind_X/Wind_Y but keeps originals.
    if "wind_u" not in df.columns and "Wind_X" in df.columns:
        df["wind_u"] = df["Wind_X"]
    if "wind_v" not in df.columns and "Wind_Y" in df.columns:
        df["wind_v"] = df["Wind_Y"]

    df["year"]         = df["Date"].dt.year
    df["month"]        = df["Date"].dt.month
    df["day"]          = df["Date"].dt.day
    df["day_of_year"]  = df["Date"].dt.dayofyear
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]  = df["Date"].dt.dayofweek
    df["is_weekend"]   = (df["Date"].dt.weekday >= 5).astype(int)

    # Season -> int; handles any incoming dtype (object, string[pyarrow], numeric).
    season_to_int = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    season_str = df["season"].astype(str).str.capitalize()
    mapped = season_str.map(season_to_int)
    month_season = df["Date"].dt.month.map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    df["season"] = mapped.fillna(month_season).fillna(0).astype(int)

    # station_code must use the global training mapping.
    if station_codes is not None:
        df["station_code"] = df["station"].map(station_codes).fillna(-1).astype(int)
    else:
        df["station_code"] = df["station"].astype("category").cat.codes

    # roll_mean_14 is not built by load_data (which uses windows 7/30/90/365).
    for pollutant in POLLUTANTS:
        prefix = pollutant.replace(".", "")
        col = f"{prefix}_roll_mean_14"
        if col not in df.columns:
            df[col] = (
                df.groupby("station")[pollutant]
                .transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
            )

    # Interaction features.
    # NOTE: wind_x_precip and temp_x_humid are in NONE of the current models
    # feature lists (feature-engineering / model drift). Built here for compatibility;
    # tracked as ALLOWED_UNUSED in tests/test_model_contracts.py.
    df["wind_x_precip"] = df["WindSpeed"] * df["Precipitation"]
    df["temp_x_humid"]  = df["Temperature"] * df["Humidity"]

    # Ensure all required features exist and are plain numeric.
    # Python 3.14: pyarrow string dtypes must be coerced before XGBoost predict.
    for feat in required_features:
        if feat not in df.columns:
            df[feat] = 0.0
        col = df[feat]
        if col.dtype == bool:
            df[feat] = col.astype(int)
        elif not pd.api.types.is_numeric_dtype(col):
            df[feat] = pd.to_numeric(col, errors="coerce").fillna(0)

    return df
