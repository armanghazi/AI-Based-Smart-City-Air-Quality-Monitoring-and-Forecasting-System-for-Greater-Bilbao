from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

# --------------------------------------------------
# DATA FILE PATH — مسیر نسبت به محل config.py
# --------------------------------------------------

# ROOT = پوشه‌ای که config.py در آن قرار دارد (یعنی dashboard/)
_ROOT = Path(__file__).parent

# مسیر واقعی فایل parquet را اینجا تنظیم کن:
DATA_FILE = _ROOT.parent / "data" / "processed" / "air_quality_weather.parquet"

# --------------------------------------------------
# WHO GUIDELINES (2021)
# --------------------------------------------------

WHO_ANNUAL: dict[str, float] = {
    "PM2.5": 5.0,
    "PM10":  15.0,
    "NO2":   10.0,
}
WHO_SO2_DAILY: float = 40.0          # µg/m³ — 24-hour guideline
CORE_POLLUTANTS: list[str] = ["PM2.5", "PM10", "NO2"]

# --------------------------------------------------
# VISUAL CONSTANTS
# --------------------------------------------------

POLLUTANT_COLOR: dict[str, str] = {
    "NO2":   "#e74c3c",
    "PM10":  "#e67e22",
    "PM2.5": "#9b59b6",
    "SO2":   "#3498db",
}

MONTH_NAMES: dict[int, str] = {
    1: "Jan",  2: "Feb",  3: "Mar",  4: "Apr",
    5: "May",  6: "Jun",  7: "Jul",  8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

RISK_COLORS: dict[str, str] = {
    "Below WHO guideline": "#2ecc71",
    "1–2× WHO guideline":  "#f39c12",
    ">2× WHO guideline":   "#e74c3c",
}

RISK_ORDER: list[str] = [
    "Below WHO guideline",
    "1–2× WHO guideline",
    ">2× WHO guideline",
]

# --------------------------------------------------
# ZONE CLASSIFICATION
# --------------------------------------------------

ZONE_MAP: dict[str, str] = {
    "Bilbao":    "Urban",
    "Erandio":   "Urban",
    "Basauri":   "Industrial",
    "Barakaldo": "Industrial",
    "Santurtzi": "Port",
    "Getxo":     "Coastal",
    "Muskiz":    "Refinery",
}

ZONE_META: dict[str, dict] = {
    "Urban": {
        "icon":          "🏙️",
        "color":         "#8e44ad",
        "border":        "#6c3483",
        "description":   "Highest NO₂, Strong traffic influence, Urban canyon effects",
        "key_pollutant": "NO2",
    },
    "Industrial": {
        "icon":          "🏭",
        "color":         "#e67e22",
        "border":        "#d35400",
        "description":   "High PM2.5, High PM10, Elevated NO₂",
        "key_pollutant": "PM2.5",
    },
    "Port": {
        "icon":          "⚓",
        "color":         "#2980b9",
        "border":        "#1a6fa0",
        "description":   "Mixed traffic & marine emissions, Elevated SO₂",
        "key_pollutant": "SO2",
    },
    "Coastal": {
        "icon":          "🌊",
        "color":         "#1abc9c",
        "border":        "#148f77",
        "description":   "Better atmospheric dispersion, Marine influence on PM10",
        "key_pollutant": "PM10",
    },
    "Refinery": {
        "icon":          "🛢️",
        "color":         "#c0392b",
        "border":        "#a93226",
        "description":   "Elevated SO₂ and PM, Petrochemical emission profile",
        "key_pollutant": "SO2",
    },
}

def get_zone(town: str) -> str:
    """Map a town name to its environmental zone."""
    for key, zone in ZONE_MAP.items():
        if key.lower() in str(town).lower():
            return zone
    return "Unknown"

# --------------------------------------------------
# RISK HELPERS
# --------------------------------------------------

def classify_core_risk(score: float) -> str:
    if score < 100:   return "Below WHO guideline"
    elif score < 200: return "1–2× WHO guideline"
    return ">2× WHO guideline"

def risk_color(score: float) -> str:
    if score < 100:   return "#2ecc71"
    elif score < 200: return "#f39c12"
    return "#e74c3c"

def short_term_flag(rate: float) -> str:
    if rate == 0:      return "No exceedance"
    elif rate < 0.05:  return "Occasional"
    return "Frequent"

def who_ratio_label(val: float, pollutant: str) -> str:
    limit = WHO_ANNUAL.get(pollutant)
    if not limit: return "—"
    return f"{val / limit:.1f}×"

def who_delta(val: float, pollutant: str) -> tuple[str | None, str]:
    """Returns (delta_label, delta_color) for st.metric."""
    limit = WHO_ANNUAL.get(pollutant)
    if not limit:
        return None, "off"
    ratio = val / limit
    return f"{ratio:.1f}× WHO limit", "inverse" if ratio > 1 else "normal"

# --------------------------------------------------
# SHARED DATA LOADER
# --------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    df["Date"]      = pd.to_datetime(df["Date"])
    df["Zone"]      = df["Town"].apply(get_zone)

    # Always derive time columns fresh — avoids lowercase/uppercase mismatch on Cloud
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Day"]       = df["Date"].dt.date
    df["YearMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    # Normalise season column — handle any casing from parquet
    if "Season" in df.columns:
        df["season"] = df["Season"].str.capitalize()
    elif "season" in df.columns:
        df["season"] = df["season"].str.capitalize()
    else:
        month_to_season = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring",  4: "Spring", 5: "Spring",
            6: "Summer",  7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn",
        }
        df["season"] = df["Month"].map(month_to_season)

    # Normalise wind component columns — parquet uses lowercase wind_u / wind_v
    for old, new in [("wind_u", "Wind_X"), ("wind_v", "Wind_Y")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # --------------------------------------------------
    # Feature Engineering — lag, rolling, target
    # computed at runtime, no need to store in parquet
    # --------------------------------------------------
    df = df.sort_values(["station", "Date"]).copy()

    # Target: next-day PM2.5 per station
    for pollutant in ["PM2.5", "PM10", "NO2", "SO2"]:
        prefix = pollutant.replace(".", "")
        df[f"target_{prefix}"] = df.groupby("station")[pollutant].shift(-1)

    # Lag features
    LAG_DAYS = [1, 3, 7, 30, 90, 365]
    for pollutant in ["PM2.5", "PM10", "NO2", "SO2"]:
        prefix = pollutant.replace(".", "")
        for d in LAG_DAYS:
            df[f"{prefix}_lag_{d}"] = df.groupby("station")[pollutant].shift(d)

    # Rolling mean features
    ROLL_WINDOWS = [7, 30, 90, 365]
    for pollutant in ["PM2.5", "PM10", "NO2", "SO2"]:
        prefix = pollutant.replace(".", "")
        for w in ROLL_WINDOWS:
            df[f"{prefix}_roll_mean_{w}"] = (
                df.groupby("station")[pollutant]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

    return df