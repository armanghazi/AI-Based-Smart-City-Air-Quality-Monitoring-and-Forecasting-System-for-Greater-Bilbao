from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st


# --------------------------------------------------
# DATA FILE PATH   
# --------------------------------------------------
_ROOT = Path(__file__).parent

DATA_FILE = _ROOT.parent / "data" / "processed" / "forecasting_dataset.parquet"

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
    "Barakaldo": "Industrial Corridor",
    "Basauri":   "Industrial Corridor",
    "Bilbao": "Urban Core",
    "Erandio":   "Urban Core",
    "Algorta":   "Coastal Buffer Zone",
    "Muskiz":    "Coastal Buffer Zone",
    "Santurtzi": "Coastal Buffer Zone",
}

ZONE_META: dict[str, dict] = {
    "Industrial Corridor": {
        "icon":         "🏭",
        "color":        "#e67e22",
        "border":       "#d35400",
        "description":  "High PM2.5, High PM10, Elevated NO₂",
        "key_pollutant": "PM2.5",
    },
    "Urban Core": {
        "icon":         "🚗",
        "color":        "#8e44ad",
        "border":       "#6c3483",
        "description":  "Highest NO₂, Strong traffic influence, Urban canyon effects",
        "key_pollutant": "NO2",
    },
    "Coastal Buffer Zone": {
        "icon":         "🌊",
        "color":        "#1abc9c",
        "border":       "#148f77",
        "description":  "Better dispersion, Lower NO₂, Marine influence on PM10",
        "key_pollutant": "PM10",
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
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Day"]       = df["Date"].dt.date
    df["YearMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["Zone"]      = df["Town"].apply(get_zone)
    return df