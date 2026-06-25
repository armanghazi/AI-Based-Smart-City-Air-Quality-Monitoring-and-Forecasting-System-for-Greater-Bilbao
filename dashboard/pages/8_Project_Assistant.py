"""
8_Project_Assistant.py  —  GeoAI Smart City Air Quality Dashboard
──────────────────────────────────────────────────────────────────
Merged assistant: combines the rich Project Knowledge + Data Digest
from the original chatbot with the full 9-tool agent (GIS, XGBoost
forecasting, statistics, correlations) from the standalone agent.

Backend : OpenRouter API (meta-llama/llama-3.3-70b-instruct)
API key : st.secrets["OPENROUTER_API_KEY"]  (Streamlit Cloud)
          or OPENROUTER_API_KEY env variable (local dev)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Streamlit Cloud: pages cannot use package-relative imports ──────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import load_data, WHO_ANNUAL, WHO_SO2_DAILY, CORE_POLLUTANTS, get_zone  # noqa: E402

try:
    from config import EU_ANNUAL, ALERT_LIMITS  # noqa: E402
except ImportError:
    EU_ANNUAL, ALERT_LIMITS = {}, {}

try:
    from aqi import overall_aqi  # noqa: E402
except ImportError:
    overall_aqi = None

# ── Paths ────────────────────────────────────────────────────────────────────
# File lives at: <repo_root>/dashboard/pages/8_Project_Assistant.py
# config.py  at: <repo_root>/dashboard/config.py      → parent.parent
# data       at: <repo_root>/data/processed/           → parent.parent.parent / data
# models     at: <repo_root>/models/                   → parent.parent.parent / models
_DASHBOARD  = Path(__file__).parent.parent                # dashboard/
_REPO_ROOT  = _DASHBOARD.parent                           # repo root
_DATA_DIR   = _REPO_ROOT / "data" / "processed"
_MODELS_DIR = _REPO_ROOT / "models"

# ── Constants ────────────────────────────────────────────────────────────────
MODEL       = "meta-llama/llama-3.3-70b-instruct"
MAX_HISTORY = 4          # reduced from 10 to cut token usage ~50%
POLLUTANTS  = ["PM2.5", "PM10", "NO2", "SO2"]
WEATHER_VARS = ["Temperature", "Humidity", "Precipitation", "WindSpeed", "WindDirection"]

# ═══════════════════════════════════════════════════════════════════════════
# PROJECT KNOWLEDGE  (methodology, GIS findings, caveats)
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_TITLE = "GeoAI Smart City Air Quality Dashboard — Greater Bilbao"

PROJECT_KNOWLEDGE = """
PURPOSE: End-to-end GeoAI platform that monitors, analyzes, visualizes, and forecasts
urban air quality across Greater Bilbao (Bizkaia, Basque Country, Spain).

DATA: 7 monitoring stations, 4 pollutants (PM2.5, PM10, NO2, SO2), daily resolution, 2015-2026.
~29,000 daily records. Air quality from Open Data Euskadi; weather from Open-Meteo (ERA5).
Stored as Parquet. D-1 constraint: pipeline rejects the current incomplete day.

ZONES: Mazarredo/Erandio = Urban (traffic, highest NO2); Basauri/Barakaldo = Industrial (high PM);
Santurtzi = Port (marine + traffic, SO2); Algorta = Coastal (best dispersion);
Muskiz = Refinery (Petronor petrochemical profile).

GUIDELINES: PM2.5/PM10/NO2 vs WHO 2021 ANNUAL limits (5 / 15 / 10 ug/m3).
SO2 vs WHO 24-HOUR guideline (40 ug/m3) — episodic industrial/port behavior.

AIR QUALITY INDEX: DUAL index. Primary = EAQI/ICA (European/Spanish). Secondary = US EPA AQI.
Overall AQI = WORST pollutant category (not an average — official EAQI/ICA rule).

FORECASTING (XGBoost, one model per pollutant, 62 features, time-based split):
- NO2 R2=0.560 | PM2.5 R2=0.479 | PM10 R2=0.460 | SO2 R2=0.390 (held-out test 2024-2026)
- SO2 lower R2 is EXPECTED (episodic source), not a bug.
- Row-based split once gave fake R2=0.84 (leakage) — fixed to honest 0.479.
- Today's pollutant value is VALID as feature (available at prediction time).
- SHAP: wind speed/precipitation push DOWN (dispersion). NO2: day-of-week = traffic signal.
- Models FROZEN: live daily data improves predictions without retraining.

GIS SPATIAL ANALYSIS (notebooks 10a-10d):
NOTE: All spatial correlations across n=7 stations are exploratory/indicative only.
Top predictors: road density→NO2 (r=+0.83), dist city centre→NO2 (r=-0.77),
green cover→PM10 (r=-0.66), terrain relief→PM10 (r=-0.63), dist AP-8→PM2.5 (r=-0.54).

Station profiles (use get_station_profile tool for full details):
- BARAKALDO: road density 21,267 m/km2 + 354m from AP-8 → highest PM2.5/NO2
- MAZARREDO: 77% residential yet highest NO2 (25.8 ug/m3): road density 19,060 m/km2
- BASAURI:   industrial 32% within 500m → local emission concentration
- MUSKIZ:    34.6% industrial (Petronor) yet LOWEST PM2.5 (6.5 ug/m3): TRI 343m + NW breeze
- SANTURCE:  784m from Port + elevation 93m + TRI 445m → port drives SO2 episodes
- ALGORTA:   lowest road density (9,933 m/km2) + 2.6km coast → cleanest station
- ERANDIO:   1,264m from AP-8 + 18,631 m/km2 road density → traffic corridor

SVI (Structural Vulnerability Index):
BARAKALDO=100 | MAZARREDO=89.8 | ERANDIO=87.4 | BASAURI=51.7 | ALGORTA=34.7 | SANTURCE=10.5 | MUSKIZ=0

WIND TRANSPORT (ERA5 single grid cell ~31km, regional only — not station-specific):
- NO2 drops 57% calm→strong wind. NW/W (37.2% days, NO2=16.3) vs S/SW (32%, NO2=21.4, +31%)
- MUSKIZ NE wind: SO2=7.32 ug/m3 (1.93x vs S-wind). MAZARREDO SE wind: NO2=35.3 ug/m3 (+70% vs NW)
""".strip()

EXAMPLES = [
    "What is the latest air quality status across all stations?",
    "Why does MUSKIZ have low pollution despite being next to Petronor?",
    "Which station has the worst structural vulnerability index (SVI)?",
    "How does wind direction affect SO2 at MUSKIZ?",
    "Compare all stations by industrial land use",
    "Forecast NO2 at BARAKALDO for Today",
    "What is the yearly NO2 trend at MAZARREDO?",
    "Why is SO2 R² lower than the other models?",
]

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING  (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_all():
    """Load all datasets and models. Returns (df, df_forecast, gis_dfs, models, err)."""
    err = []
    # Main dataset
    try:
        df = load_data()
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception as e:
        df = pd.DataFrame()
        err.append(f"Main data: {e}")

    # Forecasting dataset
    try:
        df_fc = pd.read_parquet(_DATA_DIR / "forecasting_dataset.parquet")
        df_fc["Date"] = pd.to_datetime(df_fc["Date"])
    except Exception as e:
        df_fc = pd.DataFrame()
        err.append(f"Forecast data: {e}")

    # GIS tables
    gis = {}
    for name, fname in [
        ("spatial",   "station_spatial_features_v3.csv"),   # v3 = current canonical file
        ("distance",  "station_distance_features.csv"),
        ("elevation", "station_elevation_features.csv"),
        ("buffer",    "buffer_summary_table.csv"),
    ]:
        try:
            gis[name] = pd.read_csv(_DATA_DIR / fname)
        except Exception as e:
            gis[name] = pd.DataFrame()
            err.append(f"GIS/{name}: {e}")

    # XGBoost models
    mdls = {}
    for pollutant, fname in [
        ("NO2",   "xgb_no2_forecast.joblib"),
        ("PM2.5", "xgb_pm25_forecast.joblib"),
        ("PM10",  "xgb_pm10_forecast.joblib"),
        ("SO2",   "xgb_so2_forecast.joblib"),
    ]:
        try:
            mdls[pollutant] = joblib.load(_MODELS_DIR / fname)
        except Exception as e:
            err.append(f"Model/{pollutant}: {e}")

    return df, df_fc, gis, mdls, err


# ═══════════════════════════════════════════════════════════════════════════
# DATA DIGEST  (shown in "Project facts" expander)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_data_digest() -> tuple[str, str]:
    """Return (digest_text, freshness_label)."""
    try:
        df = load_data()
    except Exception as exc:
        return f"DATA DIGEST UNAVAILABLE ({exc}).", "unknown"

    lines: list[str] = []
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    lines.append(f"Coverage: {df['station'].nunique()} stations, {len(df):,} rows, {dmin} to {dmax}.")
    lines.append(f"Most recent date: {dmax}.")

    nan_bits = [f"{p}={int(df[p].isna().sum())}" for p in POLLUTANTS if p in df.columns]
    lines.append("Missing values (raw, not interpolated): " + ", ".join(nan_bits) + ".")

    lines.append("\nPer-station (latest | period mean), ug/m3:")
    for stn, g in df.sort_values("Date").groupby("station"):
        last = g.iloc[-1]
        town = last.get("Town", stn)
        zone = get_zone(town)
        parts = []
        for p in POLLUTANTS:
            if p in g.columns:
                lv = last[p]
                lv_s = "NA" if pd.isna(lv) else f"{lv:.1f}"
                parts.append(f"{p} {lv_s}|{g[p].mean():.1f}")
        lines.append(f"  {stn} ({town}, {zone}): " + "; ".join(parts))

    lines.append("\nWHO exceedance (period mean vs guideline):")
    for p in CORE_POLLUTANTS:
        if p in df.columns:
            limit = WHO_ANNUAL[p]
            means = df.groupby("station")[p].mean().sort_values(ascending=False)
            top = means.index[0]
            lines.append(
                f"  {p} (WHO annual {limit}): highest mean {top} = {means.iloc[0]:.1f} "
                f"({means.iloc[0]/limit:.1f}x limit)."
            )
    if "SO2" in df.columns:
        pct = (
            df.groupby("station")["SO2"]
            .apply(lambda s: (s > WHO_SO2_DAILY).mean() * 100)
            .sort_values(ascending=False)
        )
        lines.append(
            f"  SO2 (WHO 24h {WHO_SO2_DAILY}): % days over limit, "
            f"highest {pct.index[0]} = {pct.iloc[0]:.1f}%."
        )

    if overall_aqi is not None:
        lines.append("\nAQI (EAQI/ICA, latest reading, overall = WORST pollutant):")
        ranking = []
        for stn, g in df.sort_values("Date").groupby("station"):
            last = g.iloc[-1]
            vals = {p: (None if pd.isna(last[p]) else float(last[p])) for p in POLLUTANTS if p in g.columns}
            res = overall_aqi(vals)
            if res:
                ranking.append((stn, last.get("Town", stn), res))
        ranking.sort(key=lambda r: (r[2]["level"], r[2].get("epa_aqi") or 0))
        for stn, town, res in ranking:
            epa = res.get("epa_aqi")
            epa_s = f", EPA {epa} ({res.get('epa_label')})" if epa is not None else ""
            lines.append(
                f"  {stn} ({town}): EAQI {res['level']}/6 \"{res['label']}\", "
                f"driver {res['driver']}{epa_s}."
            )

    return "\n".join(lines[:15]), str(dmax)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _tools_from_data(df, df_fc, gis, mdls):
    """Return (tool_map, tool_schemas) built from loaded data."""

    ALL_NUMERIC = POLLUTANTS + WEATHER_VARS
    STATIONS    = df["station"].unique().tolist() if not df.empty else []

    # ── Statistics ────────────────────────────────────────────────────────

    def get_station_stats(station: str, variable: str) -> str:
        filtered = df[df["station"].str.contains(station, case=False, na=False)]
        if filtered.empty:
            return f"Station '{station}' not found. Available: {STATIONS}"
        if variable not in df.columns:
            return f"Variable '{variable}' not found. Available: {ALL_NUMERIC}"
        s = filtered[variable].describe()
        return (
            f"Station: {station} | Variable: {variable}\n"
            f"Mean: {s['mean']:.1f} | Max: {s['max']:.1f} | "
            f"Min: {s['min']:.1f} | Std: {s['std']:.1f}"
        )

    def get_worst_days(variable: str, top_n: int = 5) -> str:
        if variable not in df.columns:
            return f"Variable '{variable}' not found. Available: {ALL_NUMERIC}"
        worst = df.nlargest(top_n, variable)[["Date", "station", "Town", variable]].copy()
        worst["Date"] = worst["Date"].dt.strftime("%Y-%m-%d")
        return worst.to_string(index=False)

    def compare_all_stations(variable: str) -> str:
        if variable not in df.columns:
            return f"Variable '{variable}' not found. Available: {ALL_NUMERIC}"
        result = df.groupby("station")[variable].mean().sort_values(ascending=False)
        return result.round(1).to_string()

    def get_yearly_trend(station: str, variable: str) -> str:
        filtered = df[df["station"].str.contains(station, case=False, na=False)]
        if filtered.empty:
            return f"Station '{station}' not found."
        if variable not in df.columns:
            return f"Variable '{variable}' not found."
        trend = filtered.groupby(filtered["Date"].dt.year)[variable].mean()
        return trend.round(1).to_string()

    def get_correlation(pollutant: str, weather_var: str) -> str:
        if pollutant not in POLLUTANTS:
            return f"Pollutant '{pollutant}' not found. Available: {POLLUTANTS}"
        if weather_var not in WEATHER_VARS:
            return f"Weather variable '{weather_var}' not found. Available: {WEATHER_VARS}"
        corr = df[[pollutant, weather_var]].corr().iloc[0, 1]
        direction = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        return f"Correlation {pollutant} vs {weather_var}: {corr:.3f} ({direction}, {strength})"

    # ── Forecasting ───────────────────────────────────────────────────────

    def forecast_pollutant(station: str, pollutant: str, date: str) -> str:
        if pollutant not in mdls:
            return f"Model for '{pollutant}' not found. Available: {list(mdls.keys())}"
        if df_fc.empty:
            return "Forecasting dataset not available."
        filtered = df_fc[df_fc["station"].str.contains(station, case=False, na=False)].copy()
        if filtered.empty:
            return f"Station '{station}' not found."
        target_date = pd.to_datetime(date)
        filtered["date_diff"] = (filtered["Date"] - target_date).abs()
        row = filtered.loc[filtered["date_diff"].idxmin()]
        bundle   = mdls[pollutant]
        features = bundle["features"]
        model    = bundle["model"]
        metrics  = bundle["metrics"]
        X = row[features].values.reshape(1, -1).astype(float)
        pred = model.predict(X)[0]
        return (
            f"Forecast | Station: {station} | Pollutant: {pollutant} | "
            f"Date: {row['Date'].strftime('%Y-%m-%d')}\n"
            f"Predicted: {pred:.1f} µg/m³\n"
            f"Model R²: {metrics['R2']:.3f} | MAE: {metrics['MAE']:.2f}"
        )

    # ── GIS ───────────────────────────────────────────────────────────────

    def get_station_profile(station: str) -> str:
        df_s = gis.get("spatial",   pd.DataFrame())
        df_d = gis.get("distance",  pd.DataFrame())
        df_e = gis.get("elevation", pd.DataFrame())
        df_b = gis.get("buffer",    pd.DataFrame())

        rows = df_s[df_s["station"].str.contains(station, case=False, na=False)] if not df_s.empty else pd.DataFrame()
        if rows.empty:
            return f"Station '{station}' not found. Available: {STATIONS}"

        r_s = rows.iloc[0]
        r_d = df_d[df_d["station"].str.contains(station, case=False, na=False)].iloc[0] if not df_d.empty else None
        r_e = df_e[df_e["station"].str.contains(station, case=False, na=False)].iloc[0] if not df_e.empty else None
        r_b = df_b[df_b["Station"].str.contains(station, case=False, na=False)].iloc[0] if not df_b.empty else None

        out = (
            f"=== GIS Profile: {r_s['station']} ({r_s['town']}) ===\n"
            f"Zone        : {r_s['zone']}\n"
            f"Coordinates : {r_s['lat']:.4f}N, {r_s['lon']:.4f}E\n"
        )
        if r_e is not None:
            out += f"Elevation   : {r_e['elev_point_m']:.1f} m | Slope: {r_e['slope_deg']:.1f}°\n"
        out += (
            f"\n-- Land Use (1km buffer) --\n"
            f"Green       : {r_s['green_pct_1000m']:.1f}%\n"
            f"Industrial  : {r_s['industrial_pct_1000m']:.1f}%\n"
            f"Residential : {r_s['residential_pct_1000m']:.1f}%\n"
            f"Road density: {r_s['road_density_1000m']:.0f} m/km²\n"
        )
        if r_d is not None:
            out += (
                f"\n-- Key Distances --\n"
                f"Port Bilbao : {r_d['dist_port_bilbao_m']/1000:.1f} km\n"
                f"Petronor    : {r_d['dist_petronor_m']/1000:.1f} km\n"
                f"City centre : {r_d['dist_bilbao_centre_m']/1000:.1f} km\n"
                f"Coast       : {r_d['dist_coast_getxo_m']/1000:.1f} km\n"
            )
        if r_b is not None:
            out += (
                f"\n-- Mean Pollution --\n"
                f"PM2.5       : {r_b['Mean PM2.5']:.2f} µg/m³\n"
                f"NO2         : {r_b['Mean NO2']:.2f} µg/m³\n"
            )
        return out

    def get_spatial_comparison(feature: str) -> str:
        for df_try, scol in [
            (gis.get("spatial",   pd.DataFrame()), "station"),
            (gis.get("distance",  pd.DataFrame()), "station"),
            (gis.get("elevation", pd.DataFrame()), "station"),
            (gis.get("buffer",    pd.DataFrame()), "Station"),
        ]:
            if not df_try.empty and feature in df_try.columns:
                res = df_try[[scol, feature]].copy()
                res.columns = ["station", feature]
                return res.sort_values(feature, ascending=False).to_string(index=False)
        return (
            f"Feature '{feature}' not found. Available: elev_point_m, slope_deg, "
            f"green_pct_1000m, industrial_pct_1000m, residential_pct_1000m, "
            f"road_density_1000m, dist_port_bilbao_m, dist_petronor_m, dist_bilbao_centre_m"
        )

    def explain_pollution_by_geography(station: str) -> str:
        df_s = gis.get("spatial",   pd.DataFrame())
        df_d = gis.get("distance",  pd.DataFrame())
        df_e = gis.get("elevation", pd.DataFrame())
        df_b = gis.get("buffer",    pd.DataFrame())

        rows = df_s[df_s["station"].str.contains(station, case=False, na=False)] if not df_s.empty else pd.DataFrame()
        if rows.empty:
            return f"Station '{station}' not found. Available: {STATIONS}"

        r_s = rows.iloc[0]
        r_d = df_d[df_d["station"].str.contains(station, case=False, na=False)].iloc[0] if not df_d.empty else None
        r_e = df_e[df_e["station"].str.contains(station, case=False, na=False)].iloc[0] if not df_e.empty else None
        r_b = df_b[df_b["Station"].str.contains(station, case=False, na=False)].iloc[0] if not df_b.empty else None

        facts = [
            f"Station: {r_s['station']} | Zone: {r_s['zone']} | Town: {r_s['town']}",
            f"Industrial land use (1km): {r_s['industrial_pct_1000m']:.1f}%",
            f"Green cover (1km): {r_s['green_pct_1000m']:.1f}%",
            f"Road density (1km): {r_s['road_density_1000m']:.0f} m/km²",
        ]
        if r_d is not None:
            facts += [
                f"Distance to Petronor refinery: {r_d['dist_petronor_m']/1000:.1f} km",
                f"Distance to Port of Bilbao: {r_d['dist_port_bilbao_m']/1000:.1f} km",
                f"Distance to city centre: {r_d['dist_bilbao_centre_m']/1000:.1f} km",
            ]
        if r_e is not None:
            facts.append(f"Elevation: {r_e['elev_point_m']:.1f} m | Slope: {r_e['slope_deg']:.1f}°")
        if r_b is not None:
            facts.append(
                f"Mean PM2.5: {r_b['Mean PM2.5']:.2f} µg/m³ | "
                f"Mean NO2: {r_b['Mean NO2']:.2f} µg/m³"
            )
        return "\n".join(facts)

    # ── Tool map ──────────────────────────────────────────────────────────

    tool_map = {
        "get_station_stats":            get_station_stats,
        "get_worst_days":               get_worst_days,
        "compare_all_stations":         compare_all_stations,
        "get_yearly_trend":             get_yearly_trend,
        "get_correlation":              get_correlation,
        "forecast_pollutant":           forecast_pollutant,
        "get_station_profile":          get_station_profile,
        "get_spatial_comparison":       get_spatial_comparison,
        "explain_pollution_by_geography": explain_pollution_by_geography,
    }

    # ── Tool schemas (sent to Groq) ───────────────────────────────────────

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "get_station_stats",
                "description": "Get statistics (mean, max, min, std) for a pollutant or weather variable at a specific station",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "station": {"type": "string"},
                        "variable": {"type": "string", "description": "NO2, PM10, PM2.5, SO2, Temperature, Humidity, Precipitation, WindSpeed, WindDirection"},
                    },
                    "required": ["station", "variable"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_worst_days",
                "description": "Get the days with the highest pollution or weather values",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string"},
                        "top_n": {"type": "integer", "description": "Number of days (default 5)"},
                    },
                    "required": ["variable"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_all_stations",
                "description": "Compare all stations by average value of a pollutant or weather variable",
                "parameters": {
                    "type": "object",
                    "properties": {"variable": {"type": "string"}},
                    "required": ["variable"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_yearly_trend",
                "description": "Get yearly average trend of a variable for a station",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "station": {"type": "string"},
                        "variable": {"type": "string"},
                    },
                    "required": ["station", "variable"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_correlation",
                "description": "Get Pearson correlation between a pollutant and a weather variable",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pollutant": {"type": "string"},
                        "weather_var": {"type": "string"},
                    },
                    "required": ["pollutant", "weather_var"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forecast_pollutant",
                "description": "Predict a pollutant level for a station on a specific date using XGBoost ML model",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "station": {"type": "string"},
                        "pollutant": {"type": "string", "description": "NO2, PM2.5, PM10, SO2"},
                        "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    },
                    "required": ["station", "pollutant", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_station_profile",
                "description": "Get full GIS profile of a station: zone, elevation, land use percentages, distances to port/refinery/city",
                "parameters": {
                    "type": "object",
                    "properties": {"station": {"type": "string"}},
                    "required": ["station"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_spatial_comparison",
                "description": "Compare all stations by a GIS feature. Use exact column names: elev_point_m, slope_deg, green_pct_1000m, industrial_pct_1000m, residential_pct_1000m, road_density_1000m, dist_port_bilbao_m, dist_petronor_m, dist_bilbao_centre_m",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string", "description": "Exact GIS column name e.g. green_pct_1000m, industrial_pct_1000m, dist_petronor_m"},
                    },
                    "required": ["feature"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "explain_pollution_by_geography",
                "description": "Explain why a station has its pollution levels based on its geographic context (land use, distances, elevation)",
                "parameters": {
                    "type": "object",
                    "properties": {"station": {"type": "string"}},
                    "required": ["station"],
                },
            },
        },
    ]

    return tool_map, schemas


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

def build_system_prompt(digest: str, stations: list[str]) -> str:
    return (
        f'You are the Project Assistant for "{PROJECT_TITLE}".\n'
        "You help visitors understand both the air-quality DATA and the project's METHODOLOGY.\n\n"
        "GROUND RULES:\n"
        "- Always use tools to answer data questions — never invent numbers.\n"
        "- For GIS features use exact column names: green_pct_1000m, industrial_pct_1000m,\n"
        "  elev_point_m, dist_petronor_m, dist_port_bilbao_m, road_density_1000m.\n"
        f"- Station codes: {stations}\n"
        "- Respond in the same language the user writes in (English, Spanish, or Persian/Farsi).\n"
        "- Keep technical terms unchanged: WHO, R², XGBoost, SHAP, EAQI, ICA, AQI, SVI.\n"
        "- Be concise; concentrations in µg/m³.\n"
        "- For methodology questions, answer from PROJECT KNOWLEDGE below.\n\n"
        f"=== PROJECT KNOWLEDGE ===\n{PROJECT_KNOWLEDGE}\n\n"
        f"=== LIVE DATA DIGEST ===\n{digest}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# GROQ CALL
# ═══════════════════════════════════════════════════════════════════════════

def get_reply(history: list[dict], digest: str, tool_map: dict, schemas: list, stations: list) -> tuple[bool, str]:
    api_key = (
        st.secrets.get("OPENROUTER_API_KEY", None)
        or os.environ.get("OPENROUTER_API_KEY", None)
    )
    if not api_key:
        return False, (
            "The assistant is offline — no `OPENROUTER_API_KEY` configured. "
            "Add it under Streamlit Cloud → Settings → Secrets."
        )

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        messages = [{"role": "system", "content": build_system_prompt(digest, stations)}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history[-MAX_HISTORY:]]

        # Up to 3 tool round-trips
        for _ in range(3):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=600,
                    tools=schemas,
                    tool_choice="auto",
                )
            except Exception:
                # Retry without tools (handles Groq 400 on bad tool-call generation)
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=600,
                )
                return True, (resp.choices[0].message.content or "").strip()

            msg = resp.choices[0].message
            if not getattr(msg, "tool_calls", None):
                return True, (msg.content or "").strip()

            # Record assistant tool-call turn
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute tools
            for tc in msg.tool_calls:
                fn = tool_map.get(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = fn(**args) if fn else f"Tool '{tc.function.name}' not found."
                except Exception as e:
                    result = f"Tool error: {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result)[:3000],
                })

        # Safety net
        resp = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=600)
        return True, (resp.choices[0].message.content or "").strip()

    except Exception as exc:
        return False, (
            f"The assistant could not answer right now (possibly a rate limit). "
            f"Details: `{exc}`"
        )


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Project Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Project Assistant")
st.caption(
    "Ask about the air quality data, forecasts, GIS analysis, or the project methodology. "
    "Data questions are answered from the live dataset — the assistant never guesses numbers."
)

# Load data (once, cached)
with st.spinner("Loading data and models…"):
    df, df_fc, gis, mdls, load_errors = _load_all()

if load_errors:
    with st.expander("⚠️ Some resources failed to load", expanded=False):
        for e in load_errors:
            st.warning(e)

# Build tools from loaded data
stations = df["station"].unique().tolist() if not df.empty else []
tool_map, schemas = _tools_from_data(df, df_fc, gis, mdls)

# Data digest
digest_text, freshness = build_data_digest()
st.caption(f"Data through **{freshness}** · {MODEL} via OpenRouter")

# ── Chat input ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stChatInput"] textarea {
    border: 2px solid #2563eb !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    background: #f0f4ff !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #1d4ed8 !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.15) !important;
    background: #fff !important;
}
</style>
""", unsafe_allow_html=True)

if "assistant_msgs" not in st.session_state:
    st.session_state.assistant_msgs = []

typed  = st.chat_input("💬 Ask about air quality, forecasts, GIS, or methodology…")
prompt = typed or st.session_state.pop("queued_prompt", None)

# Quick-start buttons
st.markdown("##### ⚡ Quick start")
cols = st.columns(2)
for i, q in enumerate(EXAMPLES):
    if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
        st.session_state.queued_prompt = q
        st.rerun()

st.divider()

# Render history
for m in st.session_state.assistant_msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle new prompt
if prompt:
    st.session_state.assistant_msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ok, reply = get_reply(
                st.session_state.assistant_msgs,
                digest_text,
                tool_map,
                schemas,
                stations,
            )
        st.markdown(reply)
    st.session_state.assistant_msgs.append({"role": "assistant", "content": reply})

# Clear button
if st.session_state.assistant_msgs:
    if st.button("🗑️ Clear conversation"):
        st.session_state.assistant_msgs = []
        st.rer