"""
weather_panel.py — shared weather visualization + interpretation helpers.

Used by page 1 (Air Quality Monitoring) and page 2 (Temporal Trends) to show
weather conditions alongside pollutants, with a plain-language interpretation of
how the weather affects pollutant dispersion.

Physical basis (confirmed by the project's SHAP analysis): higher wind speed and
precipitation lower pollutant levels (dispersion + wet deposition); calm, dry
conditions allow accumulation.

Import pattern (pages already insert dashboard/ on sys.path):
    from weather_panel import weather_snapshot, weather_trend
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

WEATHER_COLS = ["Temperature", "Humidity", "Precipitation", "WindSpeed", "WindDirection"]

# How each weather variable relates to pollution (for captions)
WEATHER_META = {
    "Temperature":   ("🌡️", "°C",   "Affects photochemistry; extremes can trap or lift pollutants."),
    "Humidity":      ("💧", "%",    "High humidity can aid secondary particle formation."),
    "Precipitation": ("🌧️", "mm",   "Rain washes out particles (wet deposition) — lowers PM."),
    "WindSpeed":     ("💨", "km/h", "Stronger wind disperses pollutants — lowers concentrations."),
}


def _interpret(wind: float | None, precip: float | None) -> tuple[str, str]:
    """Return (verdict_text, color) describing dispersion conditions."""
    favorable = []
    if wind is not None:
        if wind >= 20:
            favorable.append(True)
        elif wind <= 5:
            favorable.append(False)
    if precip is not None and precip > 0:
        favorable.append(True)

    if favorable and all(favorable):
        return "Favorable — wind/rain disperse pollutants, so levels tend to be lower.", "#2ecc71"
    if favorable and not any(favorable):
        return "Unfavorable — calm, dry air lets pollutants accumulate.", "#e74c3c"
    return "Mixed conditions — moderate effect on pollutant dispersion.", "#f39c12"


def weather_snapshot(df: pd.DataFrame, title: str = "Current Weather") -> None:
    """Page 1: metric row of the latest weather + a dispersion verdict.
    `df` should already be filtered to the stations/scope of interest."""
    if df.empty or "Date" not in df.columns:
        return
    latest_date = df["Date"].max()
    today = df[df["Date"] == latest_date]

    st.subheader(f"🌦️ {title}")
    cols = st.columns(4)
    vals = {}
    for col, key in zip(cols, ["Temperature", "Humidity", "Precipitation", "WindSpeed"]):
        icon, unit, _ = WEATHER_META[key]
        v = today[key].mean() if key in today.columns else None
        vals[key] = None if (v is None or pd.isna(v)) else round(float(v), 1)
        col.metric(f"{icon} {key}", f"{vals[key]} {unit}" if vals[key] is not None else "—")

    verdict, color = _interpret(vals.get("WindSpeed"), vals.get("Precipitation"))
    st.markdown(
        f"<div style='padding:.6rem 1rem;border-left:4px solid {color};"
        f"background:rgba(0,0,0,.02);border-radius:4px;'>"
        f"<b>Air-quality outlook:</b> {verdict}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Based on the latest reading ({latest_date.date()}), averaged across the selected stations.")


def weather_trend(df: pd.DataFrame, pollutant: str, freq: str = "Year") -> None:
    """Page 2: dual-axis trend of the pollutant vs a weather driver over time,
    plus an interpretation of their relationship.
    `freq` is the grouping column already present in df ("Year" or "Month")."""
    if df.empty or pollutant not in df.columns:
        return

    st.subheader(f"🌦️ {pollutant} vs Weather — over time")

    driver = st.selectbox(
        "Weather driver",
        ["WindSpeed", "Precipitation", "Temperature", "Humidity"],
        key=f"weather_driver_{freq}",
    )
    icon, unit, note = WEATHER_META[driver]

    grouped = (
        df.groupby(freq)[[pollutant, driver]]
        .mean()
        .reset_index()
    )

    # Dual-axis: pollutant (left) + weather driver (right)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grouped[freq], y=grouped[pollutant], name=pollutant,
        line=dict(color="#9b59b6", width=2.5), yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=grouped[freq], y=grouped[driver], name=f"{driver} ({unit})",
        line=dict(color="#3498db", width=2, dash="dot"), yaxis="y2",
    ))
    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title=f"{pollutant} (µg/m³)"),
        yaxis2=dict(title=f"{driver} ({unit})", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch")

    # Interpretation via correlation
    corr = grouped[pollutant].corr(grouped[driver])
    if pd.isna(corr):
        rel = "no clear relationship in this view"
    elif corr <= -0.3:
        rel = f"an **inverse** relationship (r = {corr:.2f}) — higher {driver.lower()} goes with lower {pollutant}"
    elif corr >= 0.3:
        rel = f"a **direct** relationship (r = {corr:.2f}) — higher {driver.lower()} goes with higher {pollutant}"
    else:
        rel = f"a weak relationship (r = {corr:.2f}) at this aggregation"

    st.caption(f"{icon} {note}  \nOver this period, {pollutant} and {driver} show {rel}.")