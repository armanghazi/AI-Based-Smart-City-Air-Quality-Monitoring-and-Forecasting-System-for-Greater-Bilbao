"""
weather_panel.py — shared weather visualization + interpretation helpers.

Used by page 1 (Air Quality Monitoring) and page 2 (Temporal Trends) to show
weather conditions alongside pollutants, with a plain-language interpretation of
how the weather affects pollutant dispersion.

Physical basis (confirmed by SHAP analysis): higher wind speed and precipitation
lower pollutant levels (dispersion + wet deposition); calm, dry air allows accumulation.

IMPORTANT: all data is D-1 (yesterday's reading is the latest available, since the
pipeline rejects the current incomplete day). Labels must never say "Today" or "Current".
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from i18n_auto import tr

WEATHER_COLS = ["Temperature", "Humidity", "Precipitation", "WindSpeed", "WindDirection"]

WEATHER_META = {
    "Temperature":    ("🌡️", "°C",    "Affects photochemistry; high temps can intensify O₃ and secondary PM formation."),
    "Humidity":       ("💧", "%",     "High humidity aids secondary particle formation."),
    "Precipitation":  ("🌧️", "mm",    "Rain washes out particles (wet deposition) — lowers PM levels."),
    "WindSpeed":      ("💨", "km/h",  "Stronger wind disperses pollutants — lowers concentrations."),
    "WindDirection":  ("🧭", "°",     "Determines where emissions travel."),
}

CARDINAL = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _cardinal(deg: float | None) -> str:
    if deg is None or pd.isna(deg):
        return "—"
    return CARDINAL[int((float(deg) % 360) / 45 + 0.5) % 8]


def _interpret(wind: float | None, precip: float | None) -> tuple[str, str]:
    """Return (verdict_text, color)."""
    favorable = []
    factors = []
    if wind is not None:
        if wind >= 20:
            favorable.append(True)
            factors.append(tr(f"strong wind ({wind:.1f} km/h) aids dispersion"))
        elif wind <= 5:
            favorable.append(False)
            factors.append(tr(f"very low wind ({wind:.1f} km/h) — accumulation risk"))
        else:
            factors.append(tr(f"moderate wind ({wind:.1f} km/h)"))
    if precip is not None and precip > 0:
        favorable.append(True)
        factors.append(tr(f"precipitation ({precip:.1f} mm) washes out particles"))
    if not factors:
        factors.append(tr("calm, dry conditions"))

    if favorable and all(favorable):
        verdict = tr("Favorable — pollutants tend to disperse")
    elif favorable and not any(favorable):
        verdict = tr("Unfavorable — pollutants may accumulate")
    else:
        verdict = tr("Mixed conditions")

    return f"{verdict}. {'; '.join(factors).capitalize()}.", \
           "#2ecc71" if (favorable and all(favorable)) else \
           "#e74c3c" if (favorable and not any(favorable)) else "#f39c12"


def weather_snapshot(
    df: pd.DataFrame,
    show_per_station: bool = False,
) -> None:
    """Snapshot of the latest available weather reading (D-1)."""
    if df.empty or "Date" not in df.columns:
        return

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date]

    st.subheader(tr("🌦️ Weather conditions"))
    st.caption(
        tr("Latest available reading:") + f" **{latest_date.strftime('%d %b %Y')}** "
        + tr("(D-1 — pipeline rejects the current incomplete day). Averaged across the selected stations.")
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    def _v(col):
        v = latest[col].mean() if col in latest.columns else None
        return None if (v is None or pd.isna(v)) else round(float(v), 1)

    temp   = _v("Temperature")
    humid  = _v("Humidity")
    precip = _v("Precipitation")
    wind   = _v("WindSpeed")
    wdir   = _v("WindDirection")

    c1.metric(tr("🌡️ Temperature"),   f"{temp} °C"   if temp   is not None else "—")
    c2.metric(tr("💧 Humidity"),      f"{humid} %"   if humid  is not None else "—")
    c3.metric(tr("🌧️ Precipitation"), f"{precip} mm" if precip is not None else "—")
    c4.metric(tr("💨 Wind speed"),    f"{wind} km/h" if wind   is not None else "—")
    c5.metric(tr("🧭 Wind direction"),
            f"{_cardinal(wdir)} ({round(wdir)}°)")

    verdict, color = _interpret(wind, precip)
    st.markdown(
        f"<div style='padding:.6rem 1rem;border-left:4px solid {color};"
        f"background:rgba(0,0,0,.02);border-radius:4px;margin:.4rem 0'>"
        f"<b>{tr('Dispersion outlook')}:</b> {verdict}<br>"
        f"<small style='color:#666'>{tr('SHAP-confirmed: wind speed and precipitation are the strongest meteorological drivers of pollutant levels in this model.')}</small></div>",
        unsafe_allow_html=True,
    )

    if show_per_station and "station" in latest.columns:
        with st.expander(tr("📍 Per-station weather breakdown")):
            rows = []
            for stn, g in latest.groupby("station"):
                def sv(col):
                    v = g[col].mean() if col in g.columns else None
                    return None if (v is None or pd.isna(v)) else round(float(v), 1)
                wd = sv("WindDirection")
                rows.append({
                    tr("Station"):        stn,
                    tr("Temp (°C)"):      sv("Temperature"),
                    tr("Humidity (%)"):   sv("Humidity"),
                    tr("Precip (mm)"):    sv("Precipitation"),
                    tr("Wind (km/h)"):    sv("WindSpeed"),
                    tr("Wind dir"):       f"{_cardinal(wd)} ({int(wd)}°)" if wd else "—",
                    tr("Outlook"):        _interpret(sv("WindSpeed"), sv("Precipitation"))[0].split(".")[0],
                })
            st.dataframe(pd.DataFrame(rows).set_index(tr("Station")), width="stretch")


def weather_trend(df: pd.DataFrame, pollutant: str, freq: str = "Year") -> None:
    """Dual-axis trend: pollutant vs a user-selected weather driver."""
    if df.empty or pollutant not in df.columns:
        return

    st.subheader(f"🌦️ {pollutant} vs {tr('Weather')} — {tr('trend over time')}")
    st.caption(
        tr("How has the selected weather variable tracked with this pollutant over time? "
           "Based on D-1 daily readings aggregated to the selected granularity.")
    )

    driver = st.selectbox(
        tr("Weather driver to compare"),
        ["WindSpeed", "Precipitation", "Temperature", "Humidity"],
        key=f"weather_driver_{pollutant}_{freq}",
    )
    icon, unit, note = WEATHER_META[driver]

    freq_map = {"Year": "Year", "Month": "Month", "Day": "Day"}
    col = freq_map.get(freq, freq)

    if col not in df.columns:
        st.caption(tr(f"Column '{col}' not found — switch to Year or Month view."))
        return

    grouped = (
        df.groupby(col)[[pollutant, driver]]
        .mean()
        .dropna()
        .reset_index()
    )
    if grouped.empty:
        st.caption(tr("Not enough data for this combination."))
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grouped[col], y=grouped[pollutant], name=f"{pollutant} (µg/m³)",
        line=dict(color="#9b59b6", width=2.5), yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=grouped[col], y=grouped[driver], name=f"{driver} ({unit})",
        line=dict(color="#3498db", width=2, dash="dot"), yaxis="y2",
    ))
    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title=f"{pollutant} (µg/m³)"),
        yaxis2=dict(title=f"{driver} ({unit})", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    corr = grouped[pollutant].corr(grouped[driver])
    if pd.isna(corr):
        rel = tr("no clear relationship at this aggregation")
    elif corr <= -0.3:
        rel = (tr(f"an **inverse** relationship (r = {corr:.2f}) — "
               f"higher {driver.lower()} tends to go with **lower** {pollutant}. "
               f"This is consistent with {driver.lower()} aiding pollutant dispersion."))
    elif corr >= 0.3:
        rel = (tr(f"a **direct** relationship (r = {corr:.2f}) — "
               f"higher {driver.lower()} tends to go with **higher** {pollutant}."))
    else:
        rel = tr(f"a weak relationship (r = {corr:.2f}) at this aggregation level")

    st.caption(f"{icon} {tr(note)}  \n{tr('Over this period,')} **{pollutant}** {tr('and')} **{driver}** {tr('show')} {rel}.")