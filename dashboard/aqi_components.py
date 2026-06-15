"""
aqi_components.py — Visual AQI components for the dashboard.

Inspired by the Alhambra Living Lab dashboard style:
  1. render_aqi_donut()        — big ring with today's AQI level + label
  2. render_station_aqi_cards()— per-station cards with a human message
  3. render_aqi_calendar()     — GitHub-style calendar heatmap of daily AQI

Place in dashboard/ next to aqi.py. Import from pages:
    from aqi_components import render_aqi_donut, render_station_aqi_cards, render_aqi_calendar

Requires: aqi.py (same folder).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aqi import (
    AQI_CATEGORIES, AQI_POLLUTANTS,
    compute_aqi_category, overall_aqi,
)

# Short human messages per AQI level (Alhambra-style friendly tone)
AQI_MESSAGES = {
    1: "🙂 Air is healthy — enjoy the outdoors!",
    2: "🙂 Air is fine for most activities.",
    3: "😐 Sensitive groups: take it easy outside.",
    4: "😷 Consider limiting outdoor exertion.",
    5: "😷 Reduce time outdoors today.",
    6: "🚨 Health alert — avoid outdoor activity.",
}


# ==================================================================
# 1. DONUT — today's overall AQI (big ring + number, Alhambra style)
# ==================================================================

def render_aqi_donut(values: dict, title: str = "Today's Air Quality"):
    """
    values: {"PM2.5": x, "PM10": x, "NO2": x, "SO2": x}  (latest-day means)
    Renders a donut whose filled fraction = level/6, colored by category,
    with the label in the middle.
    """
    aqi = overall_aqi(values)
    if aqi is None:
        st.info("Not enough data for AQI.")
        return

    level = aqi["level"]
    frac = level / 6.0

    fig = go.Figure(go.Pie(
        values=[frac, 1 - frac],
        hole=0.72,
        marker_colors=[aqi["color"], "#ececec"],
        direction="clockwise",
        rotation=0,
        sort=False,
        textinfo="none",
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_annotation(
        text=(
            f"<b style='font-size:26px'>{aqi['label']}</b><br>"
            f"<span style='font-size:13px;color:#666'>"
            f"driven by {aqi['driver']} · {aqi['value']} µg/m³</span>"
        ),
        x=0.5, y=0.5, showarrow=False,
    )

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 15}},
        height=260,
        margin=dict(t=46, b=8, l=8, r=8),
    )

    st.plotly_chart(fig, width="stretch")
    st.markdown(
        f"<div style='text-align:center; font-size:0.95rem;'>"
        f"{AQI_MESSAGES[level]}</div>",
        unsafe_allow_html=True,
    )


# ==================================================================
# 2. STATION CARDS — one card per station with AQI color + message
# ==================================================================

def render_station_aqi_cards(df: pd.DataFrame, n_cols: int = 4):
    """
    df: dashboard dataframe (needs Date, station, pollutant columns).
    Uses each station's latest-day values. Renders a responsive card grid.
    """
    latest = df[df["Date"] == df["Date"].max()]

    stations = sorted(latest["station"].unique().tolist())
    cards = []
    for s in stations:
        row = latest[latest["station"] == s]
        vals = {
            p: float(row[p].mean())
            for p in AQI_POLLUTANTS
            if p in row.columns and pd.notna(row[p].mean())
        }
        aqi = overall_aqi(vals)
        if aqi:
            cards.append((s, aqi))

    for i in range(0, len(cards), n_cols):
        cols = st.columns(n_cols)
        for j, (station, aqi) in enumerate(cards[i:i + n_cols]):
            with cols[j]:
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 14px;
                        border: 1px solid #e3e3e3;
                        padding: 14px 12px 10px;
                        text-align: center;
                        background: white;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
                        margin-bottom: 10px;
                    ">
                      <div style="font-weight:700; font-size:0.95rem; color:#2c3e50;">
                        {station}
                      </div>
                      <div style="
                          background:{aqi['color']};
                          border-radius: 10px;
                          margin: 8px 0 6px;
                          padding: 8px 4px;
                          color: white;
                          text-shadow: 0 1px 2px rgba(0,0,0,0.35);
                          font-weight: 800;
                          font-size: 1.05rem;
                      ">
                        {aqi['label']}
                      </div>
                      <div style="font-size:0.74rem; color:#888;">
                        {aqi['driver']} · {aqi['value']} µg/m³
                      </div>
                      <div style="font-size:0.78rem; margin-top:6px;">
                        {AQI_MESSAGES[aqi['level']]}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==================================================================
# 3. CALENDAR HEATMAP — daily AQI for one station, one year
# ==================================================================

def render_aqi_calendar(df: pd.DataFrame, station: str, year: int):
    """
    GitHub-style calendar: x = week of year, y = weekday, color = AQI level.
    Computes the daily overall AQI (worst pollutant) per day for the station.
    """
    sdf = df[(df["station"] == station) & (df["Date"].dt.year == year)].copy()
    if sdf.empty:
        st.warning(f"No data for {station} in {year}.")
        return

    # Daily overall AQI level
    def day_level(row):
        vals = {p: row[p] for p in AQI_POLLUTANTS if p in row and pd.notna(row[p])}
        aqi = overall_aqi(vals)
        return aqi["level"] if aqi else np.nan

    sdf["aqi_level"] = sdf.apply(day_level, axis=1)
    sdf = sdf.dropna(subset=["aqi_level"])

    sdf["week"] = sdf["Date"].dt.isocalendar().week.astype(int)
    sdf["weekday"] = sdf["Date"].dt.dayofweek  # 0=Mon
    # Handle ISO week 52/53 wrap at year boundaries
    sdf.loc[(sdf["Date"].dt.month == 1) & (sdf["week"] > 50), "week"] = 0
    sdf.loc[(sdf["Date"].dt.month == 12) & (sdf["week"] == 1), "week"] = 53

    # Grid: weekday rows × week columns
    grid = np.full((7, 54), np.nan)
    text = np.full((7, 54), "", dtype=object)
    for _, r in sdf.iterrows():
        grid[int(r["weekday"]), int(r["week"])] = r["aqi_level"]
        cat = AQI_CATEGORIES[int(r["aqi_level"]) - 1]
        text[int(r["weekday"]), int(r["week"])] = (
            f"{r['Date'].strftime('%d %b %Y')}<br>{cat['label']}"
        )

    # Discrete 6-color scale mapped over levels 1..6
    colors = [c["color"] for c in AQI_CATEGORIES]
    colorscale = []
    for i, c in enumerate(colors):
        colorscale.append([i / 6, c])
        colorscale.append([(i + 1) / 6, c])

    fig = go.Figure(go.Heatmap(
        z=grid,
        text=text,
        hoverinfo="text",
        colorscale=colorscale,
        zmin=0.5, zmax=6.5,
        xgap=2, ygap=2,
        showscale=False,
    ))

    # Month labels at approximate week positions
    month_weeks = []
    for m in range(1, 13):
        first = pd.Timestamp(year=year, month=m, day=1)
        wk = int(first.isocalendar().week)
        month_weeks.append(0 if (m == 1 and wk > 50) else wk)
    fig.update_xaxes(
        tickvals=month_weeks,
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        side="top",
    )
    fig.update_yaxes(
        tickvals=[0, 1, 2, 3, 4, 5, 6],
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        autorange="reversed",
    )
    fig.update_layout(
        title=f"Daily AQI — {station} · {year}",
        height=240,
        margin=dict(t=70, b=10, l=40, r=10),
    )

    st.plotly_chart(fig, width="stretch")

    # Legend row (Less → More, Alhambra style)
    legend_html = "".join(
        f"<span style='display:inline-block; width:30px; height:14px; "
        f"background:{c['color']}; margin:0 2px; border-radius:3px;' "
        f"title=\"{c['label']}\"></span>"
        for c in AQI_CATEGORIES
    )
    st.markdown(
        f"<div style='text-align:center; font-size:0.78rem; color:#777;'>"
        f"Good {legend_html} Extremely poor</div>",
        unsafe_allow_html=True,
    )