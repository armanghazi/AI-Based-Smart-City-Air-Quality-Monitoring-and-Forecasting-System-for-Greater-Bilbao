"""
dashboard/pages/0_Daily_Briefing.py

Daily briefing page — the first thing a municipal manager opens every morning.

D-1 rule: latest_date is the most recent complete day in the pipeline.
          The model forecasts for latest_date + 1 (a specific date, never labelled
          "tomorrow" or "today" in the UI — always shown as the actual date).

Sections:
  1. Hero + alert banner
  2. D-1 readings vs forecast KPIs (4 pollutants)
  3. Forecast heatmap by station × pollutant
  4. Exceedance detail table + mailto share
  5. 7-day sparklines
  6. Recommended action (priority zone)
  7. PDF download
"""

from __future__ import annotations

import sys
import urllib.parse
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure dashboard/ is on the path when running as a page
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ALERT_LIMITS,
    EU_ANNUAL,
    POLLUTANT_COLOR,
    WHO_ANNUAL,
    WHO_SO2_DAILY,
    ZONE_META,
    center_tables,
    load_data,
)
from forecast_utils import POLLUTANTS, prepare_features
from i18n_auto import tr
from pdf_report import generate_daily_report

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Daily Briefing — Bilbao Air Intelligence",
    page_icon="🌅",
    layout="wide",
)
center_tables()

# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink:   #0c1521; --slate: #1b2b3a; --mist:  #5b7185;
    --line:  #e3e8ee; --paper: #ffffff; --haze:  #f4f7fa;
    --atm-1: #0ea5b5; --atm-2: #2563eb;
    --good:  #16a34a; --warn:  #d97706; --bad:   #dc2626;
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; color: var(--ink); }
h1, h2, h3, h4 { letter-spacing: -0.02em; font-weight: 600; }

.hero {
    position: relative;
    background:
        radial-gradient(1000px 400px at 20% -10%, rgba(14,165,181,0.18), transparent 55%),
        radial-gradient(800px 360px at 92% 0%,  rgba(99,102,241,0.20),  transparent 50%),
        linear-gradient(160deg, #4b63eb 0%, #3b4fc4 100%);
    border-radius: 18px; padding: 1.9rem 2.4rem 1.7rem;
    overflow: hidden; margin-bottom: 1rem; text-align: center;
}
.hero::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image: repeating-linear-gradient(115deg, rgba(255,255,255,0.035) 0 1px, transparent 1px 38px);
}
.hero-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.28em;
                text-transform: uppercase; color: #c7d2fe !important; margin: 0 0 0.6rem; }
.hero-title   { font-size: 2.6rem; font-weight: 700; line-height: 1.05; color: #ffffff !important; margin: 0; }
.hero-sub     { color: #dbe4ff !important; font-size: 1.02rem; margin-top: 0.7rem; }
.hero-meta    { display: flex; gap: 1.6rem; flex-wrap: wrap; margin-top: 1.5rem;
                font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
                color: #e0e7ff !important; justify-content: center; }
.hero-meta b  { color: #ffffff; font-weight: 600; }

.alert { border-radius: 12px; padding: 14px 20px; font-weight: 500; font-size: 0.96rem;
         margin: 1rem 0 0.4rem; display: flex; align-items: center; gap: 10px;
         border: 1px solid transparent; }
.alert-good { background: linear-gradient(135deg,#2ecc71,#27ae60); color: white; box-shadow: 0 2px 8px #2ecc7144; }
.alert-warn  { background: linear-gradient(135deg,#f39c12,#e67e22); color: white; box-shadow: 0 2px 8px #f39c1244; }
.alert-bad   { background: linear-gradient(135deg,#e74c3c,#c0392b); color: white; box-shadow: 0 2px 8px #e74c3c44; }

.eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.2em;
           text-transform: uppercase; color: var(--mist); margin-bottom: 0.2rem; }
.section-title { font-size: 1.1rem; font-weight: 700; color: #2c3e50;
                 margin: 0 0 4px 0; letter-spacing: 0.3px; }
div[data-testid="stMetric"] { background: var(--haze); border: 1px solid var(--line);
                               border-radius: 12px; padding: 1rem 1.1rem; }
div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; }
hr { border-color: var(--line); }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATA & CONSTANTS
# --------------------------------------------------

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

df            = load_data()
latest_date   = df["Date"].max()
forecast_date = latest_date + timedelta(days=1)
all_stations  = sorted(df["station"].unique().tolist())
n_years       = df["Year"].nunique()
n_records     = len(df)

# Global station → code mapping (must match training-time encoding)
STATION_CODES: dict[str, int] = {s: i for i, s in enumerate(sorted(df["station"].unique()))}

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_bundle(pollutant: str):
    prefix = pollutant.replace(".", "").lower()
    path   = MODELS_DIR / f"xgb_{prefix}_forecast.joblib"
    return joblib.load(path) if path.exists() else None


# --------------------------------------------------
# FORECAST COMPUTATION
# --------------------------------------------------

@st.cache_data(ttl=3600)
def compute_forecasts() -> pd.DataFrame:
    """Run all models for all stations. Returns one row per station × pollutant."""
    rows = []
    for station in all_stations:
        sdf = df[df["station"] == station].sort_values("Date")
        for pollutant in POLLUTANTS:
            bundle = load_bundle(pollutant)
            if bundle is None:
                continue
            feats = bundle["features"]
            prep  = prepare_features(sdf, feats, station_codes=STATION_CODES)
            valid = prep.dropna(subset=feats)
            if valid.empty:
                continue
            pred  = max(float(bundle["model"].predict(valid[feats].iloc[[-1]])[0]), 0.0)
            limit = ALERT_LIMITS.get(pollutant)
            ratio = pred / limit if limit else None
            rows.append({
                "station":   station,
                "Zone":      sdf["Zone"].iloc[-1] if not sdf.empty else "Unknown",
                "Pollutant": pollutant,
                "Forecast":  round(pred, 1),
                "Limit":     limit,
                "Ratio":     ratio,
                "Exceeds":   (ratio > 1) if ratio is not None else False,
            })
    return pd.DataFrame(rows)


fc_df     = compute_forecasts()
exceed_df = fc_df[fc_df["Exceeds"]] if not fc_df.empty else pd.DataFrame()
n_exceed  = len(exceed_df)

# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown(f"""
<div class="hero">
    <p class="hero-eyebrow">GeoAI Smart City Platform · Greater Bilbao · Bizkaia</p>
    <h1 class="hero-title">{tr("Daily Air Quality Briefing")}</h1>
    <p class="hero-sub">
        {tr("Latest data")}: <b>{latest_date.strftime('%d %B %Y')}</b> ·
        {tr("Forecast")}: <b>{forecast_date.strftime('%d %B %Y')}</b> ·
        {tr("Updated automatically every morning")}
    </p>
    <div class="hero-meta">
        <span><b>7</b> {tr("stations")} · <b>5</b> {tr("zones")}</span>
        <span><b>{n_years}</b> {tr("years")} · {n_records:,} {tr("daily records")}</span>
        <span><b>GeoAI spatial</b> · 13 {tr("notebooks")} · 35 {tr("features")}</span>
        <span><b>XGBoost</b> {tr("forecast")} · {tr("EU Directive alert standard")}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# ALERT BANNER
# --------------------------------------------------

if n_exceed == 0:
    st.markdown(
        f'<div class="alert alert-good">✅ '
        f'{tr("All stations within EU Directive limits — no exceedances forecast for")} '
        f'<b>{forecast_date.strftime("%d %b %Y")}</b>.</div>',
        unsafe_allow_html=True,
    )
elif n_exceed <= 4:
    exc_stations = ", ".join(sorted({e["station"].split("_")[0] for _, e in exceed_df.iterrows()}))
    st.markdown(
        f'<div class="alert alert-warn">⚠️ '
        f'{n_exceed} {tr("EU Directive exceedance")}{"s" if n_exceed > 1 else ""} '
        f'{tr("forecast for")} <b>{forecast_date.strftime("%d %b %Y")}</b> · '
        f'{tr("Stations")}: {exc_stations}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="alert alert-bad">🚨 '
        f'{n_exceed} {tr("EU Directive exceedances forecast for")} '
        f'<b>{forecast_date.strftime("%d %b %Y")}</b> · '
        f'{tr("multiple zones affected")}</div>',
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------
# SECTION 1 — D-1 READINGS vs FORECAST KPIs
# --------------------------------------------------

st.markdown(f'<p class="eyebrow">{tr("D-1 readings vs forecast")}</p>', unsafe_allow_html=True)
st.markdown(
    f"### {tr('Network-wide averages')}",
)
st.caption(
    f"{tr('Left value')}: {tr('latest complete day')} ({latest_date.strftime('%d %b %Y')}) · "
    f"{tr('Right value')}: {tr('forecast for')} {forecast_date.strftime('%d %b %Y')} · "
    f"{tr('Alert standard: EU Directive 2008/50/EC')}"
)

d1_means = (
    df[df["Date"] == latest_date]
    .groupby("station")[POLLUTANTS]
    .mean()
    .mean()
)
fc_means = fc_df.groupby("Pollutant")["Forecast"].mean() if not fc_df.empty else pd.Series(dtype=float)

cols = st.columns(4)
for col, p in zip(cols, POLLUTANTS):
    d1_val  = float(d1_means.get(p, float("nan")))
    fc_val  = float(fc_means.get(p, float("nan")))
    limit   = ALERT_LIMITS.get(p)
    ratio   = fc_val / limit if limit and not np.isnan(fc_val) else None
    color   = POLLUTANT_COLOR.get(p, "#888")
    delta   = fc_val - d1_val if not (np.isnan(fc_val) or np.isnan(d1_val)) else float("nan")

    with col:
        st.markdown(
            f"""
            <div style="border-left:4px solid {color};background:{color}0d;
                        border-radius:8px;padding:12px 14px;">
                <div style="font-size:0.8rem;color:#888;font-weight:600;
                            letter-spacing:0.5px">{p}</div>
                <div style="font-size:1.5rem;font-weight:700;color:#2c3e50;margin:4px 0">
                    {fc_val:.1f}
                    <span style="font-size:0.75rem;color:#888;font-weight:400">µg/m³</span>
                </div>
                <div style="font-size:0.75rem;color:#888">
                    {latest_date.strftime('%d %b')}: {d1_val:.1f} µg/m³
                </div>
                <div style="font-size:0.75rem;
                    color:{'#e74c3c' if delta > 0 else '#27ae60'};font-weight:600">
                    {delta:+.1f} µg/m³ {tr('vs')} {latest_date.strftime('%d %b')}
                </div>
                {f'<div style="font-size:0.75rem;color:#888;margin-top:3px">EU: {ratio:.1f}×</div>' if ratio else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# --------------------------------------------------
# SECTION 2 — FORECAST HEATMAP
# --------------------------------------------------

st.markdown(
    f'<p class="section-title">🔮 {tr("Forecast by station & pollutant")} — {forecast_date.strftime("%d %b %Y")}</p>',
    unsafe_allow_html=True,
)
st.caption(
    tr("Colour = ratio vs EU Directive limit · >1.0 = legal exceedance · "
       "For WHO-based health analysis see Urban Risk Index page.")
)

if not fc_df.empty:
    pivot = fc_df.pivot(index="station", columns="Pollutant", values="Ratio").reindex(columns=POLLUTANTS)
    fig_hm = px.imshow(
        pivot,
        color_continuous_scale=["#2ecc71", "#f9ca24", "#e74c3c"],
        zmin=0, zmax=2,
        text_auto=".2f",
        aspect="auto",
        labels={"color": "× EU limit"},
    )
    fig_hm.update_layout(
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_colorbar=dict(
            title="× EU",
            tickvals=[0, 1, 2],
            ticktext=["0×", "1× (EU limit)", "2×"],
        ),
    )
    st.plotly_chart(fig_hm, width="stretch", key="fig_hm")

st.divider()

# --------------------------------------------------
# SECTION 3 — EXCEEDANCE DETAIL TABLE
# --------------------------------------------------

if n_exceed > 0:
    st.markdown(
        f'<p class="section-title">⚠️ {tr("EU Directive exceedances")} — '
        f'{forecast_date.strftime("%d %b %Y")} ({n_exceed} {tr("found")})</p>',
        unsafe_allow_html=True,
    )

    show = exceed_df[["station", "Zone", "Pollutant", "Forecast", "Limit", "Ratio"]].copy()
    show["Ratio"]    = show["Ratio"].map(lambda x: f"{x:.2f}×")
    show["Forecast"] = show["Forecast"].map(lambda x: f"{x:.1f} µg/m³")
    show["Limit"]    = show["Limit"].map(lambda x: f"{x:.0f} µg/m³")
    show.columns     = ["Station", "Zone", "Pollutant", "Forecast", "EU Limit", "Ratio vs EU"]

    st.dataframe(show, hide_index=True, width="stretch")

    # mailto share button
    exceed_lines = "\n".join(
        f"  • {r['Station']} — {r['Pollutant']}: {r['Forecast']} ({r['Ratio vs EU']} EU limit)"
        for _, r in show.iterrows()
    )
    mailto_subject = f"Air Quality Alert — {forecast_date.strftime('%d %b %Y')}"
    mailto_body = (
        f"Air quality forecast alert for Greater Bilbao\n"
        f"Forecast date: {forecast_date.strftime('%d %b %Y')}\n"
        f"Based on data: {latest_date.strftime('%d %b %Y')}\n\n"
        f"EU Directive exceedances forecast:\n{exceed_lines}\n\n"
        f"Dashboard: https://geoai-dashboard.streamlit.app/\n"
    )
    mailto_link = (
        f"mailto:?subject={urllib.parse.quote(mailto_subject)}"
        f"&body={urllib.parse.quote(mailto_body)}"
    )
    st.link_button(tr("📧 Share alert by email"), mailto_link, type="primary")

else:
    st.success(
        f"✅ {tr('No EU Directive exceedances forecast for')} {forecast_date.strftime('%d %b %Y')}. "
        f"{tr('All stations within limits.')}"
    )

st.divider()

# --------------------------------------------------
# SECTION 4 — 7-DAY SPARKLINES (D-1 window)
# --------------------------------------------------

st.markdown(
    f'<p class="section-title">📈 {tr("Last 7 days — city-wide trend")}</p>',
    unsafe_allow_html=True,
)
st.caption(tr("Daily city-wide averages (D-1 data). WHO 2021 annual guidelines shown as reference lines."))

last7 = (
    df[df["Date"] >= latest_date - timedelta(days=6)]
    .groupby("Date")[POLLUTANTS]
    .mean()
    .reset_index()
    .sort_values("Date")
)

fig_spark = go.Figure()
for p in ["PM2.5", "PM10", "NO2"]:
    fig_spark.add_trace(go.Scatter(
        x=last7["Date"], y=last7[p],
        name=p, mode="lines+markers",
        line=dict(color=POLLUTANT_COLOR[p], width=2),
        marker=dict(size=6),
    ))
    if p in WHO_ANNUAL:
        fig_spark.add_hline(
            y=WHO_ANNUAL[p], line_dash="dot",
            line_color=POLLUTANT_COLOR[p], opacity=0.4,
            annotation_text=f"WHO {p}", annotation_font_size=9,
        )

fig_spark.update_layout(
    height=260, margin=dict(t=10, b=10, l=10, r=60),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.1),
    xaxis=dict(tickformat="%d %b"),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_spark, width="stretch", key="fig_spark")

st.divider()

# --------------------------------------------------
# SECTION 5 — RECOMMENDED ACTION
# --------------------------------------------------

ZONE_ACTIONS: dict[str, str] = {
    "Urban":      "Consider traffic flow management and public transport promotion. "
                  "NO₂ levels are traffic-driven — rush-hour peaks are most critical.",
    "Industrial": "Coordinate with industrial operators. "
                  "PM2.5 and PM10 levels may warrant stack monitoring review.",
    "Port":       "Monitor SO₂ from vessel activity. "
                  "Shore-power availability reduces marine emissions significantly.",
    "Coastal":    "Conditions are generally favourable. "
                  "Maintain standard monitoring — marine PM10 events are possible.",
    "Refinery":   "SO₂ episode risk. Check Petronor operational schedule. "
                  "Low-wind days increase local concentration risk.",
}

st.markdown(
    f'<p class="section-title">🎯 {tr("Recommended action")} — {forecast_date.strftime("%d %b %Y")}</p>',
    unsafe_allow_html=True,
)

if not fc_df.empty:
    zone_ratios = (
        fc_df[fc_df["Pollutant"].isin(["PM2.5", "PM10", "NO2"])]
        .groupby("Zone")["Ratio"]
        .mean()
        .sort_values(ascending=False)
    )

    if not zone_ratios.empty:
        worst_zone    = zone_ratios.index[0]
        worst_ratio   = zone_ratios.iloc[0]
        zone_meta     = ZONE_META.get(worst_zone, {})
        action_text   = ZONE_ACTIONS.get(worst_zone, "Monitor closely.")
        urgency_color = "#e74c3c" if worst_ratio > 1.5 else "#f39c12" if worst_ratio > 1.0 else "#2ecc71"
        urgency_label = tr("High priority") if worst_ratio > 1.5 else tr("Moderate") if worst_ratio > 1.0 else tr("Routine monitoring")

        st.markdown(
            f"""
            <div style="border:2px solid {zone_meta.get('border','#ccc')};border-radius:12px;
                        padding:18px 20px;
                        background:linear-gradient(135deg,{zone_meta.get('color','#888')}15,
                        {zone_meta.get('color','#888')}05);">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                    <span style="font-size:1.6rem">{zone_meta.get('icon','🏙️')}</span>
                    <div>
                        <div style="font-weight:700;font-size:1rem;color:#2c3e50">
                            {tr('Priority zone')}: {worst_zone}
                        </div>
                        <span style="background:{urgency_color};color:white;
                                     padding:2px 10px;border-radius:10px;
                                     font-size:0.75rem;font-weight:600">
                            {urgency_label} · {worst_ratio:.1f}× EU {tr('avg')}
                        </span>
                    </div>
                </div>
                <div style="color:#555;font-size:0.9rem;line-height:1.6">{action_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# --------------------------------------------------
# SECTION 6 — PDF DOWNLOAD
# --------------------------------------------------

st.markdown(f'<p class="section-title">📄 {tr("Download Report")}</p>', unsafe_allow_html=True)

col_pdf1, col_pdf2 = st.columns([2, 3])
with col_pdf1:
    d1_means_dict = {
        p: float(df[df["Date"] == latest_date][p].mean())
        for p in POLLUTANTS
    }
    worst_zone_pdf  = zone_ratios.index[0] if not fc_df.empty and not zone_ratios.empty else "Urban"
    action_text_pdf = ZONE_ACTIONS.get(worst_zone_pdf, "Monitor closely.")

    pdf_bytes = generate_daily_report(
        latest_date    = latest_date,
        current_values = d1_means_dict,
        fc_df          = fc_df,
        zone_action    = action_text_pdf,
        worst_zone     = worst_zone_pdf,
        who_annual     = WHO_ANNUAL,
        eu_annual      = EU_ANNUAL,
        alert_limits   = ALERT_LIMITS,
    )
    st.download_button(
        label         = tr("📄 Download Daily Alert Report (PDF)"),
        data          = pdf_bytes,
        file_name     = f"daily_alert_{latest_date.strftime('%Y%m%d')}.pdf",
        mime          = "application/pdf",
        type          = "primary",
        use_container_width=True,
    )

with col_pdf2:
    st.caption(
        tr("One-page summary: city-wide D-1 averages, "
           "EU Directive forecast exceedances for ") +
        forecast_date.strftime("%d %b %Y") +
        tr(", and zone-level recommended action.")
    )

st.divider()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown(
        "**Air quality**  \n"
        "Basque Government — RVCA network  \n"
        "[opendata.euskadi.eus](https://opendata.euskadi.eus/api-air-quality/?api=air-quality)  \n"
        "7 stations · © Gobierno Vasco · CC BY 4.0"
    )
with f2:
    st.markdown(
        "**Meteorology**  \n"
        "Open-Meteo ERA5 archive  \n"
        "[open-meteo.com](https://open-meteo.com) · CC BY 4.0"
    )
with f3:
    st.markdown(
        "**Standards**  \n"
        "WHO 2021 guidelines (health analysis)  \n"
        "EU Directive 2008/50/EC (operational alerts)"
    )

st.caption(
    f"{tr('Data')}: {tr('Basque Government (CC BY 4.0) + Open-Meteo (CC BY 4.0)')} · "
    f"{tr('Forecasts: XGBoost models (test R²=0.39–0.56)')} · "
    f"{tr('Latest data')}: {latest_date.strftime('%d %b %Y')} · "
    f"{tr('Next pipeline run: ~06:00 UTC daily')}"
)

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:0.88rem;color:#64748b;line-height:1.7">
        <b style="color:#0c1521">Arman Ghaziaskari Naeini</b><br>
        GIS &amp; Spatial Data Science<br><br>
        <a href="https://armanghazi.github.io/portfolio"
           style="color:#0ea5b5;text-decoration:none">Portfolio</a>&nbsp;&nbsp;
        <a href="https://github.com/armanghazi"
           style="color:#0ea5b5;text-decoration:none">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )