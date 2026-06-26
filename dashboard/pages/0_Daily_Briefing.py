"""
dashboard/pages/0_Daily_Briefing.py

Daily briefing page — the first thing a municipal manager opens every morning.
Shows: latest available reading (D-1), next-day forecast, 7-day sparklines,
and a mailto share button for WHO exceedances.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import timedelta
import sys

import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    load_data, WHO_ANNUAL, WHO_SO2_DAILY,
    POLLUTANT_COLOR, ZONE_META, get_fav_station,
    EU_ANNUAL,ALERT_LIMITS, center_tables
)

from pdf_report import generate_daily_report

from i18n_auto import tr

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Daily Briefing",
    page_icon="🌅",
    layout="wide",
)
center_tables()

# --------------------------------------------------
# CUSTOM CSS — clean, modern look
# --------------------------------------------------

st.markdown("""
<style>
    /* Status badge */
    .status-ok {
        background: linear-gradient(135deg,#2ecc71,#27ae60);
        color: white; border-radius: 12px;
        padding: 18px 22px; text-align: center;
        font-size: 1rem; font-weight: 600;
        box-shadow: 0 2px 8px #2ecc7144;
    }
    .status-warn {
        background: linear-gradient(135deg,#f39c12,#e67e22);
        color: white; border-radius: 12px;
        padding: 18px 22px; text-align: center;
        font-size: 1rem; font-weight: 600;
        box-shadow: 0 2px 8px #f39c1244;
    }
    .status-alert {
        background: linear-gradient(135deg,#e74c3c,#c0392b);
        color: white; border-radius: 12px;
        padding: 18px 22px; text-align: center;
        font-size: 1rem; font-weight: 600;
        box-shadow: 0 2px 8px #e74c3c44;
    }
    /* Station pill */
    .station-pill {
        display: inline-block;
        padding: 3px 10px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600;
        margin: 2px;
    }
    /* Metric card */
    .briefing-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 14px 16px;
        border-left: 4px solid #3498db;
        margin-bottom: 8px;
    }
    /* Section title */
    .section-title {
        font-size: 1.1rem; font-weight: 700;
        color: #2c3e50; margin: 0 0 4px 0;
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)




st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink:        #0c1521;
    --slate:      #1b2b3a;
    --mist:       #5b7185;
    --line:       #e3e8ee;
    --paper:      #ffffff;
    --haze:       #f4f7fa;
    --atm-1:      #0ea5b5;   /* atmospheric teal */
    --atm-2:      #2563eb;   /* deep sky */
    --good:       #16a34a;
    --warn:       #d97706;
    --bad:        #dc2626;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: var(--ink);
}

h1, h2, h3, h4 { letter-spacing: -0.02em; font-weight: 600; }

/* ---- Hero ---- */
.hero {
    position: relative;
    background:
        radial-gradient(1000px 400px at 20% -10%, rgba(14,165,181,0.18), transparent 55%),
        radial-gradient(800px 360px at 92% 0%, rgba(99,102,241,0.20), transparent 50%),
        linear-gradient(160deg, #4b63eb 0%, #3b4fc4 100%);
    border-radius: 18px;
    padding: 1.9rem 2.4rem 1.7rem;
    overflow: hidden;
    margin-bottom: 1rem;
    text-align: center;
}
.hero::after {
    /* faint isobar lines — atmospheric chart motif */
    content: "";
    position: absolute; inset: 0;
    background-image:
        repeating-linear-gradient(115deg, rgba(255,255,255,0.035) 0 1px, transparent 1px 38px);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.28em;
    text-transform: uppercase; color: #c7d2fe;
    margin: 0 0 0.6rem;
}
.hero-title {
    font-size: 2.6rem; font-weight: 700; line-height: 1.05;
    color: #ffffff !important; margin: 0;
}
.hero-title, .hero-eyebrow, .hero-sub, .hero-meta,
.hero-meta b, .hero h1 {
    color: #ffffff !important;
}
.hero-eyebrow { color: #c7d2fe !important; }
.hero-sub     { color: #dbe4ff !important; }
.hero-meta    { color: #e0e7ff !important; }
.hero-sub {
    color: #dbe4ff; font-size: 1.02rem; margin-top: 0.7rem;
    margin-left: auto; margin-right: auto;
    white-space: nowrap;
    max-width: none;
            }
.hero-meta {
    display: flex; gap: 1.6rem; flex-wrap: wrap;
    margin-top: 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; color: #e0e7ff;
    justify-content: center;     
}
.hero-meta b { color: #ffffff; font-weight: 600; }

/* ---- Alert strip ---- */
.alert {
    border-radius: 12px; padding: 14px 20px;
    font-weight: 500; font-size: 0.96rem;
    margin: 1rem 0 0.4rem;
    display: flex; align-items: center; gap: 10px;
    border: 1px solid transparent;
}
.alert a { color: inherit; font-weight: 600; text-decoration: underline; }
.alert-good {
    background: linear-gradient(135deg,#2ecc71,#27ae60);
    color: white; border-color: transparent;
    box-shadow: 0 2px 8px #2ecc7144;
}
.alert-warn {
    background: linear-gradient(135deg,#f39c12,#e67e22);
    color: white; border-color: transparent;
    box-shadow: 0 2px 8px #f39c1244;
}
.alert-bad {
    background: linear-gradient(135deg,#e74c3c,#c0392b);
    color: white; border-color: transparent;
    box-shadow: 0 2px 8px #e74c3c44;
}

/* ---- Section eyebrow ---- */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--mist); margin-bottom: 0.2rem;
}

/* ---- Zone card ---- */
.zone-card {
    border: 1px solid var(--line);
    border-radius: 14px; padding: 16px 18px;
    background: var(--paper); height: 100%;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.zone-card:hover {
    border-color: var(--atm-1);
    box-shadow: 0 8px 24px -12px rgba(14,165,181,0.4);
}
.zone-head { font-size: 1.15rem; font-weight: 600; margin-bottom: 3px; }
.zone-desc { color: var(--mist); font-size: 0.8rem; margin-bottom: 6px; line-height: 1.4; }
.zone-loc  { color: #94a3b8; font-size: 0.74rem; margin-bottom: 6px;
             font-family: 'IBM Plex Mono', monospace; }
.zone-spatial { color: #0ea5b5; font-size: 0.77rem; margin-bottom: 8px;
                font-style: italic; line-height: 1.4; }
.zone-row  { display:flex; justify-content:space-between;
             font-size: 0.84rem; padding: 2px 0; }
.zone-row .k { color: var(--mist); }
.zone-row .v { font-family: 'IBM Plex Mono', monospace; font-weight: 500; }

/* ---- Nav tile ---- */
.nav-tile {
    border: 1px solid var(--line);
    border-radius: 14px; padding: 1.4rem 1.1rem;
    text-align: left; background: var(--paper);
    transition: all 0.22s cubic-bezier(.4,0,.2,1);
    height: 100%;
}
.nav-tile:hover {
    border-color: var(--atm-2);
    transform: translateY(-3px);
    box-shadow: 0 14px 30px -18px rgba(37,99,235,0.5);
}
.nav-icon { font-size: 1.7rem; }
.nav-title { font-weight: 600; font-size: 0.98rem; margin: 8px 0 4px; }
.nav-desc { color: var(--mist); font-size: 0.8rem; line-height: 1.45; }

/* ---- Metric tuning ---- */
div[data-testid="stMetric"] {
    background: var(--haze);
    border: 1px solid var(--line);
    border-radius: 12px; padding: 1rem 1.1rem;
}
div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem;
}

hr { border-color: var(--line); }
</style>
""", unsafe_allow_html=True)
# ==================================================
# QUICK FORECAST — alert banner only (EU thresholds)
# ==================================================

MODELS_DIR = Path(__file__).parent.parent / "models"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]


@st.cache_resource
def _load_bundle(pollutant: str):
    prefix = pollutant.replace(".", "").lower()
    path   = MODELS_DIR / f"xgb_{prefix}_forecast.joblib"
    return joblib.load(path) if path.exists() else None


def _prepare_last_row(sdf: pd.DataFrame, feats: list) -> pd.DataFrame:
    """Minimal feature prep for the homepage alert banner."""
    df2 = sdf.copy()
    if "wind_u" not in df2.columns and "Wind_X" in df2.columns:
        df2["wind_u"] = df2["Wind_X"]
    if "wind_v" not in df2.columns and "Wind_Y" in df2.columns:
        df2["wind_v"] = df2["Wind_Y"]
    df2["year"]         = df2["Date"].dt.year
    df2["month"]        = df2["Date"].dt.month
    df2["day"]          = df2["Date"].dt.day
    df2["day_of_year"]  = df2["Date"].dt.dayofyear
    df2["week_of_year"] = df2["Date"].dt.isocalendar().week.astype(int)
    df2["day_of_week"]  = df2["Date"].dt.dayofweek
    df2["is_weekend"]   = (df2["Date"].dt.weekday >= 5).astype(int)
    s2i = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    mapped = df2["season"].astype(str).str.capitalize().map(s2i)
    msea   = df2["Date"].dt.month.map(
        {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    df2["season"]       = mapped.fillna(msea).fillna(0).astype(int)
    df2["station_code"] = df2["station"].astype("category").cat.codes
    for p in POLLUTANTS:
        pcol = p.replace(".", "")
        col  = f"{pcol}_roll_mean_14"
        if col not in df2.columns:
            df2[col] = (df2.groupby("station")[p]
                        .transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean()))
    df2["wind_x_precip"] = df2["WindSpeed"] * df2["Precipitation"]
    df2["temp_x_humid"]  = df2["Temperature"] * df2["Humidity"]
    for f in feats:
        if f not in df2.columns:
            df2[f] = 0.0
        if not pd.api.types.is_numeric_dtype(df2[f]):
            df2[f] = pd.to_numeric(df2[f], errors="coerce").fillna(0)
    return df2


@st.cache_data(ttl=3600)
def _get_tomorrow_exceedances() -> list:
    results = []
    for station in station_list:
        sdf = df[df["station"] == station].sort_values("Date")
        for pollutant in POLLUTANTS:
            bundle = _load_bundle(pollutant)
            if bundle is None:
                continue
            feats = bundle["features"]
            prep  = _prepare_last_row(sdf, feats).dropna(subset=feats)
            if prep.empty:
                continue
            pred  = max(float(bundle["model"].predict(prep[feats].iloc[[-1]])[0]), 0.0)
            limit = ALERT_LIMITS.get(pollutant, 25.0)
            if pred > limit:
                results.append({"station": station, "pollutant": pollutant,
                                "forecast": pred, "ratio": pred / limit})
    return results


exceed = _get_tomorrow_exceedances()
n_exc  = len(exceed)
tomorrow_str = (latest_date + timedelta(days=1)).strftime("%d %b %Y")

# ==================================================
# HERO
# ==================================================

st.markdown(f"""
<div class="hero">
    <p class="hero-eyebrow">GeoAI Smart City Platform · Greater Bilbao · Bizkaia</p>
    <h1 class="hero-title">Air Quality Intelligence for Greater Bilbao</h1>
    <p class="hero-sub">
        Monitoring and next-day forecasting across the region's air — seven stations, four pollutants, updated automatically every morning.
    </p>
    <div class="hero-meta">
        <span><b>7</b> stations · <b>5</b> zones</span>
        <span><b>{n_years}</b> years · {n_records:,} daily records</span>
        <span><b>GeoAI spatial</b> · 4 notebooks · 35 features</span>
        <span><b>Next-day</b> XGBoost forecast</span>
        <span><b>WHO 2021</b> + EU Directive 2008/50/EC</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Alert strip ───────────────────────────────────
if n_exc == 0:
    st.markdown(
        f'<div class="alert alert-good">✅ '
        f'All stations within EU Directive limits — '
        f'no legal exceedances forecast for {tomorrow_str}.</div>',
        unsafe_allow_html=True)
elif n_exc <= 4:
    s_exc = list({e["station"].split("_")[0] for e in exceed})
    st.markdown(
        f'<div class="alert alert-warn">⚠️ '
        f'{n_exc} EU Directive exceedance{"s" if n_exc>1 else ""} '
        f'forecast for {tomorrow_str} · Stations: {", ".join(s_exc)} · '
        f'<a href="/Daily_Briefing">Open the daily briefing →</a></div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="alert alert-bad">🚨 '
        f'{n_exc} EU Directive exceedances forecast for {tomorrow_str} · '
        f'multiple zones affected · '
        f'<a href="/Daily_Briefing">Open the daily briefing →</a></div>',
        unsafe_allow_html=True)

st.write("")
st.write("")

# ==================================================
# QUICK STATUS
# ==================================================

st.markdown('<p class="eyebrow">Latest reading</p>', unsafe_allow_html=True)
st.markdown("### " + tr("Latest readings across the network") + f" — {latest_date.strftime('%d %b %Y')}")

col_fav, col_snap = st.columns([2, 5], gap="large")

with col_fav:
    default_idx = (station_list.index(st.session_state.fav_station)
                   if st.session_state.fav_station in station_list else 0)
    selected_fav = st.selectbox(
        "Your default station",
        options=station_list,
        index=default_idx,
        help="Remembered on this device for your next visit.",
    )
    if st.button("Save as default", type="primary", use_container_width=True):
        st.session_state.fav_station = selected_fav
        if _cookies is not None:
            _cookies["fav_station"] = selected_fav
            _cookies.save()
            st.success(f"{selected_fav} saved.")
        else:
            st.info("Saved for this session.")

with col_snap:
    latest_means = (
        df[df["Date"] == latest_date]
        .groupby("station")[["PM2.5", "PM10", "NO2", "SO2"]]
        .mean().mean()
    )
    c1, c2, c3, c4 = st.columns(4)
    for col, poll in zip([c1, c2, c3, c4], ["PM2.5", "PM10", "NO2", "SO2"]):
        val = latest_means.get(poll, 0)
        delta_label, delta_color = who_delta(val, poll)
        eu_lim = EU_ANNUAL.get(poll)
        eu_str = f"  ·  EU {val/eu_lim:.1f}×" if eu_lim else ""
        col.metric(
            label=f"{poll}{eu_str}",
            value=f"{val:.1f} µg/m³",
            delta=delta_label,
            delta_color=delta_color,
            help=f"WHO {WHO_ANNUAL.get(poll, WHO_SO2_DAILY)} µg/m³"
                 + (f"  ·  EU {eu_lim} µg/m³" if eu_lim else ""),
        )

st.divider()

# ==================================================
# ENVIRONMENTAL ZONES
# ==================================================

st.markdown(f'<p class="eyebrow">{tr("The network, by character")}</p>', unsafe_allow_html=True)
st.markdown("### " + tr("Five environmental zones"))
st.caption(
    "Each station sits in a zone defined by its dominant emission source — "
    "traffic, industry, port, coast, or refinery. Latest-year averages shown."
)

zone_summary = (
    df[df["Year"] == int(df["Year"].max())]
    .groupby("Zone")[["PM2.5", "PM10", "NO2", "SO2"]]
    .mean().round(1)
)


# Spatial driver context from GIS analysis (notebooks 10a/10b/10c)
ZONE_SPATIAL = {
    "Urban":      "Road density 19,060 m/km² · 501 m from city centre → structural NO₂ source",
    "Industrial": "354 m from AP-8 motorway · industrial land use 10–21% within 1 km",
    "Port":       "784 m from Port of Bilbao · TRI 445 m provides terrain dispersion buffer",
    "Coastal":    "Lowest road density (9,933 m/km²) · 2.6 km coast → NW sea breeze flushing",
    "Refinery":   "2.4 km from Petronor · TRI 343 m + coastal position → dispersion advantage",
}

def render_zone_card(zone_name, meta, zone_summary, df):
    z = zone_summary.loc[zone_name] if zone_name in zone_summary.index else None
    stations_in = df[df["Zone"] == zone_name]["station"].unique().tolist()
    short = ", ".join(s.split("_")[0] for s in stations_in)
    key_poll = meta["key_pollutant"]
    if z is not None:
        kv  = z[key_poll]
        lim = WHO_ANNUAL.get(key_poll)
        vs  = f"{kv/lim:.1f}×" if lim else "—"
        pm25, pm10, no2, so2 = z["PM2.5"], z["PM10"], z["NO2"], z["SO2"]
    else:
        vs = "—"; pm25 = pm10 = no2 = so2 = 0.0

    st.markdown(f"""
    <div class="zone-card" style="border-top:3px solid {meta['color']}">
        <div class="zone-head">{meta['icon']} {zone_name}</div>
        <div class="zone-desc">{meta['description']}</div>
        <div class="zone-loc">{short}</div>
        <div class="zone-spatial">{ZONE_SPATIAL.get(zone_name, "")}</div>
        <div class="zone-row"><span class="k">PM2.5</span><span class="v">{pm25:.1f}</span></div>
        <div class="zone-row"><span class="k">PM10</span><span class="v">{pm10:.1f}</span></div>
        <div class="zone-row"><span class="k">NO₂</span><span class="v">{no2:.1f}</span></div>
        <div class="zone-row"><span class="k">SO₂</span><span class="v">{so2:.1f}</span></div>
        <div class="zone-row" style="border-top:1px solid var(--line);margin-top:6px;padding-top:6px">
            <span class="k">Key ({key_poll}) vs WHO</span>
            <span class="v" style="color:{meta['color']}">{vs}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


zones_list = list(ZONE_META.items())
cols_r1 = st.columns(3, gap="medium")
for i, (zn, meta) in enumerate(zones_list[:3]):
    with cols_r1[i]:
        render_zone_card(zn, meta, zone_summary, df)

if len(zones_list) > 3:
    cols_r2 = st.columns(3, gap="medium")
    for i, (zn, meta) in enumerate(zones_list[3:]):
        with cols_r2[i]:
            render_zone_card(zn, meta, zone_summary, df)

st.divider()

# ==================================================
# CITY-WIDE INSIGHTS
# ==================================================

st.markdown(f'<p class="eyebrow">{tr("A decade in view")}</p>', unsafe_allow_html=True)
st.markdown("### " + tr("City-wide trends"))

col_left, col_right = st.columns([3, 2], gap="large")
latest_year = int(df["Year"].max())

with col_left:
    st.markdown("#### Annual mean concentration")
    annual = df.groupby("Year")[["PM2.5", "PM10", "NO2"]].mean().reset_index()
    annual_long = annual.melt(id_vars="Year", var_name="Pollutant",
                              value_name="Concentration")
    fig_trend = px.line(
        annual_long, x="Year", y="Concentration", color="Pollutant",
        color_discrete_map=POLLUTANT_COLOR, markers=True,
    )
    for poll, limit in WHO_ANNUAL.items():
        fig_trend.add_hline(
            y=limit, line_dash="dot",
            line_color=POLLUTANT_COLOR.get(poll, "#666"), opacity=0.35,
            annotation_text=f"WHO {poll}", annotation_font_size=9,
            annotation_position="right",
        )
    fig_trend.update_layout(dragmode=False, 
        height=340, margin=dict(t=10, b=10, l=10, r=60),
        hovermode="x unified", legend=dict(orientation="h", y=1.1),
        font=dict(family="IBM Plex Sans"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_trend, width="stretch", config={"scrollZoom": False, "displayModeBar": False})

with col_right:
    st.markdown("#### Station risk ranking")

    station_latest = (
        df[df["Year"] == latest_year]
        .groupby(["station", "Zone"])[["PM2.5", "PM10", "NO2"]]
        .mean().reset_index()
    )

    def core_risk(row):
        ratios = [row["PM2.5"]/WHO_ANNUAL["PM2.5"],
                  row["PM10"]/WHO_ANNUAL["PM10"],
                  row["NO2"]/WHO_ANNUAL["NO2"]]
        return 100 * sum(ratios) / 3

    station_latest["Score"]   = station_latest.apply(core_risk, axis=1)
    station_latest            = station_latest.sort_values("Score", ascending=False)
    station_latest["Station"] = station_latest["station"].str.split("_").str[0]

    fig_status = px.bar(
        station_latest, x="Score", y="Station",
        color="Zone",
        color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
        orientation="h", text="Score",
    )
    fig_status.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_status.add_vline(x=100, line_dash="dash", line_color="#94a3b8", opacity=0.7,
                         annotation_text="WHO", annotation_font_size=9)
    fig_status.update_layout(dragmode=False, 
            height=360, margin=dict(t=40, b=10, l=10, r=30),
            showlegend=True,
            legend=dict(orientation="h", y=1.22, x=0, font=dict(size=9)),
            yaxis=dict(autorange="reversed"),
            xaxis_range=[0, max(station_latest["Score"].max()*1.25, 250)],
            font=dict(family="IBM Plex Sans"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
    st.plotly_chart(fig_status, width="stretch", config={"scrollZoom": False, "displayModeBar": False})

st.caption(
    "Risk score = mean of (concentration ÷ WHO 2021 limit) across PM2.5, PM10, NO₂, ×100. "
    "100 = exactly at the WHO guideline."
)

st.divider()

# ==================================================
# NAVIGATION
# ==================================================

st.markdown(f'<p class="eyebrow">{tr("Where to next")}</p>', unsafe_allow_html=True)
st.markdown("### " + tr("Explore the platform"))

NAV = [
    {"icon":"🌅","title":"Daily Briefing","desc":"Today's status, tomorrow's alerts, one-page PDF","page":"pages/0_Daily_Briefing.py"},
    {"icon":"🗺️","title":"Air Quality Monitoring","desc":"Interactive GIS map · station comparison · SVI structural context","page":"pages/1_Air_Quality_Monitoring.py"},
    {"icon":"📈","title":"Temporal Trends","desc":"Long-term patterns, seasonality, COVID impact","page":"pages/2_Temporal_Trends.py"},
    {"icon":"🌍","title":"GeoAI Spatial Analysis","desc":"Spatial drivers · terrain · wind transport · station DNA","page":"pages/3_GeoAI_Spatial_Analysis.py"},
    {"icon":"🌤️","title":"Weather Drivers","desc":"Wind transport analysis · dispersion effects · seasonal patterns","page":"pages/4_Weather_Drivers_&_Air_Pollution_Dynamics.py"},
    {"icon":"🔮","title":"Forecasting","desc":"Next-day XGBoost predictions with SHAP","page":"pages/5_Forecasting.py"},
    {"icon":"🏛️","title":"Decision Support","desc":"GeoAI spatial intelligence · wind transport · structural risk index","page":"pages/6_Smart_City_Decision_Support.py"},
    {"icon":"📋","title":"Scope & Limitations","desc":"Coverage, model accuracy, known gaps","page":"pages/7_Scope_and_Limitations.py"},
]

for i in range(0, len(NAV), 4):
    cols = st.columns(4, gap="medium")
    for j, m in enumerate(NAV[i:i+4]):
        with cols[j]:
            st.markdown(f"""
            <div class="nav-tile">
                <div class="nav-icon">{m['icon']}</div>
                <div class="nav-title">{m['title']}</div>
                <div class="nav-desc">{m['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open →", key=f"nav_{i+j}", use_container_width=True):
                st.switch_page(m["page"])

st.divider()

# ==================================================
# FOOTER
# ==================================================

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown(
        "**Air quality**  \n"
        "Basque Government — RVCA network  \n"
        "[opendata.euskadi.eus](https://opendata.euskadi.eus/api-air-quality/?api=air-quality)\n"
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
        "WHO 2021 guidelines (analysis)  \n"
        "EU Directive 2008/50/EC (alerts)"
    )

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:0.88rem;color:#64748b;line-height:1.7">
        <b style="color:#0c1521">Arman Ghaziaskari Naeini</b><br>
        GIS &amp; Spatial Data Science<br><br>

        <a href="https://armanghazi.github.io/portfolio"
           style="color:#0ea5b5;text-decoration:none">Portfolio</a> \n
        <a href="https://github.com/armanghazi"
           style="color:#0ea5b5;text-decoration:none">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]

# Alerts use EU Directive (legally binding in Spain)
# Analysis pages use WHO 2021 (stricter, health-optimal)
ALERT_LIMITS = {
    "PM2.5": 25.0,    # EU annual
    "PM10":  40.0,    # EU annual
    "NO2":   40.0,    # EU annual
    "SO2":   125.0,   # EU 24-hour
}


def model_path(pollutant: str) -> Path:
    prefix = pollutant.replace(".", "").lower()
    return MODELS_DIR / f"xgb_{prefix}_forecast.joblib"


@st.cache_resource
def load_model(pollutant: str):
    path = model_path(pollutant)
    return joblib.load(path) if path.exists() else None


# --------------------------------------------------
# FEATURE PREPARATION (mirrors 5_Forecasting.py)
# --------------------------------------------------

def prepare_features(df: pd.DataFrame, required_features: list) -> pd.DataFrame:
    """Build all features the XGBoost models need."""
    df = df.copy()

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

    season_to_int = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    mapped = df["season"].astype(str).str.capitalize().map(season_to_int)
    month_season = df["Date"].dt.month.map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    df["season"] = mapped.fillna(month_season).fillna(0).astype(int)
    df["station_code"] = df["station"].astype("category").cat.codes

    for pollutant in POLLUTANTS:
        prefix = pollutant.replace(".", "")
        col = f"{prefix}_roll_mean_14"
        if col not in df.columns:
            df[col] = (
                df.groupby("station")[pollutant]
                .transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
            )

    df["wind_x_precip"] = df["WindSpeed"] * df["Precipitation"]
    df["temp_x_humid"]  = df["Temperature"] * df["Humidity"]

    for feat in required_features:
        if feat not in df.columns:
            df[feat] = 0.0
        if not pd.api.types.is_numeric_dtype(df[feat]):
            df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0)

    return df


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df           = load_data()
latest_date  = df["Date"].max()
all_stations = sorted(df["station"].unique().tolist())

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    f"""
    <div style="padding:1rem 0 0.5rem">
        <h1 style="margin:0;font-size:1.9rem">
            🌅 Daily Air Quality Briefing
        </h1>
        <p style="color:#666;margin-top:4px;font-size:0.95rem">
            Greater Bilbao · Latest data:
            <b>{latest_date.strftime('%A, %d %B %Y')}</b> ·
            Updated automatically every day
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# COMPUTE TOMORROW'S FORECASTS (all stations × all pollutants)
# --------------------------------------------------

bundles = {p: load_model(p) for p in POLLUTANTS}

forecasts = []   # list of dicts: station, pollutant, prediction, limit, ratio, status
for station in all_stations:
    sdf = df[df["station"] == station].sort_values("Date")
    for pollutant in POLLUTANTS:
        bundle = bundles.get(pollutant)
        if bundle is None:
            continue
        feats = bundle["features"]
        prep  = prepare_features(sdf, feats)
        valid = prep.dropna(subset=feats)
        if valid.empty:
            continue
        pred  = float(bundle["model"].predict(valid[feats].iloc[[-1]])[0])
        pred  = max(pred, 0.0)
        limit =ALERT_LIMITS.get(pollutant)
        ratio = pred / limit if limit else None
        forecasts.append({
            "station":   station,
            "Zone":      sdf["Zone"].iloc[-1],
            "Pollutant": pollutant,
            "Forecast":  round(pred, 1),
            "Limit":     limit,
            "Ratio":     ratio,
            "Exceeds":   ratio > 1 if ratio else False,
        })

fc_df      = pd.DataFrame(forecasts)
exceed_df  = fc_df[fc_df["Exceeds"]]
n_exceed   = len(exceed_df)
n_stations = len(all_stations)

# --------------------------------------------------
# SECTION 1 — CITY STATUS BANNER
# --------------------------------------------------

tomorrow = latest_date + timedelta(days=1)

if n_exceed == 0:
    banner_class = "status-ok"
    banner_icon  = "✅"
    banner_text  = f"All {n_stations} stations within EU Directive limits for · {tomorrow.strftime('%d %b %Y')}"
elif n_exceed <= 3:
    banner_class = "status-warn"
    banner_icon  = "⚠️"
    exceed_stations = exceed_df["station"].unique()
    banner_text  = (
        f"{n_exceed} exceedance{'s' if n_exceed>1 else ''} forecast {tomorrow.strftime('%d %b %Y') } · "
        f"Stations: {', '.join(s.split('_')[0] for s in exceed_stations)}"
    )
else:
    banner_class = "status-alert"
    banner_icon  = "🚨"
    banner_text  = f"{n_exceed} EU Directive limits exceedances forecast for {tomorrow.strftime('%d %b %Y')} — review priority zones"

st.markdown(
    f'<div class="{banner_class}">{banner_icon} {banner_text}</div>',
    unsafe_allow_html=True,
)
st.markdown("")

# --------------------------------------------------
# SECTION 2 — TODAY vs TOMORROW KPIs
# --------------------------------------------------

st.markdown(f'<p class="section-title">{tr("📊 City-wide averages")}</p>', unsafe_allow_html=True)

# Today — latest available day
today_means = (
    df[df["Date"] == latest_date]
    .groupby("station")[POLLUTANTS]
    .mean()
    .mean()
)

# Tomorrow — from forecasts
tomorrow_means = fc_df.groupby("Pollutant")["Forecast"].mean()

cols = st.columns(4)
for col, p in zip(cols, POLLUTANTS):
    today_val    = today_means.get(p, float("nan"))
    tomorrow_val = tomorrow_means.get(p, float("nan"))
    limit        = ALERT_LIMITS.get(p)
    ratio        = tomorrow_val / limit if limit and not np.isnan(tomorrow_val) else None
    color        = POLLUTANT_COLOR.get(p, "#888")

    delta_val  = tomorrow_val - today_val
    delta_str  = f"{delta_val:+.1f} µg/m³ vs today"

    with col:
        st.markdown(
            f"""
            <div style="border-left:4px solid {color};
                        background:{color}0d;
                        border-radius:8px;padding:12px 14px;">
                <div style="font-size:0.8rem;color:#888;font-weight:600;
                            letter-spacing:0.5px">{p}</div>
                <div style="font-size:1.5rem;font-weight:700;
                            color:#2c3e50;margin:4px 0">
                    {tomorrow_val:.1f} <span style="font-size:0.75rem;
                    color:#888;font-weight:400">µg/m³</span>
                </div>
                <div style="font-size:0.75rem;color:#888">
                    Today: {today_val:.1f} µg/m³
                </div>
                <div style="font-size:0.75rem;
                    color:{'#e74c3c' if delta_val > 0 else '#27ae60'};
                    font-weight:600">
                    {delta_str}
                </div>
                {f'<div style="font-size:0.75rem;color:#888;margin-top:3px">WHO: {ratio:.1f}×</div>' if ratio else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# --------------------------------------------------
# SECTION 3 — TOMORROW'S STATION HEATMAP
# --------------------------------------------------

st.markdown(f'<p class="section-title">{tr("🔮 Next-day forecast")} — {tomorrow.strftime("%d %b %Y")} — {tr("by station & pollutant")}</p>',
            unsafe_allow_html=True)
st.caption(
    tr("Colour = ratio vs EU Directive limit (legally binding in Spain) · "
       ">1.0 = legal exceedance") +
    f" · {tr('Forecast date')}: {tomorrow.strftime('%d %b %Y')} · " +
    tr("For WHO-based analysis see Urban Risk Index")
)

if not fc_df.empty:
    pivot = fc_df.pivot(index="station", columns="Pollutant", values="Ratio")[POLLUTANTS]

    fig_hm = px.imshow(
        pivot,
        color_continuous_scale=["#2ecc71", "#f9ca24", "#e74c3c"],
        zmin=0, zmax=2,
        text_auto=".2f",
        aspect="auto",
        labels={"color": "× WHO limit"},
    )
    fig_hm.update_layout(
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_colorbar=dict(
            title="× WHO",
            tickvals=[0, 1, 2],
            ticktext=["0×", "1× (WHO)", "2×"],
        ),
    )
    fig_hm.add_shape(   # WHO threshold line indicator
        type="line", x0=-0.5, x1=3.5, y0=0, y1=0,
        line=dict(color="red", width=0),
    )
    st.plotly_chart(fig_hm,  width="stretch", key="fig_hm")

# --------------------------------------------------
# SECTION 4 — EXCEEDANCE DETAIL TABLE
# --------------------------------------------------

if n_exceed > 0:
    st.markdown(
        f'<p class="section-title">⚠️ WHO exceedances {tomorrow.strftime("%d %b %Y")} '
        f'({n_exceed} found)</p>',
        unsafe_allow_html=True,
    )

    show = exceed_df[["station", "Zone", "Pollutant", "Forecast", "Limit", "Ratio"]].copy()
    show["Ratio"]    = show["Ratio"].map(lambda x: f"{x:.2f}×")
    show["Forecast"] = show["Forecast"].map(lambda x: f"{x:.1f} µg/m³")
    show["Limit"]    = show["Limit"].map(lambda x: f"{x:.0f} µg/m³")
    show.columns     = ["Station", "Zone", "Pollutant",
                        "Forecast", "WHO Limit", "Ratio vs WHO"]

    st.dataframe(show, hide_index=True,  width="stretch")

    # mailto share button
    exceed_lines = "\n".join(
        f"  • {r['Station']} — {r['Pollutant']}: {r['Forecast']} µg/m³ ({r['Ratio vs WHO']} WHO)"
        for _, r in show.iterrows()
    )
    mailto_subject = f"Air Quality Alert — {tomorrow.strftime('%d %b %Y')}"
    mailto_body = (
        f"Air quality forecast alert for Greater Bilbao\n"
        f"Date: {tomorrow.strftime('%d %b %Y')}\n\n"
        f"WHO exceedances forecast:\n{exceed_lines}\n\n"
        f"Dashboard: https://geoai-dashboard.streamlit.app/\n"
    )
    import urllib.parse
    mailto_link = (
        f"mailto:?subject={urllib.parse.quote(mailto_subject)}"
        f"&body={urllib.parse.quote(mailto_body)}"
    )
    st.link_button(tr("📧 Share alert by email"), mailto_link, type="primary")

else:
    st.success(
        f"✅ {tr('No WHO exceedances forecast for')} {tomorrow.strftime('%d %b %Y')}. "
        f"{tr('All stations within safe limits.')}"
    )

st.divider()

# --------------------------------------------------
# SECTION 5 — 7-DAY SPARKLINES
# --------------------------------------------------

st.markdown(f'<p class="section-title">{tr("📈 Last 7 days — city-wide trend")}</p>',
            unsafe_allow_html=True)

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
        x=last7["Date"],
        y=last7[p],
        name=p,
        mode="lines+markers",
        line=dict(color=POLLUTANT_COLOR[p], width=2),
        marker=dict(size=6),
    ))
    if p in WHO_ANNUAL:
        fig_spark.add_hline(
            y=WHO_ANNUAL[p],
            line_dash="dot",
            line_color=POLLUTANT_COLOR[p],
            opacity=0.4,
            annotation_text=f"WHO {p}",
            annotation_font_size=9,
        )

fig_spark.update_layout(
    height=260,
    margin=dict(t=10, b=10, l=10, r=60),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.1),
    xaxis=dict(tickformat="%d %b"),
)
st.plotly_chart(fig_spark,  width="stretch", key="fig_spark")

st.divider()

# --------------------------------------------------
# SECTION 6 — RECOMMENDED ACTION
# --------------------------------------------------

st.markdown(f'<p class="section-title">{tr("🎯 Recommended action for today")}</p>',
            unsafe_allow_html=True)

# Find worst zone by forecast ratio
zone_ratios = (
    fc_df[fc_df["Pollutant"].isin(["PM2.5", "PM10", "NO2"])]
    .groupby("Zone")["Ratio"]
    .mean()
    .sort_values(ascending=False)
)

ZONE_ACTIONS = {
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

if not zone_ratios.empty:
    worst_zone      = zone_ratios.index[0]
    worst_ratio     = zone_ratios.iloc[0]
    zone_meta       = ZONE_META.get(worst_zone, {})
    action_text     = ZONE_ACTIONS.get(worst_zone, "Monitor closely.")

    urgency_color   = "#e74c3c" if worst_ratio > 1.5 else \
                      "#f39c12" if worst_ratio > 1.0 else "#2ecc71"
    urgency_label   = "High priority" if worst_ratio > 1.5 else \
                      "Moderate" if worst_ratio > 1.0 else "Routine monitoring"

    st.markdown(
        f"""
        <div style="border:2px solid {zone_meta.get('border','#ccc')};
                    border-radius:12px;padding:18px 20px;
                    background:linear-gradient(135deg,
                    {zone_meta.get('color','#888')}15,
                    {zone_meta.get('color','#888')}05);">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                <span style="font-size:1.6rem">{zone_meta.get('icon','🏙️')}</span>
                <div>
                    <div style="font-weight:700;font-size:1rem;color:#2c3e50">
                        Priority zone: {worst_zone}
                    </div>
                    <span style="background:{urgency_color};color:white;
                                 padding:2px 10px;border-radius:10px;
                                 font-size:0.75rem;font-weight:600">
                        {urgency_label} · {worst_ratio:.1f}× WHO avg
                    </span>
                </div>
            </div>
            <div style="color:#555;font-size:0.9rem;line-height:1.6">
                {action_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()



st.markdown(f'<p class="section-title">{tr("📄 Download Report")}</p>',
            unsafe_allow_html=True)
 
col_pdf1, col_pdf2 = st.columns([2, 3])
with col_pdf1:
    # Build current_values for today
    today_means_dict = {
        p: float(df[df["Date"] == latest_date][p].mean())
        for p in ["PM2.5", "PM10", "NO2", "SO2"]
    }
 
    # Build zone action text
    zone_ratios_pdf = (
        fc_df[fc_df["Pollutant"].isin(["PM2.5", "PM10", "NO2"])]
        .groupby("Zone")["Ratio"]
        .mean()
        .sort_values(ascending=False)
    )
    worst_zone_pdf  = zone_ratios_pdf.index[0] if not zone_ratios_pdf.empty else "Urban"
    action_text_pdf = ZONE_ACTIONS.get(worst_zone_pdf, "Monitor closely.")
 
    pdf_bytes = generate_daily_report(
        latest_date    = latest_date,
        current_values = today_means_dict,
        fc_df          = fc_df,
        zone_action    = action_text_pdf,
        worst_zone     = worst_zone_pdf,
        who_annual     = WHO_ANNUAL,
        eu_annual      = EU_ANNUAL,
        alert_limits   = ALERT_LIMITS,
    )
 
    st.download_button(
        label    = tr("📄 Download Daily Alert Report (PDF)"),
        data     = pdf_bytes,
        file_name= f"daily_alert_{latest_date.strftime('%Y%m%d')}.pdf",
        mime     = "application/pdf",
        type     = "primary",
        width="stretch",
    )

with col_pdf2:
    st.caption(
        tr("One-page summary: today's city-wide averages, "
           "tomorrow's EU Directive exceedances, "
           "and zone-level recommended action.")
    )

    
 
st.divider()
# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    f"{tr('Data: Basque Government (CC BY 4.0) + Open-Meteo (CC BY 4.0)')} · "
    f"{tr('Forecasts: XGBoost models (test R²=0.39–0.56)')} · "
    f"{tr('Last pipeline run')}: {latest_date.strftime('%d %b %Y')} · "
    f"{tr('Next update: ~06:00 UTC daily')}"
)