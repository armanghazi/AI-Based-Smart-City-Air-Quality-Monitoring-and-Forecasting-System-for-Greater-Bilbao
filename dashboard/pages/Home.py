# pages/Home.py
# Platform homepage: Hero + Alert banner + Quick status + Zone cards + Trends + Nav tiles
# Extracted from app.py during st.navigation refactor.

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path
from datetime import timedelta

from config import (
    load_data,
    WHO_ANNUAL, WHO_SO2_DAILY, POLLUTANT_COLOR,
    ZONE_META, who_delta, EU_ANNUAL, ALERT_LIMITS,
)
from i18n_auto import tr

# --------------------------------------------------
# CSS design system — injected here so it applies to
# the Home page. Other pages that need it should also
# call this block (or move inject_css() to config.py).
# --------------------------------------------------
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
    --atm-1:      #0ea5b5;
    --atm-2:      #2563eb;
    --good:       #16a34a;
    --warn:       #d97706;
    --bad:        #dc2626;
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; color: var(--ink); }
h1, h2, h3, h4 { letter-spacing: -0.02em; font-weight: 600; }
.hero {
    position: relative;
    background:
        radial-gradient(1000px 400px at 20% -10%, rgba(14,165,181,0.18), transparent 55%),
        radial-gradient(800px 360px at 92% 0%, rgba(99,102,241,0.20), transparent 50%),
        linear-gradient(160deg, #4b63eb 0%, #3b4fc4 100%);
    border-radius: 18px; padding: 1.9rem 2.4rem 1.7rem;
    overflow: hidden; margin-bottom: 1rem; text-align: center;
}
.hero::after {
    content: ""; position: absolute; inset: 0;
    background-image: repeating-linear-gradient(115deg, rgba(255,255,255,0.035) 0 1px, transparent 1px 38px);
    pointer-events: none;
}
.hero-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.28em; text-transform: uppercase; color: #c7d2fe !important; margin: 0 0 0.6rem; }
.hero-title   { font-size: 2.6rem; font-weight: 700; line-height: 1.05; color: #ffffff !important; margin: 0; }
.hero-sub     { color: #dbe4ff !important; font-size: 1.02rem; margin-top: 0.7rem; white-space: nowrap; }
.hero-meta    { display: flex; gap: 1.6rem; flex-wrap: wrap; margin-top: 1.5rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #e0e7ff !important; justify-content: center; }
.hero-meta b  { color: #ffffff; font-weight: 600; }
.alert { border-radius: 12px; padding: 14px 20px; font-weight: 500; font-size: 0.96rem; margin: 1rem 0 0.4rem; display: flex; align-items: center; gap: 10px; border: 1px solid transparent; }
.alert a { color: inherit; font-weight: 600; text-decoration: underline; }
.alert-good { background: linear-gradient(135deg,#2ecc71,#27ae60); color: white; box-shadow: 0 2px 8px #2ecc7144; }
.alert-warn  { background: linear-gradient(135deg,#f39c12,#e67e22); color: white; box-shadow: 0 2px 8px #f39c1244; }
.alert-bad   { background: linear-gradient(135deg,#e74c3c,#c0392b); color: white; box-shadow: 0 2px 8px #e74c3c44; }
.eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--mist); margin-bottom: 0.2rem; }
.zone-card { border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; background: var(--paper); height: 100%; transition: border-color 0.2s, box-shadow 0.2s; }
.zone-card:hover { border-color: var(--atm-1); box-shadow: 0 8px 24px -12px rgba(14,165,181,0.4); }
.zone-head { font-size: 1.15rem; font-weight: 600; margin-bottom: 3px; }
.zone-desc { color: var(--mist); font-size: 0.8rem; margin-bottom: 6px; line-height: 1.4; }
.zone-loc  { color: #94a3b8; font-size: 0.74rem; margin-bottom: 6px; font-family: 'IBM Plex Mono', monospace; }
.zone-spatial { color: #0ea5b5; font-size: 0.77rem; margin-bottom: 8px; font-style: italic; line-height: 1.4; }
.zone-row  { display:flex; justify-content:space-between; font-size: 0.84rem; padding: 2px 0; }
.zone-row .k { color: var(--mist); }
.zone-row .v { font-family: 'IBM Plex Mono', monospace; font-weight: 500; }
.nav-tile { border: 1px solid var(--line); border-radius: 14px; padding: 1.4rem 1.1rem; text-align: left; background: var(--paper); transition: all 0.22s cubic-bezier(.4,0,.2,1); height: 100%; }
.nav-tile:hover { border-color: var(--atm-2); transform: translateY(-3px); box-shadow: 0 14px 30px -18px rgba(37,99,235,0.5); }
.nav-icon  { font-size: 1.7rem; }
.nav-title { font-weight: 600; font-size: 0.98rem; margin: 8px 0 4px; }
.nav-desc  { color: var(--mist); font-size: 0.8rem; line-height: 1.45; }
div[data-testid="stMetric"] { background: var(--haze); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; }
div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; }
hr { border-color: var(--line); }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df           = load_data()
latest_date  = df["Date"].max()
station_list = sorted(df["station"].unique().tolist())
n_records    = len(df)
n_years      = df["Year"].nunique()

# --------------------------------------------------
# Cookie manager — favourite station
# --------------------------------------------------
_cookies = None
try:
    from streamlit_cookies_manager import EncryptedCookieManager
    _cookies = EncryptedCookieManager(
        prefix="smart_city_air",
        password=st.secrets.get("cookie_password", "local-dev-only-change-me"),
    )
    if not _cookies.ready():
        st.stop()
except ImportError:
    _cookies = None

if "fav_station" not in st.session_state:
    saved = _cookies.get("fav_station") if _cookies else None
    st.session_state.fav_station = saved if saved in station_list else station_list[0]

# --------------------------------------------------
# Quick forecast — alert banner only (EU thresholds)
# --------------------------------------------------
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


exceed       = _get_tomorrow_exceedances()
n_exc        = len(exceed)
# D-1 rule: latest_date is yesterday; forecast is for latest_date + 1
forecast_str = (latest_date + timedelta(days=1)).strftime("%d %b %Y")

# --------------------------------------------------
# Hero
# --------------------------------------------------
st.markdown(f"""
<div class="hero">
    <p class="hero-eyebrow">GeoAI Smart City Platform · Greater Bilbao · Bizkaia</p>
    <h1 class="hero-title">{tr("Air Quality Intelligence for Greater Bilbao")}</h1>
    <p class="hero-sub">
        {tr("Monitoring and next-day forecasting across the region's air — seven stations, four pollutants, updated automatically every morning.")}
    </p>
    <div class="hero-meta">
        <span><b>7</b> {tr("stations")} · <b>5</b> {tr("zones")}</span>
        <span><b>{n_years}</b> {tr("years")} · {n_records:,} {tr("daily records")}</span>
        <span><b>GeoAI spatial</b> · 4 {tr("notebooks")} · 35 {tr("features")}</span>
        <span><b>{tr("Next-day")}</b> XGBoost {tr("forecast")}</span>
        <span><b>WHO 2021</b> + EU Directive 2008/50/EC</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Alert strip
if n_exc == 0:
    st.markdown(
        f'<div class="alert alert-good">✅ '
        f'{tr("All stations within EU Directive limits — no legal exceedances forecast for")} {forecast_str}.</div>',
        unsafe_allow_html=True)
elif n_exc <= 4:
    s_exc = list({e["station"].split("_")[0] for e in exceed})
    st.markdown(
        f'<div class="alert alert-warn">⚠️ '
        f'{n_exc} {tr("EU Directive exceedance")}{"s" if n_exc>1 else ""} '
        f'{tr("forecast for")} {forecast_str} · {tr("Stations")}: {", ".join(s_exc)} · '
        f'<a href="/Daily_Briefing">{tr("Open the daily briefing")} →</a></div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="alert alert-bad">🚨 '
        f'{n_exc} {tr("EU Directive exceedances forecast for")} {forecast_str} · '
        f'{tr("multiple zones affected")} · '
        f'<a href="/Daily_Briefing">{tr("Open the daily briefing")} →</a></div>',
        unsafe_allow_html=True)

st.write("")
st.write("")

# --------------------------------------------------
# Quick status
# --------------------------------------------------
st.markdown(f'<p class="eyebrow">{tr("Latest reading (D-1)")}</p>', unsafe_allow_html=True)
st.markdown("### " + tr("Latest readings across the network") + f" — {latest_date.strftime('%d %b %Y')}")

col_fav, col_snap = st.columns([2, 5], gap="large")

with col_fav:
    default_idx = (
        station_list.index(st.session_state.fav_station)
        if st.session_state.fav_station in station_list else 0
    )
    selected_fav = st.selectbox(
        tr("Your default station"),
        options=station_list,
        index=default_idx,
        help=tr("Remembered on this device for your next visit."),
    )
    if st.button(tr("Save as default"), type="primary", use_container_width=True):
        st.session_state.fav_station = selected_fav
        if _cookies is not None:
            _cookies["fav_station"] = selected_fav
            _cookies.save()
            st.success(f"{selected_fav} {tr('saved')}.")
        else:
            st.info(tr("Saved for this session."))

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

# --------------------------------------------------
# Project Assistant — refined premium feature card
# --------------------------------------------------
st.markdown(f'<p class="eyebrow">{tr("AI capability")}</p>', unsafe_allow_html=True)
st.markdown("### " + tr("Ask the Project Assistant"))

card_left, card_right = st.columns([5.2, 1.3], gap="medium")

with card_left:
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg, rgba(14,165,181,0.045) 0%, rgba(37,99,235,0.055) 100%);
        border:1px solid rgba(148,163,184,0.28);
        border-radius:14px;
        padding:1.15rem 1.25rem;
        min-height:112px;
        box-shadow:0 10px 24px -20px rgba(15,23,42,0.22);
    ">
        <div style="display:flex;align-items:flex-start;gap:0.95rem">
            <div style="
                width:46px;
                height:46px;
                border-radius:12px;
                background:rgba(37,99,235,0.08);
                border:1px solid rgba(37,99,235,0.12);
                display:flex;
                align-items:center;
                justify-content:center;
                flex-shrink:0;
                font-size:1.2rem;
            ">
                ✦
            </div>

            <div style="min-width:0">
                <div style="
                    display:inline-flex;
                    align-items:center;
                    gap:0.35rem;
                    font-size:0.68rem;
                    font-weight:600;
                    text-transform:uppercase;
                    letter-spacing:0.06em;
                    color:#2563eb;
                    background:rgba(37,99,235,0.08);
                    border:1px solid rgba(37,99,235,0.10);
                    border-radius:999px;
                    padding:0.22rem 0.55rem;
                    margin-bottom:0.45rem;
                ">
                    AI Assistant
                </div>

                <p style="
                    font-weight:600;
                    font-size:1rem;
                    color:#0c1521;
                    margin:0 0 0.18rem 0;
                    letter-spacing:-0.01em;
                ">
                    {tr("Ask about data, forecasts, GIS, or methodology")}
                </p>

                <p style="
                    font-size:0.82rem;
                    color:#5b7185;
                    margin:0;
                    line-height:1.55;
                    max-width:62ch;
                ">
                    {tr("Use natural language to explore air quality trends, station differences, forecasting logic, and GeoAI findings across the platform.")}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with card_right:
    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
    if st.button(
        tr("Open AI Assistant") + " →",
        key="home_assistant_cta",
        type="primary",
        use_container_width=True,
    ):
        st.switch_page("pages/8_Project_Assistant.py")

st.divider()
# --------------------------------------------------
# Environmental zones
# --------------------------------------------------
st.markdown(f'<p class="eyebrow">{tr("The network, by character")}</p>', unsafe_allow_html=True)
st.markdown("### " + tr("Five environmental zones"))
st.caption(
    tr("Each station sits in a zone defined by its dominant emission source — "
       "traffic, industry, port, coast, or refinery. Latest-year averages shown.")
)

zone_summary = (
    df[df["Year"] == int(df["Year"].max())]
    .groupby("Zone")[["PM2.5", "PM10", "NO2", "SO2"]]
    .mean().round(1)
)

# Spatial driver context from GIS analysis (notebooks 10a/10b/10c)
ZONE_SPATIAL = {
    "Urban": tr("Road density 19,060 m/km² · 501 m from city centre → strong traffic-driven NO₂ pressure"),
    "Industrial": tr("Near AP-8 corridor · elevated industrial land use within 1 km"),
    "Port": tr("784 m from Port of Bilbao · TRI 445 m provides partial terrain-dispersion buffering"),
    "Coastal": tr("Lowest road density (9,933 m/km²) · 2.6 km from coast → NW sea-breeze flushing"),
    "Refinery": tr("2.4 km from Petronor · TRI 343 m plus coastal setting → dispersion advantage"),
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

# --------------------------------------------------
# City-wide insights
# --------------------------------------------------
st.markdown(f'<p class="eyebrow">{tr("A decade in view")}</p>', unsafe_allow_html=True)
st.markdown("### " + tr("City-wide trends"))

col_left, col_right = st.columns([3, 2], gap="large")
latest_year = int(df["Year"].max())

with col_left:
    st.markdown("#### " + tr("Annual mean concentration"))
    annual = df.groupby("Year")[["PM2.5", "PM10", "NO2"]].mean().reset_index()
    annual_long = annual.melt(id_vars="Year", var_name="Pollutant", value_name="Concentration")
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
    fig_trend.update_layout(
        dragmode=False, height=340, margin=dict(t=10, b=10, l=10, r=60),
        hovermode="x unified", legend=dict(orientation="h", y=1.1),
        font=dict(family="IBM Plex Sans"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_trend, width="stretch",
                    config={"scrollZoom": False, "displayModeBar": False},
                    key="home_trend_chart")

with col_right:
    st.markdown("#### " + tr("Station risk ranking"))
    station_latest = (
        df[df["Year"] == latest_year]
        .groupby(["station", "Zone"])[["PM2.5", "PM10", "NO2"]]
        .mean().reset_index()
    )

    def core_risk(row):
        ratios = [row["PM2.5"] / WHO_ANNUAL["PM2.5"],
                  row["PM10"]  / WHO_ANNUAL["PM10"],
                  row["NO2"]   / WHO_ANNUAL["NO2"]]
        return 100 * sum(ratios) / 3

    station_latest["Score"]   = station_latest.apply(core_risk, axis=1)
    station_latest            = station_latest.sort_values("Score", ascending=False)
    station_latest["Station"] = station_latest["station"].str.split("_").str[0]

    import plotly.graph_objects as go  # noqa: PLC0415 (local import OK here)
    fig_status = px.bar(
        station_latest, x="Score", y="Station",
        color="Zone",
        color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
        orientation="h", text="Score",
    )
    fig_status.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_status.add_vline(x=100, line_dash="dash", line_color="#94a3b8", opacity=0.7,
                         annotation_text="WHO", annotation_font_size=9)
    fig_status.update_layout(
        dragmode=False, height=360, margin=dict(t=40, b=10, l=10, r=30),
        showlegend=True,
        legend=dict(orientation="h", y=1.22, x=0, font=dict(size=9)),
        yaxis=dict(autorange="reversed"),
        xaxis_range=[0, max(station_latest["Score"].max() * 1.25, 250)],
        font=dict(family="IBM Plex Sans"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_status, width="stretch",
                    config={"scrollZoom": False, "displayModeBar": False},
                    key="home_risk_chart")

st.caption(
    tr("Risk score = mean of (concentration ÷ WHO 2021 limit) across PM2.5, PM10, NO₂, ×100. "
       "100 = exactly at the WHO guideline."))
st.divider()

# --------------------------------------------------
# Navigation tiles — updated paths match new filenames
# --------------------------------------------------
st.markdown(f'<p class="eyebrow">{tr("Where to next")}</p>', unsafe_allow_html=True)
st.markdown("### " + tr("Explore the platform"))

NAV = [
    {"icon": "🌅", "title": "Daily Briefing",
     "desc": "Status snapshot, next-day alerts, one-page PDF",
     "page": "pages/0_Daily_Briefing.py"},
    {"icon": "📡", "title": "Air Quality Monitoring",
     "desc": "Interactive GIS map · station comparison · SVI structural context",
     "page": "pages/1_Air_Quality_Monitoring.py"},
    {"icon": "📈", "title": "Temporal Trends",
     "desc": "Long-term patterns, seasonality, COVID impact",
     "page": "pages/2_Temporal_Trends.py"},
    {"icon": "🔮", "title": "Forecast Explorer",
     "desc": "Next-day XGBoost predictions with SHAP explanations",
     "page": "pages/5_Forecasting.py"},
    {"icon": "🗺️", "title": "Spatial Deep-Dive",
     "desc": "Station DNA · spatial drivers · terrain · wind transport",
     "page": "pages/3_GeoAI_Spatial_Analysis.py"},
    {"icon": "💨", "title": "Weather Drivers",
     "desc": "Wind transport · dispersion effects · seasonal patterns",
     "page": "pages/4_Weather_Drivers.py"},
    {"icon": "🏙️", "title": "Smart City Decision Support",
     "desc": "GeoAI spatial intelligence · wind transport · structural risk index",
     "page": "pages/6_Smart_City_Decision_Support.py"},
    {"icon": "📖", "title": "Methodology",
     "desc": "Coverage, model accuracy, known gaps, honest limitations",
     "page": "pages/7_Scope_and_Limitations.py"},
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

# --------------------------------------------------
# Footer
# --------------------------------------------------
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
           style="color:#0ea5b5;text-decoration:none">Portfolio</a>&nbsp;&nbsp;
        <a href="https://github.com/armanghazi"
           style="color:#0ea5b5;text-decoration:none">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )