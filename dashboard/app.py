import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

from config import (
    load_data,
    WHO_ANNUAL, WHO_SO2_DAILY, POLLUTANT_COLOR,
    ZONE_META, who_delta, EU_ANNUAL, ALERT_LIMITS
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Smart City Air Intelligence",
    page_icon="🌍",
    layout="wide",
)

# --------------------------------------------------
# GLOBAL CSS
# --------------------------------------------------

st.markdown("""
<style>
    /* Smooth card hover */
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 3px solid #3498db;
    }
    /* Nav card hover effect */
    .nav-card:hover { opacity: 0.92; transform: translateY(-1px); }
    /* Alert banner */
    .alert-banner-red {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white; border-radius: 12px; padding: 14px 20px;
        font-weight: 600; font-size: 0.95rem;
        box-shadow: 0 3px 10px #e74c3c33;
    }
    .alert-banner-yellow {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white; border-radius: 12px; padding: 14px 20px;
        font-weight: 600; font-size: 0.95rem;
        box-shadow: 0 3px 10px #f39c1233;
    }
    .alert-banner-green {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white; border-radius: 12px; padding: 14px 20px;
        font-weight: 600; font-size: 0.95rem;
        box-shadow: 0 3px 10px #2ecc7133;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df           = load_data()
latest_date  = df["Date"].max()
station_list = sorted(df["station"].unique().tolist())

# --------------------------------------------------
# COOKIE MANAGER — favourite station
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
# QUICK FORECAST — for homepage alert banner only
# Uses last available data row per station, all 4 models.
# --------------------------------------------------

MODELS_DIR = Path(__file__).parent.parent / "models"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]
WHO_LIMITS = {**WHO_ANNUAL, "SO2": WHO_SO2_DAILY}


@st.cache_resource
def _load_bundle(pollutant: str):
    prefix = pollutant.replace(".", "").lower()
    path   = MODELS_DIR / f"xgb_{prefix}_forecast.joblib"
    return joblib.load(path) if path.exists() else None


def _prepare_last_row(sdf: pd.DataFrame, feats: list) -> pd.DataFrame:
    """Minimal feature prep for homepage banner — mirrors forecast pages."""
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
        px_ = p.replace(".", "")
        col = f"{px_}_roll_mean_14"
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
def _get_tomorrow_exceedances() -> list[dict]:
    """Returns list of dicts for stations/pollutants exceeding WHO tomorrow."""
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
            ALERT_LIMITS = {"PM2.5": 25.0, "PM10": 40.0, "NO2": 40.0, "SO2": 125.0}
            limit = ALERT_LIMITS.get(pollutant)
            if limit and pred > limit:
                results.append({
                    "station":   station,
                    "pollutant": pollutant,
                    "forecast":  pred,
                    "ratio":     pred / limit,
                })
    return results


exceed = _get_tomorrow_exceedances()
n_exc  = len(exceed)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div style="padding:1.5rem 0 0.3rem">
    <h1 style="margin:0;font-size:2rem;color:#1a1a2e">
        🌍 Smart City Air Intelligence Platform
    </h1>
    <p style="color:#666;margin-top:5px;font-size:1rem">
        Greater Bilbao · Real-time air quality monitoring,
        analysis & next-day forecasting · 2015–present
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# ALERT BANNER (new)
# --------------------------------------------------

from datetime import timedelta
tomorrow_str = (latest_date + timedelta(days=1)).strftime("%d %b %Y")

if n_exc == 0:
    st.markdown(
        f'<div class="alert-banner-green">'
        f'✅ &nbsp; All stations within EU Directive — '
        f'no exceedances forecast for {tomorrow_str}'
        f'</div>',
        unsafe_allow_html=True,
    )
elif n_exc <= 4:
    stations_exc = list({e["station"].split("_")[0] for e in exceed})
    st.markdown(
        f'<div class="alert-banner-yellow">'
        f'⚠️ &nbsp; {n_exc} EU Directive exceedance{"s" if n_exc>1 else ""} '
        f'forecast for {tomorrow_str} · '
        f'Stations: {", ".join(stations_exc)} · '
        f'<a href="/Daily_Briefing" style="color:white;text-decoration:underline">'
        f'View full briefing →</a>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="alert-banner-red">'
        f'🚨 &nbsp; {n_exc} EU Directive exceedances forecast for {tomorrow_str} · '
        f'Multiple zones at risk · '
        f'<a href="/Daily_Briefing" style="color:white;text-decoration:underline">'
        f'View full briefing →</a>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# --------------------------------------------------
# FAVOURITE STATION selector
# --------------------------------------------------

fav_col1, fav_col2 = st.columns([3, 1])
with fav_col1:
    _default_idx = (
        station_list.index(st.session_state.fav_station)
        if st.session_state.fav_station in station_list else 0
    )
    _selected_fav = st.selectbox(
        "⭐ Your default monitoring station "
        "(remembered on this device for next visit)",
        options=station_list,
        index=_default_idx,
    )
with fav_col2:
    st.write("")
    st.write("")
    if st.button("💾 Save as default",  width="stretch"):
        st.session_state.fav_station = _selected_fav
        if _cookies is not None:
            _cookies["fav_station"] = _selected_fav
            _cookies.save()
            st.success(f"Saved {_selected_fav} as your default station.")
        else:
            st.info(f"Saved {_selected_fav} for this session.")

st.divider()

# --------------------------------------------------
# LATEST SNAPSHOT
# --------------------------------------------------

latest_df    = df[df["Date"] == latest_date]
latest_means = (
    latest_df
    .groupby("station")[["PM2.5", "PM10", "NO2", "SO2"]]
    .mean()
    .mean()
)

st.markdown(
    f"### 📡 Latest Snapshot — {latest_date.strftime('%d %b %Y')}",
)
st.caption("City-wide average across all stations · WHO 2021 + EU Directive 2008/50/EC")

c1, c2, c3, c4 = st.columns(4)
for col, poll in zip([c1, c2, c3, c4], ["PM2.5", "PM10", "NO2", "SO2"]):
    val                      = latest_means.get(poll, 0)
    delta_label, delta_color = who_delta(val, poll)
    eu_lim                   = EU_ANNUAL.get(poll)
    eu_str                   = f"EU: {val/eu_lim:.1f}×" if eu_lim else ""
    col.metric(
        label      = f"{poll}  {eu_str}",
        value      = f"{val:.1f} µg/m³",
        delta      = delta_label,
        delta_color= delta_color,
        help       = f"WHO limit: {WHO_ANNUAL.get(poll, WHO_SO2_DAILY)} µg/m³"
                     + (f"  |  EU limit: {eu_lim} µg/m³" if eu_lim else ""),
    )

st.divider()

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.markdown("### 📊 Dataset Overview")

total_records  = len(df)
total_stations = df["station"].nunique()
date_range     = f"{df['Date'].min().strftime('%Y')} – {df['Date'].max().strftime('%Y')}"
total_years    = df["Year"].nunique()

o1, o2, o3, o4 = st.columns(4)
o1.metric("Total daily records",  f"{total_records:,}")
o2.metric("Monitoring stations",  str(total_stations))
o3.metric("Years of data",        f"{total_years} yrs ({date_range})")
o4.metric("Pollutants tracked",   "4  (PM2.5, PM10, NO₂, SO₂)")

st.divider()

# --------------------------------------------------
# UNDERSTANDING THE INDICATORS
# --------------------------------------------------

st.markdown("### 📖 Understanding the indicators")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        <div style="border:2px solid #3498db;border-radius:12px;padding:14px;">
        <div style="font-weight:700;font-size:1rem;color:#2c3e50">
            🌍 Air Quality Index (AQI)
        </div>
        <div style="font-size:0.82rem;color:#666;margin:6px 0">
            "Is the air safe to be outside today?" — easy 6-level public scale.
        </div>
        <div style="display:flex;gap:3px;margin-top:6px">
            <span style="flex:1;background:#50f0e6;height:14px;border-radius:2px"></span>
            <span style="flex:1;background:#50ccaa;height:14px"></span>
            <span style="flex:1;background:#f0e641;height:14px"></span>
            <span style="flex:1;background:#ff5050;height:14px"></span>
            <span style="flex:1;background:#960032;height:14px"></span>
            <span style="flex:1;background:#7d2181;height:14px;border-radius:2px"></span>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.68rem;color:#999;margin-top:2px">
            <span>Good</span><span>Extremely poor</span>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div style="border:2px solid #9b59b6;border-radius:12px;padding:14px;">
        <div style="font-weight:700;font-size:1rem;color:#2c3e50">
            🎯 WHO Risk Score
        </div>
        <div style="font-size:0.82rem;color:#666;margin:6px 0">
            "How far from the ideal WHO health target?" — strict, for analysis.
        </div>
        <div style="display:flex;gap:3px;margin-top:6px">
            <span style="flex:1;background:#2ecc71;height:14px;
                         border-radius:2px 0 0 2px"></span>
            <span style="flex:1;background:#f39c12;height:14px"></span>
            <span style="flex:1;background:#e74c3c;height:14px;
                         border-radius:0 2px 2px 0"></span>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.68rem;color:#999;margin-top:2px">
            <span>Below WHO</span><span>&gt;2× WHO</span>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Both appear across the dashboard. "
    "AQI flags daily public health risk; WHO Risk Score measures distance from strict ideals. "
    "Most European cities routinely exceed WHO targets."
)

# --------------------------------------------------
# ENVIRONMENTAL ZONE OVERVIEW
# --------------------------------------------------

st.markdown("### 🗺️ Environmental Zone Overview")
st.caption(
    "Five zones derived from spatial analysis of emission sources "
    "and pollution profiles · Latest year averages shown"
)

latest_year  = int(df["Year"].max())
zone_summary = (
    df[df["Year"] == latest_year]
    .groupby("Zone")[["PM2.5", "PM10", "NO2", "SO2"]]
    .mean()
    .round(2)
)


def render_zone_card(zone_name, meta, zone_summary, df):
    z_data      = zone_summary.loc[zone_name] if zone_name in zone_summary.index else None
    stations_in = df[df["Zone"] == zone_name]["station"].unique().tolist()
    short_names = ", ".join([s.split("_")[0] for s in stations_in])

    if z_data is not None:
        pm25_val = z_data["PM2.5"]
        pm10_val = z_data["PM10"]
        no2_val  = z_data["NO2"]
        so2_val  = z_data["SO2"]
        key_poll = meta["key_pollutant"]
        key_val  = z_data[key_poll]
        who_lim  = WHO_ANNUAL.get(key_poll)
        vs_who   = f"{key_val / who_lim:.1f}×" if who_lim else "—"
    else:
        pm25_val = pm10_val = no2_val = so2_val = 0.0
        vs_who   = "—"
        key_poll = meta["key_pollutant"]

    st.markdown(
        f"""
        <div style="
            border-left:5px solid {meta['border']};
            background:linear-gradient(135deg,{meta['color']}18,{meta['color']}06);
            border-radius:10px;padding:16px 18px;margin-bottom:8px;height:100%;
        ">
            <div style="font-size:20px;margin-bottom:4px">
                {meta['icon']} <strong>{zone_name}</strong>
            </div>
            <div style="color:#555;font-size:11px;margin-bottom:6px">
                {meta['description']}
            </div>
            <div style="color:#777;font-size:11px;margin-bottom:8px">
                📍 {short_names}
            </div>
            <table style="width:100%;font-size:12px;border-collapse:collapse">
                <tr>
                    <td style="color:#777;padding:2px 0">PM2.5</td>
                    <td style="text-align:right;font-weight:600">{pm25_val:.1f} µg/m³</td>
                </tr>
                <tr>
                    <td style="color:#777;padding:2px 0">PM10</td>
                    <td style="text-align:right;font-weight:600">{pm10_val:.1f} µg/m³</td>
                </tr>
                <tr>
                    <td style="color:#777;padding:2px 0">NO₂</td>
                    <td style="text-align:right;font-weight:600">{no2_val:.1f} µg/m³</td>
                </tr>
                <tr>
                    <td style="color:#777;padding:2px 0">SO₂</td>
                    <td style="text-align:right;font-weight:600">{so2_val:.1f} µg/m³</td>
                </tr>
                <tr>
                    <td style="color:#777;padding:2px 0">
                        Key ({key_poll}) vs WHO
                    </td>
                    <td style="text-align:right;font-weight:600;
                               color:{meta['color']}">{vs_who}</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


zones_list = list(ZONE_META.items())
row1       = zones_list[:3]
row2       = zones_list[3:]

cols_r1 = st.columns(3)
for idx, (zone_name, meta) in enumerate(row1):
    with cols_r1[idx]:
        render_zone_card(zone_name, meta, zone_summary, df)

if row2:
    cols_r2 = st.columns(3)
    offsets = [0, 1] if len(row2) == 2 else [1]
    for i, (zone_name, meta) in zip(offsets, row2):
        with cols_r2[i]:
            render_zone_card(zone_name, meta, zone_summary, df)

st.divider()

# --------------------------------------------------
# CITY-WIDE TREND + STATION STATUS
# --------------------------------------------------

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### 📈 City-wide Annual Trend")

    annual = (
        df
        .groupby("Year")[["PM2.5", "PM10", "NO2"]]
        .mean()
        .reset_index()
    )
    annual_long = annual.melt(
        id_vars="Year", var_name="Pollutant", value_name="Concentration"
    )
    fig_trend = px.line(
        annual_long,
        x="Year", y="Concentration",
        color="Pollutant",
        markers=True,
        color_discrete_map=POLLUTANT_COLOR,
        labels={"Concentration": "µg/m³"},
    )
    for poll, limit in WHO_ANNUAL.items():
        fig_trend.add_hline(
            y=limit, line_dash="dot",
            line_color=POLLUTANT_COLOR[poll],
            opacity=0.4,
            annotation_text=f"WHO {poll}",
            annotation_font_size=9,
            annotation_position="right",
        )
    fig_trend.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=60),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_trend,  width="stretch")

with col_right:
    st.markdown("#### 🏙️ Station Status — Latest Year")

    station_latest = (
        df[df["Year"] == latest_year]
        .groupby(["station", "Zone"])[["PM2.5", "PM10", "NO2"]]
        .mean()
        .reset_index()
    )

    def core_risk(row):
        ratios = [
            row["PM2.5"] / WHO_ANNUAL["PM2.5"],
            row["PM10"]  / WHO_ANNUAL["PM10"],
            row["NO2"]   / WHO_ANNUAL["NO2"],
        ]
        score = 100 * sum(ratios) / 3
        if score < 100:   return score, "Below WHO"
        elif score < 200: return score, "1–2× WHO"
        return score, ">2× WHO"

    station_latest[["Score", "Status"]] = station_latest.apply(
        lambda r: pd.Series(core_risk(r)), axis=1
    )
    station_latest          = station_latest.sort_values("Score", ascending=False)
    station_latest["Station"] = station_latest["station"].str.split("_").str[0]

    fig_status = px.bar(
        station_latest,
        x="Score", y="Station",
        color="Zone",
        color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
        orientation="h",
        labels={"Score": "Core Risk Score"},
        text="Score",
    )
    fig_status.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_status.add_vline(x=100, line_dash="dash", line_color="#f39c12", opacity=0.6)
    fig_status.add_vline(x=200, line_dash="dash", line_color="#e74c3c", opacity=0.6)
    fig_status.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=40),
        showlegend=True,
        legend=dict(orientation="h", y=1.08, font=dict(size=10)),
        xaxis_range=[0, max(station_latest["Score"].max() * 1.2, 250)],
    )
    st.plotly_chart(fig_status,  width="stretch")

st.divider()

# --------------------------------------------------
# NAVIGATION CARDS
# --------------------------------------------------

st.markdown("### 🧭 Navigate to Module")
st.caption(
    "Six analytical modules — from raw monitoring to ML forecasting "
    "and Smart City decision support"
)

NAV_MODULES = [
    {
        "icon": "🌅", "title": "Daily Briefing",
        "description": "Today's status · Tomorrow's alerts · 7-day trend",
        "page": "pages/0_Daily_Briefing.py",
        "color": "#16a085", "border": "#0e6655",
    },
    {
        "icon": "🗺️", "title": "Air Quality Monitoring",
        "description": "Interactive map · Station comparison · WHO risk levels",
        "page": "pages/1_Air_Quality_Monitoring.py",
        "color": "#2980b9", "border": "#1a6fa0",
    },
    {
        "icon": "📈", "title": "Temporal Trends",
        "description": "Annual trends · Seasonality · COVID impact analysis",
        "page": "pages/2_Temporal_Trends.py",
        "color": "#27ae60", "border": "#1e8449",
    },
    {
        "icon": "🌍", "title": "Urban Risk Index",
        "description": "WHO & EU risk scoring · Heatmaps · Station rankings",
        "page": "pages/3_Urban_Risk_Index.py",
        "color": "#c0392b", "border": "#a93226",
    },
    {
        "icon": "🌤️", "title": "Weather Drivers",
        "description": "Wind · Rain · Temperature · Lag analysis",
        "page": "pages/4_Weather_Drivers_&_Air_Pollution_Dynamics.py",
        "color": "#d35400", "border": "#b94600",
    },
    {
        "icon": "🔮", "title": "Forecasting",
        "description": "Next-day ML predictions · XGBoost · SHAP explainability",
        "page": "pages/5_Forecasting.py",
        "color": "#8e44ad", "border": "#6c3483",
    },
    {
        "icon": "🏛️", "title": "Smart City Decision Support",
        "description": "Risk prioritization · Zone actions · CSV export",
        "page": "pages/6_Smart_City_Decision_Support.py",
        "color": "#f1c40f", "border": "#d4ac0d",
    },
    {
        "icon": "📋", "title": "Scope & Limitations",
        "description": "Coverage · Model accuracy · Standards · Known gaps",
        "page": "pages/7_Scope_and_Limitations.py",
        "color": "#7f8c8d", "border": "#636e72",
    },
]

N_COLS = 4
for i in range(0, len(NAV_MODULES), N_COLS):
    nav_cols    = st.columns(N_COLS)
    row_modules = NAV_MODULES[i:i + N_COLS]
    for j, module in enumerate(row_modules):
        idx = i + j
        with nav_cols[j]:
            st.markdown(
                f"""
                <div style="
                    border:2px solid {module['border']};
                    border-radius:12px;
                    padding:16px 14px 10px;
                    text-align:center;
                    background:linear-gradient(135deg,
                        {module['color']}14,{module['color']}05);
                    margin-bottom:8px;
                    transition:0.2s;
                ">
                    <div style="font-size:2rem;margin-bottom:5px">
                        {module['icon']}
                    </div>
                    <div style="font-weight:700;font-size:0.9rem;
                                color:#2c3e50;margin-bottom:5px">
                        {module['title']}
                    </div>
                    <div style="color:#777;font-size:0.75rem;line-height:1.4">
                        {module['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                f"Open →",
                key=f"nav_{idx}",
                 width="stretch",
            ):
                st.switch_page(module["page"])

st.divider()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown(
        "**🌬️ Air Quality Data**  \n"
        "Basque Government Air Quality Network  \n"
        "*(Red de Control de Calidad del Aire)*  \n"
        "7 stations · Greater Bilbao · © Gobierno Vasco · CC BY 4.0  \n"
        "WHO 2021 guidelines · EU Directive 2008/50/EC"
    )

with col_s2:
    st.markdown(
        "**🌤️ Meteorological Data**  \n"
        "Open-Meteo · [open-meteo.com](https://open-meteo.com)  \n"
        "Historical Weather API · CC BY 4.0  \n"
        "Temperature, Humidity, Precipitation, Wind Speed, Direction"
    )

with st.sidebar:
    st.divider()
    st.markdown(
        """
        <div style="font-size:0.9rem;color:#888;line-height:1.6">
        <b style="color:#555">GeoAI Smart City Platform</b><br>
        Air quality monitoring & forecasting<br>
        Greater Bilbao · 2015–present<br><br>
        <b style="color:#555">Arman Ghaziaskari Naeini</b><br>
        GIS & Spatial Data Science<br><br>
        <a href="https://armanghazi.github.io/portfolio"
           style="color:#16a085;text-decoration:none">🔗 Portfolio</a> ·
        <a href="https://github.com/armanghazi"
           style="color:#16a085;text-decoration:none">💻 GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )