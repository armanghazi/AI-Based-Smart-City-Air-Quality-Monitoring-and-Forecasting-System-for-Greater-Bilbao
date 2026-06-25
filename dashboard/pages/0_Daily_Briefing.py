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

from i18n_auto import language_selector, apply_lang_styles, tr

from forecast_utils import plotly_touch_config, PLOTLY_CONFIG

plotly_touch_config()  


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Daily Briefing",
    page_icon="🌅",
    layout="wide",
)
language_selector()
apply_lang_styles()
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
    st.plotly_chart(fig_hm,  width="stretch")


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
st.plotly_chart(fig_spark,  width="stretch")
   
st.plotly_chart(fig, config=PLOTLY_CONFIG)

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