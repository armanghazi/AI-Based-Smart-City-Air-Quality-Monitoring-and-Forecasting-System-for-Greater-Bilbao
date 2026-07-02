"""
dashboard/pages/0_Daily_Briefing.py

Daily briefing page — the first thing a municipal manager opens every morning.
Shows: D-1 latest reading, zone breakdown, city-wide trends, next-day EAQI
forecast, 7-day sparklines, recommended action, and PDF export.

NOTE: st.set_page_config is intentionally absent — called once in app.py router.
"""

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EU_ANNUAL, POLLUTANT_COLOR, WHO_ANNUAL, WHO_SO2_DAILY,
    ZONE_META, center_tables, load_data, who_delta,
)
from aqi import compute_aqi_category
from i18n_auto import tr
from pdf_report import generate_daily_report

center_tables()

# ==================================================
# CSS — shared design system
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {
    --ink:  #0c1521; --slate: #1b2b3a; --mist: #5b7185;
    --line: #e3e8ee; --paper: #ffffff; --haze: #f4f7fa;
    --atm-1: #0ea5b5; --atm-2: #2563eb;
    --good: #16a34a; --warn: #d97706; --bad: #dc2626;
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; color: var(--ink); }
h1,h2,h3,h4 { letter-spacing: -0.02em; font-weight: 600; }

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--mist); margin-bottom: 0.2rem;
}
.section-title { font-size: 1.1rem; font-weight: 700; color: #2c3e50; margin: 0 0 4px 0; }

.alert { border-radius: 12px; padding: 14px 20px; font-weight: 500; font-size: 0.96rem;
         margin: 1rem 0 0.4rem; display: flex; align-items: center; gap: 10px;
         border: 1px solid transparent; }
.alert a { color: inherit; font-weight: 600; text-decoration: underline; }
.alert-good { background: linear-gradient(135deg,#2ecc71,#27ae60); color: white; box-shadow: 0 2px 8px #2ecc7144; }
.alert-warn  { background: linear-gradient(135deg,#f39c12,#e67e22); color: white; box-shadow: 0 2px 8px #f39c1244; }
.alert-bad   { background: linear-gradient(135deg,#e74c3c,#c0392b); color: white; box-shadow: 0 2px 8px #e74c3c44; }

.zone-card { border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px;
             background: var(--paper); height: 100%; }
.zone-card:hover { border-color: var(--atm-1); box-shadow: 0 8px 24px -12px rgba(14,165,181,0.4); }
.zone-head { font-size: 1.15rem; font-weight: 600; margin-bottom: 3px; }
.zone-desc { color: var(--mist); font-size: 0.8rem; margin-bottom: 6px; line-height: 1.4; }
.zone-loc  { color: #94a3b8; font-size: 0.74rem; margin-bottom: 6px; font-family: 'IBM Plex Mono', monospace; }
.zone-spatial { color: #0ea5b5; font-size: 0.77rem; margin-bottom: 8px; font-style: italic; line-height: 1.4; }
.zone-row { display: flex; justify-content: space-between; font-size: 0.84rem; padding: 2px 0; }
.zone-row .k { color: var(--mist); }
.zone-row .v { font-family: 'IBM Plex Mono', monospace; font-weight: 500; }

div[data-testid="stMetric"] { background: var(--haze); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; }
div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; }
hr { border-color: var(--line); }
</style>
""", unsafe_allow_html=True)

# ==================================================
# DATA LOADING
# ==================================================
df          = load_data()
latest_date = df["Date"].max()
n_years     = df["Date"].dt.year.nunique()
n_records   = len(df)
all_stations = sorted(df["station"].unique().tolist())
POLLUTANTS  = ["PM2.5", "PM10", "NO2", "SO2"]

# ==================================================
# FORECAST COMPUTATION (single source — EAQI-based)
# ==================================================
MODELS_DIR = Path(__file__).parent.parent.parent / "models"   # repo_root/models/


@st.cache_resource
def _load_bundle(pollutant: str):
    prefix = pollutant.replace(".", "").lower()
    path   = MODELS_DIR / f"xgb_{prefix}_forecast.joblib"
    return joblib.load(path) if path.exists() else None


from forecast_utils import prepare_features as _prepare_features_util

# Station codes computed after df is loaded — passed explicitly to _compute_forecasts
STATION_CODES = {s: i for i, s in enumerate(sorted(df["station"].unique()))}


@st.cache_data(ttl=3600)
def _compute_forecasts(_df: pd.DataFrame) -> pd.DataFrame:
    """Run all station×pollutant forecasts. Returns a DataFrame with EAQI info.
    _df is passed explicitly so st.cache_data can hash it properly.
    """
    _stations = sorted(_df["station"].unique().tolist())
    _sc = {s: i for i, s in enumerate(_stations)}
    rows = []
    for station in _stations:
        sdf = _df[_df["station"] == station].sort_values("Date")
        for pollutant in POLLUTANTS:
            bundle = _load_bundle(pollutant)
            if bundle is None:
                continue
            feats = bundle["features"]
            prep  = _prepare_features_util(sdf, feats, station_codes=_sc).dropna(subset=feats)
            if prep.empty:
                continue
            pred = max(float(bundle["model"].predict(prep[feats].iloc[[-1]])[0]), 0.0)
            cat  = compute_aqi_category(pollutant, pred)
            eu_lim = EU_ANNUAL.get(pollutant)
            rows.append({
                "station":    station,
                "Zone":       sdf["Zone"].iloc[-1],
                "Pollutant":  pollutant,
                "Forecast":   round(pred, 1),
                "EAQI_level": cat["level"]   if cat else 0,
                "EAQI_label": cat["label"]   if cat else "—",
                "EAQI_color": cat["color"]   if cat else "#888",
                "WHO_ratio":  pred / WHO_ANNUAL[pollutant] if pollutant in WHO_ANNUAL else None,
                "EU_ratio":   pred / eu_lim if eu_lim else None,
                "EU_exceeds": (pred > eu_lim) if eu_lim else False,
            })
    return pd.DataFrame(rows)


fc_df    = _compute_forecasts(df)
tomorrow = latest_date + timedelta(days=1)

# Guard: if no models found, fc_df is empty — add expected columns to avoid KeyErrors
_EXPECTED_COLS = ["station", "Zone", "Pollutant", "Forecast",
                  "EAQI_level", "EAQI_label", "EAQI_color",
                  "WHO_ratio", "EU_ratio", "EU_exceeds"]
for _col in _EXPECTED_COLS:
    if _col not in fc_df.columns:
        fc_df[_col] = None

# Alias for pdf_report.py compatibility — expects "Exceeds" not "EU_exceeds"
fc_df["Exceeds"] = fc_df["EU_exceeds"]

# Summary for alert banner — EAQI Poor+ (level >= 4)
poor_df = fc_df[fc_df["EAQI_level"].fillna(0) >= 4]
n_poor  = len(poor_df)

# ==================================================
# HEADER
# ==================================================
st.markdown(
    f"""
    <div style="padding:1rem 0 0.5rem">
        <h1 style="margin:0;font-size:1.9rem">🌅 {tr("Daily Air Quality Briefing")}</h1>
        <p style="color:#666;margin-top:4px;font-size:0.95rem">
            {tr("Greater Bilbao")} · {tr("Latest data")}:
            <b>{latest_date.strftime('%A, %d %B %Y')}</b> ·
            {tr("Updated automatically every day")}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# ALERT BANNER — EAQI-based
# ==================================================
forecast_str = tomorrow.strftime("%d %b %Y")

if n_poor == 0:
    st.markdown(
        f'<div class="alert alert-good">✅ '
        f'{tr("All stations forecast Good or Fairly good (EAQI) for")} {forecast_str}.</div>',
        unsafe_allow_html=True)
elif n_poor <= 4:
    s_poor      = list({r["station"].split("_")[0] for _, r in poor_df.iterrows()})
    worst_label = poor_df.loc[poor_df["EAQI_level"].idxmax(), "EAQI_label"]
    st.markdown(
        f'<div class="alert alert-warn">⚠️ '
        f'{n_poor} EAQI <b>{worst_label}</b> '
        f'{tr("forecast for")} {forecast_str} · '
        f'{tr("Stations")}: {", ".join(s_poor)}</div>',
        unsafe_allow_html=True)
else:
    worst_label = poor_df.loc[poor_df["EAQI_level"].idxmax(), "EAQI_label"]
    st.markdown(
        f'<div class="alert alert-bad">🚨 '
        f'{n_poor} EAQI <b>{worst_label}</b> {tr("forecasts for")} {forecast_str} · '
        f'{tr("multiple zones affected")}</div>',
        unsafe_allow_html=True)

st.write("")
st.divider()

# ==================================================
# LATEST READINGS (D-1)
# ==================================================
st.markdown(f'<p class="eyebrow">{tr("Latest reading (D-1)")}</p>', unsafe_allow_html=True)
st.markdown("### " + tr("Latest readings across the network") + f" — {latest_date.strftime('%d %b %Y')}")

latest_means = (
    df[df["Date"] == latest_date]
    .groupby("station")[POLLUTANTS].mean().mean()
)
c1, c2, c3, c4 = st.columns(4)
for col, poll in zip([c1, c2, c3, c4], POLLUTANTS):
    val = latest_means.get(poll, 0)
    delta_label, delta_color = who_delta(val, poll)
    eu_lim = EU_ANNUAL.get(poll)
    eu_str = f"  ·  EU {val/eu_lim:.1f}×" if eu_lim else ""
    col.metric(
        label=f"{poll}{eu_str}",
        value=f"{val:.1f} µg/m³",
        delta=delta_label, delta_color=delta_color,
        help=f"WHO {WHO_ANNUAL.get(poll, WHO_SO2_DAILY)} µg/m³"
             + (f"  ·  EU {eu_lim} µg/m³" if eu_lim else ""),
    )

st.divider()

# ==================================================
# ENVIRONMENTAL ZONES
# ==================================================
st.markdown(f'<p class="eyebrow">{tr("The network, by character")}</p>', unsafe_allow_html=True)
st.markdown("### " + tr("Five environmental zones"))
st.caption(tr(
    "Each station sits in a zone defined by its dominant emission source — "
    "traffic, industry, port, coast, or refinery. Latest-year averages shown."
))

zone_summary = (
    df[df["Year"] == int(df["Year"].max())]
    .groupby("Zone")[POLLUTANTS].mean().round(1)
)

ZONE_SPATIAL = {
    "Urban":      tr("Road density 19,060 m/km² · 501 m from city centre → structural NO₂ source"),
    "Industrial": tr("354 m from AP-8 motorway · industrial land use 10–21% within 1 km"),
    "Port":       tr("784 m from Port of Bilbao · TRI 445 m provides terrain dispersion buffer"),
    "Coastal":    tr("Lowest road density (9,933 m/km²) · 2.6 km coast → NW sea breeze flushing"),
    "Refinery":   tr("2.4 km from Petronor · TRI 343 m + coastal position → dispersion advantage"),
}


def _render_zone_card(zone_name: str, meta: dict) -> None:
    z = zone_summary.loc[zone_name] if zone_name in zone_summary.index else None
    short    = ", ".join(s.split("_")[0] for s in df[df["Zone"] == zone_name]["station"].unique())
    key_poll = meta["key_pollutant"]
    if z is not None:
        kv = z[key_poll]; lim = WHO_ANNUAL.get(key_poll)
        vs = f"{kv/lim:.1f}×" if lim else "—"
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
            <span class="k">{tr("Key")} ({key_poll}) vs WHO</span>
            <span class="v" style="color:{meta['color']}">{vs}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


zones_list = list(ZONE_META.items())
for row_start in range(0, len(zones_list), 3):
    cols = st.columns(3, gap="medium")
    for i, (zn, meta) in enumerate(zones_list[row_start:row_start + 3]):
        with cols[i]:
            _render_zone_card(zn, meta)

st.divider()

# ==================================================
# CITY-WIDE TRENDS
# ==================================================
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
                    key="db_trend_chart")

with col_right:
    st.markdown("#### " + tr("Station risk ranking"))
    station_latest = (
        df[df["Year"] == latest_year]
        .groupby(["station", "Zone"])[["PM2.5", "PM10", "NO2"]].mean().reset_index()
    )
    station_latest["Score"] = station_latest.apply(
        lambda r: 100 * (r["PM2.5"] / WHO_ANNUAL["PM2.5"] +
                         r["PM10"]  / WHO_ANNUAL["PM10"] +
                         r["NO2"]   / WHO_ANNUAL["NO2"]) / 3, axis=1)
    station_latest = station_latest.sort_values("Score", ascending=False)
    station_latest["Station"] = station_latest["station"].str.split("_").str[0]

    fig_risk = px.bar(
        station_latest, x="Score", y="Station",
        color="Zone",
        color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
        orientation="h", text="Score",
    )
    fig_risk.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_risk.add_vline(x=100, line_dash="dash", line_color="#94a3b8", opacity=0.7,
                       annotation_text="WHO", annotation_font_size=9)
    fig_risk.update_layout(
        dragmode=False, height=360, margin=dict(t=40, b=10, l=10, r=30),
        showlegend=True, legend=dict(orientation="h", y=1.22, x=0, font=dict(size=9)),
        yaxis=dict(autorange="reversed"),
        xaxis_range=[0, max(station_latest["Score"].max() * 1.25, 250)],
        font=dict(family="IBM Plex Sans"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_risk, width="stretch",
                    config={"scrollZoom": False, "displayModeBar": False},
                    key="db_risk_chart")

st.caption(tr(
    "Risk score = mean of (concentration ÷ WHO 2021 limit) across PM2.5, PM10, NO₂, ×100. "
    "100 = exactly at the WHO guideline."
))
st.divider()

# ==================================================
# SECTION — TODAY vs TOMORROW KPIs
# ==================================================
# SECTION — D-1 vs FORECAST KPIs
# ==================================================
st.markdown(
    f'<p class="section-title">{tr("📊 City-wide averages")} — '
    f'{latest_date.strftime("%d %b %Y")} (D-1) vs '
    f'{tomorrow.strftime("%d %b %Y")} ({tr("forecast")})</p>',
    unsafe_allow_html=True
)

d1_means  = df[df["Date"] == latest_date].groupby("station")[POLLUTANTS].mean().mean()
fc_means  = fc_df.groupby("Pollutant")["Forecast"].mean() if not fc_df.empty else pd.Series(dtype=float)

cols = st.columns(4)
for col, p in zip(cols, POLLUTANTS):
    d1_val  = d1_means.get(p, float("nan"))
    fc_val  = fc_means.get(p, float("nan"))
    color   = POLLUTANT_COLOR.get(p, "#888")

    # D-1 card always shown; forecast only if models available
    if not np.isnan(fc_val):
        delta_val  = fc_val - d1_val
        delta_str  = f"{delta_val:+.1f} µg/m³ {tr('vs D-1')}"
        delta_color = "#e74c3c" if delta_val > 0 else "#27ae60"
        who_ratio   = fc_val / WHO_ANNUAL[p] if p in WHO_ANNUAL else None
        fc_str = f"{fc_val:.1f}"
        who_str = f'<div style="font-size:0.75rem;color:#888;margin-top:3px">WHO: {who_ratio:.1f}×</div>' if who_ratio else ""
    else:
        delta_str   = tr("Forecast unavailable")
        delta_color = "#aaa"
        fc_str      = "—"
        who_str     = ""

    with col:
        st.markdown(f"""
        <div style="border-left:4px solid {color};background:{color}0d;
                    border-radius:8px;padding:12px 14px;">
            <div style="font-size:0.8rem;color:#888;font-weight:600;letter-spacing:0.5px">{p}</div>
            <div style="font-size:1.5rem;font-weight:700;color:#2c3e50;margin:4px 0">
                {fc_str} <span style="font-size:0.75rem;color:#888;font-weight:400">µg/m³</span>
            </div>
            <div style="font-size:0.75rem;color:#888">
                D-1 ({latest_date.strftime("%d %b")}): {d1_val:.1f} µg/m³
            </div>
            <div style="font-size:0.75rem;color:{delta_color};font-weight:600">
                {delta_str}
            </div>
            {who_str}
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ==================================================
# SECTION — NEXT-DAY FORECAST HEATMAP (EU ratios)
# ==================================================
st.markdown(f'<p class="section-title">{tr("🔮 Next-day forecast")} — {tomorrow.strftime("%d %b %Y")} — {tr("by station & pollutant")}</p>',
            unsafe_allow_html=True)
st.caption(
    tr("Colour = ratio vs EU Directive limit (legally binding in Spain) · >1.0 = legal exceedance") +
    f" · {tr('Forecast date')}: {tomorrow.strftime('%d %b %Y')}"
)

if not fc_df.empty:
    pivot = fc_df.pivot(index="station", columns="Pollutant", values="EU_ratio").reindex(columns=POLLUTANTS)
    fig_hm = px.imshow(
        pivot,
        color_continuous_scale=["#2ecc71", "#f9ca24", "#e74c3c"],
        zmin=0, zmax=2, text_auto=".2f", aspect="auto",
        labels={"color": "× EU limit"},
    )
    fig_hm.update_layout(
        height=280, margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_colorbar=dict(
            title="× EU", tickvals=[0, 1, 2],
            ticktext=["0×", "1× (EU)", "2×"],
        ),
    )
    st.plotly_chart(fig_hm, width="stretch",
                    config={"scrollZoom": False, "displayModeBar": False},
                    key="db_heatmap")

# EU exceedance detail table
eu_exceed = fc_df[fc_df["EU_exceeds"]]
if not eu_exceed.empty:
    st.markdown(f'<p class="section-title">⚠️ {tr("EU Directive exceedances")} {tomorrow.strftime("%d %b %Y")} ({len(eu_exceed)} {tr("found")})</p>',
                unsafe_allow_html=True)
    show = eu_exceed[["station", "Zone", "Pollutant", "Forecast", "EU_ratio", "EAQI_label"]].copy()
    show["EU_ratio"] = show["EU_ratio"].map(lambda x: f"{x:.2f}×")
    show["Forecast"] = show["Forecast"].map(lambda x: f"{x:.1f} µg/m³")
    show.columns     = ["Station", "Zone", "Pollutant", "Forecast", "EU Ratio", "EAQI"]
    st.dataframe(show, hide_index=True, width="stretch")

    exceed_lines = "\n".join(
        f"  • {r['Station']} — {r['Pollutant']}: {r['Forecast']} ({r['EU Ratio']} EU)"
        for _, r in show.iterrows()
    )
    mailto_subject = f"Air Quality Alert — {tomorrow.strftime('%d %b %Y')}"
    mailto_body = (
        f"Air quality forecast alert for Greater Bilbao\n"
        f"Date: {tomorrow.strftime('%d %b %Y')}\n\n"
        f"EU Directive exceedances:\n{exceed_lines}\n\n"
        f"Dashboard: https://geoai-dashboard.streamlit.app/\n"
    )
    mailto_link = (
        f"mailto:?subject={urllib.parse.quote(mailto_subject)}"
        f"&body={urllib.parse.quote(mailto_body)}"
    )
    st.link_button(tr("📧 Share alert by email"), mailto_link, type="primary")
else:
    st.success(f"✅ {tr('No EU Directive exceedances forecast for')} {tomorrow.strftime('%d %b %Y')}.")

st.divider()

# ==================================================
# SECTION — 7-DAY SPARKLINES
# ==================================================
st.markdown(f'<p class="section-title">{tr("📈 Last 7 days — city-wide trend")}</p>',
            unsafe_allow_html=True)

last7 = (
    df[df["Date"] >= latest_date - timedelta(days=6)]
    .groupby("Date")[POLLUTANTS].mean().reset_index().sort_values("Date")
)

fig_spark = go.Figure()
for p in ["PM2.5", "PM10", "NO2"]:
    fig_spark.add_trace(go.Scatter(
        x=last7["Date"], y=last7[p], name=p,
        mode="lines+markers",
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
    hovermode="x unified", legend=dict(orientation="h", y=1.1),
    xaxis=dict(tickformat="%d %b"), dragmode=False,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_spark, width="stretch",
                config={"scrollZoom": False, "displayModeBar": False},
                key="db_spark")

st.divider()

# ==================================================
# SECTION — RECOMMENDED ACTION
# ==================================================
st.markdown(f'<p class="section-title">{tr("🎯 Recommended action")} — {latest_date.strftime("%d %b %Y")}</p>',
            unsafe_allow_html=True)

ZONE_ACTIONS = {
    "Urban":      tr("Consider traffic flow management and public transport promotion. "
                     "NO₂ levels are traffic-driven — rush-hour peaks are most critical."),
    "Industrial": tr("Coordinate with industrial operators. "
                     "PM2.5 and PM10 levels may warrant stack monitoring review."),
    "Port":       tr("Monitor SO₂ from vessel activity. "
                     "Shore-power availability reduces marine emissions significantly."),
    "Coastal":    tr("Conditions are generally favourable. "
                     "Maintain standard monitoring — marine PM10 events are possible."),
    "Refinery":   tr("SO₂ episode risk. Check Petronor operational schedule. "
                     "Low-wind days increase local concentration risk."),
}

zone_ratios = (
    fc_df[fc_df["Pollutant"].isin(["PM2.5", "PM10", "NO2"])]
    .groupby("Zone")["WHO_ratio"].mean().sort_values(ascending=False)
)

if not zone_ratios.empty:
    worst_zone  = zone_ratios.index[0]
    worst_ratio = zone_ratios.iloc[0]
    zone_meta   = ZONE_META.get(worst_zone, {})
    action_text = ZONE_ACTIONS.get(worst_zone, tr("Monitor closely."))
    urgency_color = "#e74c3c" if worst_ratio > 1.5 else "#f39c12" if worst_ratio > 1.0 else "#2ecc71"
    urgency_label = (tr("High priority") if worst_ratio > 1.5 else
                     tr("Moderate")      if worst_ratio > 1.0 else tr("Routine monitoring"))
    st.markdown(f"""
    <div style="border:2px solid {zone_meta.get('color','#ccc')};border-radius:12px;padding:18px 20px;
                background:linear-gradient(135deg,{zone_meta.get('color','#888')}15,{zone_meta.get('color','#888')}05);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <span style="font-size:1.6rem">{zone_meta.get('icon','🏙️')}</span>
            <div>
                <div style="font-weight:700;font-size:1rem;color:#2c3e50">
                    {tr("Priority zone")}: {worst_zone}
                </div>
                <span style="background:{urgency_color};color:white;padding:2px 10px;
                             border-radius:10px;font-size:0.75rem;font-weight:600">
                    {urgency_label} · {worst_ratio:.1f}× WHO {tr("avg")}
                </span>
            </div>
        </div>
        <div style="color:#555;font-size:0.9rem;line-height:1.6">{action_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==================================================
# SECTION — PDF EXPORT
# ==================================================
st.markdown(f'<p class="section-title">{tr("📄 Download Report")}</p>',
            unsafe_allow_html=True)

col_pdf1, col_pdf2 = st.columns([2, 3])
with col_pdf1:
    today_means_dict = {
        p: float(df[df["Date"] == latest_date][p].mean()) for p in POLLUTANTS
    }
    worst_zone_pdf  = zone_ratios.index[0] if not zone_ratios.empty else "Urban"
    action_text_pdf = ZONE_ACTIONS.get(worst_zone_pdf, tr("Monitor closely."))

    pdf_bytes = generate_daily_report(
        latest_date    = latest_date,
        current_values = today_means_dict,
        fc_df          = fc_df,
        zone_action    = action_text_pdf,
        worst_zone     = worst_zone_pdf,
        who_annual     = WHO_ANNUAL,
        eu_annual      = EU_ANNUAL,
        alert_limits   = {p: EU_ANNUAL.get(p, 25.0) for p in POLLUTANTS},
    )
    st.download_button(
        label     = tr("📄 Download Daily Alert Report (PDF)"),
        data      = pdf_bytes,
        file_name = f"daily_alert_{latest_date.strftime('%Y%m%d')}.pdf",
        mime      = "application/pdf",
        type      = "primary",
        width="stretch",
    )

with col_pdf2:
    st.caption(tr(
        "One-page summary: D-1 city-wide averages, "
        "next-day EU Directive exceedances, "
        "and zone-level recommended action."
    ))

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
        "EAQI/ICA · WHO 2021 · EU Directive 2008/50/EC"
    )

st.caption(
    f"{tr('Data: Basque Government (CC BY 4.0) + Open-Meteo (CC BY 4.0)')} · "
    f"{tr('Forecasts: XGBoost models (test R²=0.39–0.56)')} · "
    f"{tr('Latest data')}: {latest_date.strftime('%d %b %Y')} · "
    f"{tr('Next update: ~06:00 UTC daily')}"
)

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:0.88rem;color:#64748b;line-height:1.7">
        <b style="color:#0c1521">Arman Ghaziaskari Naeini</b><br>
        GIS &amp; Spatial Data Science<br><br>
        <a href="https://armanghazi.github.io/portfolio"
           style="color:#0ea5b5;text-decoration:none">🔗 Portfolio</a><br>
        <a href="https://github.com/armanghazi"
           style="color:#0ea5b5;text-decoration:none">🐙 GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )