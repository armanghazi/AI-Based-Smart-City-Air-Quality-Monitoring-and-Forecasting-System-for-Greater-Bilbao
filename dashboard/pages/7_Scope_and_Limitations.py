"""
dashboard/pages/7_Scope_and_Limitations.py

GeoAI Methodology & Scope — Greater Bilbao Air Quality Platform

Transforms the former plain-text disclaimer page into a full
methodology reference: spatial analysis, ML model architecture,
data pipeline, and honest scientific caveats.

Four sections:
  1. Platform Overview    — data, coverage, pipeline
  2. GIS & Spatial        — notebooks 10a-10d findings + top correlations
  3. ML Models            — XGBoost metrics, benchmark comparison, SHAP
  4. Honest Caveats       — n=7, ERA5, IDW, leakage fix, D-1 constraint
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

from config import load_data, WHO_ANNUAL, ZONE_META
from glossary import G
from i18n_auto import language_selector, apply_lang_styles, tr

from forecast_utils import plotly_touch_config, PLOTLY_CONFIG

plotly_touch_config()  


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GeoAI Methodology",
    page_icon="📋",
    layout="wide",
)
language_selector()
apply_lang_styles()

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS — all key numbers hardcoded from verified project sources
# ──────────────────────────────────────────────────────────────────────────────

MODEL_METRICS = {
    "NO2":   {"R2": 0.560, "RMSE": 5.97,  "MAE": 4.43,  "note": "Best — strong weekly traffic cycle"},
    "PM2.5": {"R2": 0.479, "RMSE": 3.33,  "MAE": 2.50,  "note": "Persistence + weather drivers"},
    "PM10":  {"R2": 0.460, "RMSE": 6.26,  "MAE": 4.45,  "note": "Dust events add variability"},
    "SO2":   {"R2": 0.390, "RMSE": 1.87,  "MAE": 1.23,  "note": "Hardest — episodic industrial/port"},
}

BENCHMARK = [
    {"Model": "XGBoost",         "Type": "ML (trees)",    "Scope": "All stations",   "R2": 0.479, "production": True},
    {"Model": "MLP (128,64)",    "Type": "Deep learning", "Scope": "All stations",   "R2": 0.208, "production": False},
    {"Model": "SARIMA one-step", "Type": "Classical TS",  "Scope": "Mazarredo only", "R2": 0.463, "production": False},
    {"Model": "SARIMA long-h.",  "Type": "Classical TS",  "Scope": "Mazarredo only", "R2": 0.000, "production": False},
]

SPATIAL_FINDINGS = [
    ("road_density_1000m",    "NO₂",   "+0.83", "Road density (1km)",      "Traffic infrastructure is the primary structural driver of NO₂."),
    ("dist_bilbao_centre_m",  "NO₂",   "−0.77", "Distance to city centre", "Closer to centre = higher NO₂. Urban canyon + traffic concentration."),
    ("vegetation_proxy_1000m","PM10",  "−0.66", "Green cover (1km)",       "More vegetation = lower PM10. Dry deposition + reduced impervious surface."),
    ("elev_tri_2000m",        "PM10",  "−0.63", "Terrain Relief Index",    "Complex terrain = stronger turbulent mixing = lower PM10."),
    ("dist_ap8_barakaldo_m",  "PM2.5", "−0.54", "Distance to AP-8",        "BARAKALDO (354 m from AP-8) records highest PM2.5."),
]

SPATIAL_NOTEBOOKS = [
    ("10a", "Buffer Analysis",          "OSM land use + road density at 500m/1km/2km buffers",       "GeoPandas · OSMnx"),
    ("10b", "Distance Features",        "Haversine distances to Port, Petronor, AP-8, coast, centre","GeoPandas · math"),
    ("10c", "Elevation & Terrain",      "Copernicus GLO-30 DEM: elevation, TRI, slope per station",  "rasterio · rasterstats"),
    ("10d", "Wind Transport",           "ERA5 direction × station cross-analysis, dispersion effect", "pandas · plotly"),
]

ZONE_CLR = {
    "Urban": "#8e44ad", "Industrial": "#e67e22",
    "Port": "#2980b9",  "Coastal": "#1abc9c", "Refinery": "#c0392b",
}

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

df = load_data()
latest_date   = df["Date"].max()
n_records     = len(df)
n_years       = df["Year"].nunique()
date_range    = f"{df['Date'].min().year}–{df['Date'].max().year}"

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("## 📋 " + tr("GeoAI Methodology & Scope"))
st.markdown(
    tr(
        "Full methodology reference for the GeoAI Smart City Air Quality Platform — "
        "Greater Bilbao. Covers data architecture, GIS spatial analysis, "
        "machine learning models, and honest scientific scope and limitations."
    )
)
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PLATFORM OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### " + tr("1 · Platform Overview"))

ov1, ov2, ov3, ov4, ov5 = st.columns(5)
ov1.metric(tr("Monitoring stations"),  "7")
ov2.metric(tr("Environmental zones"),  "5")
ov3.metric(tr("Daily records"),        f"{n_records:,}",
           help=G["D1_data"])
ov4.metric(tr("Data period"),          date_range)
ov5.metric(tr("Spatial features"),     "~35",
           help=G["buffer"])

st.markdown("---")

# Data pipeline
pc1, pc2 = st.columns([2, 1])

with pc1:
    st.markdown("**" + tr("Data pipeline") + "**")
    st.markdown(
        tr(
            "- **Air quality:** Basque Government RVCA network → opendata.euskadi.eus API "
            "(7 stations, 4 pollutants: PM2.5, PM10, NO₂, SO₂)  \n"
            "- **Meteorology:** Open-Meteo ERA5 archive (Temperature, Humidity, "
            "Precipitation, Wind speed/direction)  \n"
            "- **Update frequency:** Automated GitHub Actions cron — daily at 06:00 UTC  \n"
            "- **D-1 constraint:** Pipeline rejects the current incomplete day — "
            "all data is at least one day old. Labels throughout the dashboard "
            "reflect this (no 'Today' or 'Current').  \n"
            "- **Storage:** Parquet (air_quality_weather.parquet) — "
            "feature engineering (lags, rolling, targets) computed at runtime in config.load_data()"
        )
    )

with pc2:
    st.markdown("**" + tr("Monitoring stations") + "**")
    zone_summary = (
        df.groupby(["station", "Zone"])[["PM2.5", "NO2"]].mean().round(1).reset_index()
    )
    zone_summary["Station"] = zone_summary["station"].str.split("_").str[0]
    fig_ov = px.bar(
        zone_summary.sort_values("NO2"),
        x="NO2", y="Station", color="Zone",
        color_discrete_map=ZONE_CLR,
        orientation="h",
        labels={"NO2": "Mean NO₂ (µg/m³)", "Station": ""},
        title=tr("Long-term mean NO₂ by station"),
    )
    fig_ov.add_vline(x=WHO_ANNUAL["NO2"], line_dash="dot", line_color="#e74c3c",
                     annotation_text="WHO", annotation_font_size=9)
    fig_ov.update_layout(dragmode=False, 
        height=280, margin=dict(t=40, b=10, l=10, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_ov, width="stretch", config=PLOTLY_CONFIG)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — GIS & SPATIAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### " + tr("2 · GIS & Spatial Analysis"))
st.caption(
    tr(
        "Four GIS notebooks deliver ~35 spatial features per station. "
        "Correlations computed across n = 7 stations — exploratory and indicative only."
    )
)
st.info(
    "💡 " + tr(
        "**In plain English:** We analysed what physically surrounds each sensor — "
        "nearby roads, distance to industrial sites, and terrain complexity — "
        "to understand why some stations are structurally more polluted than others, "
        "regardless of daily weather."
    )
)

# Notebooks table
st.markdown("**" + tr("Spatial analysis notebooks") + "**")
nb_cols = st.columns(4)
nb_colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22"]
for col, (nb_id, title, desc, tools), color in zip(nb_cols, SPATIAL_NOTEBOOKS, nb_colors):
    col.markdown(
        f"""
        <div style="border:1px solid {color};border-radius:10px;padding:12px;
                    border-top:3px solid {color};height:140px">
            <div style="font-family:monospace;font-size:11px;color:{color};
                        font-weight:600">notebook {nb_id}</div>
            <div style="font-weight:600;font-size:13px;margin:4px 0">{title}</div>
            <div style="font-size:11px;color:#666;line-height:1.4">{desc}</div>
            <div style="font-size:10px;color:#888;margin-top:6px">{tools}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Top spatial findings
st.markdown("**" + tr("Strongest spatial predictors") + "**")

sp1, sp2 = st.columns([3, 2])

with sp1:
    rows = []
    for feat, poll, r, label, interp in SPATIAL_FINDINGS:
        rows.append({
            tr("Feature"): label,
            tr("Pollutant"): poll,
            "r": r,
            tr("Interpretation"): interp,
        })
    df_sp_tbl = pd.DataFrame(rows)
    st.dataframe(df_sp_tbl, width="stretch", hide_index=True)

with sp2:
    # Bar chart of r values
    _r_vals  = [float(r.replace("−", "-").replace("+", "")) for *_, r, _, _ in SPATIAL_FINDINGS]
    _r_lbls  = [lbl for _, _, _, lbl, _ in SPATIAL_FINDINGS]
    _r_clrs  = ["#2980b9" if v > 0 else "#e74c3c" for v in _r_vals]

    fig_r = go.Figure(go.Bar(
        x=_r_vals,
        y=_r_lbls,
        orientation="h",
        marker_color=_r_clrs,
        text=[f"r = {r}" for *_, r, _, _ in SPATIAL_FINDINGS],
        textposition="outside",
    ))
    fig_r.add_vline(x=0, line_color="#555", line_width=1)
    fig_r.update_layout(dragmode=False, 
        height=240,
        margin=dict(t=30, b=10, l=10, r=80),
        xaxis=dict(range=[-1, 1], title="Pearson r"),
        title=tr("Top 5 spatial correlations"),
    )
    st.plotly_chart(fig_r, width="stretch", config=PLOTLY_CONFIG)

st.markdown("---")

# MUSKIZ paradox callout
st.markdown("**" + tr("Key GeoAI finding — MUSKIZ dispersion paradox") + "**")
mx1, mx2, mx3, mx4 = st.columns(4)
mx1.metric(tr("Industrial land use (1km)"), "34.6%", tr("highest in network"),
           delta_color="inverse", help=G["buffer"])
mx2.metric(tr("Mean PM2.5"), "6.50 µg/m³", tr("lowest in network ✓"),
           delta_color="off", help=G["PM25"])
mx3.metric(tr("TRI 2km"), "343 m", tr("orographic ventilation"),
           delta_color="off", help=G["TRI"])
mx4.metric(tr("NE wind → SO₂ ratio"), "1.93×", tr("vs S-wind baseline"),
           delta_color="inverse", help=G["wind_regime"])
st.success(
    tr(
        "Three independent GIS methods (buffer analysis · distance features · DEM terrain) "
        "converge on the same explanation: MUSKIZ's coastal position + high terrain complexity "
        "override Petronor emission proximity. This is the clearest GeoAI story in the dataset."
    )
)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ML MODELS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### " + tr("3 · Machine Learning Models"))
st.caption(
    tr("XGBoost models trained on 2015–2022, validated 2023, tested 2024–2026. "
       "Time-based split only — no random row splitting.")
)
st.info(
    "💡 " + tr(
        "**In plain English:** The models learn from past data and are tested on "
        "future data they have never seen — the same way a weather forecast is judged "
        "by whether tomorrow actually matches what it predicted."
    )
)

# Model metrics
st.markdown("**" + tr("Production model performance (held-out test 2024–2026)") + "**")

mc1, mc2, mc3, mc4 = st.columns(4)
for col, (poll, m) in zip([mc1, mc2, mc3, mc4], MODEL_METRICS.items()):
    col.metric(
        f"{poll}  R²",
        f"{m['R2']:.3f}",
        f"RMSE {m['RMSE']:.2f} | MAE {m['MAE']:.2f}",
        help=m["note"] + "  ·  " + G["R2"],
        delta_color="off",
    )

st.markdown("---")

# Benchmark table
st.markdown("**" + tr("Benchmark comparison") + "**")
bc1, bc2 = st.columns([3, 2])

with bc1:
    df_bm = pd.DataFrame(BENCHMARK)
    df_bm["Production"] = df_bm["production"].map({True: "✅ Yes", False: "—"})
    st.dataframe(
        df_bm[["Model", "Type", "Scope", "R2", "Production"]].rename(
            columns={"R2": "R²"}
        ),
        width="stretch",
        hide_index=True,
    )

with bc2:
    fig_bm = go.Figure(go.Bar(
        x=[b["R2"] for b in BENCHMARK],
        y=[b["Model"] for b in BENCHMARK],
        orientation="h",
        marker_color=["#27ae60" if b["production"] else "#95a5a6" for b in BENCHMARK],
        text=[f"{b['R2']:.3f}" for b in BENCHMARK],
        textposition="outside",
    ))
    fig_bm.update_layout(dragmode=False, 
        height=220,
        margin=dict(t=30, b=10, l=10, r=60),
        xaxis=dict(range=[0, 0.65], title="R²"),
        title=tr("R² comparison — green = production model"),
    )
    st.plotly_chart(fig_bm, width="stretch", config=PLOTLY_CONFIG)

# ...
st.plotly_chart(fig, config=PLOTLY_CONFIG)
st.markdown("---")

# Feature architecture
st.markdown("**" + tr("Feature architecture — 62 features per model") + "**")
fa1, fa2 = st.columns(2)

with fa1:
    st.markdown(
        tr(
            "**Temporal features (9):** year, month, day, day_of_year, "
            "week_of_year, day_of_week, is_weekend, season  \n\n"
            "**Pollutant lags (8):** lag_1 and lag_7 for each of PM2.5, PM10, NO₂, SO₂  \n\n"
            "**Rolling means (4):** 14-day rolling mean for each pollutant  \n\n"
            "**Current pollutant values (4):** PM2.5, PM10, NO₂, SO₂ — "
            "available at prediction time, not leakage"
        )
    )
with fa2:
    st.markdown(
        tr(
            "**Weather features (7):** Temperature, Humidity, Precipitation, "
            "WindSpeed, WindDirection, wind_u, wind_v  \n\n"
            "**Station encoding (1):** station_code (integer, global map)  \n\n"
            "**Interaction features (3):** wind_x_precip, temp_x_humid, "
            "wind_speed_sq (built but not used — tracked as ALLOWED_UNUSED)  \n\n"
            "**Important:** bundle['features'] list is the single source of truth "
            "for column order at prediction time."
        )
    )

st.info(
    tr(
        "**Leakage fix (documented):** An early row-based random split produced "
        "inflated R² = 0.84. Switching to strict time-based split (train < 2023) "
        "corrected this to the honest R² = 0.479. "
        "This is documented explicitly as a methodological integrity marker."
    )
)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HONEST CAVEATS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### " + tr("4 · Scientific Scope & Honest Caveats"))
st.caption(
    tr(
        "These constraints are documented explicitly — not hidden. "
        "They define where the platform is reliable and where it is not."
    )
)
st.info(
    tr(
        "💡 **In plain English:** Every analytical tool has boundaries. "
        "This section tells you exactly where this platform stops being reliable "
        "and why — so you can use it with confidence within those boundaries."
    )
)

caveats = [
    (
        "🔢 " + tr("n = 7 stations"),
        tr(
            "Spatial correlations across 7 stations are exploratory and indicative only. "
            "Statistically significant inference typically requires ≥30 stations "
            "(standard Land Use Regression study design). "
            "All spatial correlations in this platform are labelled accordingly."
        ),
        "warning",
    ),
    (
        "🌍 " + tr("ERA5 single grid cell"),
        tr(
            "All 7 stations share one Open-Meteo ERA5 wind time series (~31 km resolution). "
            "Wind direction analysis represents regional atmospheric conditions, "
            "not station-level micrometeorology. "
            "Station-level wind would require a dense sensor network or numerical "
            "dispersion model (e.g. AERMOD, HYSPLIT)."
        ),
        "warning",
    ),
    (
        "🗺️ " + tr("IDW is visual approximation only"),
        tr(
            "The IDW interpolated forecast surface (page 6) is a visual tool, "
            "not a spatial model. With 7 stations, the interpolation between "
            "measurement points is purely mathematical — it does not account for "
            "terrain, wind fields, or emission sources. "
            "Areas between stations have no observational basis."
        ),
        "info",
    ),
    (
        "📅 " + tr("D-1 data constraint"),
        tr(
            "The automated pipeline rejects the current incomplete day. "
            "All data shown is at least 24 hours old. "
            "No page or label uses 'Today', 'Current', or 'Live'. "
            "This is by design — partial-day data creates false readings."
        ),
        "info",
    ),
    (
        "🏭 " + tr("LUR model not attempted"),
        tr(
            "A full Land Use Regression model was considered but not implemented. "
            "LUR at n=7 would produce an overfit model that cannot be validated. "
            "The spatial feature table (station_spatial_features_v3.csv) provides "
            "the same structural understanding without the statistical overreach."
        ),
        "info",
    ),
    (
        "📊 " + tr("Petronor confounding"),
        tr(
            "Distance to Petronor refinery shows positive correlation with PM2.5/PM10 "
            "(r = +0.70–0.79) — meaning stations farther from the refinery record higher PM. "
            "This is a spatial confounding artifact at n=7: MUSKIZ (closest) has the "
            "lowest PM due to coastal dispersion. This result must not be interpreted "
            "as evidence that refinery proximity reduces pollution."
        ),
        "warning",
    ),
]

for title, body, level in caveats:
    with st.expander(title, expanded=False):
        getattr(st, level)(body)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# DATA SOURCES & STANDARDS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### " + tr("5 · Data Sources & Standards"))

ds1, ds2, ds3 = st.columns(3)

with ds1:
    st.markdown(
        "**" + tr("Air quality data") + "**  \n"
        "Basque Government RVCA network  \n"
        "[opendata.euskadi.eus](https://opendata.euskadi.eus)  \n"
        "7 stations · CC BY 4.0  \n\n"
        "**" + tr("Meteorology") + "**  \n"
        "Open-Meteo ERA5 archive  \n"
        "[open-meteo.com](https://open-meteo.com) · CC BY 4.0"
    )

with ds2:
    st.markdown(
        "**" + tr("GIS data") + "**  \n"
        "OpenStreetMap (OSM) via OSMnx · ODbL  \n"
        "Copernicus GLO-30 DEM (30m) · CC BY 4.0  \n"
        "Open Data Euskadi COMARCAS shapefile  \n"
        "ETRS89/UTM 30N (EPSG:25830)"
    )

with ds3:
    st.markdown(
        "**" + tr("Standards") + "**  \n"
        "WHO 2021 annual guidelines (analysis)  \n"
        "— PM2.5: 5 µg/m³ · PM10: 15 µg/m³ · NO₂: 10 µg/m³  \n"
        "— SO₂: 40 µg/m³ (24-hour)  \n\n"
        "EU Directive 2008/50/EC (legal alerts only)  \n"
        "US EPA AQI shown as secondary reference"
    )

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TECH STACK
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### " + tr("6 · Technology Stack"))

tc1, tc2, tc3, tc4 = st.columns(4)

with tc1:
    st.markdown(
        "**GIS & Spatial**  \n"
        "GeoPandas · Shapely 2.1.2  \n"
        "OSMnx · Folium  \n"
        "rasterio · rasterstats  \n"
        "EPSG:25830 (UTM 30N)"
    )
with tc2:
    st.markdown(
        "**Machine Learning**  \n"
        "XGBoost (production)  \n"
        "Scikit-Learn · SHAP  \n"
        "statsmodels (OLS)  \n"
        "62 features · 4 models"
    )
with tc3:
    st.markdown(
        "**Dashboard**  \n"
        "Streamlit Cloud  \n"
        "Plotly ≤5.22.0  \n"
        "Folium · FPDF2  \n"
        "i18n: EN / ES"
    )
with tc4:
    st.markdown(
        "**Infrastructure**  \n"
        "GitHub Actions (CI/CD)  \n"
        "Google OAuth OIDC  \n"
        "Groq/Llama 3.3 70B  \n"
        "pytest (11 tests)"
    )