import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from spatial_utils import idw_grid, mask_idw_grid
import geopandas as gpd

import plotly.express as px
import plotly.graph_objects as go

from glossary import G
from config import (
    load_data, WHO_ANNUAL, WHO_SO2_DAILY, CORE_POLLUTANTS,
    POLLUTANT_COLOR, ZONE_META, RISK_COLORS,
    classify_core_risk, risk_color, get_fav_station, EU_ANNUAL, center_tables
)
from forecast_utils import prepare_features
from gauge_component import render_gauge_row
from aqi import overall_aqi, compute_aqi_category, AQI_POLLUTANTS, AQI_CATEGORIES
from aqi_components import (
    render_aqi_donut, render_station_aqi_cards, render_aqi_calendar,
)

from pdf_report import generate_daily_report, generate_monthly_report

from i18n_auto import tr

from forecast_utils import plotly_touch_config, PLOTLY_CONFIG

plotly_touch_config()  


# ==================================================
# 1. PAGE CONFIG
# ==================================================

# NOTE: st.set_page_config is intentionally absent — called once in app.py router.
center_tables()
# ==================================================
# 2. CONSTANTS
# ==================================================

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]

# Comarca boundary shapefile (used for IDW mask outline)
_COMARCA_SHP = (
    Path(__file__).parent.parent  # dashboard/ → project root
    / "GIS" / "boundaries"
    / "COMARCAS_5000_ETRS89.shp"
)

# Shared IDW colour scale — 0× (green) → 1× WHO (yellow) → 2× (dark red)
_IDW_CS = [
    [0.00, "#2ecc71"],
    [0.48, "#7fd957"],
    [0.50, "#f4d03f"],
    [0.65, "#e67e22"],
    [0.85, "#e74c3c"],
    [1.00, "#c0392b"],
]


def model_path(pollutant: str) -> Path:
    prefix = pollutant.replace(".", "").lower()
    return MODELS_DIR / f"xgb_{prefix}_forecast.joblib"


@st.cache_resource
def load_model(pollutant: str):
    path = model_path(pollutant)
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_comarca_boundary() -> tuple[list, list]:
    """Load Gran Bilbao comarca boundary and return (lats, lons) in WGS84."""
    # Path relative to project root (works both locally and on Streamlit Cloud)
    sph = Path(__file__).parent.parent.parent / "GIS" / "boundaries" / "COMARCAS_5000_ETRS89.shp"
    if not sph.exists():
        # fallback: try two levels up (if file structure differs)
        sph = Path(__file__).parent.parent / "GIS" / "boundaries" / "COMARCAS_5000_ETRS89.shp"
    com  = gpd.read_file(sph)
    gb   = com[com["COMARCA"] == "GRAN BILBAO"].to_crs("EPSG:4326")
    geom = gb.geometry.iloc[0]
    if geom.geom_type == "MultiPolygon":
        geom = geom.geoms[0]
    lons, lats = geom.exterior.coords.xy
    return list(lats), list(lons)


# ==================================================
# 3. LOAD DATA
# ==================================================

df          = load_data()
latest_date = df["Date"].max()
forecast_date = latest_date + pd.Timedelta(days=1)

# Stable station_code mapping — sorted order reproduces training codes
STATION_CODES = {s: i for i, s in enumerate(sorted(df["station"].unique()))}

# ==================================================
# 4. HEADER
# ==================================================

st.title(tr("🏛️ Smart City Decision Support"))
st.markdown(
    f"{tr('Actionable air-quality intelligence for urban decision-makers')} · "
    f"{tr('Latest data:')} **{latest_date.strftime('%d %b %Y')}**"
)

# ==================================================
# 5. SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## 🏛️ " + tr("Analysis Settings"))
    st.divider()

    window_label = st.radio(
        tr("Risk assessment window"),
        ["Last 30 days", "Last 90 days", "Last 365 days"],
        index=2,
    )
    window_days = {
        "Last 30 days": 30,
        "Last 90 days": 90,
        "Last 365 days": 365,
    }[window_label]

    st.divider()
    st.markdown("### 🗺️ " + tr("Zone Legend"))
    for z, meta in ZONE_META.items():
        st.markdown(f"{meta['icon']} **{z}**")

# ==================================================
# 6. SHARED COMPUTED VARIABLES
# (computed before tabs — later tabs depend on these)
# ==================================================

window_start = latest_date - pd.Timedelta(days=window_days)
recent       = df[df["Date"] >= window_start].copy()

# 6a. Station risk scores over the selected window
station_means = (
    recent.groupby(["station", "Zone"])[POLLUTANTS]
    .mean()
    .reset_index()
)


def composite_score(row) -> float:
    """Mean (concentration / WHO limit) over core pollutants × 100."""
    ratios = [row[p] / WHO_ANNUAL[p] for p in CORE_POLLUTANTS]
    return float(np.mean(ratios) * 100)


station_means["RiskScore"] = station_means.apply(composite_score, axis=1)
station_means["RiskLevel"] = station_means["RiskScore"].apply(classify_core_risk)
station_means = station_means.sort_values("RiskScore", ascending=False)

n_above       = (station_means["RiskScore"] >= 100).sum()
worst_station = station_means.iloc[0]

city_ratios   = {p: recent[p].mean() / WHO_ANNUAL[p] for p in CORE_POLLUTANTS}
worst_pollutant = max(city_ratios, key=city_ratios.get)

# 6b. Latest-day city-wide means
latest_day     = df[df["Date"] == latest_date]
current_values = {p: float(latest_day[p].mean()) for p in POLLUTANTS}

# 6c. Station priority table (reused in tab_status and export)
display = station_means.copy()
display["Rank"] = range(1, len(display) + 1)
display = display[["Rank", "station", "Zone", "PM2.5", "PM10",
                   "NO2", "SO2", "RiskScore", "RiskLevel"]]
display[POLLUTANTS]  = display[POLLUTANTS].round(1)
display["RiskScore"] = display["RiskScore"].round(0).astype(int)

# 6d. Load all 4 models once (used in tab_forecast and tab_action)
bundles       = {p: load_model(p) for p in POLLUTANTS}
missing_models = [p for p, b in bundles.items() if b is None]

# 6e. Forecast alerts table (used in tab_forecast, tab_action, and export)
alerts = []
for station in sorted(df["station"].unique()):
    sdf = df[df["station"] == station].sort_values("Date")
    for pollutant in POLLUTANTS:
        bundle = bundles.get(pollutant)
        if bundle is None:
            continue
        feats      = bundle["features"]
        prep       = prepare_features(sdf, feats, station_codes=STATION_CODES)
        prep_filled = prep.copy()
        prep_filled[feats] = prep_filled[feats].ffill()
        valid = prep_filled.dropna(subset=feats)
        if valid.empty:
            continue
        X_last = valid[feats].iloc[[-1]]
        pred   = max(float(bundle["model"].predict(X_last)[0]), 0.0)
        limit  = WHO_SO2_DAILY if pollutant == "SO2" else WHO_ANNUAL.get(pollutant)
        alerts.append({
            "Station":   station,
            "Zone":      sdf["Zone"].iloc[-1],
            "Pollutant": pollutant,
            "Forecast":  round(pred, 1),
            "WHO limit": limit,
            "Ratio":     round(pred / limit, 2) if limit else None,
            "Status":    "⚠️ Above WHO" if limit and pred > limit else "✅ OK",
        })

alerts_df = pd.DataFrame(alerts)
exceed     = (
    alerts_df[alerts_df["Status"].str.contains("Above")]
    if not alerts_df.empty else pd.DataFrame()
)

# 6f. Zone means for tab_action
zone_means = (
    recent.groupby("Zone")[POLLUTANTS].mean().round(1)
    .reindex([z for z in ZONE_META if z in recent["Zone"].unique()])
)

# ── Module-level helpers (used in tab_forecast wind section) ──────────────────

def _cardinal(d: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int((float(d) + 22.5) / 45) % 8]


def _dispersion_verdict(ws: float) -> tuple[str, str]:
    if ws < 10:  return tr("⚠️ Calm — poor dispersion, elevated pollution likely"), "warning"
    if ws < 15:  return tr("🟡 Light wind — moderate dispersion"), "warning"
    if ws < 20:  return tr("🟢 Moderate wind — good dispersion"), "success"
    return tr("✅ Strong wind — excellent dispersion"), "success"


def _svi_color(v: float) -> str:
    if v >= 70: return "#e74c3c"
    if v >= 40: return "#f39c12"
    return "#2ecc71"


# ==================================================
# 7. TABS
# ==================================================

tab_status, tab_forecast, tab_action, tab_spatial = st.tabs([
    tr("🚦 Current Status"),
    tr("🔮 Forecast & Map"),
    tr("🎯 Decisions & Actions"),
    tr("🗺️ Spatial Intelligence"),
])

# --------------------------------------------------
# TAB A — CURRENT STATUS
# --------------------------------------------------

with tab_status:

    # 7a-1. Gauge row
    st.markdown(f"## 🚦 {tr('City-Wide Status')} — {window_label}")
    fig_gauges = render_gauge_row(current_values, WHO_ANNUAL, WHO_SO2_DAILY)
    st.plotly_chart(fig_gauges, width="stretch", config=PLOTLY_CONFIG)

    # 7a-2. KPI metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(tr("Stations above WHO"), f"{n_above} / {len(station_means)}",
             help=G["WHO_annual"])
    k2.metric(
        tr("Highest-risk station"), worst_station["station"],
        f"{worst_station['RiskScore']:.0f} {tr('risk score')}",
        delta_color="inverse",
        help=G["risk_score"],
    )
    k3.metric(
        tr("Worst pollutant"), worst_pollutant,
        f"{city_ratios[worst_pollutant]:.1f}× {tr('WHO limit')}",
        delta_color="inverse",
        help=G["exceedance"],
    )
    k4.metric(tr("Records"), f"{len(recent):,}",
             help=G["D1_data"])

    st.divider()

    # 7a-3. AQI donut + station cards
    col_d, col_c = st.columns([1, 2])
    with col_d:
        render_aqi_donut(current_values, title=tr("City-wide AQI (Latest)"))
        aqi_info = overall_aqi(current_values)
        st.caption(
            tr("European/ICA standard — 6 levels from Good to Extremely poor. ") +
            f"{tr('International reference: US EPA AQI')} ≈ {aqi_info.get('epa_aqi', '—')} "
            f"({aqi_info.get('epa_label', '—')}). " +
            tr("⚠️ Daily-mean approximation — official indices use shorter windows.")
        )
    with col_c:
        st.markdown("##### " + tr("Air quality by station"))
        render_station_aqi_cards(df, n_cols=4)

    st.divider()

    # 7a-4. Pollution calendar
    st.markdown("#### 📅 " + tr("Pollution Calendar"))
    cc1, cc2 = st.columns(2)
    with cc1:
        cal_station = st.selectbox(
            tr("Station"), sorted(df["station"].unique()), key="cal_st"
        )
    with cc2:
        cal_year = st.selectbox(
            tr("Year"),
            sorted(df["Date"].dt.year.unique(), reverse=True),
            key="cal_yr",
        )
    render_aqi_calendar(df, cal_station, cal_year)

    # 7a-5. AQI explainer
    with st.expander(tr("ℹ️ Why two different air quality indicators?")):
        st.markdown("""
**AQI (European/ICA)** answers: *"Is the air safe to go outside?"*
It uses short-term thresholds calibrated for daily public communication.
Good = safe for everyone; Extremely poor = health alert.

**WHO Risk Score** answers: *"How far are we from the ideal long-term target?"*
It compares annual mean concentrations against the WHO 2021 annual guidelines
(PM2.5 = 5, PM10 = 15, NO₂ = 10 µg/m³) — which are very strict.

**Why can they differ?** PM2.5 = 12 µg/m³ →
AQI says "Fairly good" (within daily safe range) but
WHO Risk shows 2.4× the annual limit.
Both are correct — they answer different questions.

Most European cities routinely exceed WHO annual targets.
The WHO score tracks progress toward an ambitious public-health goal,
not a daily alert threshold.

*US EPA reference:* The EPA AQI (0–500 continuous scale) is shown alongside
the European index as an internationally familiar reference point.
        """)

    st.divider()

    # 7a-6. Station risk prioritization table
    st.markdown(f"## 📊 {tr('Station Risk Prioritization')} — {window_label}")
    st.caption(
        tr("Composite risk = mean of (concentration ÷ WHO limit) across PM2.5, "
           "PM10, NO₂ · ×100. Below 100 = within guidelines.")
    )

    col_t, col_c = st.columns([3, 2])
    with col_t:
        st.dataframe(
            display,
            width="stretch", hide_index=True,
            column_config={
                "RiskScore": st.column_config.ProgressColumn(
                    tr("Risk Score"), min_value=0, max_value=300, format="%d"
                )
            },
        )
    with col_c:
        fig_rank = go.Figure(go.Bar(
            x=station_means["RiskScore"],
            y=station_means["station"],
            orientation="h",
            marker_color=[risk_color(s) for s in station_means["RiskScore"]],
            text=station_means["RiskScore"].round(0).astype(int),
            textposition="outside",
        ))
        fig_rank.add_vline(
            x=100, line_dash="dash", line_color="#555", annotation_text="WHO"
        )
        fig_rank.update_layout(
            height=340, margin=dict(l=10, r=40, t=30, b=10),
            xaxis_title=tr("Composite risk score"),
            yaxis=dict(autorange="reversed"),
            title=tr("Priority ranking"),
            dragmode=False,
        )
        st.plotly_chart(fig_rank, width="stretch", config=PLOTLY_CONFIG)

# --------------------------------------------------
# TAB B — FORECAST & MAP
# --------------------------------------------------

with tab_forecast:

    # 7b-1. Forecast exceedance banner
    st.markdown(f"## 🔮 {tr('Forecast Alerts')} — {forecast_date.strftime('%d %b %Y')}")
    st.caption(
        f"{tr('XGBoost next-day forecast based on latest available data')} "
        f"({latest_date.strftime('%d %b %Y')}). "
        + tr("Limits: WHO 2021 annual guidelines (SO₂: 24-hour guideline).")
    )

    if missing_models:
        st.warning(f"{tr('Models not found for:')} {', '.join(missing_models)}")

    if not alerts_df.empty:
        if exceed.empty:
            st.success(
                f"✅ {tr('No WHO exceedances forecast for')} "
                f"{forecast_date.strftime('%d %b %Y')} {tr('at any station.')}"
            )
        else:
            st.error(
                f"⚠️ {len(exceed)} {tr('forecast exceedance(s) for')} "
                f"{forecast_date.strftime('%d %b %Y')} — "
                f"{tr('stations:')} {', '.join(exceed['Station'].unique())}"
            )

    st.divider()

    # 7b-3. GeoAI risk map — station markers
    st.markdown(
        f"## 🗺️ {tr('GeoAI Risk Map')} — {tr('Forecast for')} {forecast_date.strftime('%d %b %Y')}"
    )
    st.caption(
        f"{tr('Based on latest available data')} ({latest_date.strftime('%d %b %Y')}). "
        + tr("Marker colour = worst-case forecast as a ratio of the WHO limit "
             "(max across the 4 pollutants). > 1 = exceedance.")
    )

    if alerts_df.empty:
        st.info(tr("No forecasts available to map."))
    else:
        # Worst-case ratio per station (max over pollutants)
        station_risk = (
            alerts_df.groupby("Station", as_index=False)["Ratio"].max()
            .rename(columns={"Ratio": "MaxRatio"})
        )

        # Coordinates and zone — one row per station
        coords = (
            df.sort_values("Date")
            .groupby("station", as_index=False)
            .agg(
                Latitude=("Latitude", "last"),
                Longitude=("Longitude", "last"),
                Zone=("Zone", "last"),
            )
            .rename(columns={"station": "Station"})
        )
        map_df = station_risk.merge(coords, on="Station", how="left")

        def risk_tier(r: float) -> str:
            if r > 2:  return "🔴 High"
            if r > 1:  return "🟡 Moderate"
            return "🟢 Low"

        map_df["Tier"] = map_df["MaxRatio"].apply(risk_tier)

        fig_map = px.scatter_mapbox(
            map_df,
            lat="Latitude", lon="Longitude",
            color="MaxRatio",
            color_continuous_scale=["#2ecc71", "#f9ca24", "#e74c3c"],
            range_color=[0, 2],
            hover_name="Station",
            hover_data={
                "Zone": True, "Tier": True, "MaxRatio": ":.2f",
                "Latitude": False, "Longitude": False,
            },
            zoom=10,
            center=dict(
                lat=map_df["Latitude"].mean(),
                lon=map_df["Longitude"].mean(),
            ),
            height=460,
        )
        fig_map.update_traces(marker=dict(size=18))
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title="× WHO"),
            dragmode=False,
        )
        st.plotly_chart(fig_map, width="stretch", config=PLOTLY_CONFIG)


        st.divider()

        # 7b-2. Forecast heatmap
        pivot = alerts_df.pivot(index="Station", columns="Pollutant", values="Ratio")
        pivot = pivot[POLLUTANTS]
        fig_hm = px.imshow(
            pivot,
            color_continuous_scale=["#2ecc71", "#f9ca24", "#e74c3c"],
            zmin=0, zmax=2,
            text_auto=".2f",
            aspect="auto",
            title=(
                f"{tr('Forecast for')} {forecast_date.strftime('%d %b %Y')} — "
                + tr("ratio of WHO limit (> 1 = exceedance)")
            ),
        )
        fig_hm.update_layout(height=340, margin=dict(t=80, b=10), dragmode=False)
        st.plotly_chart(fig_hm, width="stretch", config=PLOTLY_CONFIG)

        with st.expander(tr("📋 Full forecast table")):
            st.dataframe(
                alerts_df.sort_values("Ratio", ascending=False),
                width="stretch", hide_index=True,
            )

        st.divider()

        # 7b-4. IDW interpolated surface
        st.markdown("## 🌡️ " + tr("Interpolated Forecast Surface (IDW)"))
        st.caption(
            tr("⚠️ **Interpolated surface (IDW) from 7 station forecasts — "
               "colour = worst-case forecast ratio (× WHO limit), same scale as markers. "
               "Clipped to Gran Bilbao comarca boundary. "
               "Visual approximation only: terrain, local sources, and road density "
               "are not modelled.**")
        )

        _lats   = map_df["Latitude"].values
        _lons   = map_df["Longitude"].values
        _ratios = map_df["MaxRatio"].values

        # Compute IDW grid and apply comarca mask
        _grid_lats, _grid_lons, _z_grid = idw_grid(_lats, _lons, _ratios)
        _z_masked = mask_idw_grid(_grid_lats, _grid_lons, _z_grid, _lats, _lons)

        # Flatten and remove NaN cells
        _glat_mesh, _glon_mesh = np.meshgrid(_grid_lats, _grid_lons, indexing="ij")
        _flat_lats = _glat_mesh.ravel()
        _flat_lons = _glon_mesh.ravel()
        _flat_z    = _z_masked.ravel()
        _valid      = ~np.isnan(_flat_z)
        _flat_lats  = _flat_lats[_valid]
        _flat_lons  = _flat_lons[_valid]
        _flat_z     = _flat_z[_valid]

        fig_idw = go.Figure()

        # Layer 1: IDW surface
        fig_idw.add_trace(go.Scattermapbox(
            lat=_flat_lats,
            lon=_flat_lons,
            mode="markers",
            marker=dict(
                size=20,
                color=_flat_z,
                colorscale=_IDW_CS,
                cmin=0,
                cmax=2,
                opacity=0.65,
                showscale=True,
                colorbar=dict(title="× WHO"),
            ),
            hoverinfo="skip",
            name="IDW surface",
        ))

        # Layer 2: comarca boundary outline
        try:
            _b_lats, _b_lons = load_comarca_boundary()
            fig_idw.add_trace(go.Scattermapbox(
                lat=_b_lats,
                lon=_b_lons,
                mode="lines",
                line=dict(color="white", width=1.5),
                opacity=0.7,
                hoverinfo="skip",
                name="Gran Bilbao boundary",
            ))
        except Exception:
            pass   # boundary optional — map still works without it

        # Layer 3: station markers — size encodes terrain dispersion (TRI)
        _sp_v3 = Path(__file__).parent.parent.parent / "data" / "processed" / "station_spatial_features_v3.csv"
        _sp_v2 = Path(__file__).parent.parent.parent / "data" / "processed" / "station_spatial_features_v2.csv"
        _sp_v1 = Path(__file__).parent.parent.parent / "data" / "processed" / "station_spatial_features.csv"
        _sp_f  = _sp_v3 if _sp_v3.exists() else (_sp_v2 if _sp_v2.exists() else _sp_v1)

        _terrain_available = False
        if _sp_f.exists():
            try:
                _df_sp = pd.read_csv(_sp_f).rename(columns={"station": "Station"})
                _merge_cols = [c for c in ["Station", "elev_tri_2000m",
                               "road_density_1000m", "elev_point_m"]
                               if c in _df_sp.columns]
                map_df = map_df.merge(
                    _df_sp[_merge_cols].drop_duplicates("Station"),
                    on="Station", how="left"
                )
                _terrain_available = (
                    "elev_tri_2000m" in map_df.columns
                    and map_df["elev_tri_2000m"].notna().any()
                )
            except Exception:
                _terrain_available = False

        if _terrain_available:
            _tri_min  = map_df["elev_tri_2000m"].min()
            _tri_max  = map_df["elev_tri_2000m"].max()
            _tri_norm = (map_df["elev_tri_2000m"] - _tri_min) / max(_tri_max - _tri_min, 1)
            _marker_size = (8 + _tri_norm * 16).fillna(12)
            _custom = np.column_stack([
                map_df["MaxRatio"].values,
                map_df["elev_tri_2000m"].fillna(0).values,
                map_df["road_density_1000m"].fillna(0).values,
                map_df["elev_point_m"].fillna(0).values,
            ])
            _hover_tmpl = (
                "<b>%{text}</b><br>"
                "Forecast: %{customdata[0]:.2f}× WHO<br>"
                "─────────<br>"
                "Terrain Relief (2km): %{customdata[1]:.0f} m<br>"
                "Road density (1km): %{customdata[2]:,.0f} m/km²<br>"
                "Elevation: %{customdata[3]:.0f} m"
                "<extra></extra>"
            )
        else:
            _marker_size = 14
            _custom = map_df[["MaxRatio"]].values
            _hover_tmpl = (
                "<b>%{text}</b><br>"
                "Max forecast: %{customdata[0]:.2f}× WHO<extra></extra>"
            )

        fig_idw.add_trace(go.Scattermapbox(
            lat=map_df["Latitude"],
            lon=map_df["Longitude"],
            mode="markers+text",
            marker=dict(
                size=_marker_size,
                color=map_df["MaxRatio"],
                colorscale=_IDW_CS,
                cmin=0,
                cmax=2,
                showscale=False,
                opacity=1.0,
            ),
            text=map_df["Station"],
            textposition="top center",
            customdata=_custom,
            hovertemplate=_hover_tmpl,
            name="Stations",
        ))


        fig_idw.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(
                    lat=float(map_df["Latitude"].mean()),
                    lon=float(map_df["Longitude"].mean()),
                ),
                zoom=10,
            ),
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            dragmode=False,
        )
        st.plotly_chart(fig_idw, width="stretch", config=PLOTLY_CONFIG)

        st.divider()

        # ── Wind Transport Context ────────────────────────────────────────────
        st.markdown("## 💨 " + tr("Wind & Dispersion Context"))
        st.caption(
            tr(
                "Regional wind conditions (ERA5 single grid cell, ~31 km resolution) "
                "that influence the latest available forecast. "
                "Wind speed controls atmospheric dispersion; wind direction determines "
                "which emission sources are upwind of each station."
            )
        )

        # Latest wind conditions
        _latest_wind = df[df["Date"] == latest_date][["WindSpeed","WindDirection","wind_u","wind_v"]].mean()
        _ws   = float(_latest_wind["WindSpeed"])
        _wd   = float(_latest_wind["WindDirection"])

        _verdict, _vcol = _dispersion_verdict(_ws)

        _wc1, _wc2, _wc3, _wc4 = st.columns(4)
        _wc1.metric(tr("Wind speed (D-1)"), f"{_ws:.1f} m/s",
                    help=G["dispersion"])
        _wc2.metric(tr("Wind direction"), f"{_wd:.0f}° ({_cardinal(_wd)})",
                    help=G["ERA5"])
        _wc3.metric(tr("Regime"), "NW/W (sea)" if _cardinal(_wd) in ["NW","W","N"] else "S/SW (land)",
                    help=G["wind_regime"])
        getattr(st, _vcol)(_verdict)

        st.markdown("---")

        # Wind speed × dispersion reference chart
        _wc_left, _wc_right = st.columns(2)

        with _wc_left:
            st.markdown(f"**{tr('Dispersion effect of wind speed')}**")
            st.caption(tr("Historical mean NO₂ across the network by wind speed category."))

            _ws_bins   = [0, 10, 15, 20, 25, 100]
            _ws_labels = ["< 10", "10–15", "15–20", "20–25", "> 25"]
            _no2_means = [29.85, 21.05, 18.56, 16.03, 12.81]

            _df_disp = pd.DataFrame({
                "wind_cat": _ws_labels,
                "mean_NO2": _no2_means,
            })
            _colors_disp = ["#e74c3c","#e67e22","#f39c12","#2ecc71","#27ae60"]

            # Highlight current wind speed bin
            _cur_bin = 0
            for i, (lo, hi) in enumerate(zip(_ws_bins[:-1], _ws_bins[1:])):
                if lo <= _ws < hi:
                    _cur_bin = i
                    break

            _df_disp["color"] = _colors_disp
            _df_disp.loc[_cur_bin, "color"] = "#2c3e50"  # highlight current

            fig_disp = go.Figure(go.Bar(
                x=_df_disp["wind_cat"],
                y=_df_disp["mean_NO2"],
                marker_color=_df_disp["color"],
                text=[f"{v:.1f}" for v in _df_disp["mean_NO2"]],
                textposition="outside",
            ))
            fig_disp.add_annotation(
                x=_ws_labels[_cur_bin],
                y=_no2_means[_cur_bin] + 1.5,
                text=tr("← latest"),
                showarrow=False,
                font=dict(size=11, color="#2c3e50"),
            )
            fig_disp.update_layout(
                height=280,
                margin=dict(t=20, b=10, l=10, r=10),
                xaxis_title=tr("Wind speed (m/s)"),
                yaxis_title="Mean NO₂ (µg/m³)",
                showlegend=False,
                dragmode=False,
            )
            st.plotly_chart(fig_disp, width="stretch", config=PLOTLY_CONFIG)
            st.caption(tr("Strong wind (> 25 m/s) reduces mean NO₂ by 57% vs calm (< 10 m/s)."))

        with _wc_right:
            st.markdown(f"**{tr('Key directional signatures')}**")
            st.caption(tr(
                "Historical mean concentrations by wind direction at key stations. "
                "ERA5 regional wind — indicative only."
            ))

            # MUSKIZ SO2 by direction (from notebook 10d findings)
            _dir8_order = ["N","NE","E","SE","S","SW","W","NW"]
            _muskiz_so2 = [5.81,7.32,5.93,5.19,3.80,3.86,3.42,4.50]
            _maz_no2    = [19.3,23.4,32.4,35.3,32.1,29.7,26.4,20.8]
            _cur_dir    = _cardinal(_wd)
            _cur_idx    = _dir8_order.index(_cur_dir) if _cur_dir in _dir8_order else 0

            fig_rose = go.Figure()
            fig_rose.add_trace(go.Scatterpolar(
                r=_muskiz_so2,
                theta=_dir8_order,
                fill="toself",
                name="MUSKIZ SO₂",
                line_color="#c0392b",
                fillcolor="rgba(192,57,43,0.2)",
            ))
            fig_rose.add_trace(go.Scatterpolar(
                r=[v/3.5 for v in _maz_no2],   # scale to same axis
                theta=_dir8_order,
                fill="toself",
                name="MAZARREDO NO₂ (÷3.5)",
                line_color="#8e44ad",
                fillcolor="rgba(142,68,173,0.15)",
            ))
            # Mark current direction
            fig_rose.add_trace(go.Scatterpolar(
                r=[max(_muskiz_so2)*1.2],
                theta=[_cur_dir],
                mode="markers",
                marker=dict(size=14, color="#2c3e50", symbol="arrow-up"),
                name=tr("Current wind"),
                showlegend=True,
            ))
            fig_rose.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 9]),
                    angularaxis=dict(direction="clockwise"),
                ),
                height=280,
                margin=dict(t=20, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
                showlegend=True,
                dragmode=False,
            )
            st.plotly_chart(fig_rose, width="stretch", config=PLOTLY_CONFIG)
            st.caption(
                tr(
                    "NE wind → MUSKIZ SO₂ peaks (Petronor transport). "
                    "SE wind → MAZARREDO NO₂ peaks (Nervión valley channelling). "
                    "Arrow = current wind direction."
                )
            )

        # Regime interpretation
        _cur_card = _cardinal(_wd)
        if _cur_card in ["NW","W","N"]:
            st.success(
                tr(
                    f"**NW/W regime (latest: {_cur_card})** — sea air from Bay of Biscay. "
                    "Historically: network-wide NO₂ = 16.3 µg/m³ (vs 21.4 µg/m³ under S/SW). "
                    "MUSKIZ: cleaner (Petronor emissions blown inland). "
                    "MAZARREDO: lower canyon trapping."
                )
            )
        elif _cur_card in ["S","SW","SE"]:
            st.warning(
                tr(
                    f"**S/SW regime (latest: {_cur_card})** — recirculation from inland. "
                    "Historically: network-wide NO₂ = 21.4 µg/m³ (+31% vs NW/W). "
                    "MAZARREDO: Nervión valley channelling amplifies urban NO₂. "
                    "BARAKALDO: industrial corridor stagnation risk."
                )
            )
        elif _cur_card == "NE":
            st.warning(
                tr(
                    f"**NE wind (latest)** — onshore from sea at shallow angle. "
                    "MUSKIZ SO₂ historically 1.93× higher under NE vs S wind — "
                    "Petronor emissions trapped against terrain ridge (TRI = 343 m)."
                )
            )
        else:
            st.info(tr(f"Wind direction: {_cur_card}. No dominant source-transport signature for this sector."))

        st.info(
            tr(
                "⚠️ **ERA5 data note:** Wind is a single regional grid cell (~31 km) "
                "shared by all stations. Direction signatures represent "
                "synoptic-scale patterns, not station micrometeorology. "
                "Source: Open-Meteo ERA5 archive (CC BY 4.0) · Notebook 10d."
            )
        )


# --------------------------------------------------
# TAB C — DECISIONS & ACTIONS
# --------------------------------------------------

with tab_action:

    # 7c-0. Decision-maker action summary (prominent)
    _worst_station_name = station_means.iloc[0]["station"] if not station_means.empty else "—"
    _worst_score = int(station_means.iloc[0]["RiskScore"]) if not station_means.empty else 0
    _worst_zone  = station_means.iloc[0]["Zone"] if not station_means.empty else "—"
    if _worst_score >= 100:
        st.error(
            f"🔴 **{tr('Priority action')}:** {_worst_station_name} ({_worst_zone}) — "
            f"Risk score {_worst_score}/300. "
            + tr("See zone recommendation below for suggested intervention.")
        )
    else:
        st.success(tr("✅ All stations within acceptable risk range for this period."))

    st.divider()

    # 7c-1. Scenario simulator
    st.markdown("## 🧪 " + tr("Scenario Simulator"))
    st.info(
        "💡 " + tr(
            "**In plain English:** This tool estimates how much the risk score would "  
            "change if pollution levels were reduced — for example by restricting "  
            "traffic or upgrading industrial filters. The numbers are model-based "  
            "estimates, not guarantees."
        )
    )
    st.caption(
        tr("Explore how the composite risk score responds to interventions. "
           "Two modes with different epistemic status — read the labels.")
    )

    sim_mode = st.radio(
        tr("Simulation mode"),
        ["Model-based counterfactual", "Policy elasticity assumption"],
        horizontal=True,
    )

    _sim_list = sorted(df["station"].unique().tolist())
    _fav      = get_fav_station(_sim_list)
    _sim_idx  = _sim_list.index(_fav) if _fav in _sim_list else 0
    sim_station = st.selectbox(
        tr("Station to simulate"), _sim_list, index=_sim_idx, key="sim_station"
    )

    sdf_sim = df[df["station"] == sim_station].sort_values("Date")

    # Baseline forecast for all 4 pollutants at this station
    baseline = {}
    for pollutant in POLLUTANTS:
        bundle = bundles.get(pollutant)
        if bundle is None:
            continue
        feats      = bundle["features"]
        prep       = prepare_features(sdf_sim, feats, station_codes=STATION_CODES)
        prep_filled = prep.copy()
        prep_filled[feats] = prep_filled[feats].ffill()
        valid = prep_filled.dropna(subset=feats)
        if valid.empty:
            continue
        X_last = valid[feats].iloc[[-1]]
        baseline[pollutant] = max(float(bundle["model"].predict(X_last)[0]), 0.0)

    def composite_from_preds(pred_dict: dict) -> float:
        """Composite risk score from a {pollutant: value} dict — core pollutants only."""
        ratios = [
            pred_dict[p] / WHO_ANNUAL[p]
            for p in CORE_POLLUTANTS if p in pred_dict
        ]
        return float(np.mean(ratios) * 100) if ratios else 0.0

    baseline_score = composite_from_preds(baseline)

    if sim_mode == "Model-based counterfactual":
        st.info(
            tr("🟢 **Honest what-if:** perturbs features the model actually uses "
               "(weather + latest available levels (D-1)), then re-runs `model.predict()`. "
               "This is a real model output.")
        )

        c1, c2, c3 = st.columns(3)
        wind_mult   = c1.slider(tr("Wind speed ×"),          0.5, 2.0, 1.0, 0.1)
        precip_mult = c2.slider(tr("Precipitation ×"),        0.0, 3.0, 1.0, 0.1)
        today_mult  = c3.slider(tr("Latest day's pollutant ×"), 0.5, 1.5, 1.0, 0.05)

        scenario = {}
        for pollutant in POLLUTANTS:
            bundle = bundles.get(pollutant)
            if bundle is None:
                continue
            feats      = bundle["features"]
            prep       = prepare_features(sdf_sim, feats, station_codes=STATION_CODES)
            prep_filled = prep.copy()
            prep_filled[feats] = prep_filled[feats].ffill()
            valid = prep_filled.dropna(subset=feats)
            if valid.empty:
                continue
            row = valid.iloc[[-1]].copy()
            # Scale all wind representations together: preserves direction, changes magnitude
            for _wcol in ("WindSpeed", "wind_u", "wind_v"):
                if _wcol in row.columns:
                    row[_wcol] = row[_wcol] * wind_mult
            if "Precipitation" in row.columns:
                row["Precipitation"] = row["Precipitation"] * precip_mult
            if pollutant in row.columns:
                row[pollutant] = row[pollutant] * today_mult
            X = row[feats]
            scenario[pollutant] = max(float(bundle["model"].predict(X)[0]), 0.0)

        scenario_score = composite_from_preds(scenario)

    else:  # Policy elasticity assumption
        st.warning(
            tr("🟡 **Policy elasticity — NOT a model prediction.** The model has no "
               "traffic/emission features, so this applies an *assumed* linear "
               "response. Use for illustrative policy framing only; "
               "the elasticity is a stated assumption.")
        )

        c1, c2 = st.columns(2)
        traffic_cut  = c1.slider(tr("Traffic reduction (%)"), 0, 50, 0, 5)
        emission_cut = c2.slider(tr("Industrial emission reduction (%)"), 0, 50, 0, 5)

        # Stated elasticities — not 1:1
        NO2_ELASTICITY = 0.7   # ~70% traffic-attributable
        PM_ELASTICITY  = 0.4   # partly traffic, partly background
        SO2_ELASTICITY = 0.6   # ~industrial-attributable

        scenario = dict(baseline)
        if "NO2"   in scenario:
            scenario["NO2"]   *= (1 - NO2_ELASTICITY  * traffic_cut  / 100)
        if "PM2.5" in scenario:
            scenario["PM2.5"] *= (1 - PM_ELASTICITY   * traffic_cut  / 100)
        if "PM10"  in scenario:
            scenario["PM10"]  *= (1 - PM_ELASTICITY   * traffic_cut  / 100)
        if "SO2"   in scenario:
            scenario["SO2"]   *= (1 - SO2_ELASTICITY  * emission_cut / 100)

        scenario_score = composite_from_preds(scenario)

    # 7c-2. Scenario results
    m1, m2, m3 = st.columns(3)
    m1.metric(tr("Baseline risk score"),  f"{baseline_score:.0f}",
              help=G["risk_score"])
    m2.metric(tr("Scenario risk score"),  f"{scenario_score:.0f}",
              help=G["risk_score"])
    delta = (
        (scenario_score - baseline_score) / baseline_score * 100
        if baseline_score else 0
    )
    m3.metric(tr("Change"), f"{delta:+.1f}%", delta_color="inverse")

    comp = pd.DataFrame({
        "Pollutant": [p for p in POLLUTANTS if p in baseline],
        "Baseline":  [round(baseline[p], 1) for p in POLLUTANTS if p in baseline],
        "Scenario":  [
            round(scenario.get(p, baseline[p]), 1)
            for p in POLLUTANTS if p in baseline
        ],
    })
    with st.expander(tr("📋 Scenario result detail"), expanded=False):
        st.dataframe(comp, width="stretch", hide_index=True)

    st.divider()

    # 7c-3. Executive summary
    st.markdown("## 📋 " + tr("Executive Summary"))

    worst_zone_line = ""
    if not zone_means.empty:
        zone_ratios = {}
        for zone, meta in ZONE_META.items():
            if zone not in zone_means.index:
                continue
            key_p = meta["key_pollutant"]
            lim   = WHO_SO2_DAILY if key_p == "SO2" else WHO_ANNUAL.get(key_p)
            if lim:
                zone_ratios[zone] = zone_means.loc[zone, key_p] / lim
        if zone_ratios:
            top_zone = max(zone_ratios, key=zone_ratios.get)
            worst_zone_line = (
                f"The **{top_zone}** zone shows the highest pressure on its key "
                f"pollutant ({ZONE_META[top_zone]['key_pollutant']}, "
                f"{zone_ratios[top_zone]:.1f}× WHO)."
            )

    if not alerts_df.empty and not exceed.empty:
        forecast_line = (
            f"Forecast for {forecast_date.strftime('%d %b %Y')} flags "
            f"**{len(exceed)} WHO exceedance(s)** "
            f"at: {', '.join(exceed['Station'].unique())}."
        )
    elif not alerts_df.empty:
        forecast_line = (
            f"No WHO exceedances forecast for "
            f"{forecast_date.strftime('%d %b %Y')}."
        )
    else:
        forecast_line = "Forecast data unavailable."

    summary = f"""
Over the **{window_label.lower()}**, **{n_above} of {len(station_means)}**
monitoring stations exceeded WHO annual guidelines on the composite risk index.
The highest-risk station is **{worst_station['station']}**
({worst_station['Zone']} zone) with a composite score of
**{worst_station['RiskScore']:.0f}**, and the most critical pollutant
city-wide is **{worst_pollutant}** at
**{city_ratios[worst_pollutant]:.1f}× the WHO limit**.

{worst_zone_line}

{forecast_line}
    """

    with st.container(border=True):
        st.markdown(summary)
        st.caption(
            tr("Auto-generated from the current analysis window and latest forecasts.") +
            f" {tr('Window:')} {window_label} · {tr('Latest data:')} {latest_date.strftime('%d %b %Y')}."
        )

    st.divider()

    # 7c-4. Zone-level recommendations
    st.markdown("## 🗺️ " + tr("Zone Recommendations"))

    ZONE_ACTION_TIERS = {
        "Urban": {
            "low":  "Maintain routine traffic monitoring; no intervention needed.",
            "mid":  "Promote public transport and soft mobility during peak hours; "
                    "monitor the NO₂ weekly cycle.",
            "high": "Activate low-emission-zone enforcement and rush-hour restrictions; "
                    "issue public NO₂ advisories.",
        },
        "Industrial": {
            "low":  "Routine stack monitoring; no escalation.",
            "mid":  "Increase inspection frequency; coordinate with operators on PM episodes.",
            "high": "Enforce temporary emission controls; mandatory continuous stack "
                    "monitoring; investigate PM source episodes.",
        },
        "Port": {
            "low":  "Standard port-activity monitoring.",
            "mid":  "Encourage cleaner marine fuels for docked vessels.",
            "high": "Prioritize shore-power (cold ironing) for berthed ships; "
                    "restrict high-sulphur fuel use during episodes.",
        },
        "Coastal": {
            "low":  "Favourable dispersion; maintain baseline monitoring.",
            "mid":  "Watch for marine-PM10 transport events; protect green buffers.",
            "high": "Unusual for this zone — verify sensors, then treat as a "
                    "transport/dust episode.",
        },
        "Refinery": {
            "low":  "Routine SO₂ monitoring; refinery operating normally.",
            "mid":  "Increase SO₂ sampling cadence; flag upcoming low-wind days.",
            "high": "Event-based alerting: coordinate refinery maintenance windows "
                    "with forecast; notify sensitive population near Petronor.",
        },
    }

    def action_tier(ratio: float) -> str:
        """Map a WHO ratio to an urgency tier."""
        if ratio is None: return "low"
        if ratio > 2:     return "high"
        if ratio > 1:     return "mid"
        return "low"

    for zone, meta in ZONE_META.items():
        if zone not in zone_means.index:
            continue
        zrow  = zone_means.loc[zone]
        key_p = meta["key_pollutant"]
        limit = WHO_SO2_DAILY if key_p == "SO2" else WHO_ANNUAL.get(key_p)
        ratio = zrow[key_p] / limit if limit else None

        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown(f"### {meta['icon']} {zone}")
                _poll_help = {"NO2": G["NO2"], "PM2.5": G["PM25"],
                              "PM10": G["PM10"], "SO2": G["SO2"]}
                st.metric(
                    f"{tr('Key pollutant:')} {key_p}",
                    f"{zrow[key_p]:.1f} µg/m³",
                    f"{ratio:.1f}× WHO" if ratio else None,
                    delta_color="inverse" if ratio and ratio > 1 else "normal",
                    help=_poll_help.get(key_p, G["WHO_annual"]),
                )
            with c2:
                st.markdown(f"**{tr('Profile:')}** {meta['description']}")
                tier       = action_tier(ratio)
                tier_badge = {
                    "low":  tr("🟢 Routine"),
                    "mid":  tr("🟡 Elevated"),
                    "high": tr("🔴 Action required"),
                }[tier]
                action_text = ZONE_ACTION_TIERS.get(zone, {}).get(tier, "—")
                st.markdown(f"**{tr('Status:')}** {tier_badge}")
                st.markdown(f"**{tr('Recommended action:')}** {tr(action_text)}")
                # Spatial context from GIS analysis (notebooks 10a/10b/10c)
                _sp_ctx = {
                    "Urban":      tr("🗺️ **Spatial driver:** Road density 19,060 m/km² + city centre 501 m → structural NO₂ source (r = −0.77 dist-centre × NO₂)."),
                    "Industrial": tr("🗺️ **Spatial driver:** AP-8 motorway 354 m from BARAKALDO + industrial land use 10–21% within 1 km → elevated PM2.5 (r = −0.54)."),
                    "Port":       tr("🗺️ **Spatial driver:** 784 m from Port of Bilbao. Under SW wind: vessel SO₂ concentrated near SANTURCE sensor."),
                    "Coastal":    tr("🗺️ **Spatial driver:** Lowest road density (9,933 m/km²) + 2.6 km from coast → NW sea breeze provides consistent flushing."),
                    "Refinery":   tr("🗺️ **Spatial driver:** 2.4 km from Petronor, 34.6% industrial land use — yet lowest PM2.5: coastal + TRI 343 m overrides emission proximity."),
                }.get(zone, "")
                if _sp_ctx:
                    st.markdown(_sp_ctx)

    st.divider()



    # 7c-5. Export
    st.markdown("## 📥 " + tr("Export for Reporting"))

    e1, e2 = st.columns(2)
    with e1:
        csv_priority = display.to_csv(index=False).encode("utf-8")
        st.download_button(
            tr("⬇️ Station risk ranking (CSV)"),
            data=csv_priority,
            file_name=f"risk_ranking_{latest_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width="stretch",
        )
    with e2:
        if not alerts_df.empty:
            csv_alerts = alerts_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                tr(f"⬇️ Forecast alerts — {forecast_date.strftime('%d %b %Y')} (CSV)"),
                data=csv_alerts,
                file_name=f"forecast_alerts_{latest_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch",
            )

        
    st.divider()
    e3, e4 = st.columns([2, 3])
    with e3:
        pdf_bytes_monthly = generate_monthly_report(
            latest_date      = latest_date,
            window_label     = window_label,
            station_means    = station_means,
            alerts_df        = alerts_df,
            zone_means       = zone_means,
            summary_text     = summary,          # متغیر summary که قبلاً ساخته شده
            n_above          = n_above,
            worst_station    = worst_station,
            worst_pollutant  = worst_pollutant,
            city_ratios      = city_ratios,
            who_annual       = WHO_ANNUAL,
            eu_annual        = EU_ANNUAL,
            zone_meta        = ZONE_META,
            zone_action_tiers= ZONE_ACTION_TIERS,
        )
    
        st.download_button(
            label    = tr("📄 Download Monthly Risk Report (PDF)"),
            data     = pdf_bytes_monthly,
            file_name= f"risk_report_{latest_date.strftime('%Y%m%d')}.pdf",
            mime     = "application/pdf",
            type     = "primary",
            width    = "stretch",
        )

    with e4:
        st.caption(
            tr("3-page report: executive summary, station risk ranking, "
               "WHO vs EU comparison, forecast alerts for the next day, "
               "and zone-level recommended actions.")
        )

    st.caption(
        tr("Data: Basque Government air-quality network + Open-Meteo (CC BY 4.0) · "
           "Forecasts: XGBoost next-day models (see Forecasting page for validation).")
    )


# ==================================================
# TAB D — SPATIAL INTELLIGENCE
# ==================================================

with tab_spatial:

    st.markdown("## 🗺️ " + tr("Spatial Intelligence"))
    st.caption(
        tr(
            "Structural spatial context for each monitoring station — "
            "land use, distances to emission sources, and terrain. "
            "Based on OSM buffer analysis (notebooks 10a/10b) and "
            "Copernicus GLO-30 DEM (notebook 10c). "
            "n = 7 stations — correlations are indicative, not inferential."
        )
    )

    # ── Load spatial features v3 (buffer + distance + elevation) ─────────────
    _SP_V3  = Path(__file__).parent.parent.parent / "data" / "processed" / "station_spatial_features_v3.csv"
    _SP_V2  = Path(__file__).parent.parent.parent / "data" / "processed" / "station_spatial_features_v2.csv"
    _SP_V1  = Path(__file__).parent.parent.parent / "data" / "processed" / "station_spatial_features.csv"
    _SP_FILE = _SP_V3 if _SP_V3.exists() else (_SP_V2 if _SP_V2.exists() else _SP_V1)

    if not _SP_FILE.exists():
        st.info(
            tr("Spatial features file not found. "
               "Run notebooks 10a, 10b, 10c locally and commit "
               "station_spatial_features_v3.csv to data/processed/.")
        )
        st.stop()

    @st.cache_data
    def _load_spatial_v3() -> pd.DataFrame:
        return pd.read_csv(_SP_FILE)

    df_sp = _load_spatial_v3()

    # Merge with air quality means for scatter plots
    _poll_means = (
        df.groupby("station")[POLLUTANTS].mean()
        .reset_index()
        .rename(columns={"PM2.5": "mean_PM25", "PM10": "mean_PM10",
                         "NO2": "mean_NO2", "SO2": "mean_SO2"})
    )
    df_sp = df_sp.merge(
        _poll_means.rename(columns={"PM2.5": "mean_PM25", "PM10": "mean_PM10",
                                     "NO2": "mean_NO2", "SO2": "mean_SO2"}),
        on="station", how="left"
    )
    # Handle case where merge columns may already exist
    for old, new in [("PM2.5","mean_PM25"),("PM10","mean_PM10"),
                     ("NO2","mean_NO2"),("SO2","mean_SO2")]:
        if old in df_sp.columns and new not in df_sp.columns:
            df_sp[new] = df_sp[old]
    _poll_means2 = df.groupby("station")[POLLUTANTS].mean().reset_index()
    df_sp = df_sp.merge(_poll_means2, on="station", how="left", suffixes=("","_aq"))

    _ZONE_CLR = {
        "Urban":      "#8e44ad", "Industrial": "#e67e22",
        "Port":       "#2980b9", "Coastal":    "#1abc9c",
        "Refinery":   "#c0392b",
    }

    # ── IDEA 1 — Spatial DNA cards per station ────────────────────────────────
    st.markdown("### " + tr("Station Spatial Profile"))
    st.caption(tr("Select a station to expand its full spatial context."))
    st.info(
        "💡 " + tr(
            "**In plain English:** This tab answers 'why is this station always high?' "  
            "— by looking at what structurally surrounds it: roads, industrial sites, "  
            "and terrain. These factors do not change day to day."
        )
    )

    for _, sp_row in df_sp.iterrows():
        zone_clr = _ZONE_CLR.get(sp_row.get("zone",""), "#555")
        with st.expander(
            f"{sp_row['station']}  —  {sp_row.get('zone','')}  |  "
            f"NO₂ {sp_row.get('NO2', sp_row.get('mean_NO2', 0)):.1f} µg/m³  ·  "
            f"PM2.5 {sp_row.get('PM2.5', sp_row.get('mean_PM25', 0)):.1f} µg/m³",
            expanded=False,
        ):
            dna1, dna2, dna3 = st.columns(3)

            with dna1:
                st.markdown(f"**{tr('Land Use')}** *(1 000 m buffer)*")
                for lbl, col, icon in [
                    ("Green",       "green_pct_1000m",       "🌿"),
                    ("Industrial",  "industrial_pct_1000m",  "🏭"),
                    ("Residential", "residential_pct_1000m", "🏘️"),
                    ("Road density","road_density_1000m",     "🛣️"),
                ]:
                    val = sp_row.get(col)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        unit = " m/km²" if "density" in col else "%"
                        st.markdown(f"{icon} **{lbl}:** {val:.1f}{unit}")

            with dna2:
                st.markdown(f"**{tr('Distances to Sources')}**")
                for lbl, col, icon in [
                    ("Port of Bilbao",  "dist_port_bilbao_m",   "⚓"),
                    ("Petronor",        "dist_petronor_m",      "🛢️"),
                    ("AP-8 Motorway",   "dist_ap8_barakaldo_m", "🚗"),
                    ("City Centre",     "dist_bilbao_centre_m", "🏙️"),
                    ("Coast",           "dist_coast_getxo_m",   "🌊"),
                ]:
                    val = sp_row.get(col)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        st.markdown(f"{icon} **{lbl}:** {val/1000:.1f} km")

            with dna3:
                st.markdown(f"**{tr('Terrain')}** *(Copernicus GLO-30)*")
                for lbl, col, icon in [
                    ("Elevation",  "elev_point_m",  "⛰️"),
                    ("Slope",      "slope_deg",      "📐"),
                    ("TRI 2km",    "elev_tri_2000m", "🏔️"),
                    ("Mean elev",  "elev_mean_1000m","📊"),
                ]:
                    val = sp_row.get(col)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        unit = "°" if col == "slope_deg" else " m"
                        st.markdown(f"{icon} **{lbl}:** {val:.1f}{unit}")

            # ── Structural driver narrative (GeoAI story) ─────────────────
            _drivers = {
                "BARAKALDO":      tr("📍 Highest road density (21,267 m/km²) + 354 m from AP-8 → structural driver of elevated PM2.5 and NO₂."),
                "MAZARREDO":      tr("📍 77% residential land use yet highest NO₂: road density 19,060 m/km² + urban canyon + 501 m from city centre."),
                "BASAURI":        tr("📍 Industrial land use 32% within 500 m (drops to 21% at 1 km) — emission sources concentrated immediately around sensor."),
                "ERANDIO":        tr("📍 1,264 m from AP-8 + 18,631 m/km² road density — traffic corridor exposure without urban canyon effect."),
                "MUSKIZ":         tr("📍 34.6% industrial (Petronor) yet lowest PM2.5 (6.5 µg/m³): TRI 343 m + coastal NW breeze override emission proximity."),
                "SANTURCE":       tr("📍 784 m from Port of Bilbao + elevation 93 m + TRI 445 m — port proximity drives SO₂; terrain aids dispersion."),
                "ALGORTA_BBIZI2": tr("📍 Lowest road density (9,933 m/km²) + 2.6 km coast + NW sea breeze → structural cleanest station in the network."),
            }
            _drv = _drivers.get(sp_row.get('station',''), '')
            if _drv:
                st.info(_drv)

    st.divider()

    # ── IDEA 2 — Interactive scatter: spatial driver vs pollutant ─────────────
    st.markdown("### " + tr("Spatial Driver Explorer"))
    st.caption(
        tr(
            "Choose a spatial feature and a pollutant to explore the structural "
            "relationship. OLS trendline shown. n = 7 — indicative only."
        )
    )

    # Feature options — only include columns that exist in df_sp
    _FEATURE_OPTIONS = {
        "Road density (1 000 m)":      "road_density_1000m",
        "Industrial % (1 000 m)":      "industrial_pct_1000m",
        "Green % (1 000 m)":           "green_pct_1000m",
        "Distance to City Centre (km)": "dist_bilbao_centre_m",
        "Distance to Port (km)":        "dist_port_bilbao_m",
        "Distance to AP-8 (km)":        "dist_ap8_barakaldo_m",
        "Elevation (m)":               "elev_point_m",
        "Terrain Relief Index 2km (m)": "elev_tri_2000m",
        "Slope (°)":                   "slope_deg",
    }
    _available_features = {
        lbl: col for lbl, col in _FEATURE_OPTIONS.items()
        if col in df_sp.columns
    }

    _POLLUTANT_OPTIONS = {
        "NO₂ (µg/m³)":   "NO2",
        "PM2.5 (µg/m³)": "PM2.5",
        "PM10 (µg/m³)":  "PM10",
        "SO₂ (µg/m³)":   "SO2",
    }

    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        x_label = st.selectbox(
            tr("Spatial feature (X axis)"),
            list(_available_features.keys()),
            index=0,
        )
    with sc_col2:
        y_label = st.selectbox(
            tr("Pollutant (Y axis)"),
            list(_POLLUTANT_OPTIONS.keys()),
            index=0,
        )

    x_col   = _available_features[x_label]
    y_col   = _POLLUTANT_OPTIONS[y_label]

    # Build scatter dataframe
    df_scatter = df_sp[["station", "zone", x_col, y_col]].dropna().copy()

    # Convert distance columns from m to km for readability
    if "dist_" in x_col:
        df_scatter[x_col] = df_scatter[x_col] / 1000
        x_display = x_label  # already says "km"
    else:
        x_display = x_label

    if len(df_scatter) < 3:
        st.warning(tr("Not enough data for this combination."))
    else:
        # Compute Pearson r
        _r = df_scatter[[x_col, y_col]].corr().iloc[0, 1]

        fig_sc = px.scatter(
            df_scatter,
            x=x_col,
            y=y_col,
            text="station",
            trendline="ols",
            color="zone",
            color_discrete_map=_ZONE_CLR,
            labels={x_col: x_display, y_col: y_label},
            title=f"{x_display} vs {y_label}  |  Pearson r = {_r:.2f}  (n = {len(df_scatter)})",
        )
        fig_sc.update_traces(
            textposition="top center",
            selector=dict(mode="markers+text"),
        )
        fig_sc.update_layout(
            height=420,
            margin=dict(t=55, b=20),
            legend=dict(orientation="h", y=-0.15),
            dragmode=False,
        )
        st.plotly_chart(fig_sc, width="stretch", config=PLOTLY_CONFIG)

        # Interpretation hint
        if abs(_r) >= 0.7:
            hint = tr("Strong signal") + f" (|r| ≥ 0.7)"
        elif abs(_r) >= 0.4:
            hint = tr("Moderate signal") + f" (|r| ≥ 0.4)"
        else:
            hint = tr("Weak or no signal") + f" (|r| < 0.4)"
        color_hint = "success" if abs(_r) >= 0.7 else ("warning" if abs(_r) >= 0.4 else "info")
        getattr(st, color_hint)(hint)


    st.divider()

    # ── IDEA 4 — Structural Vulnerability Index (SVI) ────────────────────────
    st.markdown("### " + tr("Structural Vulnerability Index (SVI)"))
    st.caption(
        tr(
            "A composite spatial index combining the three strongest structural "
            "predictors of air pollution: road density (1km buffer), distance to "
            "city centre, and Terrain Relief Index (2km). "
            "Higher score = structurally more exposed to poor air quality, "
            "independent of daily weather. "
            "n = 7 — indicative only. "
            "SVI does not capture port or industrial point-source emissions."
        )
    )

    # Compute SVI from available spatial features
    _svi_cols = {
        "road_density_1000m":   +1,   # higher road density = worse
        "dist_bilbao_centre_m": -1,   # closer to centre = worse (invert)
        "elev_tri_2000m":       -1,   # higher TRI = better dispersion (invert)
    }
    _svi_available = all(c in df_sp.columns for c in _svi_cols)

    if not _svi_available:
        st.info(tr("SVI requires v3 spatial features (road density + distances + terrain). "
                   "Run notebooks 10a, 10b, 10c and commit station_spatial_features_v3.csv."))
    else:
        _df_svi = df_sp[["station", "zone"] + list(_svi_cols.keys())].copy().dropna()

        # Z-score normalise each feature, apply directionality sign
        for col, sign in _svi_cols.items():
            _mean = _df_svi[col].mean()
            _std  = _df_svi[col].std()
            _df_svi[f"_z_{col}"] = sign * (_df_svi[col] - _mean) / max(_std, 1e-9)

        _z_cols = [f"_z_{c}" for c in _svi_cols]
        _df_svi["_svi_raw"] = _df_svi[_z_cols].mean(axis=1)

        # Rescale to 0–100
        _svi_min = _df_svi["_svi_raw"].min()
        _svi_max = _df_svi["_svi_raw"].max()
        _df_svi["SVI"] = (
            (_df_svi["_svi_raw"] - _svi_min) / max(_svi_max - _svi_min, 1e-9) * 100
        ).round(1)
        _df_svi = _df_svi.sort_values("SVI", ascending=False).reset_index(drop=True)

        # Merge with observed pollutant means for validation scatter
        _obs_means = df.groupby("station")[POLLUTANTS].mean().reset_index()
        _df_svi = _df_svi.merge(_obs_means, on="station", how="left")

        # ── Row 1: KPI bar chart (ranking) ────────────────────────────────────
        svi_col1, svi_col2 = st.columns([3, 2])

        with svi_col1:
            fig_svi_bar = go.Figure(go.Bar(
                x=_df_svi["SVI"],
                y=_df_svi["station"],
                orientation="h",
                marker_color=[_svi_color(v) for v in _df_svi["SVI"]],
                text=_df_svi["SVI"].apply(lambda v: f"{v:.0f}"),
                textposition="outside",
                customdata=_df_svi[["zone", "road_density_1000m",
                                     "dist_bilbao_centre_m",
                                     "elev_tri_2000m"]].values,
                hovertemplate=(
                    "<b>%{y}</b> (%{customdata[0]})<br>"
                    "SVI: %{x:.1f} / 100<br>"
                    "Road density: %{customdata[1]:,.0f} m/km²<br>"
                    "Dist. centre: %{customdata[2]:.0f} m<br>"
                    "TRI 2km: %{customdata[3]:.0f} m"
                    "<extra></extra>"
                ),
            ))
            fig_svi_bar.add_vline(
                x=50, line_dash="dash", line_color="#7f8c8d",
                annotation_text=tr("threshold"), annotation_position="top",
            )
            fig_svi_bar.update_layout(
                height=320,
                margin=dict(l=10, r=60, t=30, b=10),
                xaxis=dict(title="SVI (0 = best, 100 = most exposed)", range=[0, 115]),
                yaxis=dict(autorange="reversed"),
                title=tr("Station Ranking — Structural Vulnerability"),
                dragmode=False,
            )
            st.plotly_chart(fig_svi_bar, width="stretch", config=PLOTLY_CONFIG)

        with svi_col2:
            st.markdown(f"**{tr('Index components')}**")
            st.markdown(
                tr(
                    "- **Road density** (1km): traffic emission exposure\n"
                    "- **Distance to city centre**: urban traffic intensity\n"
                    "- **Terrain Relief Index** (2km): atmospheric dispersion capacity\n"
                    "\n"
                    "Each feature Z-score normalised, mean-aggregated, "
                    "rescaled 0–100."
                )
            )
            st.markdown("---")
            st.markdown(f"**{tr('Highest exposure')}**")
            _top = _df_svi.iloc[0]
            st.error(f"🔴 {_top['station']} — SVI {_top['SVI']:.0f}/100")
            st.markdown(f"**{tr('Lowest exposure')}**")
            _bot = _df_svi.iloc[-1]
            st.success(f"🟢 {_bot['station']} — SVI {_bot['SVI']:.0f}/100")

        # ── Row 2: Validation scatter — SVI vs observed NO₂ ──────────────────
        st.markdown(f"#### {tr('Validation: SVI vs observed mean NO₂')}")
        st.caption(
            tr(
                "If SVI is a meaningful structural predictor, stations with higher SVI "
                "should record higher long-term mean NO₂. "
                "Pearson r shown."
            )
        )

        _r_svi_no2  = _df_svi["SVI"].corr(_df_svi["NO2"])
        _r_svi_pm25 = _df_svi["SVI"].corr(_df_svi["PM2.5"])

        sc_v1, sc_v2 = st.columns(2)

        with sc_v1:
            fig_val1 = px.scatter(
                _df_svi,
                x="SVI", y="NO2",
                text="station",
                trendline="ols",
                color="zone",
                color_discrete_map=_ZONE_CLR,
                labels={"SVI": "Structural Vulnerability Index (0–100)",
                        "NO2": "Mean NO₂ (µg/m³)"},
                title=f"SVI vs Mean NO₂  |  r = {_r_svi_no2:.2f}",
            )
            fig_val1.update_traces(
                textposition="top center",
                selector=dict(mode="markers+text"),
            )
            fig_val1.update_layout(
                height=360, margin=dict(t=50, b=10), showlegend=False,
                dragmode=False,
            )
            st.plotly_chart(fig_val1, width="stretch", config=PLOTLY_CONFIG)

        with sc_v2:
            fig_val2 = px.scatter(
                _df_svi,
                x="SVI", y="PM2.5",
                text="station",
                trendline="ols",
                color="zone",
                color_discrete_map=_ZONE_CLR,
                labels={"SVI": "Structural Vulnerability Index (0–100)",
                        "PM2.5": "Mean PM2.5 (µg/m³)"},
                title=f"SVI vs Mean PM2.5  |  r = {_r_svi_pm25:.2f}",
            )
            fig_val2.update_traces(
                textposition="top center",
                selector=dict(mode="markers+text"),
            )
            fig_val2.update_layout(
                height=360, margin=dict(t=50, b=10), showlegend=False,
                dragmode=False,
            )
            st.plotly_chart(fig_val2, width="stretch", config=PLOTLY_CONFIG)

        st.info(
            tr(
                "⚠️ **Interpretation note:** SVI captures structural traffic "
                "and terrain drivers only. SANTURCE (port) and MUSKIZ (refinery) "
                "have low SVI scores because their pollution comes from point sources "
                "(vessel emissions, industrial stack) not captured by road density "
                "or terrain. SVI should not be used as a standalone forecast tool."
            )
        )

    # ── Full data table ───────────────────────────────────────────────────────
    with st.expander(tr("📋 Full spatial features table (v3)")):
        _show_cols = [c for c in df_sp.columns
                      if not c.endswith("_aq") and c not in ["ndvi_note"]]
        _display_sp = df_sp[_show_cols].copy()
        _float_cols = _display_sp.select_dtypes("float64").columns
        _display_sp[_float_cols] = _display_sp[_float_cols].round(2)
        st.dataframe(_display_sp, width="stretch", hide_index=True)
        st.download_button(
            tr("⬇️ Download spatial features (CSV)"),
            data=_display_sp.to_csv(index=False).encode("utf-8"),
            file_name="station_spatial_features_v3.csv",
            mime="text/csv",
        )

    st.info(
        tr(
            "**Sources:** OSM land use via OSMnx · Haversine distances · "
            "Copernicus GLO-30 DEM (30 m). "
            "Notebooks: 10a (buffer) · 10b (distance) · 10c (elevation). "
            "n = 7 stations — correlations are exploratory only."
        )
    )