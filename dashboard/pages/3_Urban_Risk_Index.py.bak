"""
dashboard/pages/3_GeoAI_Spatial_Analysis.py

GeoAI Spatial Analysis — Greater Bilbao Monitoring Network

Replaces the former Urban Risk Index page.
Four analytical tabs:
  Tab 1 — Station DNA        : full spatial profile per station
  Tab 2 — Spatial Drivers    : correlation matrix + interactive scatter
  Tab 3 — Terrain & Dispersion: DEM elevation + TRI + Copernicus GLO-30
  Tab 4 — Wind Transport     : direction × pollutant + dispersion effect

Data sources:
  - station_spatial_features_v3.csv  (notebooks 10a/10b/10c)
  - air_quality_weather.parquet      (config.load_data)
  - ERA5 wind via Open-Meteo         (CC BY 4.0)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from config import load_data, WHO_ANNUAL, ZONE_META

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GeoAI Spatial Analysis",
    page_icon="🌍",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]

ZONE_CLR = {
    "Urban":      "#8e44ad",
    "Industrial": "#e67e22",
    "Port":       "#2980b9",
    "Coastal":    "#1abc9c",
    "Refinery":   "#c0392b",
}

STATION_ZONES = {
    "ALGORTA_BBIZI2": "Coastal",
    "BARAKALDO":      "Industrial",
    "BASAURI":        "Industrial",
    "ERANDIO":        "Urban",
    "MAZARREDO":      "Urban",
    "MUSKIZ":         "Refinery",
    "SANTURCE":       "Port",
}

# Structural driver sentences (from GIS analysis findings)
STATION_DRIVER = {
    "BARAKALDO":      "Highest road density in network (21,267 m/km²) + 354 m from AP-8 interchange → structural driver of elevated PM2.5 and NO₂.",
    "MAZARREDO":      "77% residential land use yet highest NO₂ (25.8 µg/m³): road density 19,060 m/km² + urban canyon + 501 m from city centre.",
    "BASAURI":        "Industrial land use 32% within 500 m (drops to 21% at 1 km) — emission sources concentrated immediately around the sensor.",
    "ERANDIO":        "1,264 m from AP-8 + 18,631 m/km² road density — traffic corridor exposure without the urban canyon attenuation.",
    "MUSKIZ":         "34.6% industrial land use (Petronor) yet lowest PM2.5 (6.5 µg/m³): TRI 343 m + coastal NW sea breeze override emission proximity.",
    "SANTURCE":       "784 m from Port of Bilbao + elevation 93 m + TRI 445 m — port proximity drives SO₂ episodes; terrain aids dispersion of other pollutants.",
    "ALGORTA_BBIZI2": "Lowest road density (9,933 m/km²) + 2.6 km from Cantabrian coast → consistent NW sea breeze flushing → structurally cleanest station.",
}

# Top spatial findings for display
TOP_FINDINGS = [
    ("road_density_1000m", "NO2",  "+0.83", "Road density → NO₂",
     "Strongest spatial predictor. Traffic infrastructure is the primary structural driver of NO₂."),
    ("dist_bilbao_centre_m", "NO2", "−0.77", "Distance to city centre → NO₂",
     "Closer to city centre = higher NO₂. Confirms urban canyon + traffic concentration effect."),
    ("vegetation_proxy_1000m", "PM10", "−0.66", "Green cover → PM10",
     "More vegetation within 1 km = lower PM10. Dry deposition + lower impervious surface."),
    ("elev_tri_2000m", "PM10", "−0.63", "Terrain Relief Index → PM10",
     "More complex terrain = stronger turbulent mixing = lower PM10. MUSKIZ TRI = 343 m."),
    ("dist_ap8_barakaldo_m", "PM2.5", "−0.54", "Distance to AP-8 motorway → PM2.5",
     "BARAKALDO (354 m from AP-8) records the highest PM2.5. Road traffic proximity confirmed."),
]

DIR8_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

df = load_data()
latest_date = df["Date"].max()

# Spatial features — try v3 → v2 → v1
_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
_SP_FILE = (
    _DATA_DIR / "station_spatial_features_v3.csv"
    if (_DATA_DIR / "station_spatial_features_v3.csv").exists()
    else _DATA_DIR / "station_spatial_features_v2.csv"
    if (_DATA_DIR / "station_spatial_features_v2.csv").exists()
    else _DATA_DIR / "station_spatial_features.csv"
)


@st.cache_data
def load_spatial() -> pd.DataFrame:
    if not _SP_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(_SP_FILE)


@st.cache_data
def load_means() -> pd.DataFrame:
    return df.groupby("station")[POLLUTANTS].mean().reset_index()


df_sp    = load_spatial()
df_means = load_means()

if not df_sp.empty and "station" in df_sp.columns:
    df_sp = df_sp.merge(df_means, on="station", how="left")

spatial_ok = not df_sp.empty

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("## 🌍 GeoAI Spatial Analysis")
st.markdown(
    "Structural spatial context for the Greater Bilbao air quality monitoring network.  \n"
    "**Sources:** OSM buffer analysis (notebook 10a) · Haversine distances (10b) · "
    "Copernicus GLO-30 DEM (10c) · ERA5 wind transport (10d).  \n"
    "n = 7 stations — all correlations are exploratory and indicative only."
)

if not spatial_ok:
    st.warning(
        "Spatial features file not found. "
        "Run notebooks 10a–10c locally and commit `station_spatial_features_v3.csv` "
        "to `data/processed/`. Some tabs will show reduced content."
    )

# Key finding strip
f1, f2, f3, f4, f5 = st.columns(5)
for col, (feat, poll, r, label, _) in zip([f1, f2, f3, f4, f5], TOP_FINDINGS):
    col.metric(label, r, help=f"{feat} × {poll}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tab_dna, tab_drivers, tab_terrain, tab_wind = st.tabs([
    "🗺️ Station DNA",
    "📊 Spatial Drivers",
    "⛰️ Terrain & Dispersion",
    "💨 Wind Transport",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STATION DNA
# ══════════════════════════════════════════════════════════════════════════════

with tab_dna:
    st.markdown("### Station Spatial Profile")
    st.caption(
        "Complete structural context for each station from OSM buffer analysis, "
        "landmark distances, and Copernicus DEM terrain analysis."
    )

    if not spatial_ok:
        st.info("Spatial features file required. Run notebooks 10a–10c.")
    else:
        # Station selector
        selected = st.selectbox(
            "Select station",
            options=df_sp["station"].tolist(),
            format_func=lambda s: f"{s}  ({STATION_ZONES.get(s, '')})",
        )

        sp_row = df_sp[df_sp["station"] == selected].iloc[0]
        zone   = STATION_ZONES.get(selected, "")
        zcolor = ZONE_CLR.get(zone, "#555")

        # Structural driver narrative
        drv = STATION_DRIVER.get(selected, "")
        if drv:
            st.info(f"📍 {drv}")

        # Three columns: Land Use | Distances | Terrain
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"**🏘️ Land Use** *(1 000 m buffer)*")
            for lbl, col, icon in [
                ("Green %",       "green_pct_1000m",       "🌿"),
                ("Industrial %",  "industrial_pct_1000m",  "🏭"),
                ("Residential %", "residential_pct_1000m", "🏘️"),
                ("Commercial %",  "commercial_pct_1000m",  "🏪"),
                ("Road density",  "road_density_1000m",    "🛣️"),
            ]:
                val = sp_row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    unit = " m/km²" if "density" in col else "%"
                    st.metric(f"{icon} {lbl}", f"{val:.1f}{unit}")

        with c2:
            st.markdown("**📍 Distances to Sources**")
            for lbl, col, icon in [
                ("Port of Bilbao",  "dist_port_bilbao_m",   "⚓"),
                ("Petronor",        "dist_petronor_m",      "🛢️"),
                ("AP-8 Motorway",   "dist_ap8_barakaldo_m", "🚗"),
                ("City Centre",     "dist_bilbao_centre_m", "🏙️"),
                ("Coastline",       "dist_coast_getxo_m",   "🌊"),
            ]:
                val = sp_row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    st.metric(f"{icon} {lbl}", f"{val/1000:.1f} km")

        with c3:
            st.markdown("**⛰️ Terrain** *(Copernicus GLO-30)*")
            for lbl, col, icon in [
                ("Elevation",   "elev_point_m",  "📏"),
                ("Slope",       "slope_deg",      "📐"),
                ("TRI 2km",     "elev_tri_2000m", "🏔️"),
                ("Mean elev 1km","elev_mean_1000m","📊"),
            ]:
                val = sp_row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    unit = "°" if col == "slope_deg" else " m"
                    st.metric(f"{icon} {lbl}", f"{val:.1f}{unit}")

        st.divider()

        # Pollutant profile vs network mean
        st.markdown("**📈 Observed pollution vs network mean**")
        net_mean = df_means.set_index("station")[POLLUTANTS].mean()
        station_vals = df_means[df_means["station"] == selected].iloc[0]

        poll_cols = st.columns(4)
        for col, poll in zip(poll_cols, POLLUTANTS):
            val  = station_vals[poll]
            mean = net_mean[poll]
            diff = val - mean
            col.metric(
                poll,
                f"{val:.1f} µg/m³",
                delta=f"{diff:+.1f} vs network",
                delta_color="inverse",
            )

        st.divider()

        # Network comparison bar chart
        st.markdown("**📊 Network comparison — mean pollutants**")
        nc1, nc2 = st.columns(2)
        for col, poll in zip([nc1, nc2], ["NO2", "PM2.5"]):
            df_bar = df_means.sort_values(poll, ascending=True).copy()
            df_bar["color"] = [
                "#2c3e50" if s == selected else ZONE_CLR.get(STATION_ZONES.get(s, ""), "#95a5a6")
                for s in df_bar["station"]
            ]
            fig = go.Figure(go.Bar(
                x=df_bar[poll],
                y=df_bar["station"],
                orientation="h",
                marker_color=df_bar["color"],
                text=df_bar[poll].round(1),
                textposition="outside",
            ))
            who_lim = WHO_ANNUAL.get(poll)
            if who_lim:
                fig.add_vline(x=who_lim, line_dash="dot", line_color="#e74c3c",
                              annotation_text="WHO", annotation_font_size=9)
            fig.update_layout(
                height=280, margin=dict(t=30, b=10, l=10, r=50),
                title=f"Mean {poll} — selected station highlighted",
                xaxis_title=f"µg/m³",
                yaxis=dict(autorange="reversed"),
            )
            col.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SPATIAL DRIVERS
# ══════════════════════════════════════════════════════════════════════════════

with tab_drivers:
    st.markdown("### Spatial Driver Analysis")
    st.caption(
        "Pearson r between spatial features and long-term mean pollutant concentrations. "
        "n = 7 stations — correlations are exploratory and indicative only."
    )

    if not spatial_ok:
        st.info("Spatial features file required.")
    else:
        # ── Correlation heatmap ────────────────────────────────────────────────
        st.markdown("#### Correlation matrix — spatial features × pollutants")

        _feature_groups = {
            "Buffer — Land Use": [
                "green_pct_1000m", "industrial_pct_1000m",
                "residential_pct_1000m", "road_density_1000m",
            ],
            "Distances": [
                "dist_port_bilbao_m", "dist_petronor_m",
                "dist_ap8_barakaldo_m", "dist_bilbao_centre_m",
                "dist_coast_getxo_m",
            ],
            "Terrain": [
                "elev_point_m", "elev_tri_2000m",
                "elev_std_1000m", "slope_deg",
            ],
        }
        _all_feat = [f for grp in _feature_groups.values() for f in grp
                     if f in df_sp.columns]
        _poll_cols = [p for p in POLLUTANTS if p in df_sp.columns]

        if _all_feat and _poll_cols:
            corr = df_sp[_all_feat + _poll_cols].corr().loc[_all_feat, _poll_cols]

            _labels = {
                "green_pct_1000m":       "Green % (1km)",
                "industrial_pct_1000m":  "Industrial % (1km)",
                "residential_pct_1000m": "Residential % (1km)",
                "road_density_1000m":    "Road density (1km)",
                "dist_port_bilbao_m":    "Dist. Port (m)",
                "dist_petronor_m":       "Dist. Petronor (m)",
                "dist_ap8_barakaldo_m":  "Dist. AP-8 (m)",
                "dist_bilbao_centre_m":  "Dist. City Centre (m)",
                "dist_coast_getxo_m":    "Dist. Coast (m)",
                "elev_point_m":          "Elevation (m)",
                "elev_tri_2000m":        "TRI 2km (m)",
                "elev_std_1000m":        "Elev std 1km (m)",
                "slope_deg":             "Slope (°)",
            }

            fig_hm = go.Figure(go.Heatmap(
                z=corr.values,
                x=[p.replace(".", "") for p in _poll_cols],
                y=[_labels.get(f, f) for f in _all_feat],
                colorscale="RdBu",
                zmid=0, zmin=-1, zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorbar=dict(title="Pearson r"),
            ))
            fig_hm.update_layout(
                height=480,
                margin=dict(t=20, b=10, l=160, r=20),
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig_hm, width="stretch")

        st.divider()

        # ── Interactive scatter ────────────────────────────────────────────────
        st.markdown("#### Spatial driver explorer")
        st.caption("Choose X (spatial feature) and Y (pollutant) to explore the relationship.")

        _feat_labels = {
            "Road density (1km)":         "road_density_1000m",
            "Industrial % (1km)":         "industrial_pct_1000m",
            "Green % (1km)":              "green_pct_1000m",
            "Distance to City Centre (km)":"dist_bilbao_centre_m",
            "Distance to Port (km)":       "dist_port_bilbao_m",
            "Distance to AP-8 (km)":       "dist_ap8_barakaldo_m",
            "Elevation (m)":              "elev_point_m",
            "TRI 2km (m)":               "elev_tri_2000m",
            "Slope (°)":                 "slope_deg",
        }
        _avail = {k: v for k, v in _feat_labels.items() if v in df_sp.columns}

        _sc1, _sc2 = st.columns(2)
        x_lbl = _sc1.selectbox("Spatial feature (X)", list(_avail.keys()), index=0)
        y_lbl = _sc2.selectbox("Pollutant (Y)", POLLUTANTS, index=2)

        x_col = _avail[x_lbl]
        y_col = y_lbl

        _df_sc = df_sp[["station", "zone", x_col, y_col]].dropna().copy()
        if "dist_" in x_col:
            _df_sc[x_col] = _df_sc[x_col] / 1000

        if len(_df_sc) >= 3:
            _r = _df_sc[[x_col, y_col]].corr().iloc[0, 1]
            fig_sc = px.scatter(
                _df_sc, x=x_col, y=y_col, text="station",
                trendline="ols",
                color="zone",
                color_discrete_map=ZONE_CLR,
                labels={x_col: x_lbl, y_col: f"Mean {y_lbl} (µg/m³)"},
                title=f"{x_lbl} vs Mean {y_lbl}  |  Pearson r = {_r:.2f}  (n = {len(_df_sc)})",
            )
            fig_sc.update_traces(textposition="top center",
                                 selector=dict(mode="markers+text"))
            fig_sc.update_layout(
                height=400, margin=dict(t=55, b=10),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_sc, width="stretch")

            _badge = ("🟢 Strong" if abs(_r) >= 0.7
                      else "🟡 Moderate" if abs(_r) >= 0.4
                      else "⚪ Weak")
            st.info(f"{_badge} signal  |  r = {_r:.2f}")

        st.divider()

        # Top findings table
        st.markdown("#### Summary — strongest spatial predictors")
        _rows = []
        for feat, poll, r, label, explanation in TOP_FINDINGS:
            _rows.append({"Feature": label, "Pollutant": poll, "r": r, "Interpretation": explanation})
        st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TERRAIN & DISPERSION
# ══════════════════════════════════════════════════════════════════════════════

with tab_terrain:
    st.markdown("### Terrain & Atmospheric Dispersion")
    st.caption(
        "Elevation and terrain complexity derived from Copernicus GLO-30 DEM (30 m resolution). "
        "Higher Terrain Relief Index (TRI) = more complex terrain = stronger turbulent mixing."
    )

    if not spatial_ok or "elev_point_m" not in df_sp.columns:
        st.info(
            "Terrain features require station_spatial_features_v3.csv. "
            "Run notebook 10c (elevation_terrain_features)."
        )
    else:
        df_ter = df_sp.copy().dropna(subset=["elev_point_m"])

        # ── Three-panel chart: elevation + TRI + slope ─────────────────────────
        t1, t2, t3 = st.columns(3)

        with t1:
            df_e = df_ter.sort_values("elev_point_m")
            fig_e = go.Figure(go.Bar(
                x=df_e["elev_point_m"], y=df_e["station"],
                orientation="h",
                marker_color=[ZONE_CLR.get(STATION_ZONES.get(s, ""), "#95a5a6")
                              for s in df_e["station"]],
                text=df_e["elev_point_m"].round(0).astype(int).astype(str) + " m",
                textposition="outside",
            ))
            fig_e.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=60),
                                title="Station elevation (m)")
            st.plotly_chart(fig_e, width="stretch")

        with t2:
            if "elev_tri_2000m" in df_ter.columns:
                df_tri = df_ter.dropna(subset=["elev_tri_2000m"]).sort_values("elev_tri_2000m")
                fig_tri = go.Figure(go.Bar(
                    x=df_tri["elev_tri_2000m"], y=df_tri["station"],
                    orientation="h",
                    marker_color=[ZONE_CLR.get(STATION_ZONES.get(s, ""), "#95a5a6")
                                  for s in df_tri["station"]],
                    text=df_tri["elev_tri_2000m"].round(0).astype(int).astype(str) + " m",
                    textposition="outside",
                ))
                fig_tri.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=60),
                                      title="TRI — 2km buffer (m)")
                st.plotly_chart(fig_tri, width="stretch")

        with t3:
            if "slope_deg" in df_ter.columns:
                df_sl = df_ter.sort_values("slope_deg")
                fig_sl = go.Figure(go.Bar(
                    x=df_sl["slope_deg"], y=df_sl["station"],
                    orientation="h",
                    marker_color=[ZONE_CLR.get(STATION_ZONES.get(s, ""), "#95a5a6")
                                  for s in df_sl["station"]],
                    text=df_sl["slope_deg"].round(1).astype(str) + "°",
                    textposition="outside",
                ))
                fig_sl.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=60),
                                     title="Slope at station (°)")
                st.plotly_chart(fig_sl, width="stretch")

        st.divider()

        # ── Terrain vs pollutant scatter ───────────────────────────────────────
        st.markdown("#### TRI × mean PM10  — terrain complexity reduces particulate levels")
        if "elev_tri_2000m" in df_sp.columns and "PM10" in df_sp.columns:
            df_tsc = df_sp[["station", "zone", "elev_tri_2000m", "PM10",
                            "elev_point_m"]].dropna()
            r_tri = df_tsc["elev_tri_2000m"].corr(df_tsc["PM10"])
            fig_tsc = px.scatter(
                df_tsc, x="elev_tri_2000m", y="PM10", text="station",
                trendline="ols",
                color="zone", color_discrete_map=ZONE_CLR,
                labels={"elev_tri_2000m": "Terrain Relief Index 2km (m)",
                        "PM10": "Mean PM10 (µg/m³)"},
                title=f"TRI × Mean PM10  |  r = {r_tri:.2f}  (n = {len(df_tsc)})",
            )
            fig_tsc.update_traces(textposition="top center",
                                  selector=dict(mode="markers+text"))
            fig_tsc.update_layout(height=380, margin=dict(t=55,b=10),
                                  showlegend=False)
            st.plotly_chart(fig_tsc, width="stretch")

        st.divider()

        # ── Key terrain finding ────────────────────────────────────────────────
        st.markdown("#### MUSKIZ dispersion paradox — confirmed by terrain analysis")
        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.metric("Industrial land use (1km)", "34.6%", "Highest in network",
                    delta_color="inverse")
        _mc2.metric("Mean PM2.5", "6.50 µg/m³", "Lowest in network ✓",
                    delta_color="off")
        _mc3.metric("TRI 2km", "343 m", "2nd highest — orographic ventilation",
                    delta_color="off")
        st.success(
            "Three independent GIS methods confirm the MUSKIZ paradox: "
            "coastal location (notebook 10a) + farthest from city centre (10b) + "
            "high TRI from terrain ridge (10c) together override Petronor emission proximity."
        )

        st.info(
            "**DEM source:** Copernicus GLO-30, 30 m resolution · EPSG:25830 · "
            "Tiles: N43_W003 + N43_W004. Full analysis: notebook 10c."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — WIND TRANSPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_wind:
    st.markdown("### Wind Transport Analysis")
    st.caption(
        "Regional wind conditions from ERA5 (single grid cell, ~31 km resolution). "
        "Direction = sector wind is coming FROM. All 7 stations share the same ERA5 time series."
    )

    def dir8(wd):
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        return dirs[int((float(wd) + 22.5) / 45) % 8]

    df["dir8"] = df["WindDirection"].apply(dir8)

    # ── Section A: Wind speed × dispersion ────────────────────────────────────
    st.markdown("#### A — Wind speed reduces pollution monotonically")

    _ws_labels = ["< 10", "10–15", "15–20", "20–25", "> 25"]
    _no2_means = [29.85, 21.05, 18.56, 16.03, 12.81]
    _pm25_means = [13.76, 11.27,  9.55,  7.84,  6.44]
    _pm10_means = [22.70, 19.30, 16.84, 14.64, 12.84]
    _so2_means  = [ 6.32,  5.03,  4.73,  4.39,  4.20]

    _colors_ws = ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#27ae60"]

    _wa1, _wa2 = st.columns(2)

    with _wa1:
        fig_ws = go.Figure()
        for poll, vals, dash in [
            ("NO₂", _no2_means, "solid"),
            ("PM2.5", _pm25_means, "dash"),
            ("PM10", _pm10_means, "dot"),
        ]:
            fig_ws.add_trace(go.Scatter(
                x=_ws_labels, y=vals, mode="lines+markers",
                name=poll, line=dict(dash=dash),
            ))
        fig_ws.update_layout(
            height=300, title="Mean concentration by wind speed (m/s)",
            margin=dict(t=45, b=10),
            xaxis_title="Wind speed", yaxis_title="µg/m³",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_ws, width="stretch")

    with _wa2:
        _pcts = {
            "NO₂":  "-57%", "PM2.5": "-53%",
            "PM10": "-43%", "SO₂":   "-34%",
        }
        st.markdown("**Reduction: calm → strong wind**")
        for poll, pct in _pcts.items():
            st.metric(f"{poll} reduction", pct,
                      help="Calm < 10 m/s vs strong > 25 m/s")
        st.caption(
            "SO₂ least responsive (−34%) — point-source emissions "
            "driven more by source intensity than wind speed."
        )

    st.divider()

    # ── Section B: Direction × station heatmaps ───────────────────────────────
    st.markdown("#### B — Wind direction × station cross-analysis")
    st.caption(
        "Under the same regional wind, stations respond differently — "
        "revealing each station's position relative to emission sources."
    )

    _poll_wind = st.selectbox("Pollutant", POLLUTANTS, index=2, key="wind_poll")

    _pivot = (
        df.groupby(["dir8", "station"])[_poll_wind]
        .mean()
        .unstack("station")
        .reindex(DIR8_ORDER)
    )
    _st_order = ["MUSKIZ", "SANTURCE", "ALGORTA_BBIZI2",
                 "ERANDIO", "BARAKALDO", "MAZARREDO", "BASAURI"]
    _st_order = [s for s in _st_order if s in _pivot.columns]
    _pivot = _pivot[_st_order]

    fig_dir = go.Figure(go.Heatmap(
        z=_pivot.values.T,
        x=DIR8_ORDER,
        y=[f"{s}\n({STATION_ZONES.get(s,'')})" for s in _st_order],
        colorscale="RdYlGn_r",
        text=np.round(_pivot.values.T, 1),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="µg/m³"),
    ))
    fig_dir.update_layout(
        height=380,
        margin=dict(t=20, b=10, l=140, r=20),
        xaxis_title="Wind direction (FROM)",
    )
    st.plotly_chart(fig_dir, width="stretch")

    st.divider()

    # ── Section C: MUSKIZ + MAZARREDO deep-dive ────────────────────────────────
    st.markdown("#### C — Station signatures: MUSKIZ SO₂ + MAZARREDO NO₂")

    _muskiz_so2 = [5.81, 7.32, 5.93, 5.19, 3.80, 3.86, 3.42, 4.50]
    _maz_no2    = [19.31, 23.36, 32.36, 35.32, 32.09, 29.68, 26.37, 20.79]

    _wc1, _wc2 = st.columns(2)

    with _wc1:
        _daily_freq = (
            df.drop_duplicates("Date")["dir8"]
            .value_counts(normalize=True).reindex(DIR8_ORDER).fillna(0) * 100
        )
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(
            x=DIR8_ORDER, y=_muskiz_so2,
            name="Mean SO₂ (µg/m³)",
            marker_color="#c0392b", opacity=0.85,
        ))
        _ax2 = go.Bar(
            x=DIR8_ORDER, y=_daily_freq.values,
            name="Wind frequency (%)",
            marker_color="#3498db", opacity=0.5,
            yaxis="y2",
        )
        fig_m.add_trace(_ax2)
        fig_m.update_layout(
            height=320,
            title="MUSKIZ — SO₂ by wind direction (Petronor refinery)",
            yaxis=dict(title="Mean SO₂ (µg/m³)", color="#c0392b"),
            yaxis2=dict(title="Wind freq (%)", overlaying="y", side="right",
                        color="#3498db"),
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=50, b=10),
            barmode="group",
        )
        st.plotly_chart(fig_m, width="stretch")
        st.caption(
            "NE wind → SO₂ = 7.32 µg/m³ (Petronor trapped against terrain, TRI = 343 m). "
            "S wind → SO₂ = 3.80 µg/m³ (sea breeze clears emissions). Ratio: **1.93×**."
        )

    with _wc2:
        fig_maz = go.Figure()
        fig_maz.add_trace(go.Bar(
            x=DIR8_ORDER, y=_maz_no2,
            name="Mean NO₂ (µg/m³)",
            marker_color="#8e44ad", opacity=0.85,
        ))
        fig_maz.add_trace(go.Bar(
            x=DIR8_ORDER, y=_daily_freq.values,
            name="Wind frequency (%)",
            marker_color="#3498db", opacity=0.5,
            yaxis="y2",
        ))
        fig_maz.update_layout(
            height=320,
            title="MAZARREDO — NO₂ by wind direction (Bilbao city centre)",
            yaxis=dict(title="Mean NO₂ (µg/m³)", color="#8e44ad"),
            yaxis2=dict(title="Wind freq (%)", overlaying="y", side="right",
                        color="#3498db"),
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=50, b=10),
            barmode="group",
        )
        st.plotly_chart(fig_maz, width="stretch")
        st.caption(
            "SE wind → NO₂ = 35.3 µg/m³ (+70% above NW baseline). "
            "Nervión valley channels SE flow, trapping urban traffic emissions."
        )

    st.divider()

    # ── Wind regime summary ────────────────────────────────────────────────────
    st.markdown("#### D — Two dominant regimes")
    _r1, _r2 = st.columns(2)
    _r1.success(
        "**NW/W regime (37.2% of days) — Bay of Biscay sea air**  \n"
        "Network-wide mean NO₂ = 16.3 µg/m³  \n"
        "MUSKIZ cleanest — Petronor emissions blown inland  \n"
        "MAZARREDO lower — valley ventilated"
    )
    _r2.warning(
        "**S/SW regime (32.0% of days) — inland recirculation**  \n"
        "Network-wide mean NO₂ = 21.4 µg/m³ (+31%)  \n"
        "MAZARREDO worst — Nervión valley trapping  \n"
        "BARAKALDO elevated — industrial corridor stagnation"
    )

    st.info(
        "⚠️ **ERA5 note:** Single regional grid cell (~31 km) — "
        "all stations share the same wind time series. "
        "Direction signatures represent synoptic-scale patterns, "
        "not station-level micrometeorology. "
        "Source: Open-Meteo ERA5 archive (CC BY 4.0) · Notebook 10d."
    )