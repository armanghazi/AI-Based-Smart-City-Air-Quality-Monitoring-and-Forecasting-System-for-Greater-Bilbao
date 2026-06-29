"""
dashboard/pages/1_Air_Quality_Monitoring.py

Air Quality Monitoring page.

D-1 rule: all labels show explicit dates — never "today", "current", or "live".
          The weather snapshot always reflects the latest complete day (D-1),
          clearly labelled as such regardless of the selected period filter.

AQI standard: European Air Quality Index (EAQI / ICA) — 6 bands, imported from
              aqi.py (single source of truth). Three simplified display bands used
              for map colouring: Good (lvl 1–2), Moderate (lvl 3), Poor (lvl 4–6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent.parent))

from aqi import AQI_CATEGORIES, AQI_THRESHOLDS, compute_aqi_category
from aqi_components import render_station_aqi_cards
from config import (
    MONTH_NAMES,
    POLLUTANT_COLOR,
    WHO_ANNUAL,
    ZONE_MAP,
    ZONE_META,
    center_tables,
    get_zone,
    load_data,
)
from i18n_auto import tr
from weather_panel import weather_snapshot

# --------------------------------------------------
# NOTE: st.set_page_config is intentionally absent —
# it is called once in app.py (st.navigation router).
# --------------------------------------------------

center_tables()
st.title(tr("📡 Air Quality Monitoring"))
st.markdown(tr("Greater Bilbao · EAQI standard · D-1 data"))

# --------------------------------------------------
# DATA
# --------------------------------------------------

df = load_data()

# --------------------------------------------------
# AQI HELPERS (built on aqi.py — EAQI 6-band)
# --------------------------------------------------
# For map display we collapse 6 bands into 3 groups:
#   Good     = EAQI levels 1–2  (teal / green)
#   Moderate = EAQI level 3     (yellow)
#   Poor     = EAQI levels 4–6  (red spectrum)

_DISPLAY_BANDS = [
    {"label": "Good",     "levels": {1, 2}, "color": "#50ccaa"},
    {"label": "Moderate", "levels": {3},    "color": "#f0e641"},
    {"label": "Poor",     "levels": {4, 5, 6}, "color": "#ff5050"},
]

# Build sidebar threshold legend from AQI_THRESHOLDS
def _aqi_display_label(value: float, pollutant: str) -> tuple[str, str]:
    """Return (display_label, hex_color) for a concentration using EAQI."""
    cat = compute_aqi_category(pollutant, value)
    if cat is None:
        return "—", "#cccccc"
    level = cat["level"]
    for band in _DISPLAY_BANDS:
        if level in band["levels"]:
            return band["label"], band["color"]
    return cat["label"], cat["color"]


def _aqi_map_color(value: float, pollutant: str) -> str:
    return _aqi_display_label(value, pollutant)[1]


def _svi_color(v: float | None) -> str:
    """Colour for SVI value (defined once, outside loops)."""
    if v is None:
        return "#95a5a6"
    if v >= 70:
        return "#e74c3c"
    if v >= 40:
        return "#f39c12"
    return "#2ecc71"


# --------------------------------------------------
# SIDEBAR — FILTERS
# --------------------------------------------------

st.sidebar.header(tr("Filters"))

pollutant = st.sidebar.selectbox(
    tr("Select Pollutant"),
    ["PM2.5", "PM10", "NO2", "SO2"],
)

time_mode = st.sidebar.radio(
    tr("Time Granularity"),
    options=["Year", "Month", "Day"],
    index=0,
    horizontal=True,
)

MONTH_NAMES_FULL = {
    1: "January",  2: "February", 3: "March",    4: "April",
    5: "May",      6: "June",     7: "July",      8: "August",
    9: "September",10: "October", 11: "November", 12: "December",
}

if time_mode == "Year":
    year_options  = ["All"] + sorted(df["Year"].dropna().unique().tolist())
    selected_year = st.sidebar.selectbox(tr("Select Year"), year_options, index=0)
    if selected_year == "All":
        filtered     = df.copy()
        period_label = "All Years (2015–2026)"
    else:
        filtered     = df[df["Year"] == int(selected_year)].copy()
        period_label = str(selected_year)

elif time_mode == "Month":
    year_options    = sorted(df["Year"].dropna().unique().tolist())
    selected_year_m = st.sidebar.selectbox(
        tr("Select Year"), year_options, index=len(year_options) - 1
    )
    available_months    = sorted(df[df["Year"] == selected_year_m]["Month"].dropna().unique())
    month_options       = ["All"] + [MONTH_NAMES_FULL[m] for m in available_months]
    selected_month_label = st.sidebar.selectbox(tr("Select Month"), month_options, index=0)
    if selected_month_label == "All":
        filtered     = df[df["Year"] == selected_year_m].copy()
        period_label = f"All months of {selected_year_m}"
    else:
        month_num    = {v: k for k, v in MONTH_NAMES_FULL.items()}[selected_month_label]
        filtered     = df[
            (df["Year"] == selected_year_m) & (df["Month"] == month_num)
        ].copy()
        period_label = f"{selected_month_label} {selected_year_m}"

else:  # Day
    year_options    = sorted(df["Year"].dropna().unique().tolist())
    selected_year_d = st.sidebar.selectbox(
        tr("Select Year"), year_options, index=len(year_options) - 1
    )
    available_months_d = sorted(
        df[df["Year"] == selected_year_d]["Month"].dropna().unique()
    )
    month_options_d       = [MONTH_NAMES_FULL[m] for m in available_months_d]
    selected_month_label_d = st.sidebar.selectbox(tr("Select Month"), month_options_d, index=0)
    month_num_d           = {v: k for k, v in MONTH_NAMES_FULL.items()}[selected_month_label_d]
    df_month              = df[(df["Year"] == selected_year_d) & (df["Month"] == month_num_d)]
    available_days        = sorted(df_month["Day"].dropna().unique())
    selected_day          = st.sidebar.selectbox(
        tr("Select Day"), available_days,
        format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"),
    )
    filtered     = df[df["Day"] == selected_day].copy()
    period_label = pd.Timestamp(selected_day).strftime("%d %B %Y")

# --------------------------------------------------
# SIDEBAR — AQI LEGEND (EAQI bands for this pollutant)
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📊 {tr('EAQI Bands')} — {pollutant} (µg/m³)")

thresholds = AQI_THRESHOLDS.get(pollutant, [])
cats       = AQI_CATEGORIES
for i, cat in enumerate(cats):
    lo = thresholds[i - 1] + 0.1 if i > 0 else 0
    hi = f"≤ {thresholds[i]}" if i < len(thresholds) else f"> {thresholds[-1]}"
    lo_str = f"{lo:.0f}–" if i > 0 else "0–"
    st.sidebar.markdown(
        f"<span style='color:{cat['color']};font-size:1.1rem'>●</span> "
        f"**{cat['label']}** : {lo_str}{hi}",
        unsafe_allow_html=True,
    )

st.sidebar.markdown(
    "<small style='color:#888'>Source: European Air Quality Index (EAQI / ICA Spain)</small>",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR — ZONE LEGEND
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 🗺️ {tr('Environmental Zones')}")
for zone, meta in ZONE_META.items():
    st.sidebar.markdown(
        f"{meta['icon']} **{zone}**  \n"
        f"<span style='color:{meta['color']};font-size:11px'>{meta['description']}</span>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# SAFETY CHECK
# --------------------------------------------------

if filtered.empty:
    st.warning(tr("No data available for the selected filters."))
    st.stop()

# --------------------------------------------------
# WEATHER SNAPSHOT — latest day in selected period
# --------------------------------------------------

weather_snapshot(filtered, show_per_station=True)

st.markdown("---")

# --------------------------------------------------
# STATION MEAN FOR SELECTED PERIOD
# --------------------------------------------------

station_mean = (
    filtered
    .groupby(["station", "Town", "Latitude", "Longitude"])[pollutant]
    .mean()
    .reset_index()
)

if station_mean.empty:
    st.warning(tr("No station data after aggregation."))
    st.stop()

station_mean["Zone"] = station_mean["Town"].apply(get_zone)

# --------------------------------------------------
# SVI DATA — load from spatial features
# --------------------------------------------------

_SP_V3 = Path(__file__).parent.parent.parent / "data" / "processed" / "station_spatial_features_v3.csv"
_SP_V1 = Path(__file__).parent.parent.parent / "data" / "processed" / "station_spatial_features.csv"
_SP_F  = _SP_V3 if _SP_V3.exists() else (_SP_V1 if _SP_V1.exists() else None)

_SVI_DATA: dict[str, float] = {}

if _SP_F is not None:
    try:
        _df_sp   = pd.read_csv(_SP_F)
        _svi_cols = ["road_density_1000m", "dist_bilbao_centre_m", "elev_tri_2000m"]
        if all(c in _df_sp.columns for c in _svi_cols):
            _signs = {"road_density_1000m": +1, "dist_bilbao_centre_m": -1, "elev_tri_2000m": -1}
            for col, sign in _signs.items():
                _m = _df_sp[col].mean()
                _s = _df_sp[col].std()
                _df_sp[f"_z_{col}"] = sign * (_df_sp[col] - _m) / max(_s, 1e-9)
            _df_sp["_svi_raw"] = _df_sp[[f"_z_{c}" for c in _svi_cols]].mean(axis=1)
            _mn = _df_sp["_svi_raw"].min()
            _mx = _df_sp["_svi_raw"].max()
            _df_sp["SVI"] = ((_df_sp["_svi_raw"] - _mn) / max(_mx - _mn, 1e-9) * 100).round(1)
            _SVI_DATA = dict(zip(_df_sp["station"], _df_sp["SVI"]))
    except Exception:
        pass

# GIS Phase C spatial driver notes (notebook 10a / 10b)
_DRIVER: dict[str, str] = {
    "BARAKALDO":      "Road density 21,267 m/km² + 354 m from AP-8",
    "MAZARREDO":      "Road density 19,060 m/km² + 501 m from city centre",
    "BASAURI":        "Industrial land use 32% within 500 m",
    "ERANDIO":        "1,264 m from AP-8 + 18,631 m/km² road density",
    "MUSKIZ":         "TRI 343 m + coastal NW breeze → dispersion advantage",
    "SANTURCE":       "784 m from Port of Bilbao + TRI 445 m",
    "ALGORTA_BBIZI2": "Lowest road density (9,933 m/km²) + 2.6 km from coast",
}

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

avg_value     = station_mean[pollutant].mean()
worst_station = station_mean.loc[station_mean[pollutant].idxmax()]
best_station  = station_mean.loc[station_mean[pollutant].idxmin()]

labels        = station_mean[pollutant].apply(lambda v: _aqi_display_label(v, pollutant)[0])
good_count    = int((labels == "Good").sum())
mod_count     = int((labels == "Moderate").sum())
poor_count    = int((labels == "Poor").sum())

st.subheader(f"{pollutant} — {period_label}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{tr('Network avg')} {pollutant}", f"{avg_value:.1f} µg/m³")
c2.metric(
    tr("Most polluted"),
    worst_station["station"].split("_")[0],
    f"{worst_station[pollutant]:.1f} µg/m³",
    delta_color="inverse",
)
c3.metric(
    tr("Cleanest"),
    best_station["station"].split("_")[0],
    f"{best_station[pollutant]:.1f} µg/m³",
)
c4.metric(f"🟢 {tr('Good')} / 🟡 {tr('Moderate')}", f"{good_count} / {mod_count}")
c5.metric(f"🔴 {tr('Poor stations')}", poor_count)

# --------------------------------------------------
# ZONE SUMMARY CARDS
# --------------------------------------------------

st.markdown("---")
st.subheader(tr("🗺️ Environmental Zone Summary"))

zones_list = list(ZONE_META.items())
row1       = zones_list[:3]
row2       = zones_list[3:]


def _zone_card(zone_name: str, meta: dict) -> None:
    zone_stations = station_mean[station_mean["Zone"] == zone_name]
    if zone_stations.empty:
        zone_avg     = "N/A"
        zone_worst   = "N/A"
        zone_quality = "–"
        n_stations   = 0
    else:
        zone_avg       = f"{zone_stations[pollutant].mean():.1f} µg/m³"
        zone_worst_row = zone_stations.loc[zone_stations[pollutant].idxmax()]
        zone_worst     = (
            f"{zone_worst_row['station'].split('_')[0]} "
            f"({zone_worst_row[pollutant]:.1f})"
        )
        zone_labels  = zone_stations[pollutant].apply(
            lambda v: _aqi_display_label(v, pollutant)[0]
        )
        zone_quality = zone_labels.mode()[0] if not zone_labels.empty else "–"
        n_stations   = len(zone_stations)

    st.markdown(
        f"""
        <div style="
            border-left:5px solid {meta['border']};
            background:linear-gradient(135deg,{meta['color']}18,{meta['color']}06);
            border-radius:10px; padding:16px 18px;
            margin-bottom:8px; height:100%;
        ">
            <div style="font-size:20px;margin-bottom:4px">
                {meta['icon']} <strong>{zone_name}</strong>
            </div>
            <div style="color:#555;font-size:11px;margin-bottom:10px">
                {meta['description']}
            </div>
            <table style="width:100%;font-size:12px;border-collapse:collapse">
                <tr>
                    <td style="color:#777;padding:3px 0">{tr('Stations')}</td>
                    <td style="text-align:right;font-weight:600">{n_stations}</td>
                </tr>
                <tr>
                    <td style="color:#777;padding:3px 0">Avg {pollutant}</td>
                    <td style="text-align:right;font-weight:600">{zone_avg}</td>
                </tr>
                <tr>
                    <td style="color:#777;padding:3px 0">{tr('Highest station')}</td>
                    <td style="text-align:right;font-weight:600">{zone_worst}</td>
                </tr>
                <tr>
                    <td style="color:#777;padding:3px 0">{tr('Typical quality')}</td>
                    <td style="text-align:right;font-weight:600">{zone_quality}</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


cols_r1 = st.columns(3)
for idx, (zone_name, meta) in enumerate(row1):
    with cols_r1[idx]:
        _zone_card(zone_name, meta)

if row2:
    n_rem   = len(row2)
    pad     = (3 - n_rem) // 2
    spacers = [None] * pad + list(row2) + [None] * (3 - n_rem - pad)
    cols_r2 = st.columns(3)
    for i, item in enumerate(spacers):
        if item is not None:
            with cols_r2[i]:
                _zone_card(item[0], item[1])

st.markdown("---")

# --------------------------------------------------
# FOLIUM MAP
# --------------------------------------------------

st.subheader(f"🗺️ {tr('Station Map')} — {pollutant} — {period_label}")

_map_mode = st.radio(
    tr("Map colour mode"),
    [tr("Air Quality (EAQI)"), tr("Structural Vulnerability (SVI)")],
    horizontal=True,
    key="map_mode_radio",
    help=tr("EAQI = pollution level · SVI = structural exposure from GIS analysis"),
)
_use_svi = _map_mode == tr("Structural Vulnerability (SVI)") and bool(_SVI_DATA)

if _use_svi:
    st.caption(
        tr("SVI = Structural Vulnerability Index — composite of road density, "
           "distance to city centre, and Terrain Relief Index (TRI). "
           "Higher = structurally more exposed to poor air quality.")
    )

center_lat = station_mean["Latitude"].mean()
center_lon = station_mean["Longitude"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

for zone_name, meta in ZONE_META.items():
    zone_stations = station_mean[station_mean["Zone"] == zone_name]
    if zone_stations.empty:
        continue

    fg      = folium.FeatureGroup(name=f"{meta['icon']} {zone_name}")
    z_color = meta["color"]

    for _, row in zone_stations.iterrows():
        value       = row[pollutant]
        aqi_label, aqi_color = _aqi_display_label(value, pollutant)
        _svi_val    = _SVI_DATA.get(row["station"])
        _driver_txt = _DRIVER.get(row["station"], "")

        _fill_color = _svi_color(_svi_val) if _use_svi else aqi_color
        _radius     = max(8, min(22, (_svi_val / 4.5) if (_use_svi and _svi_val) else value / 2))

        _svi_html = (
            f"<br><span style='font-size:11px;color:#555'>"
            f"SVI: <b>{_svi_val:.0f}/100</b></span>"
            if _svi_val is not None else ""
        )
        _drv_html = (
            f"<br><span style='font-size:10px;color:#888;font-style:italic'>"
            f"{_driver_txt}</span>"
            if _driver_txt else ""
        )

        popup_html = (
            f"<div style='font-family:sans-serif;min-width:190px;padding:4px'>"
            f"<b style='font-size:14px'>{row['station']}</b><br>"
            f"<span style='color:#555'>{row['Town']}</span><br>"
            f"<span style='background:{z_color};color:white;padding:1px 8px;"
            f"border-radius:8px;font-size:11px'>{meta['icon']} {zone_name}</span><br>"
            f"<hr style='margin:4px 0'>"
            f"<b>{pollutant}:</b> {value:.2f} &micro;g/m&sup3;<br>"
            f"<span style='background:{aqi_color};color:white;"
            f"padding:2px 10px;border-radius:10px;font-size:12px'>{aqi_label}</span>"
            f"{_svi_html}{_drv_html}"
            f"<br><span style='font-size:11px;color:#888'>{tr('Period')}: {period_label}</span>"
            f"</div>"
        )

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=_radius,
            popup=folium.Popup(popup_html, max_width=270, parse_html=False),
            tooltip=(
                f"{row['station'].split('_')[0]} | {meta['icon']} {zone_name} | "
                + (f"SVI {_svi_val:.0f}/100" if _use_svi and _svi_val is not None
                   else f"{value:.1f} µg/m³ ({aqi_label})")
            ),
            color=z_color, weight=3,
            fill=True, fill_color=_fill_color, fill_opacity=0.88,
        ).add_to(fg)

        folium.Marker(
            location=[row["Latitude"] + 0.003, row["Longitude"]],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:10px;font-weight:600;color:#2c3e50;'
                    f'white-space:nowrap">{row["station"].split("_")[0]}</div>'
                ),
                icon_size=(120, 20), icon_anchor=(60, 0),
            ),
        ).add_to(fg)

    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# Map legend
zone_legend_items = "".join([
    f'<span style="color:{meta["color"]};">●</span> {meta["icon"]} {zone_name}<br>'
    for zone_name, meta in ZONE_META.items()
])
thresholds_p = AQI_THRESHOLDS.get(pollutant, [])
band_legend  = "".join([
    f'<span style="color:{b["color"]};">●</span> {b["label"]}<br>'
    for b in _DISPLAY_BANDS
])

legend_html = f"""
<div style="
    position:fixed;bottom:30px;left:30px;z-index:1000;
    background:white;border-radius:10px;padding:14px 18px;
    box-shadow:0 2px 12px rgba(0,0,0,0.2);
    font-family:sans-serif;font-size:13px;line-height:2;max-width:240px;
">
    <b style="font-size:14px;">EAQI — {pollutant}</b><br>
    {band_legend}
    <hr style="margin:6px 0">
    <b style="font-size:12px;">Zone (border colour)</b><br>
    {zone_legend_items}
    <span style="color:#888;font-size:11px;">Fill = EAQI · Border = Zone</span>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(
    m,
    width=None,
    height=600,
    returned_objects=[],
    key=f"map_{pollutant}_{time_mode}_{period_label}",
)

# --------------------------------------------------
# AQI STATION CARDS
# --------------------------------------------------

st.markdown("---")
st.subheader(f"🌈 {tr('Air Quality Index by station')} — {period_label}")
st.caption(
    tr("European Air Quality Index (EAQI/ICA) — overall level = worst pollutant. "
       "Based on the latest reading in the selected period.")
)
render_station_aqi_cards(filtered, n_cols=4)

# --------------------------------------------------
# STATION RANKING TABLE
# --------------------------------------------------

st.subheader(f"🏆 {tr('Station Ranking')} — {pollutant} — {period_label}")

ranking       = station_mean.sort_values(pollutant, ascending=False).reset_index(drop=True)
ranking.index = ranking.index + 1
ranking["EAQI"] = ranking[pollutant].apply(lambda v: _aqi_display_label(v, pollutant)[0])

st.dataframe(
    ranking[["station", "Town", "Zone", pollutant, "EAQI"]],
    width="stretch",
    column_config={
        pollutant: st.column_config.ProgressColumn(
            f"{pollutant} (µg/m³)",
            min_value=0,
            max_value=float(ranking[pollutant].max() * 1.1),
            format="%.1f",
        ),
        "EAQI": st.column_config.TextColumn(tr("EAQI Level")),
        "Zone": st.column_config.TextColumn(tr("Environmental Zone")),
    },
    hide_index=False,
)

# --------------------------------------------------
# ZONE COMPARISON TABLE — computed from filtered (raw rows)
# --------------------------------------------------

with st.expander(tr("📊 Zone Comparison Table")):
    st.caption(
        tr("Averages computed from all daily readings in the selected period "
           "(not from per-station means).")
    )
    # Add Zone column to filtered data for groupby
    _fc = filtered.copy()
    _fc["Zone"] = _fc["Town"].apply(get_zone)

    zone_comp = (
        _fc.groupby("Zone")[pollutant]
        .agg(
            Avg="mean",
            Min="min",
            Max="max",
            Days="count",
        )
        .round(2)
        .reset_index()
    )
    zone_comp.columns = ["Zone", f"Avg {pollutant}", f"Min {pollutant}", f"Max {pollutant}", "Days"]
    zone_comp["EAQI (avg)"] = zone_comp[f"Avg {pollutant}"].apply(
        lambda v: _aqi_display_label(v, pollutant)[0]
    )
    st.dataframe(zone_comp, hide_index=True, width="stretch")