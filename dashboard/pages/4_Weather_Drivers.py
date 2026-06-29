import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    load_data,
    WHO_ANNUAL, CORE_POLLUTANTS,
    POLLUTANT_COLOR, MONTH_NAMES,
    ZONE_META, get_zone,get_fav_station, center_tables
)

from i18n_auto import tr

from forecast_utils import plotly_touch_config, PLOTLY_CONFIG

plotly_touch_config()  


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

# NOTE: st.set_page_config is intentionally absent — called once in app.py router.
center_tables()

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

WEATHER_VARS = [
    "Temperature",
    "Humidity",
    "Precipitation",
    "WindSpeed",
    "Wind_X",   # mapped from wind_u in config.load_data()
    "Wind_Y",   # mapped from wind_v in config.load_data()
]

WEATHER_UNIT = {
    "Temperature":   "°C",
    "Humidity":      "%",
    "Precipitation": "mm",
    "WindSpeed":     "km/h",
    "Wind_X":        "m/s  (E–W component, + = eastward)",
    "Wind_Y":        "m/s  (N–S component, + = northward)",
}

WEATHER_ICON = {
    "Temperature":   "🌡️",
    "Humidity":      "💧",
    "Precipitation": "🌧️",
    "WindSpeed":     "💨",
    "Wind_X":        "➡️",
    "Wind_Y":        "⬆️",
}

WEATHER_COLOR = {
    "Temperature":   "#e74c3c",
    "Humidity":      "#3498db",
    "Precipitation": "#1abc9c",
    "WindSpeed":     "#95a5a6",
    "Wind_X":        "#34495e",
    "Wind_Y":        "#16a085",
}

ALL_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]

SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_COLOR = {
    "Winter": "#3498db",
    "Spring": "#2ecc71",
    "Summer": "#e74c3c",
    "Autumn": "#e67e22",
}

_POLL_PREFIXES = [p.replace(".", "") for p in ALL_POLLUTANTS]

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_data()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(tr("🌦️ Weather Drivers & Air Pollution Dynamics"))
st.markdown(
    tr("How meteorological conditions influence air pollution patterns "
       "across Greater Bilbao · 2015–2026")
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🌦️ " + tr("Filters"))
    st.divider()

    filter_mode = st.radio(
        tr("Filter by"), ["All Stations", "Zone", "Station"],
        horizontal=True
    )

    if filter_mode == "Zone":
        selected_zone = st.selectbox(
            tr("Zone"), list(ZONE_META.keys()),
            format_func=lambda z: f"{ZONE_META[z]['icon']} {z}"
        )
    elif filter_mode == "Station":
        _station_list = sorted(df["station"].unique().tolist())
        _fav = get_fav_station(_station_list)
        _idx = _station_list.index(_fav) if _fav in _station_list else 0
        selected_station = st.selectbox(
            tr("Station"), _station_list, index=_idx
        )

    st.markdown("### " + tr("Time Range"))
    all_years = sorted(df["Year"].dropna().unique().tolist())
    time_mode = st.radio(tr("Granularity"), ["Year", "Month", "Day"], horizontal=True)

    if time_mode == "Year":
        year_opts    = ["All"] + [str(y) for y in all_years]
        sel_year_str = st.selectbox(tr("Year"), year_opts, index=0)
        sel_year     = None if sel_year_str == "All" else int(sel_year_str)
        sel_month    = None
        sel_day      = None

    elif time_mode == "Month":
        sel_year      = st.selectbox(tr("Year"), all_years, index=len(all_years) - 1)
        month_avail   = sorted(df[df["Year"] == sel_year]["Month"].dropna().unique().tolist())
        month_opts    = ["All"] + [MONTH_NAMES[m] for m in month_avail]
        sel_month_lbl = st.selectbox(tr("Month"), month_opts, index=0)
        sel_month     = (
            {v: k for k, v in MONTH_NAMES.items()}[sel_month_lbl]
            if sel_month_lbl != "All" else None
        )
        sel_day       = None

    else:  # Day
        sel_year      = st.selectbox(tr("Year"), all_years, index=len(all_years) - 1)
        month_avail   = sorted(df[df["Year"] == sel_year]["Month"].dropna().unique().tolist())
        sel_month_lbl = st.selectbox(tr("Month"), [MONTH_NAMES[m] for m in month_avail])
        sel_month     = {v: k for k, v in MONTH_NAMES.items()}[sel_month_lbl]
        day_avail     = sorted(
            df[(df["Year"] == sel_year) & (df["Month"] == sel_month)]["Date"]
            .dt.date.dropna().unique().tolist()
        )
        sel_day = st.selectbox(
            tr("Day"), day_avail,
            format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y")
        )

    st.divider()
    st.markdown("### 🗺️ " + tr("Zone Legend"))
    for z, meta in ZONE_META.items():
        st.markdown(f"{meta['icon']} **{z}**")

# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------

base = df.copy()

if filter_mode == "Zone":
    base        = base[base["Zone"] == selected_zone]
    scope_label = f"{ZONE_META[selected_zone]['icon']} {selected_zone}"
elif filter_mode == "Station":
    base        = base[base["station"] == selected_station]
    scope_label = selected_station
else:
    scope_label = "All Stations"

if time_mode == "Year":
    if sel_year:
        base         = base[base["Year"] == sel_year]
        period_label = str(sel_year)
    else:
        period_label = "2015–2026"

elif time_mode == "Month":
    base = base[base["Year"] == sel_year]
    if sel_month:
        base         = base[base["Month"] == sel_month]
        period_label = f"{MONTH_NAMES[sel_month]} {sel_year}"
    else:
        period_label = f"All months of {sel_year}"

else:  # Day
    base         = base[base["Date"].dt.date == sel_day]
    period_label = pd.Timestamp(sel_day).strftime("%d %B %Y")

if base.empty:
    st.warning(tr("No data for selected filters."))
    st.stop()

# --------------------------------------------------
# KPI ROW
# --------------------------------------------------

st.markdown(f"### 📡 Overview — {scope_label} · {period_label}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(f"🌡️ {tr('Avg Temperature')}",    f"{base['Temperature'].mean():.1f} °C")
k2.metric(f"💧 {tr('Avg Humidity')}",        f"{base['Humidity'].mean():.1f} %")
k3.metric(f"🌧️ {tr('Avg Precipitation')}",   f"{base['Precipitation'].mean():.1f} mm")
k4.metric(f"💨 {tr('Avg Wind Speed')}",      f"{base['WindSpeed'].mean():.1f} km/h")
k5.metric(f"📋 {tr('Daily observations')}",  f"{len(base):,}")

st.divider()

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    tr("📊 Correlation Matrix"),
    tr("🌡️ Weather Relationships"),
    tr("🍂 Seasonal Effects"),
    tr("⏳ Lag Analysis"),
    tr("🤖 Forecast Features"),
])

# ==================== TAB 1: CORRELATION MATRIX ====================
# ── Wind Transport constants (used inside tab1) ───────────────────────────────
# Hardcoded from notebook 10d — verified against parquet 2015-2026
_DIR8_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_WS_LABELS  = ["< 10", "10–15", "15–20", "20–25", "> 25"]
_DISP = {
    "NO2":   [29.85, 21.05, 18.56, 16.03, 12.81],
    "PM2.5": [13.76, 11.27,  9.55,  7.84,  6.44],
    "PM10":  [22.70, 19.30, 16.84, 14.64, 12.84],
    "SO2":   [ 6.32,  5.03,  4.73,  4.39,  4.20],
}
_DIR_NO2 = {
    "ALGORTA_BBIZI2": [ 9.9, 11.7, 15.0, 17.6, 15.5, 14.1, 11.8,  9.3],
    "BARAKALDO":      [16.9, 18.3, 25.9, 27.4, 23.7, 22.5, 20.2, 17.1],
    "BASAURI":        [18.8, 20.2, 25.2, 26.0, 25.9, 23.1, 24.5, 19.5],
    "ERANDIO":        [20.5, 23.3, 31.3, 30.6, 25.9, 24.3, 22.0, 18.8],
    "MAZARREDO":      [19.3, 23.4, 32.4, 35.3, 32.1, 29.7, 26.4, 20.8],
    "MUSKIZ":         [ 8.6, 10.5, 13.2, 13.7,  8.4,  7.7,  6.5,  7.5],
    "SANTURCE":       [17.6, 18.2, 22.8, 25.6, 20.3, 19.4, 16.1, 16.7],
}
_MUSKIZ_SO2 = [5.81, 7.32, 5.93, 5.19, 3.80, 3.86, 3.42, 4.50]
_WIND_FREQ  = [11.9, 8.6, 4.6, 5.6, 23.0, 9.2, 13.1, 24.0]

with tab1:
    st.markdown("### " + tr("Correlation Matrix — Pollutants × Weather"))
    st.caption(tr("Pearson r · −1 = inverse · 0 = no relation · +1 = direct"))

    corr_cols    = ALL_POLLUTANTS + [v for v in WEATHER_VARS if v in base.columns]
    corr_matrix  = base[corr_cols].dropna().corr()
    weather_present = [v for v in WEATHER_VARS if v in corr_matrix.columns]
    corr_focus   = corr_matrix.loc[ALL_POLLUTANTS, weather_present]

    col_f, col_full = st.columns([3, 2])

    with col_f:
        fig_focus = px.imshow(
            corr_focus,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
            title="Pollutant × Weather (focus view)"
        )
        fig_focus.update_layout(dragmode=False, height=320, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_focus,  width="stretch", config=PLOTLY_CONFIG)

    with col_full:
        st.markdown("#### " + tr("Strongest pairs"))
        pairs = []
        for p in ALL_POLLUTANTS:
            for w in weather_present:
                pairs.append({
                    "Pollutant":   p,
                    "Weather Var": w,
                    "r":           round(corr_focus.loc[p, w], 3)
                })
        pairs_df = (
            pd.DataFrame(pairs)
            .assign(abs_r=lambda d: d["r"].abs())
            .sort_values("abs_r", ascending=False)
            .drop(columns="abs_r")
            .head(8)
            .reset_index(drop=True)
        )
        pairs_df.index += 1
        st.dataframe(
            pairs_df,
            width="stretch",
            column_config={
                "r": st.column_config.ProgressColumn(
                    "Correlation (r)", min_value=-1, max_value=1, format="%.3f"
                )
            }
        )

    with st.expander(tr("View full correlation matrix")):
        fig_full_m = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
            title="Full Correlation Matrix"
        )
        fig_full_m.update_layout(dragmode=False, height=500)
        st.plotly_chart(fig_full_m,  width="stretch", config=PLOTLY_CONFIG)

    # ── Section A: Wind speed × dispersion ──────────────────────────────────
    st.divider()
    st.markdown("### " + tr("A — Wind speed controls atmospheric dispersion"))
    st.caption(tr("Monotonic relationship between wind intensity and mean pollutant concentration."))

    _wa1, _wa2 = st.columns([3, 2])
    with _wa1:
        _poll_sel = st.radio(
            tr("Show pollutant"),
            ["NO₂", "PM2.5", "PM10", "SO₂"],
            horizontal=True, key="ws_poll_sel",
        )
        _poll_map = {"NO₂": "NO2", "PM2.5": "PM2.5", "PM10": "PM10", "SO₂": "SO2"}
        _vals = _DISP[_poll_map[_poll_sel]]
        _clrs = ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#27ae60"]
        _cur_ws = float(base[base["Date"] == base["Date"].max()]["WindSpeed"].mean())
        _cur_bin = 0
        for _i, (_lo, _hi) in enumerate(zip([0,10,15,20,25],[10,15,20,25,100])):
            if _lo <= _cur_ws < _hi:
                _cur_bin = _i; break
        _clrs_h = ["#2c3e50" if _i == _cur_bin else c for _i, c in enumerate(_clrs)]
        fig_ws = go.Figure(go.Bar(
            x=_WS_LABELS, y=_vals, marker_color=_clrs_h,
            text=[f"{v:.1f}" for v in _vals], textposition="outside",
        ))
        if _cur_bin < len(_WS_LABELS):
            fig_ws.add_annotation(
                x=_WS_LABELS[_cur_bin], y=_vals[_cur_bin] + 0.8,
                text=tr("← D-1"), showarrow=False, font=dict(size=11, color="#2c3e50"),
            )
        fig_ws.update_layout(
            dragmode=False, height=320, margin=dict(t=20, b=10, l=10, r=20),
            xaxis_title=tr("Wind speed (m/s)"),
            yaxis_title=f"Mean {_poll_sel} (µg/m³)", showlegend=False,
        )
        st.plotly_chart(fig_ws, width="stretch", config=PLOTLY_CONFIG)
    with _wa2:
        st.markdown("**" + tr("Reduction: calm → strong wind") + "**")
        for _p, _pct in [("NO₂","-57%"),("PM2.5","-53%"),("PM10","-43%"),("SO₂","-34%")]:
            st.metric(_p, _pct, delta_color="inverse")
        st.caption(tr("SO₂ least responsive (−34%) — point-source emissions driven more by source intensity than wind."))

    # ── Section B: Wind direction × station heatmap ─────────────────────────
    st.divider()
    st.markdown("### " + tr("B — Wind direction reveals emission source locations"))
    st.caption(tr("Under the same regional wind, stations respond differently — revealing each station's position relative to emission sources. ERA5 regional wind — direction = sector wind is coming FROM."))

    _st_order = ["MUSKIZ","SANTURCE","ALGORTA_BBIZI2","ERANDIO","BARAKALDO","MAZARREDO","BASAURI"]
    _st_zones = {
        "ALGORTA_BBIZI2":"Coastal","BARAKALDO":"Industrial","BASAURI":"Industrial",
        "ERANDIO":"Urban","MAZARREDO":"Urban","MUSKIZ":"Refinery","SANTURCE":"Port",
    }
    _z_matrix = [[_DIR_NO2[s][d] for d in range(8)] for s in _st_order]
    fig_dir = go.Figure(go.Heatmap(
        z=_z_matrix, x=_DIR8_ORDER,
        y=[f"{s} ({_st_zones[s]})" for s in _st_order],
        colorscale="RdYlGn_r",
        text=[[f"{v:.0f}" for v in row] for row in _z_matrix],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="NO₂ µg/m³"),
    ))
    fig_dir.update_layout(
        dragmode=False, height=350, margin=dict(t=20, b=10, l=160, r=20),
        xaxis_title=tr("Wind direction (FROM)"),
    )
    st.plotly_chart(fig_dir, width="stretch", config=PLOTLY_CONFIG)

    # ── Section C: MUSKIZ + MAZARREDO signatures ─────────────────────────────
    st.divider()
    st.markdown("### " + tr("C — Two directional signatures"))
    _wc1, _wc2 = st.columns(2)
    with _wc1:
        fig_musk = go.Figure()
        fig_musk.add_trace(go.Bar(x=_DIR8_ORDER, y=_MUSKIZ_SO2, name=tr("Mean SO₂ (µg/m³)"), marker_color="#c0392b", opacity=0.85, yaxis="y"))
        fig_musk.add_trace(go.Bar(x=_DIR8_ORDER, y=_WIND_FREQ, name=tr("Wind frequency (%)"), marker_color="#3498db", opacity=0.5, yaxis="y2"))
        fig_musk.update_layout(
            dragmode=False, height=320, title=tr("MUSKIZ — SO₂ by wind direction"), barmode="group",
            yaxis=dict(title="Mean SO₂ (µg/m³)", color="#c0392b"),
            yaxis2=dict(title=tr("Wind freq (%)"), overlaying="y", side="right", color="#3498db"),
            legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_musk, width="stretch", config=PLOTLY_CONFIG)
        st.caption(tr("NE wind → SO₂ = 7.32 µg/m³ — Petronor emissions trapped against terrain ridge (TRI = 343 m). S wind → SO₂ = 3.80 µg/m³ (sea breeze clears). Ratio: 1.93×."))
    with _wc2:
        _maz_no2_vals = [19.3,23.4,32.4,35.3,32.1,29.7,26.4,20.8]
        fig_maz = go.Figure()
        fig_maz.add_trace(go.Bar(x=_DIR8_ORDER, y=_maz_no2_vals, name=tr("Mean NO₂ (µg/m³)"), marker_color="#8e44ad", opacity=0.85, yaxis="y"))
        fig_maz.add_trace(go.Bar(x=_DIR8_ORDER, y=_WIND_FREQ, name=tr("Wind frequency (%)"), marker_color="#3498db", opacity=0.5, yaxis="y2"))
        fig_maz.update_layout(
            dragmode=False, height=320, title=tr("MAZARREDO — NO₂ by wind direction"), barmode="group",
            yaxis=dict(title="Mean NO₂ (µg/m³)", color="#8e44ad"),
            yaxis2=dict(title=tr("Wind freq (%)"), overlaying="y", side="right", color="#3498db"),
            legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_maz, width="stretch", config=PLOTLY_CONFIG)
        st.caption(tr("SE wind → NO₂ = 35.3 µg/m³ (+70% vs NW baseline). Nervión valley channels SE flow — urban traffic emissions trapped in canyon."))

    # ── Section D: Two regimes ────────────────────────────────────────────────
    st.divider()
    st.markdown("### " + tr("D — Two dominant wind regimes"))
    _dr1, _dr2 = st.columns(2)
    _dr1.success(
        "**" + tr("NW/W regime (37.2% of days)") + "**  \n"
        + tr("Bay of Biscay sea air — clean marine inflow.") + "  \n"
        + tr("Network mean NO₂ = 16.3 µg/m³") + "  \n"
        + tr("MUSKIZ cleanest — Petronor blown inland") + "  \n"
        + tr("MAZARREDO lower — valley ventilated")
    )
    _dr2.warning(
        "**" + tr("S/SW regime (32.0% of days)") + "**  \n"
        + tr("Inland recirculation — pollution accumulates.") + "  \n"
        + tr("Network mean NO₂ = 21.4 µg/m³ (+31%)") + "  \n"
        + tr("MAZARREDO worst — Nervión valley trapping") + "  \n"
        + tr("BARAKALDO elevated — industrial corridor stagnation")
    )
    st.info(
        "⚠️ **" + tr("ERA5 data note") + ":** "
        + tr("Single regional grid cell (~31 km) shared by all 7 stations. "
             "Wind direction signatures represent synoptic-scale patterns, "
             "not station-level micrometeorology. "
             "Source: Open-Meteo ERA5 archive (CC BY 4.0) · Notebook 10d.")
    )

# ==================== TAB 2: WEATHER RELATIONSHIPS ====================
with tab2:
    st.markdown("### " + tr("Scatter — Pollutant vs Weather Variable"))

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        x_var = st.selectbox(
            tr("Weather variable (X)"),
            [v for v in WEATHER_VARS if v in base.columns],
            index=0
        )
    with sc2:
        y_var = st.selectbox(tr("Pollutant (Y)"), ALL_POLLUTANTS, index=0)
    with sc3:
        color_by = st.radio(tr("Color by"), ["Zone", "Season", "None"], horizontal=True)

    scatter_cols = [x_var, y_var, "Zone", "season"]
    scatter_df   = base[[c for c in scatter_cols if c in base.columns]].dropna().copy()
    corr_val     = scatter_df[[x_var, y_var]].corr().iloc[0, 1]

    if color_by == "Zone":
        fig_sc = px.scatter(
            scatter_df, x=x_var, y=y_var,
            color="Zone",
            color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
            opacity=0.4,
            labels={
                x_var: f"{x_var} ({WEATHER_UNIT.get(x_var,'')})",
                y_var: f"{y_var} (µg/m³)"
            },
            title=f"{y_var} vs {x_var} · r = {corr_val:.3f}"
        )
    elif color_by == "Season" and "season" in scatter_df.columns:
        fig_sc = px.scatter(
            scatter_df, x=x_var, y=y_var,
            color="season",
            color_discrete_map=SEASON_COLOR,
            category_orders={"season": SEASON_ORDER},
            opacity=0.4,
            labels={
                x_var: f"{x_var} ({WEATHER_UNIT.get(x_var,'')})",
                y_var: f"{y_var} (µg/m³)"
            },
            title=f"{y_var} vs {x_var} · r = {corr_val:.3f}"
        )
    else:
        fig_sc = px.scatter(
            scatter_df, x=x_var, y=y_var,
            opacity=0.4,
            color_discrete_sequence=[POLLUTANT_COLOR.get(y_var, "#3498db")],
            labels={
                x_var: f"{x_var} ({WEATHER_UNIT.get(x_var,'')})",
                y_var: f"{y_var} (µg/m³)"
            },
            title=f"{y_var} vs {x_var} · r = {corr_val:.3f}"
        )

    if WHO_ANNUAL.get(y_var):
        fig_sc.add_hline(
            y=WHO_ANNUAL[y_var], line_dash="dash", line_color="red",
            annotation_text=f"WHO {y_var}", annotation_font_size=10
        )
    fig_sc.update_layout(dragmode=False, height=420, hovermode="closest")
    st.plotly_chart(fig_sc,  width="stretch", config=PLOTLY_CONFIG)

    st.info(
        f"Pearson correlation between **{x_var}** and **{y_var}**: **{corr_val:.3f}**"
    )

     # --------------------------------------------------
    # Wind Rose — WindDirection vs WindSpeed
    # --------------------------------------------------
    if "WindDirection" in base.columns and "WindSpeed" in base.columns:
        st.markdown("---")
        st.markdown("### 🌹 Wind Rose")
        st.caption(
            "Each sector = frequency of wind from that direction · "
            "Color = average pollution · Radius = number of days (left) / avg speed (right)"
        )

        wr_poll = st.selectbox(
            tr("Color wind rose by pollutant"),
            ALL_POLLUTANTS, index=0, key="wr_poll"
        )

        wd_df = base[["WindDirection", "WindSpeed", wr_poll]].dropna().copy()

        # 16-sector binning
        wd_df["Dir16"] = (
            ((wd_df["WindDirection"] + 11.25) // 22.5 * 22.5) % 360
        ).astype(int)

        DIR_LABELS = {
            0: "N",   22: "NNE",  45: "NE",  67: "ENE",
            90: "E",  112: "ESE", 135: "SE", 157: "SSE",
            180: "S", 202: "SSW", 225: "SW", 247: "WSW",
            270: "W", 292: "WNW", 315: "NW", 337: "NNW",
        }

        wd_grp = (
            wd_df.groupby("Dir16")
            .agg(
                Days        =("WindDirection", "count"),
                AvgSpeed    =("WindSpeed",      "mean"),
                AvgPollution=(wr_poll,           "mean"),
            )
            .reset_index()
        )
        wd_grp["Label"] = wd_grp["Dir16"].map(
            lambda d: DIR_LABELS.get(d, f"{d}°")
        )

        col_r1, col_r2 = st.columns(2)

        # Compute shared color range for both roses
        _poll_min = float(wd_grp["AvgPollution"].min())
        _poll_max = float(wd_grp["AvgPollution"].max())

        with col_r1:
            fig_rose = px.bar_polar(
                wd_grp, r="Days", theta="Label",
                color="AvgPollution",
                color_continuous_scale="YlOrRd",
                range_color=[_poll_min, _poll_max],
                title=f"Wind frequency · color = avg {wr_poll}",
                labels={
                    "Days":         "Days",
                    "AvgPollution": f"Avg {wr_poll} (µg/m³)",
                    "Label":        "Direction"
                }
            )
            fig_rose.update_layout(
                dragmode=False, height=420,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_rose, width="stretch", config=PLOTLY_CONFIG)

        with col_r2:
            fig_speed_rose = px.bar_polar(
                wd_grp, r="AvgSpeed", theta="Label",
                color="AvgPollution",
                color_continuous_scale="YlOrRd",
                range_color=[_poll_min, _poll_max],
                title=f"Avg wind speed by direction · color = avg {wr_poll}",
                labels={
                    "AvgSpeed":     "Avg Speed (km/h)",
                    "AvgPollution": f"Avg {wr_poll} (µg/m³)",
                    "Label":        "Direction"
                }
            )
            fig_speed_rose.update_layout(
                dragmode=False, height=420,
                coloraxis_colorbar=dict(
                    title=f"Avg {wr_poll}<br>(µg/m³)",
                    thickness=14,
                    len=0.7,
                ),
            )
            st.plotly_chart(fig_speed_rose, width="stretch", config=PLOTLY_CONFIG)

        st.caption(
            f"🎨 {tr('Color scale (both charts)')}: "
            f"{_poll_min:.1f} → {_poll_max:.1f} µg/m³ avg {wr_poll} · "
            f"{tr('Same scale — left = frequency, right = speed')}"
        )

    # --------------------------------------------------
    # Wind Components (Wind_X / Wind_Y) vs Pollution
    # --------------------------------------------------
    wind_cols_present = [c for c in ["Wind_X", "Wind_Y"] if c in base.columns]

    if wind_cols_present:
        st.markdown("---")
        st.markdown("### 🧭 Wind Components vs Pollution")
        st.caption(
            "Wind_X = East–West component · Wind_Y = North–South component · "
            "Derived from WindSpeed and WindDirection"
        )

        wc_cols = st.columns(len(wind_cols_present))
        for idx, wc in enumerate(wind_cols_present):
            wc_df    = base[[wc, y_var, "Zone"]].dropna()
            wc_corr  = wc_df[[wc, y_var]].corr().iloc[0, 1]
            fig_wc   = px.scatter(
                wc_df, x=wc, y=y_var,
                color="Zone",
                color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
                opacity=0.4,
                labels={
                    wc:    f"{wc} ({WEATHER_UNIT.get(wc, 'm/s')})",
                    y_var: f"{y_var} (µg/m³)"
                },
                title=f"{y_var} vs {wc} · r = {wc_corr:.3f}"
            )
            if WHO_ANNUAL.get(y_var):
                fig_wc.add_hline(
                    y=WHO_ANNUAL[y_var], line_dash="dash", line_color="red",
                    annotation_text=f"WHO {y_var}", annotation_font_size=9
                )
            fig_wc.update_layout(dragmode=False, height=380)
            with wc_cols[idx]:
                st.plotly_chart(fig_wc,  width="stretch", config=PLOTLY_CONFIG)
    else:
        st.info(
            tr("Wind_X and Wind_Y columns not found. "
               "Add them to the dataset to enable wind component analysis.")
        )

   

# ==================== TAB 3: SEASONAL EFFECTS ====================
with tab3:
    st.markdown("### 🍂 " + tr("Seasonal Pollution Patterns"))

    if "season" not in base.columns:
        st.warning(tr("Season column not found in dataset."))
    else:
        seas_poll = st.selectbox(tr("Pollutant"), ALL_POLLUTANTS, index=0, key="seas_poll")

        seasonal = (
            base.groupby("season")[
                ALL_POLLUTANTS + [v for v in WEATHER_VARS if v in base.columns]
            ]
            .mean()
            .round(2)
            .reset_index()
        )
        seasonal["season"] = pd.Categorical(
            seasonal["season"], categories=SEASON_ORDER, ordered=True
        )
        seasonal = seasonal.sort_values("season")

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            seas_long = seasonal.melt(
                id_vars="season",
                value_vars=ALL_POLLUTANTS,
                var_name="Pollutant",
                value_name="Concentration"
            )
            fig_seas = px.bar(
                seas_long, x="season", y="Concentration",
                color="Pollutant", barmode="group",
                color_discrete_map=POLLUTANT_COLOR,
                category_orders={"season": SEASON_ORDER},
                title="All pollutants by season",
                labels={"Concentration": "µg/m³"}
            )
            fig_seas.update_layout(dragmode=False, height=380)
            st.plotly_chart(fig_seas,  width="stretch", config=PLOTLY_CONFIG)

        with col_s2:
            fig_sw = make_subplots(specs=[[{"secondary_y": True}]])
            fig_sw.add_trace(
                go.Bar(
                    x=seasonal["season"], y=seasonal[seas_poll],
                    name=f"{seas_poll} (µg/m³)",
                    marker_color=POLLUTANT_COLOR.get(seas_poll, "#9b59b6"),
                    opacity=0.75
                ),
                secondary_y=False
            )
            if "Temperature" in seasonal.columns:
                fig_sw.add_trace(
                    go.Scatter(
                        x=seasonal["season"], y=seasonal["Temperature"],
                        name="Temperature (°C)",
                        line=dict(color=WEATHER_COLOR["Temperature"], width=2.5),
                        mode="lines+markers"
                    ),
                    secondary_y=True
                )
            fig_sw.update_yaxes(title_text=f"{seas_poll} (µg/m³)", secondary_y=False)
            fig_sw.update_yaxes(title_text="Temperature (°C)",      secondary_y=True)
            fig_sw.update_layout(dragmode=False, 
                title=f"{seas_poll} vs Temperature by season",
                height=380, hovermode="x unified",
                legend=dict(orientation="h", y=1.08)
            )
            st.plotly_chart(fig_sw,  width="stretch", config=PLOTLY_CONFIG)

        # Zone × Season heatmap
        st.markdown("#### " + tr("Zone × Season heatmap"))
        zone_seas = (
            base.groupby(["Zone", "season"])[seas_poll]
            .mean().round(2).reset_index()
        )
        zone_seas["season"] = pd.Categorical(
            zone_seas["season"], categories=SEASON_ORDER, ordered=True
        )
        pivot_zs = (
            zone_seas.pivot(index="Zone", columns="season", values=seas_poll)
            .reindex(columns=SEASON_ORDER)
        )
        fig_zs = px.imshow(
            pivot_zs,
            color_continuous_scale=["#2ecc71", "#f9ca24", "#e74c3c"],
            text_auto=".1f",
            aspect="auto",
            title=f"{seas_poll} µg/m³ — Zone × Season"
        )
        fig_zs.update_layout(dragmode=False, height=280)
        st.plotly_chart(fig_zs,  width="stretch", config=PLOTLY_CONFIG)

        with st.expander(tr("📊 Full seasonal statistics")):
            st.dataframe(seasonal.set_index("season"), width="stretch")

# ==================== TAB 4: LAG ANALYSIS ====================
with tab4:
    st.markdown("### ⏳ " + tr("Lag Correlation Analysis"))
    st.caption(
        tr("How well does the latest available reading (D-1) predict future pollution? "
           "High lag correlation = persistent pollution events.")
    )

    lag_poll = st.selectbox(
        tr("Pollutant"), ALL_POLLUTANTS, index=0, key="lag_poll"
    )

    # ── داینامیک: prefix و target ──────────────────────────────
    poll_prefix = lag_poll.replace(".", "")          # PM2.5 → PM25
    target_col  = f"target_{poll_prefix}"            # target_PM25 / target_SO2 ...

    if target_col not in base.columns:
        st.warning(f"`{target_col}` not found in dataset.")
        st.stop()

    # ── Lag columns و days map ─────────────────────────────────
    LAG_DAYS     = [1, 3, 7, 30, 90, 365]
    lag_cols_use = [f"{poll_prefix}_lag_{d}" for d in LAG_DAYS
                    if f"{poll_prefix}_lag_{d}" in base.columns]

    lag_days_map = {f"{poll_prefix}_lag_{d}": d for d in LAG_DAYS}

    # ── Lag chart ──────────────────────────────────────────────
    results = []
    for col in lag_cols_use:
        pair = base[[col, target_col]].dropna()
        if len(pair) > 10:
            r = pair.corr().iloc[0, 1]
            results.append({
                "Lag feature": col,
                "Days":        lag_days_map[col],
                "r":           round(r, 3),
            })

    if results:
        lag_df = pd.DataFrame(results).sort_values("Days")

        fig_lag = go.Figure()
        fig_lag.add_trace(go.Bar(
            x=lag_df["Days"].astype(str) + "d",
            y=lag_df["r"],
            marker_color=["#2ecc71" if r >= 0 else "#e74c3c" for r in lag_df["r"]],
            text=lag_df["r"].round(3),
            textposition="outside",
            name="Correlation r"
        ))
        fig_lag.add_hline(y=0, line_color="#bbb", line_width=1)
        fig_lag.update_layout(dragmode=False, 
            title=f"{lag_poll} lag features vs {target_col} · correlation",
            xaxis_title="Lag (days)",
            yaxis_title="Pearson r",
            height=400,
            yaxis=dict(range=[-0.1, max(lag_df["r"].max() * 1.2, 0.5)])
        )
        st.plotly_chart(fig_lag,  width="stretch", config=PLOTLY_CONFIG)

        # ── Rolling Mean ───────────────────────────────────────
        st.markdown("#### " + tr("Rolling Mean Features"))
        ROLL_WINDOWS  = [7, 30, 90, 365]
        roll_cols_use = [f"{poll_prefix}_roll_mean_{w}" for w in ROLL_WINDOWS
                         if f"{poll_prefix}_roll_mean_{w}" in base.columns]
        roll_days_map = {f"{poll_prefix}_roll_mean_{w}": w for w in ROLL_WINDOWS}

        roll_results = []
        for col in roll_cols_use:
            pair = base[[col, target_col]].dropna()
            if len(pair) > 10:
                r = pair.corr().iloc[0, 1]
                roll_results.append({
                    "Roll feature":  col,
                    "Window (days)": roll_days_map[col],
                    "r":             round(r, 3)
                })

        if roll_results:
            roll_df = pd.DataFrame(roll_results).sort_values("Window (days)")
            fig_roll = px.bar(
                roll_df,
                x=roll_df["Window (days)"].astype(str) + "d",
                y="r",
                color="r",
                color_continuous_scale="Blues",
                text="r",
                title=f"{lag_poll} rolling mean features vs {target_col}",
                labels={"x": "Window (days)", "r": "Pearson r"}
            )
            fig_roll.update_traces(
                texttemplate="%{text:.3f}", textposition="outside"
            )
            fig_roll.update_layout(dragmode=False, height=360, showlegend=False)
            st.plotly_chart(fig_roll,  width="stretch", config=PLOTLY_CONFIG)
        else:
            st.info(f"No rolling mean columns found for {lag_poll}.")

        with st.expander(tr("📊 Lag correlation table")):
            st.dataframe(
                lag_df.set_index("Lag feature"),
                 width="stretch"
            )
    else:
        st.warning(f"No lag columns found for {lag_poll}.")

# ==================== TAB 5: FORECAST FEATURES ====================
with tab5:
    st.markdown("### 🤖 " + tr("Feature Ranking for Forecasting"))
    st.caption(
        tr("Pearson correlation of all candidate features with target_PM25. "
           "High absolute correlation = potentially useful predictor. "
           "Computed on the full dataset — independent of sidebar filters.")
    )
    st.info(
        "💡 " + tr("This tab always uses the full dataset (2015–2026) "
                   "to reflect what the model was actually trained on. "
                   "Sidebar filters do not apply here.")
    )

    # Use full df — not base — correlations must reflect training data
    _feat_df = df.copy()

    if "target_PM25" not in _feat_df.columns:
        st.warning(tr("target_PM25 column not found in dataset."))
        st.stop()

    candidate_features = (
        [v for v in WEATHER_VARS if v in _feat_df.columns] +
        [c for c in _feat_df.columns if
         any(c.startswith(f"{pfx}_lag_") for pfx in _POLL_PREFIXES)] +
        [c for c in _feat_df.columns if
         any(c.startswith(f"{pfx}_roll_mean_") for pfx in _POLL_PREFIXES)]
    )
    candidate_features = [f for f in candidate_features if f in _feat_df.columns]

    corr_target = (
        _feat_df[candidate_features + ["target_PM25"]]
        .corr()["target_PM25"]
        .drop("target_PM25")
        .reset_index()
        .rename(columns={"index": "Feature", "target_PM25": "r"})
        .assign(abs_r=lambda d: d["r"].abs())
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )
    corr_target.index += 1

    def feat_category(f):
        if any(f.startswith(v) for v in WEATHER_VARS): return "Weather"
        if "roll_mean" in f: return "Rolling Mean"
        if "_lag_" in f:     return "Lag"
        return "Other"

    corr_target["Category"] = corr_target["Feature"].apply(feat_category)

    cat_color = {
        "Weather":      "#3498db",
        "Lag":          "#e67e22",
        "Rolling Mean": "#9b59b6",
        "Other":        "#95a5a6",
    }

    top_n = st.slider(tr("Show top N features"), 10, len(corr_target), 20)

    fig_feat = px.bar(
        corr_target.head(top_n),
        x="r", y="Feature",
        color="Category",
        color_discrete_map=cat_color,
        orientation="h",
        title=f"Top {top_n} features by |correlation| with target_PM25",
        labels={"r": "Pearson r", "Feature": ""},
        text="r"
    )
    fig_feat.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_feat.add_vline(x=0, line_color="#bbb", line_width=1)
    fig_feat.update_layout(dragmode=False, 
        height=max(400, top_n * 28),
        margin=dict(l=10, r=60, t=50, b=10),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_feat,  width="stretch", config=PLOTLY_CONFIG)

    st.markdown("#### " + tr("Summary by feature category"))
    cat_summary = (
        corr_target.groupby("Category")["abs_r"]
        .agg(["mean", "max", "count"])
        .rename(columns={"mean": "Avg |r|", "max": "Max |r|", "count": "# Features"})
        .round(3)
        .sort_values("Avg |r|", ascending=False)
    )
    st.dataframe(cat_summary, width="stretch")

    with st.expander(tr("📊 Full feature ranking table")):
        st.dataframe(
            corr_target.drop(columns="abs_r"),
            width="stretch",
            column_config={
                "r": st.column_config.ProgressColumn(
                    "Correlation (r)", min_value=-1, max_value=1, format="%.3f"
                )
            }
        )