import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    load_data,
    WHO_ANNUAL, CORE_POLLUTANTS,
    POLLUTANT_COLOR, MONTH_NAMES,
    ZONE_META, get_zone,
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Weather & Pollution Correlation",
    page_icon="🌤️",
    layout="wide"
)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

WEATHER_VARS = ["Temperature", "Humidity", "Precipitation", "WindSpeed"]

WEATHER_COLOR = {
    "Temperature":   "#e74c3c",
    "Humidity":      "#3498db",
    "Precipitation": "#1abc9c",
    "WindSpeed":     "#95a5a6",
}

WEATHER_UNIT = {
    "Temperature":   "°C",
    "Humidity":      "%",
    "Precipitation": "mm",
    "WindSpeed":     "m/s",
}

WEATHER_ICON = {
    "Temperature":   "🌡️",
    "Humidity":      "💧",
    "Precipitation": "🌧️",
    "WindSpeed":     "💨",
}

ALL_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]

# --------------------------------------------------
# LOAD & PREPARE
# --------------------------------------------------

df = load_data()

# Verify weather columns exist
missing = [c for c in WEATHER_VARS if c not in df.columns]
if missing:
    st.error(f"Weather columns not found in dataset: {missing}")
    st.stop()

# Daily aggregation per station
daily = (
    df
    .groupby(["station", "Town", "Zone", "Latitude", "Longitude", "Date", "Year", "Month"],
             as_index=False)
    .agg({
        "PM2.5":         "mean",
        "PM10":          "mean",
        "NO2":           "mean",
        "SO2":           "mean",
        "Temperature":   "mean",
        "Humidity":      "mean",
        "Precipitation": "mean",
        "WindSpeed":     "mean",
    })
)

all_stations = sorted(daily["station"].unique().tolist())
all_years    = sorted(daily["Year"].dropna().unique().tolist())

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🌤️ Weather & Pollution")
    st.markdown("Greater Bilbao · Meteorological Drivers")
    st.divider()

    st.markdown("### Filters")

    filter_mode = st.radio("Filter by", ["All Stations", "Zone", "Station"],
                           horizontal=True)

    if filter_mode == "Zone":
        selected_zone = st.selectbox(
            "Zone",
            list(ZONE_META.keys()),
            format_func=lambda z: f"{ZONE_META[z]['icon']} {z}"
        )
    elif filter_mode == "Station":
        selected_station = st.selectbox("Station", all_stations)

    year_opts = ["All"] + [str(y) for y in all_years]
    selected_year_str = st.selectbox("Year", year_opts, index=0)
    selected_year = None if selected_year_str == "All" else int(selected_year_str)

    st.divider()
    st.markdown("### Zone Legend")
    for z, meta in ZONE_META.items():
        st.markdown(
            f"{meta['icon']} **{z}**",
            unsafe_allow_html=True
        )

# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------

base = daily.copy()

if filter_mode == "Zone":
    base        = base[base["Zone"] == selected_zone]
    scope_label = f"{ZONE_META[selected_zone]['icon']} {selected_zone}"
elif filter_mode == "Station":
    base        = base[base["station"] == selected_station]
    scope_label = selected_station
else:
    scope_label = "All Stations"

if selected_year:
    base         = base[base["Year"] == selected_year]
    period_label = str(selected_year)
else:
    period_label = "2015–2026"

if base.empty:
    st.warning("No data for selected filters.")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌤️ Weather & Pollution Correlation")
st.caption(
    f"Scope: **{scope_label}** · Period: **{period_label}** · "
    f"n = {len(base):,} daily observations"
)

# --------------------------------------------------
# KPI ROW — weather snapshot
# --------------------------------------------------

st.markdown("### 🌡️ Weather Overview")

wk1, wk2, wk3, wk4 = st.columns(4)
weather_kpis = [wk1, wk2, wk3, wk4]

for col, var in zip(weather_kpis, WEATHER_VARS):
    val  = base[var].mean()
    unit = WEATHER_UNIT[var]
    icon = WEATHER_ICON[var]
    col.metric(f"{icon} Avg {var}", f"{val:.1f} {unit}")

st.divider()

# --------------------------------------------------
# TAB LAYOUT
# --------------------------------------------------

tab_corr, tab_scatter, tab_wind, tab_rain, tab_zone = st.tabs([
    "🔗 Correlation Matrix",
    "📍 Scatter Analysis",
    "💨 Wind Effect",
    "🌧️ Rain & Humidity",
    "🗺️ Zone Comparison",
])

# ==================== TAB 1: CORRELATION MATRIX ====================
with tab_corr:
    st.markdown("### Correlation between weather variables and pollutants")
    st.caption("Pearson correlation coefficient · −1 = inverse · 0 = no relation · +1 = direct")

    corr_cols  = ALL_POLLUTANTS + WEATHER_VARS
    corr_df    = base[corr_cols].dropna()
    corr_matrix = corr_df.corr()

    # Focus: pollutants vs weather only
    corr_focus = corr_matrix.loc[ALL_POLLUTANTS, WEATHER_VARS]

    fig_corr = px.imshow(
        corr_focus,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
        title="Pollutant × Weather Correlation"
    )
    fig_corr.update_layout(
        height=320,
        coloraxis_colorbar=dict(title="r"),
        margin=dict(t=50, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_corr, width="stretch")

    # Full matrix in expander
    with st.expander("View full correlation matrix"):
        fig_full = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
            title="Full Correlation Matrix"
        )
        fig_full.update_layout(height=450)
        st.plotly_chart(fig_full, width="stretch")

    # Key insights
    st.markdown("#### 🔍 Strongest Correlations")
    pairs = []
    for p in ALL_POLLUTANTS:
        for w in WEATHER_VARS:
            r = corr_focus.loc[p, w]
            pairs.append({"Pollutant": p, "Weather Var": w, "r": round(r, 3)})

    pairs_df = (
        pd.DataFrame(pairs)
        .assign(abs_r=lambda d: d["r"].abs())
        .sort_values("abs_r", ascending=False)
        .drop(columns="abs_r")
        .head(6)
        .reset_index(drop=True)
    )
    pairs_df.index += 1

    st.dataframe(
        pairs_df,
        use_container_width=False,
        column_config={
            "r": st.column_config.ProgressColumn(
                "Correlation (r)",
                min_value=-1,
                max_value=1,
                format="%.3f"
            )
        }
    )

# ==================== TAB 2: SCATTER ====================
with tab_scatter:
    st.markdown("### Scatter Plot — Pollutant vs Weather Variable")

    sc1, sc2 = st.columns(2)
    with sc1:
        x_var = st.selectbox("Weather variable (X)", WEATHER_VARS, index=3)
    with sc2:
        y_var = st.selectbox("Pollutant (Y)", ALL_POLLUTANTS, index=0)

    color_by = st.radio(
        "Color by", ["Zone", "Month", "None"],
        horizontal=True
    )

    scatter_df = base[[x_var, y_var, "Zone", "Month", "station"]].dropna()

    if color_by == "Zone":
        color_col = "Zone"
        color_map = {z: m["color"] for z, m in ZONE_META.items()}
        fig_sc = px.scatter(
            scatter_df, x=x_var, y=y_var,
            color=color_col,
            color_discrete_map=color_map,
            opacity=0.5,
            trendline="ols",
            trendline_scope="overall",
            labels={
                x_var: f"{x_var} ({WEATHER_UNIT[x_var]})",
                y_var: f"{y_var} (µg/m³)"
            },
            title=f"{y_var} vs {x_var} — colored by zone"
        )

    elif color_by == "Month":
        scatter_df["MonthName"] = scatter_df["Month"].map(MONTH_NAMES)
        fig_sc = px.scatter(
            scatter_df, x=x_var, y=y_var,
            color="Month",
            color_continuous_scale="turbo",
            opacity=0.4,
            trendline="ols",
            trendline_scope="overall",
            labels={
                x_var: f"{x_var} ({WEATHER_UNIT[x_var]})",
                y_var: f"{y_var} (µg/m³)"
            },
            title=f"{y_var} vs {x_var} — colored by month"
        )
    else:
        fig_sc = px.scatter(
            scatter_df, x=x_var, y=y_var,
            opacity=0.4,
            trendline="ols",
            color_discrete_sequence=[POLLUTANT_COLOR.get(y_var, "#3498db")],
            labels={
                x_var: f"{x_var} ({WEATHER_UNIT[x_var]})",
                y_var: f"{y_var} (µg/m³)"
            },
            title=f"{y_var} vs {x_var}"
        )

    if WHO_ANNUAL.get(y_var):
        fig_sc.add_hline(
            y=WHO_ANNUAL[y_var],
            line_dash="dash",
            line_color="red",
            annotation_text=f"WHO {y_var}",
            annotation_font_size=10
        )

    fig_sc.update_layout(height=460, hovermode="closest")
    st.plotly_chart(fig_sc, width="stretch")

# ==================== TAB 3: WIND EFFECT ====================
with tab_wind:
    st.markdown("### 💨 Wind Speed Effect on Pollution")
    st.caption(
        "Higher wind speed improves atmospheric dispersion — "
        "especially relevant for Industrial Corridor stations"
    )

    wind_poll = st.selectbox("Pollutant", ALL_POLLUTANTS, index=0, key="wind_poll")

    # Bin wind speed
    wind_df = base[["WindSpeed", wind_poll, "Zone", "Month"]].dropna().copy()
    wind_df["WindBin"] = pd.cut(
        wind_df["WindSpeed"],
        bins=[0, 2, 4, 6, 8, 100],
        labels=["0–2", "2–4", "4–6", "6–8", ">8"]
    )

    wind_zone = (
        wind_df
        .groupby(["WindBin", "Zone"], observed=True)[wind_poll]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "Mean", "std": "Std", "count": "N"})
    )

    fig_wind = px.bar(
        wind_zone,
        x="WindBin",
        y="Mean",
        color="Zone",
        barmode="group",
        error_y="Std",
        color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
        labels={
            "WindBin": "Wind Speed bin (m/s)",
            "Mean":    f"{wind_poll} mean (µg/m³)"
        },
        title=f"{wind_poll} mean by wind speed bin and zone"
    )
    if WHO_ANNUAL.get(wind_poll):
        fig_wind.add_hline(
            y=WHO_ANNUAL[wind_poll],
            line_dash="dash", line_color="red",
            annotation_text=f"WHO {wind_poll}"
        )
    fig_wind.update_layout(height=400)
    st.plotly_chart(fig_wind, width="stretch")

    # Wind × month seasonality
    st.markdown("#### Monthly average wind speed vs pollution")

    wind_monthly = (
        base
        .groupby("Month")
        .agg(
            WindSpeed=(  "WindSpeed",  "mean"),
            **{wind_poll: (wind_poll, "mean")}
        )
        .reset_index()
    )
    wind_monthly["MonthName"] = wind_monthly["Month"].map(MONTH_NAMES)

    fig_wm = make_subplots(specs=[[{"secondary_y": True}]])
    fig_wm.add_trace(
        go.Bar(
            x=wind_monthly["MonthName"],
            y=wind_monthly[wind_poll],
            name=f"{wind_poll} (µg/m³)",
            marker_color=POLLUTANT_COLOR.get(wind_poll, "#9b59b6"),
            opacity=0.7
        ),
        secondary_y=False
    )
    fig_wm.add_trace(
        go.Scatter(
            x=wind_monthly["MonthName"],
            y=wind_monthly["WindSpeed"],
            name="Wind Speed (m/s)",
            line=dict(color=WEATHER_COLOR["WindSpeed"], width=2.5),
            mode="lines+markers"
        ),
        secondary_y=True
    )
    fig_wm.update_yaxes(title_text=f"{wind_poll} (µg/m³)", secondary_y=False)
    fig_wm.update_yaxes(title_text="Wind Speed (m/s)",     secondary_y=True)
    fig_wm.update_layout(
        title=f"Monthly {wind_poll} vs Wind Speed",
        height=360, hovermode="x unified",
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig_wm, width="stretch")

# ==================== TAB 4: RAIN & HUMIDITY ====================
with tab_rain:
    st.markdown("### 🌧️ Precipitation & Humidity Effect")

    rain_poll = st.selectbox("Pollutant", ALL_POLLUTANTS, index=1, key="rain_poll")

    col_r, col_h = st.columns(2)

    # ── Precipitation bins ──────────────────────────
    with col_r:
        st.markdown("#### Precipitation bins")

        rain_df = base[["Precipitation", rain_poll, "Zone"]].dropna().copy()
        rain_df["RainBin"] = pd.cut(
            rain_df["Precipitation"],
            bins=[-0.01, 0, 2, 10, 1000],
            labels=["Dry (0)", "Light (0–2mm)", "Moderate (2–10mm)", "Heavy (>10mm)"]
        )

        rain_zone = (
            rain_df
            .groupby(["RainBin", "Zone"], observed=True)[rain_poll]
            .mean()
            .reset_index()
            .rename(columns={rain_poll: "Mean"})
        )

        fig_rain = px.bar(
            rain_zone,
            x="RainBin",
            y="Mean",
            color="Zone",
            barmode="group",
            color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
            labels={
                "RainBin": "Precipitation category",
                "Mean":    f"{rain_poll} mean (µg/m³)"
            },
            title=f"{rain_poll} by precipitation level"
        )
        if WHO_ANNUAL.get(rain_poll):
            fig_rain.add_hline(
                y=WHO_ANNUAL[rain_poll],
                line_dash="dash", line_color="red",
                annotation_text=f"WHO {rain_poll}"
            )
        fig_rain.update_layout(height=360)
        st.plotly_chart(fig_rain, width="stretch")

    # ── Humidity bins ───────────────────────────────
    with col_h:
        st.markdown("#### Humidity bins")

        hum_df = base[["Humidity", rain_poll, "Zone"]].dropna().copy()
        hum_df["HumBin"] = pd.cut(
            hum_df["Humidity"],
            bins=[0, 40, 60, 80, 100],
            labels=["Dry (<40%)", "Comfortable (40–60%)",
                    "Humid (60–80%)", "Very humid (>80%)"]
        )

        hum_zone = (
            hum_df
            .groupby(["HumBin", "Zone"], observed=True)[rain_poll]
            .mean()
            .reset_index()
            .rename(columns={rain_poll: "Mean"})
        )

        fig_hum = px.bar(
            hum_zone,
            x="HumBin",
            y="Mean",
            color="Zone",
            barmode="group",
            color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
            labels={
                "HumBin": "Humidity category",
                "Mean":   f"{rain_poll} mean (µg/m³)"
            },
            title=f"{rain_poll} by humidity level"
        )
        if WHO_ANNUAL.get(rain_poll):
            fig_hum.add_hline(
                y=WHO_ANNUAL[rain_poll],
                line_dash="dash", line_color="red",
                annotation_text=f"WHO {rain_poll}"
            )
        fig_hum.update_layout(height=360)
        st.plotly_chart(fig_hum, width="stretch")

    # ── Monthly precipitation vs pollution ──────────
    st.markdown("#### Monthly precipitation vs pollution")

    prec_monthly = (
        base
        .groupby("Month")
        .agg(
            Precipitation=("Precipitation", "mean"),
            **{rain_poll: (rain_poll, "mean")}
        )
        .reset_index()
    )
    prec_monthly["MonthName"] = prec_monthly["Month"].map(MONTH_NAMES)

    fig_pm = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pm.add_trace(
        go.Bar(
            x=prec_monthly["MonthName"],
            y=prec_monthly[rain_poll],
            name=f"{rain_poll} (µg/m³)",
            marker_color=POLLUTANT_COLOR.get(rain_poll, "#e67e22"),
            opacity=0.7
        ),
        secondary_y=False
    )
    fig_pm.add_trace(
        go.Scatter(
            x=prec_monthly["MonthName"],
            y=prec_monthly["Precipitation"],
            name="Precipitation (mm)",
            line=dict(color=WEATHER_COLOR["Precipitation"], width=2.5),
            mode="lines+markers"
        ),
        secondary_y=True
    )
    fig_pm.update_yaxes(title_text=f"{rain_poll} (µg/m³)", secondary_y=False)
    fig_pm.update_yaxes(title_text="Precipitation (mm)",   secondary_y=True)
    fig_pm.update_layout(
        title=f"Monthly {rain_poll} vs Precipitation",
        height=360, hovermode="x unified",
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig_pm, width="stretch")

# ==================== TAB 5: ZONE COMPARISON ====================
with tab_zone:
    st.markdown("### 🗺️ Zone × Weather Profile")
    st.caption("How meteorological conditions differ across environmental zones")

    # Weather profile per zone
    zone_weather = (
        base
        .groupby("Zone")[WEATHER_VARS + ALL_POLLUTANTS]
        .mean()
        .round(2)
        .reset_index()
    )

    # Zone weather cards
    zones_present = [z for z in ZONE_META if z in zone_weather["Zone"].values]
    zcols = st.columns(len(zones_present) or 1)

    for idx, zone_name in enumerate(zones_present):
        meta   = ZONE_META[zone_name]
        z_row  = zone_weather[zone_weather["Zone"] == zone_name].iloc[0]

        with zcols[idx]:
            st.markdown(
                f"""
                <div style="
                    border-left:5px solid {meta['border']};
                    background:linear-gradient(135deg,{meta['color']}18,{meta['color']}06);
                    border-radius:10px; padding:14px 16px; margin-bottom:8px;
                ">
                    <div style="font-size:18px;margin-bottom:6px">
                        {meta['icon']} <strong>{zone_name}</strong>
                    </div>
                    <table style="width:100%;font-size:12px;border-collapse:collapse">
                        <tr><td style="color:#777;padding:2px 0">🌡️ Temperature</td>
                            <td style="text-align:right;font-weight:600">{z_row['Temperature']:.1f} °C</td></tr>
                        <tr><td style="color:#777;padding:2px 0">💧 Humidity</td>
                            <td style="text-align:right;font-weight:600">{z_row['Humidity']:.1f} %</td></tr>
                        <tr><td style="color:#777;padding:2px 0">🌧️ Precipitation</td>
                            <td style="text-align:right;font-weight:600">{z_row['Precipitation']:.1f} mm</td></tr>
                        <tr><td style="color:#777;padding:2px 0">💨 Wind Speed</td>
                            <td style="text-align:right;font-weight:600">{z_row['WindSpeed']:.1f} m/s</td></tr>
                        <tr><td colspan="2" style="padding:4px 0"></td></tr>
                        <tr><td style="color:#777;padding:2px 0">PM2.5</td>
                            <td style="text-align:right;font-weight:600">{z_row['PM2.5']:.1f} µg/m³</td></tr>
                        <tr><td style="color:#777;padding:2px 0">NO₂</td>
                            <td style="text-align:right;font-weight:600">{z_row['NO2']:.1f} µg/m³</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # Radar: weather profile per zone
    st.markdown("#### Zone Weather Profile — Radar")

    weather_radar_vars = WEATHER_VARS
    fig_radar = go.Figure()

    for zone_name in zones_present:
        meta  = ZONE_META[zone_name]
        z_row = zone_weather[zone_weather["Zone"] == zone_name].iloc[0]

        # Normalize each variable to 0–1 for radar
        vals_norm = []
        for var in weather_radar_vars:
            col_min = zone_weather[var].min()
            col_max = zone_weather[var].max()
            rng     = col_max - col_min
            norm    = (z_row[var] - col_min) / rng if rng > 0 else 0.5
            vals_norm.append(round(norm, 3))

        fig_radar.add_trace(go.Scatterpolar(
            r=vals_norm + [vals_norm[0]],
            theta=weather_radar_vars + [weather_radar_vars[0]],
            fill="toself",
            name=f"{meta['icon']} {zone_name}",
            line=dict(color=meta["color"], width=2),
            opacity=0.6
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Normalized weather profile by zone (0 = lowest, 1 = highest across zones)",
        showlegend=True,
        height=420,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig_radar, width="stretch")

    # Scatter matrix: weather vs pollutant per zone
    st.markdown("#### Temperature vs Pollutants by Zone")

    temp_poll = st.selectbox("Pollutant", ALL_POLLUTANTS, index=2, key="temp_poll")

    fig_temp = px.scatter(
        base[["Temperature", temp_poll, "Zone", "Month"]].dropna(),
        x="Temperature",
        y=temp_poll,
        color="Zone",
        color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
        trendline="ols",
        opacity=0.45,
        facet_col="Zone",
        facet_col_wrap=3,
        labels={
            "Temperature": "Temperature (°C)",
            temp_poll:     f"{temp_poll} (µg/m³)"
        },
        title=f"{temp_poll} vs Temperature — per zone"
    )
    if WHO_ANNUAL.get(temp_poll):
        fig_temp.add_hline(
            y=WHO_ANNUAL[temp_poll],
            line_dash="dash", line_color="red",
            annotation_text=f"WHO {temp_poll}", annotation_font_size=9
        )
    fig_temp.update_layout(height=380)
    st.plotly_chart(fig_temp, width="stretch")

    # Full summary table
    with st.expander("📊 Full zone × weather summary table"):
        st.dataframe(
            zone_weather.set_index("Zone").round(2),
            width="stretch"
        )

    st.caption(
    f"Scope: **{scope_label}** · Period: **{period_label}** · "
    f"n = {len(base):,} daily observations · "
    "Meteorological data: Open-Meteo (open-meteo.com)"
)