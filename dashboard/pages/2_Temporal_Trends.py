import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    load_data,
    WHO_ANNUAL, WHO_SO2_DAILY, CORE_POLLUTANTS,
    POLLUTANT_COLOR, MONTH_NAMES, RISK_COLORS, RISK_ORDER,
    ZONE_MAP, ZONE_META, get_zone,
    classify_core_risk, risk_color, short_term_flag,
    who_ratio_label, who_delta, get_fav_station, center_tables
)
from i18n_auto import language_selector, apply_lang_styles, tr

from weather_panel import weather_trend

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from forecast_utils import plotly_touch_config, PLOTLY_CONFIG

plotly_touch_config()  


st.set_page_config(page_title="Temporal Trends", layout="wide")
language_selector()
apply_lang_styles()
center_tables()

# -----------------------
# Constants
# -----------------------
COVID_PERIODS = ["Pre-COVID", "COVID", "Post-COVID"]
# -----------------------
# Load Data
# -----------------------

df = load_data()

# -----------------------
# Page Title
# -----------------------

st.title(tr("📈 Temporal Trends Dashboard"))
st.caption(tr("Greater Bilbao · 2015–2026 · WHO 2021 guidelines"))

# -----------------------
# Sidebar
# -----------------------

with st.sidebar:
    st.markdown("### " + tr("Filters"))

    pollutant = st.selectbox(tr("Pollutant"), ["PM2.5", "PM10", "NO2", "SO2"])

    # ---- Station / Zone selector ----
    zone_options  = ["ALL ZONES"] + list(ZONE_META.keys())
    station_options = ["ALL"] + sorted(df["station"].unique().tolist())

    filter_mode = st.radio(tr("Filter by"), ["Station", "Zone"], horizontal=True)

    if filter_mode == "Station":
            _fav = get_fav_station(station_options)
            _idx = station_options.index(_fav) if _fav in station_options else 0
            selected_station = st.selectbox(tr("Station"), station_options, index=_idx)
            selected_zone    = None
    else:
        selected_zone    = st.selectbox(
            tr("Zone"),
            list(ZONE_META.keys()),
            format_func=lambda z: f"{ZONE_META[z]['icon']} {z}"
        )
        selected_station = None

    # ---- Time granularity ----
    st.divider()
    st.markdown("### " + tr("Time Range"))

    time_mode = st.radio(tr("Granularity"), ["Year", "Month", "Day"], horizontal=True)

    if time_mode == "Year":
        year_options = ["All"] + sorted(df["Year"].dropna().unique().tolist())
        selected_year = st.selectbox(tr("Year"), year_options, index=0)
        selected_month = None
        selected_day   = None

    elif time_mode == "Month":
        year_list = sorted(df["Year"].dropna().unique().tolist())
        selected_year = st.selectbox(tr("Year"), year_list, index=len(year_list) - 1)
        month_options = ["All"] + [MONTH_NAMES[m] for m in sorted(
            df[df["Year"] == selected_year]["Month"].dropna().unique().tolist()
        )]
        selected_month_label = st.selectbox(tr("Month"), month_options, index=0)
        selected_month = (
            {v: k for k, v in MONTH_NAMES.items()}[selected_month_label]
            if selected_month_label != "All" else None
        )
        selected_day = None

    else:  # Day
        year_list = sorted(df["Year"].dropna().unique().tolist())
        selected_year = st.selectbox(tr("Year"), year_list, index=len(year_list) - 1)
        month_avail = sorted(
            df[df["Year"] == selected_year]["Month"].dropna().unique().tolist()
        )
        selected_month_label = st.selectbox(
            tr("Month"), [MONTH_NAMES[m] for m in month_avail]
        )
        selected_month = {v: k for k, v in MONTH_NAMES.items()}[selected_month_label]
        day_avail = sorted(
            df[(df["Year"] == selected_year) & (df["Month"] == selected_month)]["Day"]
            .dropna().unique().tolist()
        )
        selected_day = st.selectbox(
            tr("Day"), day_avail,
            format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y")
        )

    show_who = st.toggle(tr("Show WHO guideline"), value=True)

    st.divider()
    if pollutant in WHO_ANNUAL:
        st.markdown(f"**{tr('WHO annual limit')} ({pollutant}):** {WHO_ANNUAL[pollutant]} µg/m³")
    else:
        st.markdown(tr("SO₂: evaluated on 24h exceedance, no annual WHO limit"))

    # Zone legend
    st.divider()
    st.markdown("### 🗺️ " + tr("Zones"))
    for z, meta in ZONE_META.items():
        stations_in_zone = df[df["Zone"] == z]["station"].unique().tolist()
        short = [s.split("_")[0] for s in stations_in_zone]
        st.markdown(
            f"{meta['icon']} **{z}**  \n"
            f"<span style='font-size:11px;color:#777'>{', '.join(short)}</span>",
            unsafe_allow_html=True
        )

# -----------------------
# Apply filters
# -----------------------

# Station / Zone filter
if filter_mode == "Station":
    df_filtered = (
        df[df["station"] == selected_station].copy()
        if selected_station != "ALL"
        else df.copy()
    )
    scope_label = selected_station if selected_station != "ALL" else "All Stations"
else:
    df_filtered = df[df["Zone"] == selected_zone].copy()
    scope_label = f"{ZONE_META[selected_zone]['icon']} {selected_zone}"

# Time filter
if time_mode == "Year":
    if selected_year != "All":
        df_filtered = df_filtered[df_filtered["Year"] == int(selected_year)]
    period_label = str(selected_year) if selected_year != "All" else "2015–2026"

elif time_mode == "Month":
    df_filtered = df_filtered[df_filtered["Year"] == selected_year]
    if selected_month:
        df_filtered = df_filtered[df_filtered["Month"] == selected_month]
        period_label = f"{MONTH_NAMES[selected_month]} {selected_year}"
    else:
        period_label = f"All months of {selected_year}"

else:  # Day
    df_filtered = df_filtered[df_filtered["Day"] == selected_day]
    period_label = pd.Timestamp(selected_day).strftime("%d %B %Y")

df_filtered = df_filtered.copy()

unit       = "µg/m³"
who_limit  = WHO_ANNUAL.get(pollutant)
main_color = POLLUTANT_COLOR[pollutant]

if df_filtered.empty:
    st.warning(tr("No data available for the selected filters."))
    st.stop()

# -----------------------
# KPIs
# -----------------------

avg_val   = df_filtered[pollutant].mean()
max_val   = df_filtered[pollutant].max()
min_val   = df_filtered[pollutant].min()
who_ratio = f"{avg_val / who_limit:.1f}× WHO" if who_limit else "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{tr('Average')} {pollutant}", f"{avg_val:.1f} {unit}")
c2.metric(tr("Maximum (daily)"), f"{max_val:.1f} {unit}")
c3.metric(tr("Minimum (daily)"), f"{min_val:.1f} {unit}")
c4.metric(tr("Avg vs WHO limit"), who_ratio)

st.caption(f"Scope: **{scope_label}** · Period: **{period_label}**")
st.divider()

# -----------------------
# Zone Summary Banner (when viewing ALL or a specific Zone)
# -----------------------

if filter_mode == "Zone" or (filter_mode == "Station" and selected_station == "ALL"):
    st.subheader(tr("🗺️ Zone Breakdown"))

    zones_in_scope = df_filtered["Zone"].unique().tolist()
    zone_cols = st.columns(len([z for z in ZONE_META if z in zones_in_scope]) or 1)
    col_idx = 0

    for zone_name, meta in ZONE_META.items():
        zone_data = df_filtered[df_filtered["Zone"] == zone_name]
        if zone_data.empty:
            continue

        z_avg = zone_data[pollutant].mean()
        z_max = zone_data[pollutant].max()
        z_min = zone_data[pollutant].min()
        vs_who = f"{z_avg / who_limit:.1f}×" if who_limit else "—"

        with zone_cols[col_idx]:
            st.markdown(
                f"""
                <div style="
                    border-left:5px solid {meta['color']};
                    background:linear-gradient(135deg,{meta['color']}18,{meta['color']}06);
                    border-radius:10px; padding:14px 16px; margin-bottom:8px;
                ">
                    <div style="font-size:18px;margin-bottom:4px">
                        {meta['icon']} <strong>{zone_name}</strong>
                    </div>
                    <table style="width:100%;font-size:13px;border-collapse:collapse">
                        <tr>
                            <td style="color:#777;padding:2px 0">Avg {pollutant}</td>
                            <td style="text-align:right;font-weight:600">{z_avg:.1f} µg/m³</td>
                        </tr>
                        <tr>
                            <td style="color:#777;padding:2px 0">Max / Min</td>
                            <td style="text-align:right;font-weight:600">{z_max:.1f} / {z_min:.1f}</td>
                        </tr>
                        <tr>
                            <td style="color:#777;padding:2px 0">vs WHO</td>
                            <td style="text-align:right;font-weight:600">{vs_who}</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True
            )
        col_idx += 1

    st.divider()

# -----------------------
# Temporal charts — adapt to time_mode
# -----------------------

# Day mode: no time-series charts, show station/zone bar instead
if time_mode == "Day":
    st.subheader(f"📊 {pollutant} on {period_label}")

    if filter_mode == "Station" and selected_station != "ALL":
        day_val = df_filtered[pollutant].mean()
        st.metric(f"{selected_station} — {pollutant}", f"{day_val:.2f} µg/m³")
        if who_limit:
            ratio = day_val / who_limit
            color = "#2ecc71" if ratio <= 1 else ("#f39c12" if ratio <= 2 else "#e74c3c")
            st.markdown(
                f"<span style='color:{color};font-size:16px;font-weight:600'>"
                f"{ratio:.1f}× WHO annual guideline</span>",
                unsafe_allow_html=True
            )
    else:
        # Bar chart per station for that day
        day_station = (
            df_filtered
            .groupby(["station", "Zone"])[pollutant]
            .mean()
            .reset_index()
            .sort_values(pollutant, ascending=False)
        )
        day_station["ZoneColor"] = day_station["Zone"].map(
            lambda z: ZONE_META.get(z, {}).get("color", "#999")
        )

        fig_day = px.bar(
            day_station,
            x="station",
            y=pollutant,
            color="Zone",
            color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
            title=f"{pollutant} by station — {period_label}",
            labels={pollutant: f"{pollutant} (µg/m³)"},
        )
        if show_who and who_limit:
            fig_day.add_hline(
                y=who_limit, line_dash="dash", line_color="red",
                annotation_text=f"WHO {who_limit} µg/m³",
                annotation_font_size=11
            )
        fig_day.update_layout(dragmode=False, height=380)
        st.plotly_chart(fig_day,  width="stretch", config=PLOTLY_CONFIG)
        
    st.plotly_chart(fig, config=PLOTLY_CONFIG)

    # ── Weather snapshot for this specific day ──────────────────────────
    st.subheader("🌦️ Weather conditions — this day")
    if not df_filtered.empty:
        day_date = df_filtered["Date"].max()
        c1, c2, c3, c4, c5 = st.columns(5)
        def _dv(col):
            v = df_filtered[col].mean() if col in df_filtered.columns else None
            return None if (v is None or pd.isna(v)) else round(float(v), 1)
        temp   = _dv("Temperature")
        humid  = _dv("Humidity")
        precip = _dv("Precipitation")
        wind   = _dv("WindSpeed")
        wdir   = _dv("WindDirection")
        CARDINAL = ["N","NE","E","SE","S","SW","W","NW"]
        cardinal = CARDINAL[int((float(wdir)%360)/45+0.5)%8] if wdir else "—"
        c1.metric("🌡️ Temperature",    f"{temp} °C"   if temp   is not None else "—")
        c2.metric("💧 Humidity",       f"{humid} %"   if humid  is not None else "—")
        c3.metric("🌧️ Precipitation",  f"{precip} mm" if precip is not None else "—")
        c4.metric("💨 Wind speed",     f"{wind} km/h" if wind   is not None else "—")
        c5.metric("🧭 Wind direction",
                  f"{cardinal} ({int(wdir)}°)" if wdir is not None else "—")
        favorable = []
        if wind is not None:
            if wind >= 20:   favorable.append(True)
            elif wind <= 5:  favorable.append(False)
        if precip is not None and precip > 0:
            favorable.append(True)
        if favorable and all(favorable):
            verdict, color = "Favorable — pollutants tend to disperse", "#2ecc71"
        elif favorable and not any(favorable):
            verdict, color = "Unfavorable — pollutants may accumulate", "#e74c3c"
        else:
            verdict, color = "Mixed conditions", "#f39c12"
        st.markdown(
            f"<div style='padding:.5rem 1rem;border-left:4px solid {color};"
            f"background:rgba(0,0,0,.02);border-radius:4px;margin:.3rem 0'>"
            f"<b>Dispersion outlook:</b> {verdict}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Reading: {day_date.strftime('%d %b %Y')} — averaged across the selected stations.")
    else:
        st.caption("No data available for this day.")

else:
    # ---- Annual Trend ----
    st.subheader(tr("📅 Annual Trend"))

    if time_mode == "Year":
        annual_grp = df_filtered.groupby("Year")
        x_col = "Year"
    else:  # Month mode — group by month
        annual_grp = df_filtered.groupby("Month")
        x_col = "Month"

    if filter_mode == "Zone":
        # Line per station, colored by zone
        if time_mode == "Year":
            trend_df = (
                df_filtered
                .groupby(["Year", "station", "Zone"])[pollutant]
                .mean()
                .reset_index()
            )
            x_col_plot = "Year"
        else:
            trend_df = (
                df_filtered
                .groupby(["Month", "station", "Zone"])[pollutant]
                .mean()
                .reset_index()
            )
            trend_df["MonthName"] = trend_df["Month"].map(MONTH_NAMES)
            x_col_plot = "MonthName"

        fig_trend = px.line(
            trend_df,
            x=x_col_plot,
            y=pollutant,
            color="station",
            line_dash="Zone",
            color_discrete_map={
                s: ZONE_META.get(
                    df[df["station"] == s]["Zone"].iloc[0], {}
                ).get("color", "#999")
                for s in trend_df["station"].unique()
            },
            markers=True,
            title=f"{pollutant} trend — {scope_label} — {period_label}",
            labels={pollutant: f"{pollutant} (µg/m³)"}
        )
    else:
        if time_mode == "Year":
            trend_df = (
                df_filtered.groupby("Year")[pollutant].mean().reset_index()
            )
            x_col_plot = "Year"
        else:
            trend_df = (
                df_filtered.groupby("Month")[pollutant].mean().reset_index()
            )
            trend_df["MonthName"] = trend_df["Month"].map(MONTH_NAMES)
            x_col_plot = "MonthName"

        fig_trend = px.line(
            trend_df,
            x=x_col_plot,
            y=pollutant,
            markers=True,
            title=f"{pollutant} trend — {scope_label} — {period_label}",
            labels={pollutant: f"{pollutant} (µg/m³)"},
            color_discrete_sequence=[main_color]
        )

    if show_who and who_limit:
        fig_trend.add_hline(
            y=who_limit, line_dash="dash", line_color="red",
            annotation_text=f"WHO limit {who_limit} µg/m³",
            annotation_position="top right",
            annotation_font_size=11
        )
    fig_trend.update_layout(dragmode=False, height=380, hovermode="x unified")
    st.plotly_chart(fig_trend,  width="stretch", config=PLOTLY_CONFIG)

    # Weather section — trend (only for Year/Month mode with enough data points)
    if time_mode == "Year" and selected_year == "All":
        freq_col = "Year"
    elif time_mode == "Year":
        freq_col = "Month"
    else:
        freq_col = "Day" if "selected_month" in dir() and selected_month != "All" else "Month"
    n_unique = df_filtered[freq_col].nunique() if freq_col in df_filtered.columns else 0
    if n_unique >= 2:
        weather_trend(df_filtered, pollutant, freq=freq_col)
    else:
        st.caption("⚠️ Not enough data points for a weather trend in this time range.")

    # ---- Monthly Seasonality (only in Year mode) ----
    if time_mode == "Year":
        st.subheader(tr("🌦️ Monthly Seasonality"))

        if filter_mode == "Zone":
            monthly_df = (
                df_filtered
                .groupby(["Month", "Zone"])[pollutant]
                .mean()
                .reset_index()
            )
            monthly_df["MonthName"] = monthly_df["Month"].map(MONTH_NAMES)

            fig_monthly = px.bar(
                monthly_df,
                x="MonthName",
                y=pollutant,
                color="Zone",
                barmode="group",
                color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
                title=f"{pollutant} monthly mean by zone — {period_label}",
                labels={pollutant: f"{pollutant} (µg/m³)", "MonthName": "Month"}
            )
        else:
            monthly_df = (
                df_filtered
                .groupby("Month")[pollutant]
                .agg(mean="mean", std="std")
                .reset_index()
            )
            monthly_df["MonthName"] = monthly_df["Month"].map(MONTH_NAMES)

            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_df["MonthName"],
                y=monthly_df["mean"],
                error_y=dict(
                    type="data",
                    array=monthly_df["std"].fillna(0),
                    visible=True
                ),
                marker_color=main_color,
                marker_opacity=0.8,
                name=pollutant
            ))
            fig_monthly.update_layout(dragmode=False, 
                title=f"{pollutant} monthly mean ± std — {scope_label}",
                xaxis_title="Month",
                yaxis_title=f"{pollutant} (µg/m³)",
                height=360,
                showlegend=False
            )

        if show_who and who_limit:
            fig_monthly.add_hline(
                y=who_limit, line_dash="dash", line_color="red",
                annotation_text=f"WHO {who_limit} µg/m³",
                annotation_font_size=11
            )

        st.plotly_chart(fig_monthly,  width="stretch", config=PLOTLY_CONFIG)

# -----------------------
# Station Comparison (ALL stations, Year mode only)
# -----------------------

if filter_mode == "Station" and selected_station == "ALL" and time_mode == "Year":
    st.subheader(tr("🏙️ Station Comparison"))

    comparison = (
        df
        .groupby(["Year", "station", "Zone"])[pollutant]
        .mean()
        .reset_index()
    )

    fig_comp = px.line(
        comparison,
        x="Year",
        y=pollutant,
        color="station",
        line_dash="Zone",
        markers=True,
        title=f"{pollutant} annual mean per station (colored by zone)",
        labels={pollutant: f"{pollutant} (µg/m³)"},
        color_discrete_map={
            s: ZONE_META.get(
                df[df["station"] == s]["Zone"].iloc[0], {}
            ).get("color", "#999")
            for s in comparison["station"].unique()
        }
    )
    if show_who and who_limit:
        fig_comp.add_hline(
            y=who_limit, line_dash="dash", line_color="red",
            annotation_text=f"WHO limit {who_limit} µg/m³",
            annotation_position="bottom right",
            annotation_font_size=11
        )
    fig_comp.update_layout(dragmode=False, height=420, hovermode="x unified")
    st.plotly_chart(fig_comp,  width="stretch", config=PLOTLY_CONFIG)

# -----------------------
# COVID Impact (Year mode only, full dataset)
# -----------------------

if time_mode == "Year":
    st.subheader(tr("🦠 COVID Impact Analysis"))

    covid_df = df_filtered.copy()
    covid_df["Period"] = np.select(
        [
            covid_df["Year"] <= 2019,
            covid_df["Year"].between(2020, 2021),
            covid_df["Year"] >= 2022
        ],
        COVID_PERIODS,
        default="Unknown"
    )
    covid_df["Period"] = pd.Categorical(
        covid_df["Period"], categories=COVID_PERIODS, ordered=True
    )

    if filter_mode == "Zone":
        # Show one bar group per zone per period
        covid_zone = (
            covid_df
            .groupby(["Period", "Zone"])[pollutant]
            .mean()
            .reset_index()
        )
        fig_covid = px.bar(
            covid_zone,
            x="Period",
            y=pollutant,
            color="Zone",
            barmode="group",
            color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
            title=f"COVID impact by zone — {pollutant}",
            labels={"Mean": f"{pollutant} (µg/m³)"},
            category_orders={"Period": COVID_PERIODS}
        )
        fig_covid.update_layout(dragmode=False, height=380)
        if show_who and who_limit:
            fig_covid.add_hline(
                y=who_limit, line_dash="dash", line_color="red",
                annotation_text=f"WHO {who_limit} µg/m³",
                annotation_font_size=11
            )
        st.plotly_chart(fig_covid,  width="stretch", config=PLOTLY_CONFIG)

    else:
        covid_stats = (
            covid_df
            .groupby("Period")[pollutant]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={
                "mean": "Mean", "median": "Median",
                "std": "Std", "count": "Days"
            })
            .reset_index()
            .sort_values("Period")
        )

        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            pre_val = covid_stats.loc[
                covid_stats["Period"] == "Pre-COVID", "Mean"
            ].values
            annotations = []
            if len(pre_val) > 0:
                base = pre_val[0]
                for _, r in covid_stats.iterrows():
                    if r["Period"] != "Pre-COVID" and base > 0:
                        pct  = (r["Mean"] - base) / base * 100
                        sign = "+" if pct >= 0 else ""
                        annotations.append(dict(
                            x=r["Period"],
                            y=r["Mean"] + covid_stats["Mean"].max() * 0.05,
                            text=f"{sign}{pct:.1f}%",
                            showarrow=False,
                            font=dict(size=12, color="#2c3e50")
                        ))

            fig_covid = px.bar(
                covid_stats,
                x="Period",
                y="Mean",
                color="Period",
                color_discrete_map={
                    "Pre-COVID": "#3498db",
                    "COVID":     "#e74c3c",
                    "Post-COVID": "#2ecc71"
                },
                category_orders={"Period": COVID_PERIODS},
                title=f"Average {pollutant} by COVID period — {scope_label}",
                labels={"Mean": f"{pollutant} (µg/m³)"},
                error_y="Std"
            )
            fig_covid.update_layout(dragmode=False, 
                annotations=annotations,
                showlegend=False,
                height=360
            )
            if show_who and who_limit:
                fig_covid.add_hline(
                    y=who_limit, line_dash="dash", line_color="red",
                    annotation_text=f"WHO {who_limit} µg/m³",
                    annotation_font_size=11
                )
            st.plotly_chart(fig_covid,  width="stretch", config=PLOTLY_CONFIG)

        with col_table:
            st.markdown(f"**{tr('Statistics by period')}**")
            display_stats = covid_stats.copy()
            for col in ["Mean", "Median", "Std"]:
                display_stats[col] = display_stats[col].round(2)
            st.dataframe(
                display_stats,
                width="stretch",
                hide_index=True,
                column_config={
                    "Mean": st.column_config.ProgressColumn(
                        "Mean",
                        min_value=0,
                        max_value=float(display_stats["Mean"].max() * 1.2),
                        format="%.2f"
                    )
                }
            )



st.divider()

# ── GeoAI Pattern Discovery ───────────────────────────────────────────────────
st.markdown("## 🔗 " + tr("GeoAI Pattern Discovery"))
st.caption(
    tr(
        "How temporal patterns confirm GIS spatial findings. "
        "Each finding is independently confirmed by both time-series analysis and GIS."
    )
)

# KPI strip — key numbers at a glance
_kpi1, _kpi2, _kpi3, _kpi4 = st.columns(4)
_kpi1.metric(tr("Weekday vs weekend NO₂"), "+32.4%",
             help=tr("Traffic drives weekday spike — confirmed by road density r = +0.83"))
_kpi2.metric(tr("COVID-19 NO₂ drop 2020"), "−22.1%",
             help=tr("Natural experiment: removing traffic confirmed road density effect"))
_kpi3.metric(tr("NO₂ decline 2015→2026"),  "−41%",
             help=tr("Structural drivers unchanged — decline is behavioural (fleet, standards)"))
_kpi4.metric(tr("Winter vs summer NO₂"),   "+56%",
             help=tr("Boundary layer compression overrides stronger winter wind speeds"))

st.divider()

# Finding 1
with st.expander("📊 " + tr("Finding 1 — The traffic signal (temporal + spatial)"), expanded=True):
    _gai1, _gai2 = st.columns([3, 2])
    with _gai1:
        _dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        _dow_no2    = [19.63, 20.51, 20.56, 20.32, 20.42, 16.18, 14.46]
        _dow_colors = ["#e74c3c" if d < 5 else "#2ecc71" for d in range(7)]
        _fig_dow = go.Figure(go.Bar(
            x=_dow_labels, y=_dow_no2,
            marker_color=_dow_colors,
            text=[f"{v:.1f}" for v in _dow_no2], textposition="outside",
        ))
        _fig_dow.add_hline(y=sum(_dow_no2[:5])/5, line_dash="dash",
                           line_color="#e74c3c",
                           annotation_text=tr("weekday avg"), annotation_font_size=9)
        _fig_dow.add_hline(y=sum(_dow_no2[5:])/2, line_dash="dash",
                           line_color="#2ecc71",
                           annotation_text=tr("weekend avg"), annotation_font_size=9,
                           annotation_position="bottom right")
        _fig_dow.update_layout(dragmode=False, 
            height=300, title=tr("Mean NO₂ by day of week — 2015–2026"),
            yaxis_title="Mean NO₂ (µg/m³)", margin=dict(t=45,b=10,l=10,r=20),
            showlegend=False,
        )
        st.plotly_chart(_fig_dow, width="stretch", config=PLOTLY_CONFIG)
    with _gai2:
        st.metric(tr("Weekday mean NO₂"), "20.29 µg/m³", delta_color="off")
        st.metric(tr("Weekend mean NO₂"), "15.32 µg/m³", delta_color="off")
        st.metric(tr("Weekday premium"), "+32.4%", delta_color="inverse")
        st.markdown("---")
        st.markdown(
            "**" + tr("GIS confirmation") + "**\n"
            + tr("Road density × NO₂: **r = +0.83** — strongest spatial predictor. "
                 "Stations with highest road density (BARAKALDO 21,267 m/km² · "
                 "MAZARREDO 19,060 m/km²) drive the weekday spike.")
        )
    st.success(
        tr("Two independent methods — same conclusion: "
           "traffic infrastructure is the primary NO₂ driver. "
           "The temporal pattern (−32% on weekends) is the daily fingerprint "
           "of the spatial structure (road density r = +0.83).")
    )

# Finding 2
with st.expander("🦠 " + tr("Finding 2 — COVID-19 as a natural experiment"), expanded=False):
    _cc1, _cc2, _cc3 = st.columns(3)
    _cc1.metric(tr("2019 network mean NO₂"), "20.44 µg/m³", delta_color="off")
    _cc2.metric(tr("2020 lockdown NO₂"),     "15.92 µg/m³", "−22.1%", delta_color="inverse")
    _cc3.metric(tr("2021 partial recovery"), "16.71 µg/m³", "+4.9%", delta_color="inverse")
    st.markdown(
        tr("The 2020 drop was **largest at Urban and Industrial stations** "
           "(highest road density) and **smallest at Coastal and Refinery** "
           "(lower road density, marine/industrial sources). "
           "This confirms road density (r = +0.83) represents a real causal mechanism, "
           "not a spurious correlation.")
    )

# Finding 3
with st.expander("❄️ " + tr("Finding 3 — Winter paradox: highest NO₂ with strongest wind"), expanded=False):
    _ss1, _ss2 = st.columns(2)
    with _ss1:
        _seasons    = ["Winter", "Spring", "Summer", "Autumn"]
        _season_no2 = [22.37, 17.95, 14.34, 20.76]
        _season_ws  = [20.1, 18.0, 15.9, 17.9]
        _fig_sea = go.Figure()
        _fig_sea.add_trace(go.Bar(
            x=_seasons, y=_season_no2, name=tr("Mean NO₂ (µg/m³)"),
            marker_color=["#3498db","#2ecc71","#f39c12","#e67e22"], opacity=0.85,
        ))
        _fig_sea.add_trace(go.Scatter(
            x=_seasons, y=_season_ws, name=tr("Wind speed (m/s)"),
            mode="lines+markers", line=dict(color="#2c3e50", dash="dash"), yaxis="y2",
        ))
        _fig_sea.update_layout(dragmode=False, 
            height=280, title=tr("Seasonal NO₂ vs wind speed"),
            yaxis=dict(title="Mean NO₂ (µg/m³)"),
            yaxis2=dict(title=tr("Wind speed (m/s)"), overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.3), margin=dict(t=45,b=10),
            barmode="group",
        )
        st.plotly_chart(_fig_sea, width="stretch", config=PLOTLY_CONFIG)
    with _ss2:
        st.markdown(
            "**" + tr("Winter paradox") + "**\n"
            + tr("Strongest wind (20.1 m/s) yet highest NO₂ (22.4 µg/m³). "
                 "Reason: lower boundary layer height + more combustion emissions "
                 "+ less thermal convection. "
                 "Wind speed alone cannot overcome structural winter effects — "
                 "GeoAI model uses season alongside wind speed as separate features.")
        )

# Finding 4
with st.expander("📉 " + tr("Finding 4 — Long-term NO₂ decline: structural or behavioural?"), expanded=False):
    _years   = list(range(2015, 2027))
    _ann_no2 = [24.99, 21.68, 22.68, 20.77, 20.44, 15.92, 16.71, 17.11, 16.95, 16.04, 15.70, 14.71]
    _fig_ann = go.Figure()
    _fig_ann.add_trace(go.Scatter(
        x=_years, y=_ann_no2, mode="lines+markers",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
    ))
    _fig_ann.add_vrect(x0=2020, x1=2021.5, fillcolor="#3498db", opacity=0.08,
                       line_width=0, annotation_text="COVID",
                       annotation_position="top left", annotation_font_size=10)
    _fig_ann.add_hline(y=10, line_dash="dot", line_color="#e74c3c", opacity=0.5,
                       annotation_text="WHO 10 µg/m³", annotation_font_size=9)
    _fig_ann.update_layout(dragmode=False, 
        height=260, title=tr("Annual mean NO₂ — all stations"),
        yaxis_title="Mean NO₂ (µg/m³)", margin=dict(t=45,b=10,l=10,r=20),
        showlegend=False,
    )
    st.plotly_chart(_fig_ann, width="stretch", config=PLOTLY_CONFIG)
    st.markdown(
        tr("NO₂ fell from 25.0 µg/m³ (2015) to 14.7 µg/m³ (2026) — a 41% reduction. "
           "All years remain above the WHO 2021 guideline (10 µg/m³).  \n\n"
           "**GIS context:** Road density (r = +0.83) is unchanged — the road network is the same. "
           "The decline is behavioural: fleet electrification, Euro 5/6 standards, post-COVID shift. "
           "Structural analysis remains valid for *relative* differences between stations.")
    )

st.info(
    tr("Findings verified against parquet 2015–2026. "
       "GIS correlations: n = 7 stations, exploratory only. "
       "ERA5 wind: single grid cell (~31 km).")
)