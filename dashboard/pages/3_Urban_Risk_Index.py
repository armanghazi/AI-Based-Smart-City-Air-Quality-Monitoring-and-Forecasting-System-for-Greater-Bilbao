from config import (
    load_data,
    WHO_ANNUAL, WHO_SO2_DAILY, CORE_POLLUTANTS,
    POLLUTANT_COLOR, MONTH_NAMES, RISK_COLORS, RISK_ORDER,
    ZONE_MAP, ZONE_META, get_zone,EU_ANNUAL, 
    classify_core_risk, risk_color, short_term_flag,
    who_ratio_label, who_delta,get_fav_station, center_tables
)
 
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Urban Air Quality — Greater Bilbao",
    layout="wide",
    initial_sidebar_state="expanded"
)
center_tables()

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #e74c3c;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 4px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df           = load_data()
all_stations = sorted(df["station"].unique().tolist())
all_years    = sorted(df["Year"].dropna().unique().tolist())

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🌍 Urban Air Quality")
    st.markdown("Greater Bilbao · WHO 2021 Guidelines")
    st.divider()

    st.markdown("### Filters")

    # ── Station / Zone filter ──────────────────────
    filter_mode = st.radio("Filter by", ["Station", "Zone"], horizontal=True)

    if filter_mode == "Station":
        selected_stations = st.multiselect(
            "Stations (leave empty = all)",
            options=all_stations,
            default=[]
        )
        selected_zone = None
    else:
        selected_zone = st.selectbox(
            "Zone",
            list(ZONE_META.keys()),
            format_func=lambda z: f"{ZONE_META[z]['icon']} {z}"
        )
        selected_stations = []

    st.divider()

    # ── Time granularity ──────────────────────────
    st.markdown("### Time Range")
    time_mode = st.radio("Granularity", ["Year", "Month", "Day"], horizontal=True)

    if time_mode == "Year":
        year_options     = ["All years"] + [str(y) for y in all_years]
        selected_year_str = st.selectbox("Year", year_options, index=0)
        selected_year    = None if selected_year_str == "All years" else int(selected_year_str)
        selected_month   = None
        selected_day     = None

    elif time_mode == "Month":
        selected_year    = st.selectbox("Year", all_years, index=len(all_years) - 1)
        month_avail      = sorted(df[df["Year"] == selected_year]["Month"].dropna().unique().tolist())
        month_opts       = ["All"] + [MONTH_NAMES[m] for m in month_avail]
        sel_month_label  = st.selectbox("Month", month_opts, index=0)
        selected_month   = (
            {v: k for k, v in MONTH_NAMES.items()}[sel_month_label]
            if sel_month_label != "All" else None
        )
        selected_day     = None
        selected_year_str = str(selected_year)

    else:  # Day
        selected_year    = st.selectbox("Year", all_years, index=len(all_years) - 1)
        month_avail      = sorted(df[df["Year"] == selected_year]["Month"].dropna().unique().tolist())
        sel_month_label  = st.selectbox("Month", [MONTH_NAMES[m] for m in month_avail])
        selected_month   = {v: k for k, v in MONTH_NAMES.items()}[sel_month_label]
        day_avail        = sorted(
            df[(df["Year"] == selected_year) & (df["Month"] == selected_month)]["Day"]
            .dropna().unique().tolist()
        )
        selected_day     = st.selectbox(
            "Day", day_avail,
            format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y")
        )
        selected_year_str = pd.Timestamp(selected_day).strftime("%d %B %Y")

    st.divider()

    # ── Display options ───────────────────────────
    st.markdown("### Display")
    map_mode       = st.radio("Map layer", ["Risk score", "Heatmap — PM2.5", "Heatmap — NO2", "Heatmap — PM10"])
    show_who_lines = st.toggle("Show WHO guideline lines", value=True)

    st.divider()

    # ── Zone legend ───────────────────────────────
    st.markdown("### 🗺️ Environmental Zones")
    for z, meta in ZONE_META.items():
        stations_in_zone = df[df["Zone"] == z]["station"].unique().tolist()
        short = [s.split("_")[0] for s in stations_in_zone]
        st.markdown(
            f"{meta['icon']} **{z}**  \n"
            f"<span style='font-size:11px;color:#777'>{', '.join(short)}</span>",
            unsafe_allow_html=True
        )

    st.divider()
    st.caption("Data: 7 monitoring stations, 2015–2026 (~27k daily records)")

# --------------------------------------------------
# APPLY FILTERS — station / zone
# --------------------------------------------------

base_df = df.copy()

if filter_mode == "Zone" and selected_zone:
    base_df     = base_df[base_df["Zone"] == selected_zone]
    scope_label = f"{ZONE_META[selected_zone]['icon']} {selected_zone}"
elif filter_mode == "Station" and selected_stations:
    base_df     = base_df[base_df["station"].isin(selected_stations)]
    scope_label = ", ".join(selected_stations)
else:
    scope_label = "All Stations"

# ── Apply time filter ─────────────────────────────
if time_mode == "Year":
    if selected_year:
        base_df      = base_df[base_df["Year"] == selected_year]
    period_label = selected_year_str

elif time_mode == "Month":
    base_df = base_df[base_df["Year"] == selected_year]
    if selected_month:
        base_df      = base_df[base_df["Month"] == selected_month]
        period_label = f"{MONTH_NAMES[selected_month]} {selected_year}"
    else:
        period_label = f"All months of {selected_year}"

else:  # Day
    base_df      = base_df[base_df["Day"] == selected_day]
    period_label = selected_year_str

if base_df.empty:
    st.warning("No data for selected filters.")
    st.stop()

# --------------------------------------------------
# DAILY AGGREGATION
# --------------------------------------------------

grp_cols = ["station", "Town", "Zone", "Latitude", "Longitude", "Date", "Year", "Month", "YearMonth"]
daily_df = (
    base_df
    .groupby(grp_cols, as_index=False)
    .agg({"PM2.5": "mean", "PM10": "mean", "NO2": "mean", "SO2": "mean"})
)

# --------------------------------------------------
# STATION-YEAR ANNUAL TABLE
# --------------------------------------------------

station_year = (
    daily_df
    .groupby(["station", "Town", "Zone", "Latitude", "Longitude", "Year"], as_index=False)
    .agg({"PM2.5": "mean", "PM10": "mean", "NO2": "mean", "SO2": "mean", "Date": "nunique"})
    .rename(columns={"Date": "ValidDays"})
)

# Coverage threshold — relax for Month/Day modes
if time_mode == "Year":
    if selected_year:
        total_days = daily_df["Date"].dt.normalize().nunique()
        min_valid  = max(30, int(total_days * 0.6))
    else:
        min_valid = 200
else:
    min_valid = 1   # Month / Day: no coverage filter

station_year = station_year[station_year["ValidDays"] >= min_valid].copy()

if station_year.empty:
    st.warning("Not enough data coverage for the selected period. Try 'All years'.")
    st.stop()

# WHO ratios & core risk
for p, col in [("PM2.5","PM25_ratio"), ("PM10","PM10_ratio"), ("NO2","NO2_ratio")]:
    station_year[col] = (station_year[p] / WHO_ANNUAL[p]).clip(upper=5)

station_year["CoreRiskScore"] = 100 * (
    station_year["PM25_ratio"] + station_year["PM10_ratio"] + station_year["NO2_ratio"]
) / 3
station_year["CoreRiskLevel"] = station_year["CoreRiskScore"].apply(classify_core_risk)

# SO2 exceedance
daily_df["SO2_Exceed"] = daily_df["SO2"] > WHO_SO2_DAILY
so2_stats = (
    daily_df
    .groupby(["station", "Town", "Zone", "Latitude", "Longitude", "Year"], as_index=False)
    .agg(
        SO2_HighDays       = ("SO2_Exceed", "sum"),
        SO2_ExceedanceRate = ("SO2_Exceed", "mean"),
        SO2_AnnualMean     = ("SO2", "mean")
    )
)
station_year = station_year.merge(
    so2_stats,
    on=["station", "Town", "Zone", "Latitude", "Longitude", "Year"],
    how="left"
)
station_year["SO2_HighDays"]        = station_year["SO2_HighDays"].fillna(0).astype(int)
station_year["SO2_ExceedanceRate"]  = station_year["SO2_ExceedanceRate"].fillna(0)
station_year["SO2_AnnualMean"]      = station_year["SO2_AnnualMean"].fillna(0)
station_year["SO2_PressureLevel"]   = station_year["SO2_ExceedanceRate"].apply(short_term_flag)



# Final station risk (average across years if "All")
if time_mode == "Year" and not selected_year:
    station_risk = (
        station_year
        .groupby(["station", "Town", "Zone", "Latitude", "Longitude"], as_index=False)
        .agg(
            ValidYears         = ("Year",              "nunique"),
            AvgValidDays       = ("ValidDays",          "mean"),
            **{p: (p, "mean") for p in CORE_POLLUTANTS},
            SO2_AnnualMean     = ("SO2_AnnualMean",     "mean"),
            CoreRiskScore      = ("CoreRiskScore",      "mean"),
            SO2_HighDays       = ("SO2_HighDays",       "mean"),
            SO2_ExceedanceRate = ("SO2_ExceedanceRate", "mean"),
            PM25_ratio         = ("PM25_ratio",         "mean"),
            PM10_ratio         = ("PM10_ratio",         "mean"),
            NO2_ratio          = ("NO2_ratio",          "mean"),
        )
    )
    station_risk["CoreRiskLevel"]    = station_risk["CoreRiskScore"].apply(classify_core_risk)
    station_risk["SO2_PressureLevel"]= station_risk["SO2_ExceedanceRate"].apply(short_term_flag)
else:
    station_risk                     = station_year.copy()
    station_risk["ValidYears"]       = 1
    station_risk["AvgValidDays"]     = station_risk["ValidDays"]

ranking = station_risk.sort_values("CoreRiskScore", ascending=False).reset_index(drop=True)

# --------------------------------------------------
# HEADER & KPIs
# --------------------------------------------------

st.title("🌍 Urban Air Quality — Greater Bilbao")
st.caption(
    f"Period: **{period_label}** · Scope: **{scope_label}** · "
    f"Stations: **{len(ranking)}** · WHO 2021 annual guidelines"
)

avg_risk     = ranking["CoreRiskScore"].mean()
worst        = ranking.iloc[0]
best         = ranking.sort_values("CoreRiskScore").iloc[0]
so2_stations = int((ranking["SO2_ExceedanceRate"] > 0).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean Core Risk Score", f"{avg_risk:.1f}", help="100 = 1× WHO limit")
c2.metric("Worst Station",  worst["station"], f"Score {worst['CoreRiskScore']:.1f}")
c3.metric("Best Station",   best["station"],  f"Score {best['CoreRiskScore']:.1f}")
c4.metric("Avg NO₂",        f"{ranking['NO2'].mean():.1f} µg/m³",
          f"WHO limit: {WHO_ANNUAL['NO2']} µg/m³")
c5.metric("SO₂ exceedance stations", f"{so2_stations} / {len(ranking)}")

# ── Zone summary banner ───────────────────────────
st.markdown("---")
st.subheader("🗺️ Environmental Zone Summary")

zones_present = [z for z in ZONE_META if z in ranking["Zone"].values]
zone_cols     = st.columns(len(zones_present) or 1)

for idx, zone_name in enumerate(zones_present):
    meta       = ZONE_META[zone_name]
    zone_data  = ranking[ranking["Zone"] == zone_name]
    z_avg      = zone_data["CoreRiskScore"].mean()
    z_no2      = zone_data["NO2"].mean()
    z_pm25     = zone_data["PM2.5"].mean()
    z_worst    = zone_data.iloc[0]["station"].split("_")[0] if not zone_data.empty else "—"
    n          = len(zone_data)

    with zone_cols[idx]:
        st.markdown(
            f"""
            <div style="
                border-left:5px solid {meta['border']};
                background:linear-gradient(135deg,{meta['color']}18,{meta['color']}06);
                border-radius:10px; padding:14px 16px; margin-bottom:8px;
            ">
                <div style="font-size:18px;margin-bottom:4px">
                    {meta['icon']} <strong>{zone_name}</strong>
                </div>
                <div style="color:#666;font-size:11px;margin-bottom:8px">{meta['description']}</div>
                <table style="width:100%;font-size:13px;border-collapse:collapse">
                    <tr><td style="color:#777;padding:2px 0">Stations</td>
                        <td style="text-align:right;font-weight:600">{n}</td></tr>
                    <tr><td style="color:#777;padding:2px 0">Avg Risk Score</td>
                        <td style="text-align:right;font-weight:600">{z_avg:.1f}</td></tr>
                    <tr><td style="color:#777;padding:2px 0">Avg NO₂</td>
                        <td style="text-align:right;font-weight:600">{z_no2:.1f} µg/m³</td></tr>
                    <tr><td style="color:#777;padding:2px 0">Avg PM2.5</td>
                        <td style="text-align:right;font-weight:600">{z_pm25:.1f} µg/m³</td></tr>
                    <tr><td style="color:#777;padding:2px 0">Highest station</td>
                        <td style="text-align:right;font-weight:600">{z_worst}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------

tab_map, tab_risk, tab_trend, tab_seasonal, tab_table = st.tabs([
    "🗺️ Risk Map",
    "📊 Risk Breakdown",
    "📈 Trends over Time",
    "🌡️ Seasonal Patterns",
    "🔢 Data Table"
])

# ==================== TAB 1: MAP ====================
with tab_map:
    col_map, col_legend = st.columns([3, 1])

    with col_map:
        center_lat = ranking["Latitude"].mean()
        center_lon = ranking["Longitude"].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles="CartoDB positron"
        )

        if "Heatmap" in map_mode:
            pollutant_key = map_mode.split("— ")[1]
            heat_data = [
                [row["Latitude"], row["Longitude"], row[pollutant_key]]
                for _, row in ranking.iterrows()
                if pd.notna(row[pollutant_key])
            ]
            HeatMap(
                heat_data, radius=40, blur=25,
                gradient={0.2:"#2ecc71", 0.5:"#f39c12", 0.8:"#e74c3c", 1.0:"#7b241c"}
            ).add_to(m)

        # FeatureGroup per zone
        for zone_name, meta in ZONE_META.items():
            zone_rows = ranking[ranking["Zone"] == zone_name]
            if zone_rows.empty:
                continue
            fg = folium.FeatureGroup(name=f"{meta['icon']} {zone_name}")

            for _, row in zone_rows.iterrows():
                score    = row["CoreRiskScore"]
                color    = risk_color(score)
                z_color  = meta["color"]
                radius   = 10 + (score / 50)

                popup_lines = [
                    f"<b style='font-size:14px'>{row['station']}</b>",
                    f"<i>{row['Town']}</i>",
                    f"<span style='background:{z_color};color:white;padding:1px 8px;"
                    f"border-radius:8px;font-size:11px'>{zone_name}</span>",
                    "<hr style='margin:4px 0'>",
                    f"<b>Core Risk Score:</b> {score:.1f} ({row['CoreRiskLevel']})",
                    "<hr style='margin:4px 0'>",
                    f"PM2.5: <b>{row['PM2.5']:.1f}</b> µg/m³ ({who_ratio_label(row['PM2.5'],'PM2.5')} WHO)",
                    f"PM10: <b>{row['PM10']:.1f}</b> µg/m³ ({who_ratio_label(row['PM10'],'PM10')} WHO)",
                    f"NO₂: <b>{row['NO2']:.1f}</b> µg/m³ ({who_ratio_label(row['NO2'],'NO2')} WHO)",
                    f"SO₂: <b>{row['SO2_AnnualMean']:.1f}</b> µg/m³",
                    "<hr style='margin:4px 0'>",
                    f"SO₂ exceedance: <b>{row['SO2_ExceedanceRate']*100:.1f}%</b> ({row['SO2_PressureLevel']})",
                ]

                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=radius,
                    popup=folium.Popup("<br>".join(popup_lines), max_width=270),
                    tooltip=f"<b>{row['station']}</b> · {zone_name} · Score {score:.0f}",
                    color=z_color,
                    weight=3,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.88
                ).add_to(fg)

                folium.Marker(
                    location=[row["Latitude"] + 0.003, row["Longitude"]],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size:10px;font-weight:600;'
                            f'color:#2c3e50;white-space:nowrap">'
                            f'{row["station"].split("_")[0]}</div>'
                        ),
                        icon_size=(120, 20),
                        icon_anchor=(60, 0)
                    )
                ).add_to(fg)

            fg.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=None, height=580, returned_objects=[],
                  key=f"map_{time_mode}_{period_label}_{scope_label}")

    with col_legend:
        st.markdown("#### Air Quality")
        for level, color in RISK_COLORS.items():
            n = int((ranking["CoreRiskLevel"] == level).sum())
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
                f'<div style="width:18px;height:18px;border-radius:50%;background:{color};flex-shrink:0"></div>'
                f'<span style="font-size:13px">{level}<br><b>{n} station{"s" if n!=1 else ""}</b></span></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### Zones (border)")
        for z, meta in ZONE_META.items():
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'<div style="width:18px;height:18px;border-radius:50%;border:3px solid {meta["color"]};flex-shrink:0"></div>'
                f'<span style="font-size:12px">{meta["icon"]} {z}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("**Score guide**")
        st.markdown("""
<div style='font-size:12px;line-height:1.8'>
• <b>&lt;100</b>: below WHO<br>
• <b>100–200</b>: 1–2× WHO<br>
• <b>&gt;200</b>: &gt;2× WHO<br><br>
Fill = air quality · Border = zone
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**WHO 2021 annual limits**")
        for p, v in WHO_ANNUAL.items():
            eu_v = EU_ANNUAL.get(p, "—")
            st.markdown(
                f"<div style='font-size:12px'>"
                f"{p}: <b>{v} µg/m³</b> WHO · <span style='color:#888'>{eu_v} µg/m³ EU</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.caption("EU = Directive 2008/50/EC")


# ==================== TAB 2: RISK BREAKDOWN ====================
with tab_risk:
    c_left, c_right = st.columns(2)

    with c_left:
        fig_bar = px.bar(
            ranking.sort_values("CoreRiskScore"),
            x="CoreRiskScore",
            y="station",
            color="CoreRiskLevel",
            color_discrete_map=RISK_COLORS,
            category_orders={"CoreRiskLevel": RISK_ORDER},
            orientation="h",
            title="Annual Core Risk Score by Station",
            labels={"CoreRiskScore": "Core Risk Score", "station": ""},
            text="CoreRiskScore",
            pattern_shape="Zone",          # extra visual separation by zone
        )
        fig_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        if show_who_lines:
            fig_bar.add_vline(x=100, line_dash="dash", line_color="#f39c12",
                              annotation_text="1× WHO", annotation_position="top")
            fig_bar.add_vline(x=200, line_dash="dash", line_color="#e74c3c",
                              annotation_text="2× WHO", annotation_position="top")
        fig_bar.update_layout(showlegend=True, height=400,
                              margin=dict(l=10, r=60, t=40, b=10))
        st.plotly_chart(fig_bar,  width="stretch")

    with c_right:
        cats = ["PM2.5 ratio", "PM10 ratio", "NO₂ ratio"]
        fig_radar = go.Figure()

        for _, row in ranking.iterrows():
            zone_color = ZONE_META.get(row.get("Zone", ""), {}).get("color", "#999")
            vals = [row["PM25_ratio"], row["PM10_ratio"], row["NO2_ratio"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                name=f"{row['station'].split('_')[0]} ({row.get('Zone','')[:3]}…)",
                opacity=0.5,
                line=dict(width=1.5, color=zone_color)
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title="Pollutant Ratios vs WHO Limit<br><sup>Line color = zone</sup>",
            showlegend=True,
            height=400,
            margin=dict(l=30, r=30, t=60, b=30)
        )
        st.plotly_chart(fig_radar,  width="stretch")

    # Grouped bar — by zone
    st.markdown("#### Zone-level Pollutant Comparison")

    zone_poll_rows = []
    for zone_name in ZONE_META:
        zone_data = ranking[ranking["Zone"] == zone_name]
        if zone_data.empty:
            continue
        for p in CORE_POLLUTANTS:
            zone_poll_rows.append({
                "Zone":          zone_name,
                "Pollutant":     p,
                "Concentration": zone_data[p].mean(),
                "WHO_Limit":     WHO_ANNUAL[p]
            })
    zpdf = pd.DataFrame(zone_poll_rows)

    if not zpdf.empty:
        fig_zone_poll = px.bar(
            zpdf,
            x="Zone",
            y="Concentration",
            color="Pollutant",
            barmode="group",
            color_discrete_map=POLLUTANT_COLOR,
            labels={"Concentration": "µg/m³"},
            title="Zone-average pollutant concentrations"
        )
        if show_who_lines:
            for p, lim in WHO_ANNUAL.items():
                fig_zone_poll.add_hline(
                    y=lim, line_dash="dot", line_color=POLLUTANT_COLOR[p],
                    annotation_text=f"WHO {p}", annotation_font_size=10
                )
        fig_zone_poll.update_layout(height=320)
        st.plotly_chart(fig_zone_poll,  width="stretch")

    # Individual pollutant means per station
    st.markdown("#### Individual Pollutant Annual Means vs WHO Guideline")

    pollutant_rows = []
    for _, row in ranking.iterrows():
        for p in CORE_POLLUTANTS:
            pollutant_rows.append({
                "station":       row["station"].split("_")[0],
                "Zone":          row.get("Zone", ""),
                "Pollutant":     p,
                "Concentration": row[p],
            })
    pdf = pd.DataFrame(pollutant_rows)

    fig_grouped = px.bar(
        pdf,
        x="station",
        y="Concentration",
        color="Pollutant",
        barmode="group",
        color_discrete_map=POLLUTANT_COLOR,
        facet_col="Zone",
        facet_col_wrap=3,
        labels={"Concentration": "µg/m³", "station": ""},
        title="Annual mean concentrations by station and pollutant (grouped by zone)"
    )
    if show_who_lines:
        for p, lim in WHO_ANNUAL.items():
            fig_grouped.add_hline(
                y=lim, line_dash="dot", line_color=POLLUTANT_COLOR[p],
                annotation_text=f"WHO {p}", annotation_font_size=9
            )
    fig_grouped.update_layout(height=360)
    st.plotly_chart(fig_grouped,  width="stretch")

    # SO2
    st.markdown("#### SO₂ Short-Term Pressure")
    fig_so2 = px.bar(
        ranking.sort_values("SO2_ExceedanceRate", ascending=False),
        x="station",
        y="SO2_ExceedanceRate",
        color="Zone",
        color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
        pattern_shape="SO2_PressureLevel",
        labels={"SO2_ExceedanceRate": "Exceedance rate (fraction)", "station": ""},
        title="Fraction of days exceeding WHO SO₂ 24h guideline (40 µg/m³) — colored by zone",
        text_auto=".1%"
    )
    fig_so2.update_layout(height=320)
    st.plotly_chart(fig_so2,  width="stretch")

    # --------------------------------------------------
# WHO vs EU comparison table
# --------------------------------------------------
st.markdown("#### WHO 2021 vs EU Directive 2008/50/EC")
st.caption(
    "WHO guidelines are stricter than legally binding EU limits. "
    "Exceeding WHO does NOT mean a legal violation."
)

# Build comparison per station
eu_rows = []
for _, row in ranking.iterrows():
    for p in CORE_POLLUTANTS:
        val      = row[p]
        who_lim  = WHO_ANNUAL[p]
        eu_lim   = EU_ANNUAL[p]
        eu_rows.append({
            "Station":    row["station"].split("_")[0],
            "Zone":       row.get("Zone", ""),
            "Pollutant":  p,
            "Mean (µg/m³)": round(val, 1),
            "vs WHO":     f"{val/who_lim:.1f}×",
            "vs EU":      f"{val/eu_lim:.1f}×",
            "WHO status": "⚠️ Above" if val > who_lim else "✅ Below",
            "EU status":  "⚠️ Above" if val > eu_lim  else "✅ Below",
        })

eu_df = pd.DataFrame(eu_rows)
st.dataframe(eu_df, hide_index=True, use_container_width=True)


# ==================== TAB 3: TRENDS ====================
with tab_trend:

    # In Day mode: no time-series trend makes sense — show snapshot bar
    if time_mode == "Day":
        st.markdown(f"### {period_label} — station snapshot")

        snap = (
            daily_df
            .groupby(["station", "Zone"])[CORE_POLLUTANTS]
            .mean()
            .reset_index()
        )
        poll_day = st.selectbox("Pollutant", CORE_POLLUTANTS + ["SO2"],
                                key="trend_day_poll")
        fig_snap = px.bar(
            snap.sort_values(poll_day, ascending=False),
            x="station", y=poll_day,
            color="Zone",
            color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
            title=f"{poll_day} on {period_label}",
            labels={poll_day: f"{poll_day} (µg/m³)"}
        )
        if show_who_lines and poll_day in WHO_ANNUAL:
            fig_snap.add_hline(
                y=WHO_ANNUAL[poll_day], line_dash="dash", line_color="red",
                annotation_text=f"WHO {WHO_ANNUAL[poll_day]} µg/m³"
            )
        fig_snap.update_layout(height=380)
        st.plotly_chart(fig_snap,  width="stretch")

    else:
        st.markdown("### Trends — stations colored by zone")

        # Use full df for trend (all years), restricted to station/zone scope
        trend_base = df.copy()
        if filter_mode == "Zone" and selected_zone:
            trend_base = trend_base[trend_base["Zone"] == selected_zone]
        elif filter_mode == "Station" and selected_stations:
            trend_base = trend_base[trend_base["station"].isin(selected_stations)]

        pollutant_choice = st.selectbox(
            "Pollutant", CORE_POLLUTANTS + ["SO2"], index=0, key="trend_poll"
        )

        if time_mode == "Year":
            yearly = (
                trend_base
                .groupby(["Year", "station", "Zone"], as_index=False)
                .agg({pollutant_choice: "mean"})
            )
            fig_trend = px.line(
                yearly, x="Year", y=pollutant_choice,
                color="station",
                line_dash="Zone",
                color_discrete_map={
                    s: ZONE_META.get(
                        trend_base[trend_base["station"] == s]["Zone"].iloc[0], {}
                    ).get("color", "#999")
                    for s in yearly["station"].unique()
                },
                markers=True,
                title=f"{pollutant_choice} annual mean — {scope_label}",
                labels={pollutant_choice: f"{pollutant_choice} µg/m³"}
            )

        else:  # Month mode
            monthly_t = (
                trend_base[trend_base["Year"] == selected_year]
                .groupby(["Month", "station", "Zone"], as_index=False)
                .agg({pollutant_choice: "mean"})
            )
            monthly_t["MonthName"] = monthly_t["Month"].map(MONTH_NAMES)
            fig_trend = px.line(
                monthly_t, x="MonthName", y=pollutant_choice,
                color="station",
                line_dash="Zone",
                color_discrete_map={
                    s: ZONE_META.get(
                        trend_base[trend_base["station"] == s]["Zone"].iloc[0], {}
                    ).get("color", "#999")
                    for s in monthly_t["station"].unique()
                },
                markers=True,
                title=f"{pollutant_choice} monthly mean — {scope_label} — {period_label}",
                labels={pollutant_choice: f"{pollutant_choice} µg/m³", "MonthName": "Month"}
            )

        if show_who_lines and pollutant_choice in WHO_ANNUAL:
            fig_trend.add_hline(
                y=WHO_ANNUAL[pollutant_choice], line_dash="dash", line_color="red",
                annotation_text=f"WHO limit {WHO_ANNUAL[pollutant_choice]} µg/m³",
                annotation_position="bottom right"
            )
        fig_trend.update_layout(height=420, hovermode="x unified")
        st.plotly_chart(fig_trend,  width="stretch")

        # Zone-level trend (Year mode only)
        if time_mode == "Year":
            st.markdown("### Zone-level trend (aggregated)")

            zone_yearly = (
                trend_base
                .groupby(["Year", "Zone"], as_index=False)
                .agg({pollutant_choice: "mean"})
            )
            fig_zone_trend = px.line(
                zone_yearly, x="Year", y=pollutant_choice,
                color="Zone",
                color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
                markers=True,
                title=f"{pollutant_choice} annual mean by zone",
                labels={pollutant_choice: f"{pollutant_choice} µg/m³"}
            )
            if show_who_lines and pollutant_choice in WHO_ANNUAL:
                fig_zone_trend.add_hline(
                    y=WHO_ANNUAL[pollutant_choice], line_dash="dash", line_color="red",
                    annotation_text=f"WHO {WHO_ANNUAL[pollutant_choice]} µg/m³"
                )
            fig_zone_trend.update_layout(height=360, hovermode="x unified")
            st.plotly_chart(fig_zone_trend,  width="stretch")

            # City-wide rolling area chart
            st.markdown("### City-wide monthly average (3-month rolling)")
            monthly_all = (
                trend_base
                .groupby("YearMonth", as_index=False)
                .agg({"PM2.5": "mean", "PM10": "mean", "NO2": "mean", "SO2": "mean"})
                .sort_values("YearMonth")
            )
            fig_area = go.Figure()
            for p in CORE_POLLUTANTS:
                fig_area.add_trace(go.Scatter(
                    x=monthly_all["YearMonth"],
                    y=monthly_all[p].rolling(3, center=True, min_periods=1).mean(),
                    name=p, fill="tozeroy",
                    line=dict(color=POLLUTANT_COLOR[p], width=1.5),
                    opacity=0.4
                ))
            if show_who_lines:
                for p, lim in WHO_ANNUAL.items():
                    fig_area.add_hline(y=lim, line_dash="dot",
                                       line_color=POLLUTANT_COLOR[p],
                                       annotation_text=f"WHO {p}",
                                       annotation_font_size=9, opacity=0.6)
            fig_area.update_layout(
                title="3-month rolling average — µg/m³",
                xaxis_title="Date", yaxis_title="µg/m³",
                height=380, hovermode="x unified",
                legend=dict(orientation="h", y=1.05)
            )
            st.plotly_chart(fig_area,  width="stretch")


# ==================== TAB 4: SEASONAL ====================
with tab_seasonal:
    st.markdown("### Monthly seasonality by zone")

    poll_s = st.selectbox("Pollutant", CORE_POLLUTANTS + ["SO2"],
                          index=2, key="seas_poll")

    # Zone-level seasonality
    monthly_zone = (
        daily_df
        .groupby(["Zone", "Month"], as_index=False)
        .agg({poll_s: "mean"})
    )
    monthly_zone["MonthName"] = monthly_zone["Month"].map(MONTH_NAMES)

    fig_zone_season = px.line(
        monthly_zone, x="Month", y=poll_s,
        color="Zone",
        color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
        markers=True,
        title=f"{poll_s} monthly mean by zone",
        labels={poll_s: f"{poll_s} µg/m³", "Month": "Month"}
    )
    fig_zone_season.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=list(MONTH_NAMES.values())
    )
    if show_who_lines and poll_s in WHO_ANNUAL:
        fig_zone_season.add_hline(
            y=WHO_ANNUAL[poll_s], line_dash="dash", line_color="red",
            annotation_text="WHO limit"
        )
    fig_zone_season.update_layout(height=360, hovermode="x unified")
    st.plotly_chart(fig_zone_season,  width="stretch")

    # Station-level seasonality
    st.markdown("### Monthly seasonality by station")

    monthly_station = (
        daily_df
        .groupby(["station", "Zone", "Month"], as_index=False)
        .agg({poll_s: "mean"})
    )
    monthly_station["MonthName"] = monthly_station["Month"].map(MONTH_NAMES)

    fig_season = px.line(
        monthly_station, x="Month", y=poll_s,
        color="station",
        line_dash="Zone",
        color_discrete_map={
            s: ZONE_META.get(
                monthly_station[monthly_station["station"] == s]["Zone"].iloc[0], {}
            ).get("color", "#999")
            for s in monthly_station["station"].unique()
        },
        markers=True,
        title=f"{poll_s} monthly mean by station (line style = zone)",
        labels={poll_s: f"{poll_s} µg/m³", "Month": "Month"}
    )
    fig_season.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=list(MONTH_NAMES.values())
    )
    if show_who_lines and poll_s in WHO_ANNUAL:
        fig_season.add_hline(
            y=WHO_ANNUAL[poll_s], line_dash="dash", line_color="red",
            annotation_text="WHO limit"
        )
    fig_season.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig_season,  width="stretch")

    # Heatmap: station × month
    st.markdown("### Station × Month heatmap")

    pivot = monthly_station.pivot(index="station", columns="Month", values=poll_s)
    pivot.columns = [MONTH_NAMES[c] for c in pivot.columns]

    # Sort rows by zone
    zone_keys = list(ZONE_META.keys())

    def _zone_order(station: str) -> int:
        rows = daily_df[daily_df["station"] == station]
        if rows.empty:
            return 99
        zone = rows["Zone"].iloc[0]
        return zone_keys.index(zone) if zone in zone_keys else 99

    pivot = pivot.loc[sorted(pivot.index, key=_zone_order)]
    
    fig_hm = px.imshow(
        pivot,
        color_continuous_scale=["#2ecc71", "#f9ca24", "#e74c3c"],
        aspect="auto",
        title=f"{poll_s} µg/m³ — station × month (rows sorted by zone)",
        labels={"color": f"{poll_s} µg/m³"}
    )
    fig_hm.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_hm,  width="stretch")


# ==================== TAB 5: TABLE ====================
with tab_table:
    st.markdown("### Station risk summary table")

    display_cols = [
        "station", "Town", "Zone", "CoreRiskScore", "CoreRiskLevel",
        "PM2.5", "PM10", "NO2", "SO2_AnnualMean",
        "SO2_ExceedanceRate", "SO2_PressureLevel"
    ]
    if time_mode == "Year" and selected_year:
        display_cols = ["Year", "ValidDays"] + display_cols
    else:
        display_cols = ["ValidYears", "AvgValidDays"] + display_cols

    display_df = ranking[[c for c in display_cols if c in ranking.columns]].copy()
    float_cols = ["CoreRiskScore","PM2.5","PM10","NO2","SO2_AnnualMean",
                  "SO2_ExceedanceRate","AvgValidDays"]
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "CoreRiskScore": st.column_config.ProgressColumn(
                "Core Risk Score", min_value=0,
                max_value=float(display_df["CoreRiskScore"].max() * 1.1),
                format="%.1f"
            ),
            "SO2_ExceedanceRate": st.column_config.ProgressColumn(
                "SO₂ Exceedance Rate", min_value=0, max_value=1, format="%.1%"
            ),
            "Zone": st.column_config.TextColumn("Environmental Zone")
        }
    )

    # Zone comparison expander
    with st.expander("📊 Zone Comparison Table"):
        zone_comp = (
            ranking
            .groupby("Zone")[["CoreRiskScore", "PM2.5", "PM10", "NO2", "SO2_ExceedanceRate"]]
            .agg(["mean", "min", "max"])
            .round(2)
        )
        st.dataframe(zone_comp, width="stretch")

    st.markdown("---")
    st.markdown("### Daily raw data explorer")

    _fav = get_fav_station(all_stations)
    _idx = all_stations.index(_fav) if _fav in all_stations else 0
    station_sel = st.selectbox("Select station", options=all_stations, index=_idx)
    
    raw = daily_df[daily_df["station"] == station_sel][["Date","PM2.5","PM10","NO2","SO2"]].copy()
    raw = raw.sort_values("Date")

    station_zone = daily_df[daily_df["station"] == station_sel]["Zone"].iloc[0] \
        if not daily_df[daily_df["station"] == station_sel].empty else "Unknown"
    zone_c = ZONE_META.get(station_zone, {}).get("color", "#999")

    st.markdown(
        f"<span style='background:{zone_c};color:white;padding:2px 10px;"
        f"border-radius:8px;font-size:12px'>"
        f"{ZONE_META.get(station_zone,{}).get('icon','')} {station_zone}</span>",
        unsafe_allow_html=True
    )

    fig_raw = go.Figure()
    for p in CORE_POLLUTANTS:
        fig_raw.add_trace(go.Scatter(
            x=raw["Date"], y=raw[p], name=p,
            line=dict(color=POLLUTANT_COLOR[p], width=1), opacity=0.8
        ))
    if show_who_lines:
        for p, lim in WHO_ANNUAL.items():
            fig_raw.add_hline(y=lim, line_dash="dot", line_color=POLLUTANT_COLOR[p],
                              annotation_text=f"WHO {p}", annotation_font_size=9)
    fig_raw.update_layout(
        title=f"Daily concentrations — {station_sel} ({station_zone})",
        xaxis_title="Date", yaxis_title="µg/m³",
        height=380, hovermode="x unified",
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig_raw,  width="stretch")

    with st.expander("Show raw daily data"):
        st.dataframe(raw,  width="stretch", hide_index=True)

