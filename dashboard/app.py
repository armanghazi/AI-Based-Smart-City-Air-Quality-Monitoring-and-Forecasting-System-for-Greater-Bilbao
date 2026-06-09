import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import (
    load_data,
    WHO_ANNUAL, POLLUTANT_COLOR,
    ZONE_META, who_delta,
)

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Smart City Air Intelligence",
    page_icon="🌍",
    layout="wide"
)

# -----------------------
# Load Data
# -----------------------
df = load_data()

# -----------------------
# Header
# -----------------------
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem">
    <h1 style="margin:0; font-size:2rem">
        🌍 Smart City Air Intelligence Platform
    </h1>
    <p style="color:#666; margin-top:4px; font-size:1rem">
        Greater Bilbao · Air quality monitoring · 2015–2026
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------
# Latest Snapshot
# -----------------------
latest_date  = df["Date"].max()
latest_df    = df[df["Date"] == latest_date]
latest_means = (
    latest_df
    .groupby("station")[["PM2.5", "PM10", "NO2", "SO2"]]
    .mean()
    .mean()
)

st.markdown(f"### 📡 Latest Snapshot — {latest_date.strftime('%d %b %Y')}")
st.caption("City-wide average across all stations on the most recent recorded day")

c1, c2, c3, c4 = st.columns(4)
for col, poll in zip([c1, c2, c3, c4], ["PM2.5", "PM10", "NO2", "SO2"]):
    val                      = latest_means.get(poll, 0)
    delta_label, delta_color = who_delta(val, poll)
    col.metric(
        label=poll,
        value=f"{val:.1f} µg/m³",
        delta=delta_label,
        delta_color=delta_color
    )

st.divider()

# -----------------------
# Dataset Overview
# -----------------------
st.markdown("### 📊 Dataset Overview")

total_records  = len(df)
total_stations = df["station"].nunique()
date_range     = f"{df['Date'].min().strftime('%Y')} – {df['Date'].max().strftime('%Y')}"
total_years    = df["Year"].nunique()

o1, o2, o3, o4 = st.columns(4)
o1.metric("Total daily records", f"{total_records:,}")
o2.metric("Monitoring stations", str(total_stations))
o3.metric("Years of data",       f"{total_years} yrs ({date_range})")
o4.metric("Pollutants tracked",  "4  (PM2.5, PM10, NO₂, SO₂)")

st.divider()

# -----------------------
# Environmental Zone Overview — 5 zones, 2 rows
# -----------------------
st.markdown("### 🗺️ Environmental Zone Overview")
st.caption("Spatial zones identified from EDA based on pollution profile and emission source characteristics")

latest_year  = int(df["Year"].max())
zone_summary = (
    df[df["Year"] == latest_year]
    .groupby("Zone")[["PM2.5", "PM10", "NO2", "SO2"]]
    .mean()
    .round(2)
)

def render_zone_card(zone_name, meta, zone_summary, df):
    z_data      = zone_summary.loc[zone_name] if zone_name in zone_summary.index else None
    stations_in = df[df["Zone"] == zone_name]["station"].unique().tolist()
    short_names = ", ".join([s.split("_")[0] for s in stations_in])

    if z_data is not None:
        pm25_val = z_data["PM2.5"]
        pm10_val = z_data["PM10"]
        no2_val  = z_data["NO2"]
        so2_val  = z_data["SO2"]
        key_poll = meta["key_pollutant"]
        key_val  = z_data[key_poll]
        who_lim  = WHO_ANNUAL.get(key_poll)
        vs_who   = f"{key_val / who_lim:.1f}×" if who_lim else "—"
    else:
        pm25_val = pm10_val = no2_val = so2_val = 0.0
        vs_who   = "—"
        key_poll = meta["key_pollutant"]

    st.markdown(
        f"""
        <div style="
            border-left: 5px solid {meta['border']};
            background: linear-gradient(135deg, {meta['color']}18, {meta['color']}06);
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 8px;
            height: 100%;
        ">
            <div style="font-size:20px; margin-bottom:4px">
                {meta['icon']} <strong>{zone_name}</strong>
            </div>
            <div style="color:#555; font-size:11px; margin-bottom:6px">
                {meta['description']}
            </div>
            <div style="color:#777; font-size:11px; margin-bottom:8px">
                📍 {short_names}
            </div>
            <table style="width:100%; font-size:12px; border-collapse:collapse">
                <tr>
                    <td style="color:#777; padding:2px 0">PM2.5</td>
                    <td style="text-align:right; font-weight:600">{pm25_val:.1f} µg/m³</td>
                </tr>
                <tr>
                    <td style="color:#777; padding:2px 0">PM10</td>
                    <td style="text-align:right; font-weight:600">{pm10_val:.1f} µg/m³</td>
                </tr>
                <tr>
                    <td style="color:#777; padding:2px 0">NO₂</td>
                    <td style="text-align:right; font-weight:600">{no2_val:.1f} µg/m³</td>
                </tr>
                <tr>
                    <td style="color:#777; padding:2px 0">SO₂</td>
                    <td style="text-align:right; font-weight:600">{so2_val:.1f} µg/m³</td>
                </tr>
                <tr>
                    <td style="color:#777; padding:2px 0">Key ({key_poll}) vs WHO</td>
                    <td style="text-align:right; font-weight:600;
                               color:{meta['color']}">{vs_who}</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

zones_list = list(ZONE_META.items())
row1       = zones_list[:3]
row2       = zones_list[3:]

cols_r1 = st.columns(3)
for idx, (zone_name, meta) in enumerate(row1):
    with cols_r1[idx]:
        render_zone_card(zone_name, meta, zone_summary, df)

if row2:
    cols_r2  = st.columns(3)
    offsets  = [0, 1] if len(row2) == 2 else [1]
    for i, (zone_name, meta) in zip(offsets, row2):
        with cols_r2[i]:
            render_zone_card(zone_name, meta, zone_summary, df)

st.divider()

# -----------------------
# City-wide trend + Station status
# -----------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### 📈 City-wide Annual Trend")

    annual = (
        df
        .groupby("Year")[["PM2.5", "PM10", "NO2"]]
        .mean()
        .reset_index()
    )
    annual_long = annual.melt(
        id_vars="Year",
        var_name="Pollutant",
        value_name="Concentration"
    )
    fig_trend = px.line(
        annual_long,
        x="Year", y="Concentration",
        color="Pollutant",
        markers=True,
        color_discrete_map=POLLUTANT_COLOR,
        labels={"Concentration": "µg/m³"},
        title=""
    )
    for poll, limit in WHO_ANNUAL.items():
        fig_trend.add_hline(
            y=limit, line_dash="dot",
            line_color=POLLUTANT_COLOR[poll],
            opacity=0.4,
            annotation_text=f"WHO {poll}",
            annotation_font_size=9,
            annotation_position="right"
        )
    fig_trend.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=60),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig_trend, width="stretch")

with col_right:
    st.markdown("#### 🏙️ Station Status — Latest Year")

    station_latest = (
        df[df["Year"] == latest_year]
        .groupby(["station", "Zone"])[["PM2.5", "PM10", "NO2"]]
        .mean()
        .reset_index()
    )

    def core_risk(row):
        ratios = [
            row["PM2.5"] / WHO_ANNUAL["PM2.5"],
            row["PM10"]  / WHO_ANNUAL["PM10"],
            row["NO2"]   / WHO_ANNUAL["NO2"],
        ]
        score = 100 * sum(ratios) / 3
        if score < 100:   return score, "Below WHO"
        elif score < 200: return score, "1–2× WHO"
        return score, ">2× WHO"

    station_latest[["Score", "Status"]] = station_latest.apply(
        lambda r: pd.Series(core_risk(r)), axis=1
    )
    station_latest          = station_latest.sort_values("Score", ascending=False)
    station_latest["Station"] = station_latest["station"].str.split("_").str[0]

    fig_status = px.bar(
        station_latest,
        x="Score", y="Station",
        color="Zone",
        color_discrete_map={z: m["color"] for z, m in ZONE_META.items()},
        orientation="h",
        labels={"Score": "Core Risk Score"},
        text="Score",
        title=""
    )
    fig_status.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_status.add_vline(x=100, line_dash="dash", line_color="#f39c12", opacity=0.6)
    fig_status.add_vline(x=200, line_dash="dash", line_color="#e74c3c", opacity=0.6)
    fig_status.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=40),
        showlegend=True,
        legend=dict(orientation="h", y=1.08, font=dict(size=10)),
        xaxis_range=[0, max(station_latest["Score"].max() * 1.2, 250)]
    )
    st.plotly_chart(fig_status, width="stretch")

st.divider()

# -----------------------
# Navigation Cards
# -----------------------
st.markdown("### 🧭 Navigate to Module")
st.caption(
    "Explore detailed analyses: Air Quality Monitoring, "
    "Temporal Trends, Urban Risk Index, Weather Drivers & Air Pollution Dynamics"
)

NAV_MODULES = [
    {
        "icon":        "🗺️",
        "title":       "Air Quality Monitoring",
        "description": "Interactive map · Station comparison · WHO risk levels",
        "page":        "pages/1_Air_Quality_Monitoring.py",
        "color":       "#2980b9",
        "border":      "#1a6fa0",
    },
    {
        "icon":        "📈",
        "title":       "Temporal Trends",
        "description": "Annual trends · Seasonality · COVID impact analysis",
        "page":        "pages/2_Temporal_Trends.py",
        "color":       "#27ae60",
        "border":      "#1e8449",
    },
    {
        "icon":        "🌍",
        "title":       "Urban Risk Index",
        "description": "WHO-based risk scoring · Heatmaps · Station rankings",
        "page":        "pages/3_Urban_Risk_Index.py",
        "color":       "#c0392b",
        "border":      "#a93226",
    },
    {
        "icon":        "🌤️",
        "title":       "Weather Drivers",
        "description": "Wind · Rain · Temperature · Lag analysis · Forecast features",
        "page":        "pages/4_Weather_Drivers_&_Air_Pollution_Dynamics.py",
        "color":       "#d35400",
        "border":      "#b94600",
    },
]

nav_cols = st.columns(4)

for idx, module in enumerate(NAV_MODULES):
    with nav_cols[idx]:
        st.markdown(
            f"""
            <div style="
                border: 2px solid {module['border']};
                border-radius: 12px;
                padding: 20px 16px 12px;
                text-align: center;
                background: linear-gradient(135deg, {module['color']}12, {module['color']}04);
                margin-bottom: 8px;
            ">
                <div style="font-size: 2.2rem; margin-bottom: 6px">{module['icon']}</div>
                <div style="font-weight: 700; font-size: 1rem; color: #2c3e50;
                            margin-bottom: 6px">{module['title']}</div>
                <div style="color: #777; font-size: 0.82rem; line-height: 1.4">
                    {module['description']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button(
            f"Open {module['title']} →",
            key=f"nav_{idx}",
            use_container_width=True
        ):
            st.switch_page(module["page"])

# -----------------------
# Footer
# -----------------------
st.divider()
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown(
        "**🌬️ Air Quality Data**  \n"
        "Basque Government Air Quality Network  \n"
        "*(Red de Control de Calidad del Aire)*  \n"
        "7 stations · Greater Bilbao · © Gobierno Vasco · CC BY 4.0  \n"
        "WHO 2021 guidelines applied"
    )

with col_s2:
    st.markdown(
        "**🌤️ Meteorological Data**  \n"
        "Open-Meteo · [open-meteo.com](https://open-meteo.com)  \n"
        "Historical Weather API · CC BY 4.0  \n"
        "~29k daily records \n"   
        "\n"
        "Temperature, Humidity, Precipitation, Wind Speed, Wind Direction"
    )