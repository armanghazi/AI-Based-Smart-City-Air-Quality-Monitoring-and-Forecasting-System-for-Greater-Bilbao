import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import DATA_FILE

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

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_FILE)
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df

df = load_data()

WHO_ANNUAL = {"PM2.5": 5.0, "PM10": 15.0, "NO2": 10.0}

POLLUTANT_COLOR = {
    "NO2": "#e74c3c", "PM10": "#e67e22",
    "PM2.5": "#9b59b6", "SO2": "#3498db"
}

# -----------------------
# Header
# -----------------------

st.markdown("""
<div style="padding: 1.5rem 0 0.5rem">
    <h1 style="margin:0; font-size:2rem">
        🌍 Smart City Air Intelligence Platform
    </h1>
    <p style="color:#666; margin-top:4px; font-size:1rem">
        Greater Bilbao · Real-time air quality monitoring · 2015–2026
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------
# Live snapshot: latest available date
# -----------------------

latest_date = df["Date"].max()
latest_df   = df[df["Date"] == latest_date]

latest_means = (
    latest_df
    .groupby("station")[["PM2.5", "PM10", "NO2", "SO2"]]
    .mean()
    .mean()
)

st.markdown(f"### 📡 Latest Snapshot — {latest_date.strftime('%d %b %Y')}")
st.caption("City-wide average across all stations on the most recent recorded day")

c1, c2, c3, c4 = st.columns(4)

def who_delta(val, pollutant):
    limit = WHO_ANNUAL.get(pollutant)
    if not limit:
        return None, "off"
    ratio = val / limit
    label = f"{ratio:.1f}× WHO limit"
    color = "inverse" if ratio > 1 else "normal"
    return label, color

for col, poll in zip([c1, c2, c3, c4], ["PM2.5", "PM10", "NO2", "SO2"]):
    val = latest_means.get(poll, 0)
    delta_label, delta_color = who_delta(val, poll)
    col.metric(
        label=poll,
        value=f"{val:.1f} µg/m³",
        delta=delta_label,
        delta_color=delta_color
    )

st.divider()

# -----------------------
# Overview stats
# -----------------------

st.markdown("### 📊 Dataset Overview")

total_records  = len(df)
total_stations = df["station"].nunique()
date_range     = f"{df['Date'].min().strftime('%Y')} – {df['Date'].max().strftime('%Y')}"
total_years    = df["Year"].nunique()

o1, o2, o3, o4 = st.columns(4)
o1.metric("Total daily records",  f"{total_records:,}")
o2.metric("Monitoring stations",  str(total_stations))
o3.metric("Years of data",        f"{total_years} yrs ({date_range})")
o4.metric("Pollutants tracked",   "4  (PM2.5, PM10, NO2, SO2)")

st.divider()

# -----------------------
# Two-column layout: trend + station status
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
        x="Year",
        y="Concentration",
        color="Pollutant",
        markers=True,
        color_discrete_map=POLLUTANT_COLOR,
        labels={"Concentration": "µg/m³"},
        title=""
    )

    for poll, limit in WHO_ANNUAL.items():
        fig_trend.add_hline(
            y=limit,
            line_dash="dot",
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
    st.plotly_chart(fig_trend, use_container_width=True)

with col_right:
    st.markdown("#### 🏙️ Station Status — Latest Year")

    latest_year = int(df["Year"].max())
    station_latest = (
        df[df["Year"] == latest_year]
        .groupby("station")[["PM2.5", "PM10", "NO2"]]
        .mean()
        .reset_index()
    )

    def core_risk(row):
        ratios = [row["PM2.5"] / WHO_ANNUAL["PM2.5"],
                  row["PM10"]  / WHO_ANNUAL["PM10"],
                  row["NO2"]   / WHO_ANNUAL["NO2"]]
        score = 100 * sum(ratios) / 3
        if score < 100:
            return score, "Good"
        elif score < 200:
            return score, "Moderate"
        return score, "Poor"

    station_latest[["Score", "Status"]] = station_latest.apply(
        lambda r: pd.Series(core_risk(r)), axis=1
    )
    station_latest = station_latest.sort_values("Score", ascending=False)
    station_latest["Station"] = station_latest["station"].str.split("_").str[0]

    fig_status = px.bar(
        station_latest,
        x="Score",
        y="Station",
        color="Status",
        color_discrete_map={
            "Good":     "#2ecc71",
            "Moderate": "#f39c12",
            "Poor":     "#e74c3c"
        },
        orientation="h",
        title="",
        labels={"Score": "Core Risk Score"},
        text="Score"
    )
    fig_status.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_status.add_vline(x=100, line_dash="dash", line_color="#f39c12", opacity=0.6)
    fig_status.add_vline(x=200, line_dash="dash", line_color="#e74c3c", opacity=0.6)
    fig_status.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=40),
        showlegend=False,
        xaxis_range=[0, max(station_latest["Score"].max() * 1.2, 250)]
    )
    st.plotly_chart(fig_status, use_container_width=True)

st.divider()

# -----------------------
# Module navigation cards
# -----------------------

st.markdown("### 🧭 Navigate to Module")

m1, m2, m3 = st.columns(3)

card_style = """
    border: 0.5px solid #ddd; border-radius: 10px;
    padding: 20px; text-align: center; height: 160px;
    display: flex; flex-direction: column;
    justify-content: center; gap: 8px;
"""

with m1:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size: 2rem">🗺️</div>
        <div style="font-weight: 600; font-size: 1rem">Air Quality Monitoring</div>
        <div style="color: #888; font-size: 0.85rem">
            Interactive map · Station comparison · WHO risk levels
        </div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size: 2rem">📈</div>
        <div style="font-weight: 600; font-size: 1rem">Temporal Trends</div>
        <div style="color: #888; font-size: 0.85rem">
            Annual trends · Seasonality · COVID impact analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size: 2rem">🌍</div>
        <div style="font-weight: 600; font-size: 1rem">Urban Risk Index</div>
        <div style="color: #888; font-size: 0.85rem">
            WHO-based risk scoring · Heatmaps · Station rankings
        </div>
    </div>
    """, unsafe_allow_html=True)

st.caption("Use the left sidebar to navigate between modules.")