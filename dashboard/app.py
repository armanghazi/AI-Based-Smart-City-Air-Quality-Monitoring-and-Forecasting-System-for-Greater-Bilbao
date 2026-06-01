import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path

# Base directory: folder where app.py lives  (dashboard/)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "final_air_quality.parquet"
)

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Smart City Air Intelligence",
    layout="wide"
)
st.title("🌍 Smart City Air Quality Dashboard")
st.markdown("Greater Bilbao Air Quality Monitoring System")

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_FILE)
    df["Year"] = df["Date"].dt.year
    return df

df = load_data()

# -----------------------
# Thresholds per pollutant
# WHO / EU guideline–based thresholds
# -----------------------
THRESHOLDS = {
    "PM2.5": {"low": 15,  "mid": 25},   # µg/m³
    "PM10":  {"low": 30,  "mid": 50},   # µg/m³
    "NO2":   {"low": 40,  "mid": 100},  # µg/m³
    "SO2":   {"low": 20,  "mid": 80},   # µg/m³
}

def get_color(value, pollutant):
    """Return green / yellow / red based on thresholds."""
    t = THRESHOLDS[pollutant]
    if value <= t["low"]:
        return "#2ecc71"   # green
    elif value <= t["mid"]:
        return "#f39c12"   # yellow/orange
    else:
        return "#e74c3c"   # red

def get_quality_label(value, pollutant):
    t = THRESHOLDS[pollutant]
    if value <= t["low"]:
        return "Good"
    elif value <= t["mid"]:
        return "Moderate"
    else:
        return "Poor"

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Filters")
pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["PM2.5", "PM10", "NO2", "SO2"]
)
selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Year"].unique())
)

# Show thresholds info in sidebar
t = THRESHOLDS[pollutant]
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Thresholds (µg/m³)")
st.sidebar.markdown(
    f"🟢 **Good** : ≤ {t['low']}\n\n"
    f"🟡 **Moderate** : {t['low']+1} – {t['mid']}\n\n"
    f"🔴 **Poor** : > {t['mid']}"
)

# -----------------------
# Filter Data
# -----------------------
filtered = df[df["Year"] == selected_year]

# -----------------------
# Mean per station
# -----------------------
station_mean = (
    filtered
    .groupby(["station", "Town", "Latitude", "Longitude"])[pollutant]
    .mean()
    .reset_index()
)



# -----------------------
# KPI Statistics
# -----------------------

avg_value = station_mean[pollutant].mean()

worst_station = station_mean.loc[
    station_mean[pollutant].idxmax()
]

best_station = station_mean.loc[
    station_mean[pollutant].idxmin()
]


# -----------------------
# Map
# -----------------------
center_lat = station_mean["Latitude"].mean()
center_lon = station_mean["Longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles="CartoDB positron"
)

# -----------------------
# Markers (color-coded)
# -----------------------
for _, row in station_mean.iterrows():
    value = row[pollutant]
    color = get_color(value, pollutant)
    label = get_quality_label(value, pollutant)

    popup_html = f"""
    <div style="font-family:sans-serif; min-width:160px;">
        <b style="font-size:14px;">{row['station']}</b><br>
        <span style="color:#555;">{row['Town']}</span><hr style="margin:4px 0">
        <b>{pollutant}:</b> {value:.2f} µg/m³<br>
        <span style="
            background:{color};
            color:white;
            padding:2px 8px;
            border-radius:10px;
            font-size:12px;
        ">{label}</span>
    </div>
    """

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius = max(
            6,
                min(
                  20,
                    value / 2
                    )
),  # size by value (with limits)
        popup=folium.Popup(popup_html, max_width=220),
        tooltip=f"{row['station']} – {value:.1f} µg/m³ ({label})",
        color="white",
        weight=1.5,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
    ).add_to(m)

# -----------------------
# HTML Legend (inside map)
# -----------------------
legend_html = f"""
<div style="
    position: fixed;
    bottom: 30px; left: 30px;
    z-index: 1000;
    background: white;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    font-family: sans-serif;
    font-size: 13px;
    line-height: 1.8;
">
    <b style="font-size:14px;">Air Quality – {pollutant}</b><br>
    <span style="color:#2ecc71;">●</span> Good &nbsp;(≤ {t['low']} µg/m³)<br>
    <span style="color:#f39c12;">●</span> Moderate &nbsp;({t['low']+1}–{t['mid']} µg/m³)<br>
    <span style="color:#e74c3c;">●</span> Poor &nbsp;(> {t['mid']} µg/m³)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# -----------------------
# Show Map
# -----------------------
st.subheader(f"{pollutant} Mean Values – {selected_year}")

# Summary KPI row
# -----------------------
# KPI Cards
# -----------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        f"Average {pollutant}",
        f"{avg_value:.1f} µg/m³"
    )

with col2:
    st.metric(
        "Most Polluted Station",
        worst_station["station"]
    )

    st.caption(
        f"{worst_station[pollutant]:.1f} µg/m³"
    )

with col3:
    st.metric(
        "Cleanest Station",
        best_station["station"]
    )

    st.caption(
        f"{best_station[pollutant]:.1f} µg/m³"
    )

with col1:
    good_count = sum(
        1 for v in station_mean[pollutant]
        if get_quality_label(v, pollutant) == "Good"
    )
    st.metric("🟢 Good stations", good_count)
with col2:
    mod_count = sum(
        1 for v in station_mean[pollutant]
        if get_quality_label(v, pollutant) == "Moderate"
    )
    st.metric("🟡 Moderate stations", mod_count)
with col3:
    poor_count = sum(
        1 for v in station_mean[pollutant]
        if get_quality_label(v, pollutant) == "Poor"
    )
    st.metric("🔴 Poor stations", poor_count)

st_folium(m, width=1200, height=600)
# -----------------------
# Ranking Table
# -----------------------

st.subheader(
    f"🏆 Station Ranking - {pollutant}"
)

ranking = (
    station_mean
    .sort_values(
        pollutant,
        ascending=False
    )
    .reset_index(drop=True)
)

ranking.index = ranking.index + 1

st.dataframe(
    ranking[
        [
            "station",
            "Town",
            pollutant
        ]
    ],
    use_container_width=True
)



