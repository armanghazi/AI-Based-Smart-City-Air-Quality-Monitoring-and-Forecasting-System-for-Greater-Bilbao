import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from config import DATA_FILE

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
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    return df

df = load_data()

# -----------------------
# Thresholds per pollutant
# WHO / EU guideline-based thresholds
# -----------------------
THRESHOLDS = {
    "PM2.5": {"low": 10, "mid": 25},
    "PM10":  {"low": 25, "mid": 50},
    "NO2":   {"low": 20, "mid": 40},
    "SO2":   {"low": 20, "mid": 40},
}

def get_color(value, pollutant):
    t = THRESHOLDS[pollutant]
    if value <= t["low"]:
        return "#2ecc71"
    elif value <= t["mid"]:
        return "#f39c12"
    return "#e74c3c"

def get_quality_label(value, pollutant):
    t = THRESHOLDS[pollutant]
    if value <= t["low"]:
        return "Good"
    elif value <= t["mid"]:
        return "Moderate"
    return "Poor"

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Filters")

pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["PM2.5", "PM10", "NO2", "SO2"]
)

year_options = ["All"] + sorted(df["Year"].dropna().unique().tolist())
selected_year = st.sidebar.selectbox(
    "Select Year",
    options=year_options,
    index=0
)

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
if selected_year == "All":
    filtered = df.copy()
    period_label = "All Years (2015–2026)"
else:
    filtered = df[df["Year"] == int(selected_year)].copy()
    period_label = str(selected_year)

if filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# -----------------------
# Mean per station
# -----------------------
station_mean = (
    filtered
    .groupby(["station", "Town", "Latitude", "Longitude"])[pollutant]
    .mean()
    .reset_index()
)

if station_mean.empty:
    st.warning("No station data after aggregation.")
    st.stop()

# -----------------------
# KPI values
# -----------------------
avg_value     = station_mean[pollutant].mean()
worst_station = station_mean.loc[station_mean[pollutant].idxmax()]
best_station  = station_mean.loc[station_mean[pollutant].idxmin()]
good_count    = int((station_mean[pollutant].apply(lambda v: get_quality_label(v, pollutant)) == "Good").sum())
mod_count     = int((station_mean[pollutant].apply(lambda v: get_quality_label(v, pollutant)) == "Moderate").sum())
poor_count    = int((station_mean[pollutant].apply(lambda v: get_quality_label(v, pollutant)) == "Poor").sum())

# -----------------------
# KPI Cards — clean single block
# -----------------------
st.subheader(f"{pollutant} Mean Values — {period_label}")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(f"Avg {pollutant}", f"{avg_value:.1f} µg/m³")

c2.metric(
    "Most Polluted",
    worst_station["station"].split("_")[0],
    f"{worst_station[pollutant]:.1f} µg/m³",
    delta_color="inverse"
)

c3.metric(
    "Cleanest",
    best_station["station"].split("_")[0],
    f"{best_station[pollutant]:.1f} µg/m³"
)

c4.metric(
    "🟢 Good / 🟡 Moderate",
    f"{good_count} / {mod_count}"
)

c5.metric("🔴 Poor stations", poor_count)

# -----------------------
# Map
# -----------------------
center_lat = station_mean["Latitude"].mean()
center_lon = station_mean["Longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="CartoDB positron"
)

for _, row in station_mean.iterrows():
    value = row[pollutant]
    color = get_color(value, pollutant)
    label = get_quality_label(value, pollutant)
    radius = max(8, min(22, value / 2))

    popup_html = f"""
    <div style="font-family:sans-serif; min-width:180px;">
        <b style="font-size:14px;">{row['station']}</b><br>
        <span style="color:#555;">{row['Town']}</span>
        <hr style="margin:4px 0">
        <b>{pollutant}:</b> {value:.2f} µg/m³<br>
        <span style="
            background:{color};color:white;
            padding:2px 10px;border-radius:10px;font-size:12px;
        ">{label}</span>
        <hr style="margin:4px 0">
        <span style="font-size:11px;color:#888;">Period: {period_label}</span>
    </div>
    """

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=240),
        tooltip=f"{row['station'].split('_')[0]} – {value:.1f} µg/m³ ({label})",
        color="white",
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.88,
    ).add_to(m)

    # Station name label above marker
    folium.Marker(
        location=[row["Latitude"] + 0.003, row["Longitude"]],
        icon=folium.DivIcon(
            html=f'<div style="font-size:10px;font-weight:600;color:#2c3e50;white-space:nowrap">{row["station"].split("_")[0]}</div>',
            icon_size=(120, 20),
            icon_anchor=(60, 0)
        )
    ).add_to(m)

# HTML legend inside map
legend_html = f"""
<div style="
    position:fixed; bottom:30px; left:30px; z-index:1000;
    background:white; border-radius:10px; padding:14px 18px;
    box-shadow:0 2px 12px rgba(0,0,0,0.2);
    font-family:sans-serif; font-size:13px; line-height:2;
">
    <b style="font-size:14px;">Air Quality — {pollutant}</b><br>
    <span style="color:#2ecc71;">●</span> Good &nbsp;(≤ {t['low']} µg/m³)<br>
    <span style="color:#f39c12;">●</span> Moderate &nbsp;({t['low']+1}–{t['mid']} µg/m³)<br>
    <span style="color:#e74c3c;">●</span> Poor &nbsp;(> {t['mid']} µg/m³)<br>
    <span style="color:#888;font-size:11px;">Circle size ∝ concentration</span>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=None, height=600, returned_objects=[])

# -----------------------
# Ranking Table
# -----------------------
st.subheader(f"🏆 Station Ranking — {pollutant} — {period_label}")

ranking = (
    station_mean
    .sort_values(pollutant, ascending=False)
    .reset_index(drop=True)
)
ranking.index = ranking.index + 1

ranking["Quality"] = ranking[pollutant].apply(lambda v: get_quality_label(v, pollutant))
ranking[f"{pollutant} vs threshold"] = ranking[pollutant] / t["mid"]

st.dataframe(
    ranking[["station", "Town", pollutant, "Quality"]],
    use_container_width=True,
    column_config={
        pollutant: st.column_config.ProgressColumn(
            f"{pollutant} (µg/m³)",
            min_value=0,
            max_value=float(ranking[pollutant].max() * 1.1),
            format="%.1f"
        ),
        "Quality": st.column_config.TextColumn("Quality Level")
    },
    hide_index=False
)