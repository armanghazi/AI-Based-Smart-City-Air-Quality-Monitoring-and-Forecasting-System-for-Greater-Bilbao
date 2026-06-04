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
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"]   = df["Date"].dt.date
    return df

df = load_data()

# -----------------------
# Zone Classification
# -----------------------
ZONE_MAP = {
    "Barakaldo": "Industrial Corridor",
    "Basauri":   "Industrial Corridor",
    "Bilbao": "Urban Core",
    "Erandio":   "Urban Core",
    "Getxo":   "Coastal Buffer Zone",
    "Muskiz":    "Coastal Buffer Zone",
    "Santurtzi": "Coastal Buffer Zone",
}

ZONE_META = {
    "Industrial Corridor": {
        "icon": "🏭",
        "color": "#e67e22",
        "description": "High PM2.5, High PM10, Elevated NO₂",
        "border": "#d35400",
    },
    "Urban Core": {
        "icon": "🚗",
        "color": "#8e44ad",
        "description": "Highest NO₂, Strong traffic influence, Urban canyon effects",
        "border": "#6c3483",
    },
    "Coastal Buffer Zone": {
        "icon": "🌊",
        "color": "#1abc9c",
        "description": "Better dispersion, Lower NO₂, Marine influence on PM10",
        "border": "#148f77",
    },
}

ZONE_MARKER_COLOR = {
    "Industrial Corridor": "#e67e22",
    "Urban Core":          "#8e44ad",
    "Coastal Buffer Zone": "#1abc9c",
}

def get_zone(town: str) -> str:
    for key in ZONE_MAP:
        if key.lower() in town.lower():
            return ZONE_MAP[key]
    return "Unknown"

# -----------------------
# Thresholds
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
# Sidebar — Pollutant
# -----------------------
st.sidebar.header("Filters")

pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["PM2.5", "PM10", "NO2", "SO2"]
)

# -----------------------
# Sidebar — Time granularity
# -----------------------
time_mode = st.sidebar.radio(
    "Time Granularity",
    options=["Year", "Month", "Day"],
    index=0,
    horizontal=True
)

# ---- Year mode ----
if time_mode == "Year":
    year_options = ["All"] + sorted(df["Year"].dropna().unique().tolist())
    selected_year = st.sidebar.selectbox("Select Year", year_options, index=0)

    if selected_year == "All":
        filtered = df.copy()
        period_label = "All Years (2015–2026)"
    else:
        filtered = df[df["Year"] == int(selected_year)].copy()
        period_label = str(selected_year)

# ---- Month mode ----
elif time_mode == "Month":
    year_options = sorted(df["Year"].dropna().unique().tolist())
    selected_year_m = st.sidebar.selectbox("Select Year", year_options, index=len(year_options)-1)

    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    available_months = sorted(
        df[df["Year"] == selected_year_m]["Month"].dropna().unique().tolist()
    )
    month_options = ["All"] + [month_names[m] for m in available_months]
    selected_month_label = st.sidebar.selectbox("Select Month", month_options, index=0)

    if selected_month_label == "All":
        filtered = df[df["Year"] == selected_year_m].copy()
        period_label = f"All months of {selected_year_m}"
    else:
        month_num = {v: k for k, v in month_names.items()}[selected_month_label]
        filtered = df[
            (df["Year"] == selected_year_m) & (df["Month"] == month_num)
        ].copy()
        period_label = f"{selected_month_label} {selected_year_m}"

# ---- Day mode ----
else:
    year_options = sorted(df["Year"].dropna().unique().tolist())
    selected_year_d = st.sidebar.selectbox("Select Year", year_options, index=len(year_options)-1)

    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    available_months_d = sorted(
        df[df["Year"] == selected_year_d]["Month"].dropna().unique().tolist()
    )
    month_options_d = [month_names[m] for m in available_months_d]
    selected_month_label_d = st.sidebar.selectbox("Select Month", month_options_d, index=0)
    month_num_d = {v: k for k, v in month_names.items()}[selected_month_label_d]

    df_month = df[
        (df["Year"] == selected_year_d) & (df["Month"] == month_num_d)
    ]
    available_days = sorted(df_month["Day"].dropna().unique().tolist())
    selected_day = st.sidebar.selectbox(
        "Select Day",
        available_days,
        format_func=lambda d: d.strftime("%d %b %Y")
    )
    filtered = df[df["Day"] == selected_day].copy()
    period_label = pd.Timestamp(selected_day).strftime("%d %B %Y")

# -----------------------
# Threshold legend in sidebar
# -----------------------
t = THRESHOLDS[pollutant]
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Thresholds (µg/m³)")
st.sidebar.markdown(
    f"🟢 **Good** : ≤ {t['low']}\n\n"
    f"🟡 **Moderate** : {t['low']+1} – {t['mid']}\n\n"
    f"🔴 **Poor** : > {t['mid']}"
)

# -----------------------
# Zone legend in sidebar
# -----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🗺️ Environmental Zones")
for zone, meta in ZONE_META.items():
    st.sidebar.markdown(
        f"{meta['icon']} **{zone}**  \n"
        f"<span style='color:{meta['color']};font-size:11px'>{meta['description']}</span>",
        unsafe_allow_html=True
    )

# -----------------------
# Safety check
# -----------------------
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

# Assign zone
station_mean["Zone"] = station_mean["Town"].apply(get_zone)

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
# KPI Cards
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
c4.metric("🟢 Good / 🟡 Moderate", f"{good_count} / {mod_count}")
c5.metric("🔴 Poor stations", poor_count)

# -----------------------
# Zone Summary Cards
# -----------------------
st.markdown("---")
st.subheader("🗺️ Environmental Zone Summary")

zone_cols = st.columns(3)

for idx, (zone_name, meta) in enumerate(ZONE_META.items()):
    zone_stations = station_mean[station_mean["Zone"] == zone_name]

    if zone_stations.empty:
        zone_avg = "N/A"
        zone_worst = "N/A"
        zone_quality = "–"
        n_stations = 0
    else:
        zone_avg = f"{zone_stations[pollutant].mean():.1f} µg/m³"
        zone_worst_row = zone_stations.loc[zone_stations[pollutant].idxmax()]
        zone_worst = f"{zone_worst_row['station'].split('_')[0]} ({zone_worst_row[pollutant]:.1f})"
        zone_q_vals = zone_stations[pollutant].apply(lambda v: get_quality_label(v, pollutant))
        zone_quality = zone_q_vals.mode()[0] if not zone_q_vals.empty else "–"
        n_stations = len(zone_stations)

    with zone_cols[idx]:
        st.markdown(
            f"""
            <div style="
                border-left: 5px solid {meta['border']};
                background: linear-gradient(135deg, {meta['color']}18, {meta['color']}06);
                border-radius: 10px;
                padding: 16px 18px;
                margin-bottom: 8px;
            ">
                <div style="font-size:22px; margin-bottom:4px">{meta['icon']} <strong>{zone_name}</strong></div>
                <div style="color:#555; font-size:12px; margin-bottom:10px">{meta['description']}</div>
                <table style="width:100%; font-size:13px; border-collapse:collapse">
                    <tr>
                        <td style="color:#777; padding:3px 0">Stations</td>
                        <td style="text-align:right; font-weight:600">{n_stations}</td>
                    </tr>
                    <tr>
                        <td style="color:#777; padding:3px 0">Avg {pollutant}</td>
                        <td style="text-align:right; font-weight:600">{zone_avg}</td>
                    </tr>
                    <tr>
                        <td style="color:#777; padding:3px 0">Highest station</td>
                        <td style="text-align:right; font-weight:600">{zone_worst}</td>
                    </tr>
                    <tr>
                        <td style="color:#777; padding:3px 0">Typical quality</td>
                        <td style="text-align:right; font-weight:600">{zone_quality}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# -----------------------
# Map
# -----------------------
st.subheader(f"🗺️ Station Map — {pollutant} — {period_label}")

center_lat = station_mean["Latitude"].mean()
center_lon = station_mean["Longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="CartoDB positron"
)

# Draw light zone boundary hints using FeatureGroup per zone
for zone_name, meta in ZONE_META.items():
    zone_stations = station_mean[station_mean["Zone"] == zone_name]
    if zone_stations.empty:
        continue

    fg = folium.FeatureGroup(name=f"Zone: {zone_name}")

    for _, row in zone_stations.iterrows():
        value   = row[pollutant]
        color   = get_color(value, pollutant)
        label   = get_quality_label(value, pollutant)
        radius  = max(8, min(22, value / 2))
        z_color = ZONE_MARKER_COLOR[zone_name]

        popup_html = (
            f"<div style='font-family:sans-serif;min-width:190px;padding:4px'>"
            f"<b style='font-size:14px'>{row['station']}</b><br>"
            f"<span style='color:#555'>{row['Town']}</span><br>"
            f"<span style='background:{z_color};color:white;padding:1px 8px;"
            f"border-radius:8px;font-size:11px'>{zone_name}</span><br>"
            f"<hr style='margin:4px 0'>"
            f"<b>{pollutant}:</b> {value:.2f} &micro;g/m&sup3;<br>"
            f"<span style='background:{color};color:white;"
            f"padding:2px 10px;border-radius:10px;font-size:12px'>{label}</span><br>"
            f"<span style='font-size:11px;color:#888'>Period: {period_label}</span>"
            f"</div>"
        )

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=250, parse_html=False),
            tooltip=f"{row['station'].split('_')[0]} | {zone_name} | {value:.1f} µg/m³ ({label})",
            color=z_color,       # zone color as border
            weight=3,
            fill=True,
            fill_color=color,    # air quality color as fill
            fill_opacity=0.88,
        ).add_to(fg)

        # Station name label
        folium.Marker(
            location=[row["Latitude"] + 0.003, row["Longitude"]],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:10px;font-weight:600;color:#2c3e50;'
                    f'white-space:nowrap">{row["station"].split("_")[0]}</div>'
                ),
                icon_size=(120, 20),
                icon_anchor=(60, 0)
            )
        ).add_to(fg)

    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# Legend
legend_html = f"""
<div style="
    position:fixed; bottom:30px; left:30px; z-index:1000;
    background:white; border-radius:10px; padding:14px 18px;
    box-shadow:0 2px 12px rgba(0,0,0,0.2);
    font-family:sans-serif; font-size:13px; line-height:2; max-width:220px;
">
    <b style="font-size:14px;">Air Quality — {pollutant}</b><br>
    <span style="color:#2ecc71;">●</span> Good &nbsp;(≤ {t['low']} µg/m³)<br>
    <span style="color:#f39c12;">●</span> Moderate &nbsp;({t['low']+1}–{t['mid']} µg/m³)<br>
    <span style="color:#e74c3c;">●</span> Poor &nbsp;(> {t['mid']} µg/m³)<br>
    <hr style="margin:6px 0">
    <b style="font-size:12px;">Zone (border color)</b><br>
    <span style="color:#e67e22;">●</span> Industrial Corridor<br>
    <span style="color:#8e44ad;">●</span> Urban Core<br>
    <span style="color:#1abc9c;">●</span> Coastal Buffer Zone<br>
    <span style="color:#888;font-size:11px;">Fill = Air quality · Border = Zone</span>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(
    m,
    width=None,
    height=600,
    returned_objects=[],
    key=f"map_{pollutant}_{time_mode}_{period_label}"
)

# -----------------------
# Ranking Table — with Zone column
# -----------------------
st.subheader(f"🏆 Station Ranking — {pollutant} — {period_label}")

ranking = (
    station_mean
    .sort_values(pollutant, ascending=False)
    .reset_index(drop=True)
)
ranking.index = ranking.index + 1
ranking["Quality"] = ranking[pollutant].apply(lambda v: get_quality_label(v, pollutant))

st.dataframe(
    ranking[["station", "Town", "Zone", pollutant, "Quality"]],
    use_container_width=True,
    column_config={
        pollutant: st.column_config.ProgressColumn(
            f"{pollutant} (µg/m³)",
            min_value=0,
            max_value=float(ranking[pollutant].max() * 1.1),
            format="%.1f"
        ),
        "Quality": st.column_config.TextColumn("Quality Level"),
        "Zone":    st.column_config.TextColumn("Environmental Zone"),
    },
    hide_index=False
)

# -----------------------
# Zone breakdown table
# -----------------------
with st.expander("📊 Zone Comparison Table"):
    zone_summary = (
        station_mean
        .groupby("Zone")[pollutant]
        .agg(["mean", "min", "max", "count"])
        .rename(columns={
            "mean":  f"Avg {pollutant}",
            "min":   f"Min {pollutant}",
            "max":   f"Max {pollutant}",
            "count": "Stations"
        })
        .round(2)
        .reset_index()
    )
    st.dataframe(zone_summary, use_container_width=True)