import streamlit as st
import pandas as pd
import plotly.express as px
import folium

from config import DATA_FILE
from streamlit_folium import st_folium

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Urban Risk Index",
    layout="wide"
)

st.title("🌍 Urban Risk Index")
st.markdown(
    """
    WHO-based annual air pollution risk dashboard for Greater Bilbao.

    - Core annual risk = PM2.5 + PM10 + NO2 annual means relative to WHO 2021 annual guideline values
    - SO2 = short-term pressure indicator based on daily exceedance of WHO 24-hour guideline
    """
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    return df

df = load_data()

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

WHO_ANNUAL = {
    "PM2.5": 5.0,
    "PM10": 15.0,
    "NO2": 10.0
}

WHO_SO2_DAILY = 40.0

CORE_POLLUTANTS = ["PM2.5", "PM10", "NO2"]

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def classify_core_risk(score):
    if score < 100:
        return "Below WHO guideline"
    elif score < 200:
        return "1–2× WHO guideline"
    else:
        return ">2× WHO guideline"

def risk_color(score):
    if score < 100:
        return "#2ecc71"
    elif score < 200:
        return "#f39c12"
    else:
        return "#e74c3c"

def short_term_flag(rate):
    if rate == 0:
        return "No daily exceedance"
    elif rate < 0.05:
        return "Occasional exceedance"
    else:
        return "Frequent exceedance"

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Filters")

year_options = ["All"] + sorted(df["Year"].dropna().unique().tolist())

selected_year = st.sidebar.selectbox(
    "Select Year",
    options=year_options,
    index=0
)

if selected_year == "All":
    base_df = df.copy()
    selected_year_label = "All years"
else:
    base_df = df[df["Year"] == selected_year].copy()
    selected_year_label = str(selected_year)

# --------------------------------------------------
# SAFETY CHECK
# --------------------------------------------------

if base_df.empty:
    st.warning("No data available for the selected filter.")
    st.stop()

# --------------------------------------------------
# DAILY AGGREGATION
# --------------------------------------------------
# If dataset already has one row per station-day, this still works safely.
# If there are multiple rows per station-day, this converts them to daily means.

daily_df = (
    base_df
    .groupby(["station", "Town", "Latitude", "Longitude", "Date", "Year"], as_index=False)
    .agg({
        "PM2.5": "mean",
        "PM10": "mean",
        "NO2": "mean",
        "SO2": "mean"
    })
)

# --------------------------------------------------
# STATION-YEAR ANNUAL TABLE
# --------------------------------------------------

station_year = (
    daily_df
    .groupby(["station", "Town", "Latitude", "Longitude", "Year"], as_index=False)
    .agg({
        "PM2.5": "mean",
        "PM10": "mean",
        "NO2": "mean",
        "SO2": "mean",
        "Date": "nunique"
    })
    .rename(columns={"Date": "ValidDays"})
)

# --------------------------------------------------
# COVERAGE LOGIC
# --------------------------------------------------
# For a single selected year: threshold adapts to available days in that year.
# For All: require enough days within each station-year to represent a year fairly.

if selected_year == "All":
    min_valid_days = 200
else:
    total_days_in_scope = daily_df["Date"].dt.normalize().nunique()
    min_valid_days = max(30, int(total_days_in_scope * 0.6))

station_year["CoverageFlag"] = station_year["ValidDays"] >= min_valid_days

if selected_year == "All":
    station_year["CoverageRatio"] = station_year["ValidDays"] / 365.0
else:
    station_year["CoverageRatio"] = station_year["ValidDays"] / max(total_days_in_scope, 1)

station_year = station_year[station_year["CoverageFlag"]].copy()

if station_year.empty:
    st.warning(
        f"No stations meet the minimum data coverage for {selected_year_label}. "
        f"Try another year or choose 'All'."
    )
    st.stop()

# --------------------------------------------------
# WHO ANNUAL CORE INDEX
# --------------------------------------------------

station_year["PM25_ratio"] = station_year["PM2.5"] / WHO_ANNUAL["PM2.5"]
station_year["PM10_ratio"] = station_year["PM10"] / WHO_ANNUAL["PM10"]
station_year["NO2_ratio"]  = station_year["NO2"]  / WHO_ANNUAL["NO2"]

for col in ["PM25_ratio", "PM10_ratio", "NO2_ratio"]:
    station_year[col] = station_year[col].clip(upper=5)

station_year["CoreRiskScore"] = 100 * (
    station_year["PM25_ratio"] +
    station_year["PM10_ratio"] +
    station_year["NO2_ratio"]
) / 3

station_year["CoreRiskLevel"] = station_year["CoreRiskScore"].apply(classify_core_risk)

# --------------------------------------------------
# SO2 SHORT-TERM PRESSURE
# --------------------------------------------------

daily_df["SO2_Exceed"] = daily_df["SO2"] > WHO_SO2_DAILY

station_year_so2 = (
    daily_df
    .groupby(["station", "Town", "Latitude", "Longitude", "Year"], as_index=False)
    .agg(
        SO2_HighDays=("SO2_Exceed", "sum"),
        SO2_ExceedanceRate=("SO2_Exceed", "mean"),
        SO2_AnnualMean=("SO2", "mean")
    )
)

station_year = station_year.merge(
    station_year_so2,
    on=["station", "Town", "Latitude", "Longitude", "Year"],
    how="left"
)

station_year["SO2_HighDays"] = station_year["SO2_HighDays"].fillna(0).astype(int)
station_year["SO2_ExceedanceRate"] = station_year["SO2_ExceedanceRate"].fillna(0)
station_year["SO2_AnnualMean"] = station_year["SO2_AnnualMean"].fillna(0)
station_year["SO2_PressureLevel"] = station_year["SO2_ExceedanceRate"].apply(short_term_flag)

# --------------------------------------------------
# FINAL STATION TABLE
# --------------------------------------------------
# If "All" is selected, average annual station-year scores across valid years.
# If one year is selected, keep that year's station values.

if selected_year == "All":
    station_risk = (
        station_year
        .groupby(["station", "Town", "Latitude", "Longitude"], as_index=False)
        .agg(
            ValidYears=("Year", "nunique"),
            AvgValidDays=("ValidDays", "mean"),
            AvgCoverageRatio=("CoverageRatio", "mean"),
            **{
                "PM2.5": ("PM2.5", "mean"),
                "PM10": ("PM10", "mean"),
                "NO2": ("NO2", "mean"),
                "SO2_AnnualMean": ("SO2_AnnualMean", "mean"),
                "CoreRiskScore": ("CoreRiskScore", "mean"),
                "SO2_HighDays": ("SO2_HighDays", "mean"),
                "SO2_ExceedanceRate": ("SO2_ExceedanceRate", "mean")
            }
        )
    )

    station_risk["SO2_HighDays"] = station_risk["SO2_HighDays"].round(1)
    station_risk["CoreRiskLevel"] = station_risk["CoreRiskScore"].apply(classify_core_risk)
    station_risk["SO2_PressureLevel"] = station_risk["SO2_ExceedanceRate"].apply(short_term_flag)

else:
    station_risk = station_year.copy()
    station_risk["ValidYears"] = 1
    station_risk["AvgValidDays"] = station_risk["ValidDays"]
    station_risk["AvgCoverageRatio"] = station_risk["CoverageRatio"]

ranking = station_risk.sort_values("CoreRiskScore", ascending=False).reset_index(drop=True)

if ranking.empty:
    st.warning("No ranked stations available after processing.")
    st.stop()

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

avg_core_risk = ranking["CoreRiskScore"].mean()
highest_station = ranking.iloc[0]
lowest_station = ranking.sort_values("CoreRiskScore", ascending=True).iloc[0]
so2_exposed_stations = (ranking["SO2_ExceedanceRate"] > 0).sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Annual Core Risk (mean)",
        f"{avg_core_risk:.1f}"
    )

with col2:
    st.metric(
        "Highest Annual Exposure",
        highest_station["station"]
    )

with col3:
    st.metric(
        "Lowest Annual Exposure",
        lowest_station["station"]
    )

with col4:
    st.metric(
        "Stations with SO2 exceedance",
        int(so2_exposed_stations)
    )

# --------------------------------------------------
# INFO BOX
# --------------------------------------------------

st.caption(
    f"Current view: {selected_year_label} | Minimum valid days required per station-year: {min_valid_days}"
)

# --------------------------------------------------
# MAP
# --------------------------------------------------

st.subheader("🗺️ WHO-based Annual Risk Map")

center_lat = ranking["Latitude"].mean()
center_lon = ranking["Longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles="CartoDB positron"
)

for _, row in ranking.iterrows():
    color = risk_color(row["CoreRiskScore"])

    if selected_year == "All":
        popup_html = f"""
        <b>{row['station']}</b><br>
        Town: {row['Town']}<br>
        Annual Core Risk Score (avg): {row['CoreRiskScore']:.1f}<br>
        Annual Risk Class: {row['CoreRiskLevel']}<br>
        PM2.5 annual mean (avg): {row['PM2.5']:.2f} µg/m³<br>
        PM10 annual mean (avg): {row['PM10']:.2f} µg/m³<br>
        NO2 annual mean (avg): {row['NO2']:.2f} µg/m³<br>
        SO2 annual mean (avg): {row['SO2_AnnualMean']:.2f} µg/m³<br>
        SO2 exceedance rate (avg): {row['SO2_ExceedanceRate']*100:.1f}%<br>
        Valid years: {row['ValidYears']}<br>
        Avg valid days/year: {row['AvgValidDays']:.1f}
        """
    else:
        popup_html = f"""
        <b>{row['station']}</b><br>
        Town: {row['Town']}<br>
        Year: {row['Year']}<br>
        Annual Core Risk Score: {row['CoreRiskScore']:.1f}<br>
        Annual Risk Class: {row['CoreRiskLevel']}<br>
        PM2.5 annual mean: {row['PM2.5']:.2f} µg/m³<br>
        PM10 annual mean: {row['PM10']:.2f} µg/m³<br>
        NO2 annual mean: {row['NO2']:.2f} µg/m³<br>
        SO2 annual mean: {row['SO2_AnnualMean']:.2f} µg/m³<br>
        SO2 exceedance days: {row['SO2_HighDays']}<br>
        SO2 exceedance rate: {row['SO2_ExceedanceRate']*100:.1f}%<br>
        Valid days: {row['ValidDays']}<br>
        Coverage ratio: {row['CoverageRatio']*100:.1f}%
        """

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=15,
        popup=popup_html,
        tooltip=f"{row['station']} | Score {row['CoreRiskScore']:.1f}",
        color="white",
        weight=1.5,
        fill=True,
        fill_color=color,
        fill_opacity=0.9
    ).add_to(m)

st_folium(m, width=None, height=600)

# --------------------------------------------------
# CHARTS
# --------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("📈 Annual Core Risk by Station")

    fig_rank = px.bar(
        ranking,
        x="station",
        y="CoreRiskScore",
        color="CoreRiskLevel",
        category_orders={
            "CoreRiskLevel": [
                "Below WHO guideline",
                "1–2× WHO guideline",
                ">2× WHO guideline"
            ]
        },
        title="WHO-based Annual Core Risk Score"
    )

    fig_rank.update_layout(
        xaxis_title="Station",
        yaxis_title="Core Risk Score"
    )

    st.plotly_chart(fig_rank, use_container_width=True)

with right:
    st.subheader("🥧 Annual Risk Class Distribution")

    risk_count = (
        ranking["CoreRiskLevel"]
        .value_counts()
        .rename_axis("CoreRiskLevel")
        .reset_index(name="Count")
    )

    fig_pie = px.pie(
        risk_count,
        names="CoreRiskLevel",
        values="Count",
        hole=0.45,
        title="Station Distribution by Annual Risk Class"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

# --------------------------------------------------
# SO2 CHART
# --------------------------------------------------

st.subheader("🌫️ SO2 Short-Term Pressure")

fig_so2 = px.bar(
    ranking.sort_values("SO2_ExceedanceRate", ascending=False),
    x="station",
    y="SO2_ExceedanceRate",
    color="SO2_PressureLevel",
    title="SO2 Daily Exceedance Rate by Station"
)

fig_so2.update_layout(
    xaxis_title="Station",
    yaxis_title="Exceedance Rate"
)

st.plotly_chart(fig_so2, use_container_width=True)

# --------------------------------------------------
# TOP STATIONS
# --------------------------------------------------

st.subheader("⚠ Highest Annual Exposure Stations")

if selected_year == "All":
    top_cols = [
        "station",
        "Town",
        "ValidYears",
        "CoreRiskScore",
        "CoreRiskLevel",
        "SO2_ExceedanceRate",
        "SO2_PressureLevel"
    ]
else:
    top_cols = [
        "station",
        "Town",
        "Year",
        "ValidDays",
        "CoreRiskScore",
        "CoreRiskLevel",
        "SO2_HighDays",
        "SO2_PressureLevel"
    ]

top3 = ranking.head(3)[top_cols]
st.dataframe(top3, use_container_width=True)

# --------------------------------------------------
# FULL TABLE
# --------------------------------------------------

with st.expander("View full station risk table"):
    if selected_year == "All":
        display_cols = [
            "station",
            "Town",
            "ValidYears",
            "AvgValidDays",
            "AvgCoverageRatio",
            "PM2.5",
            "PM10",
            "NO2",
            "SO2_AnnualMean",
            "CoreRiskScore",
            "CoreRiskLevel",
            "SO2_HighDays",
            "SO2_ExceedanceRate",
            "SO2_PressureLevel"
        ]
    else:
        display_cols = [
            "station",
            "Town",
            "Year",
            "ValidDays",
            "CoverageRatio",
            "PM2.5",
            "PM10",
            "NO2",
            "SO2_AnnualMean",
            "CoreRiskScore",
            "CoreRiskLevel",
            "SO2_HighDays",
            "SO2_ExceedanceRate",
            "SO2_PressureLevel"
        ]

    st.dataframe(
        ranking[display_cols],
        use_container_width=True
    )