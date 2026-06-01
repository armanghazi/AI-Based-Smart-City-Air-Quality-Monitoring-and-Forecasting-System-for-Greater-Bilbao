import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -----------------------
# Load Data
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "final_air_quality.parquet"
)

@st.cache_data
def load_data():

    df = pd.read_parquet(DATA_FILE)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    return df

df = load_data()

# -----------------------
# Page Title
# -----------------------

st.title("📈 Temporal Trends Dashboard")

# -----------------------
# Sidebar
# -----------------------

pollutant = st.sidebar.selectbox(
    "Pollutant",
    ["PM2.5", "PM10", "NO2", "SO2"]
)

stations = ["ALL"] + sorted(
    df["station"].unique().tolist()
)

selected_station = st.sidebar.selectbox(
    "Station",
    stations
)

# -----------------------
# Filter
# -----------------------

if selected_station != "ALL":

    df_plot = df[
        df["station"] == selected_station
    ]

else:

    df_plot = df.copy()

# -----------------------
# KPI
# -----------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average",
    f"{df_plot[pollutant].mean():.1f}"
)

col2.metric(
    "Maximum",
    f"{df_plot[pollutant].max():.1f}"
)

col3.metric(
    "Minimum",
    f"{df_plot[pollutant].min():.1f}"
)

# -----------------------
# Annual Trend
# -----------------------

st.subheader("📅 Annual Trend")

annual = (
    df_plot
    .groupby("Year")[pollutant]
    .mean()
    .reset_index()
)

fig = px.line(
    annual,
    x="Year",
    y=pollutant,
    markers=True
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------
# Monthly Seasonality
# -----------------------

st.subheader("🌦 Monthly Seasonality")

monthly = (
    df_plot
    .groupby("Month")[pollutant]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    monthly,
    x="Month",
    y=pollutant
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# -----------------------
# Station Comparison
# -----------------------

if selected_station == "ALL":

    st.subheader(
        "🏙 Station Comparison"
    )

    comparison = (
        df
        .groupby(
            ["Year","station"]
        )[pollutant]
        .mean()
        .reset_index()
    )

    fig3 = px.line(
        comparison,
        x="Year",
        y=pollutant,
        color="station"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# -----------------------
# COVID Analysis
# -----------------------

st.subheader("🦠 COVID Impact")

covid_df = df_plot.copy()

covid_df["Period"] = "Post-COVID"

covid_df.loc[
    covid_df["Year"] <= 2019,
    "Period"
] = "Pre-COVID"

covid_df.loc[
    covid_df["Year"].between(2020, 2021),
    "Period"
] = "COVID"

covid_summary = (
    covid_df
    .groupby("Period")[pollutant]
    .mean()
    .reset_index()
)

fig4 = px.bar(
    covid_summary,
    x="Period",
    y=pollutant
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

st.dataframe(
    covid_summary,
    use_container_width=True
)