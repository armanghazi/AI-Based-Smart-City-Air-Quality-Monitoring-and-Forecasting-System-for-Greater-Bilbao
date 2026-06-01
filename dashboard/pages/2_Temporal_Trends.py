import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config import DATA_FILE

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

covid_df["Period"] = np.select(
    [
        covid_df["Year"] <= 2019,
        covid_df["Year"].between(2020, 2021),
        covid_df["Year"] >= 2022
    ],
    [
        "Pre-COVID",
        "COVID",
        "Post-COVID"
    ],
    default="Unknown"
)

period_order = ["Pre-COVID", "COVID", "Post-COVID"]
covid_df["Period"] = pd.Categorical(
    covid_df["Period"],
    categories=period_order,
    ordered=True
)

covid_mean = (
    covid_df.groupby("Period", as_index=False)[pollutant]
    .mean()
    .sort_values("Period")
)

fig4 = px.bar(
    covid_mean,
    x="Period",
    y=pollutant,
    color="Period",
    category_orders={"Period": period_order},
    title=f"Average {pollutant} by Period"
)

st.plotly_chart(fig4, use_container_width=True)
st.dataframe(covid_mean, use_container_width=True)

covid_stats = (
    covid_df.groupby("Period", as_index=False)
    .agg(
        mean_value=(pollutant, "mean"),
        median_value=(pollutant, "median"),
        std_value=(pollutant, "std"),
        count_value=(pollutant, "count")
    )
    .sort_values("Period")
)

st.dataframe(covid_stats, use_container_width=True)