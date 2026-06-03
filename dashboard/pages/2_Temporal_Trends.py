import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config import DATA_FILE

# -----------------------
# Constants
# -----------------------

WHO_ANNUAL = {"PM2.5": 5.0, "PM10": 15.0, "NO2": 10.0}

POLLUTANT_COLOR = {
    "NO2": "#e74c3c", "PM10": "#e67e22",
    "PM2.5": "#9b59b6", "SO2": "#3498db"
}

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

COVID_PERIODS = ["Pre-COVID", "COVID", "Post-COVID"]

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

# -----------------------
# Page Title
# -----------------------

st.title("📈 Temporal Trends Dashboard")
st.caption("Greater Bilbao · 2015–2026 · WHO 2021 guidelines")

# -----------------------
# Sidebar
# -----------------------

with st.sidebar:
    st.markdown("### Filters")

    pollutant = st.selectbox(
        "Pollutant",
        ["PM2.5", "PM10", "NO2", "SO2"]
    )

    station_options = ["ALL"] + sorted(df["station"].unique().tolist())
    selected_station = st.selectbox("Station", station_options)

    show_who = st.toggle("Show WHO guideline", value=True)

    st.divider()
    if pollutant in WHO_ANNUAL:
        st.markdown(f"**WHO annual limit ({pollutant}):** {WHO_ANNUAL[pollutant]} µg/m³")
    else:
        st.markdown("SO₂: evaluated on 24h exceedance, no annual WHO limit")

# -----------------------
# Filter
# -----------------------

df_plot = df[df["station"] == selected_station].copy() if selected_station != "ALL" else df.copy()

unit = "µg/m³"
who_limit = WHO_ANNUAL.get(pollutant)
main_color = POLLUTANT_COLOR[pollutant]

# -----------------------
# KPIs
# -----------------------

avg_val = df_plot[pollutant].mean()
max_val = df_plot[pollutant].max()
min_val = df_plot[pollutant].min()

who_ratio = f"{avg_val / who_limit:.1f}× WHO" if who_limit else "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Average {pollutant}", f"{avg_val:.1f} {unit}")
c2.metric("Maximum (daily)", f"{max_val:.1f} {unit}")
c3.metric("Minimum (daily)", f"{min_val:.1f} {unit}")
c4.metric("Avg vs WHO limit", who_ratio)

st.divider()

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

fig_annual = px.line(
    annual,
    x="Year",
    y=pollutant,
    markers=True,
    title=f"{pollutant} annual mean — {selected_station}",
    labels={pollutant: f"{pollutant} ({unit})", "Year": "Year"},
    color_discrete_sequence=[main_color]
)

if show_who and who_limit:
    fig_annual.add_hline(
        y=who_limit,
        line_dash="dash",
        line_color="red",
        annotation_text=f"WHO limit {who_limit} µg/m³",
        annotation_position="top right",
        annotation_font_size=11
    )

fig_annual.update_layout(height=360, hovermode="x unified")
st.plotly_chart(fig_annual, use_container_width=True)

# -----------------------
# Monthly Seasonality
# -----------------------

st.subheader("🌦️ Monthly Seasonality")

monthly = (
    df_plot
    .groupby("Month")[pollutant]
    .agg(mean="mean", std="std")
    .reset_index()
)
monthly["MonthName"] = monthly["Month"].map(MONTH_NAMES)

fig_monthly = go.Figure()

# Error bars (±1 std)
fig_monthly.add_trace(go.Bar(
    x=monthly["MonthName"],
    y=monthly["mean"],
    error_y=dict(type="data", array=monthly["std"].fillna(0), visible=True),
    marker_color=main_color,
    marker_opacity=0.8,
    name=pollutant
))

if show_who and who_limit:
    fig_monthly.add_hline(
        y=who_limit,
        line_dash="dash",
        line_color="red",
        annotation_text=f"WHO {who_limit} µg/m³",
        annotation_font_size=11
    )

fig_monthly.update_layout(
    title=f"{pollutant} monthly mean ± std — {selected_station}",
    xaxis_title="Month",
    yaxis_title=f"{pollutant} ({unit})",
    height=360,
    showlegend=False
)

st.plotly_chart(fig_monthly, use_container_width=True)

# -----------------------
# Station Comparison (only when ALL)
# -----------------------

if selected_station == "ALL":
    st.subheader("🏙️ Station Comparison")

    comparison = (
        df
        .groupby(["Year", "station"])[pollutant]
        .mean()
        .reset_index()
    )

    fig_comp = px.line(
        comparison,
        x="Year",
        y=pollutant,
        color="station",
        markers=True,
        title=f"{pollutant} annual mean per station",
        labels={pollutant: f"{pollutant} ({unit})"}
    )

    if show_who and who_limit:
        fig_comp.add_hline(
            y=who_limit,
            line_dash="dash",
            line_color="red",
            annotation_text=f"WHO limit {who_limit} µg/m³",
            annotation_position="bottom right",
            annotation_font_size=11
        )

    fig_comp.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig_comp, use_container_width=True)

# -----------------------
# COVID Impact
# -----------------------

st.subheader("🦠 COVID Impact Analysis")

covid_df = df_plot.copy()
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
    covid_df["Period"],
    categories=COVID_PERIODS,
    ordered=True
)

covid_stats = (
    covid_df
    .groupby("Period")[pollutant]
    .agg(["mean", "median", "std", "count"])
    .rename(columns={"mean": "Mean", "median": "Median", "std": "Std", "count": "Days"})
    .reset_index()
    .sort_values("Period")
)

col_chart, col_table = st.columns([2, 1])

with col_chart:
    # Annotate % change vs Pre-COVID
    pre_val = covid_stats.loc[covid_stats["Period"] == "Pre-COVID", "Mean"].values
    annotations = []
    if len(pre_val) > 0:
        base = pre_val[0]
        for _, r in covid_stats.iterrows():
            if r["Period"] != "Pre-COVID" and base > 0:
                pct = (r["Mean"] - base) / base * 100
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
        title=f"Average {pollutant} by COVID period",
        labels={"Mean": f"{pollutant} ({unit})"},
        error_y="Std"
    )
    fig_covid.update_layout(
        annotations=annotations,
        showlegend=False,
        height=360
    )

    if show_who and who_limit:
        fig_covid.add_hline(
            y=who_limit,
            line_dash="dash",
            line_color="red",
            annotation_text=f"WHO {who_limit} µg/m³",
            annotation_font_size=11
        )

    st.plotly_chart(fig_covid, use_container_width=True)

with col_table:
    st.markdown("**Statistics by period**")
    display_stats = covid_stats.copy()
    for col in ["Mean", "Median", "Std"]:
        display_stats[col] = display_stats[col].round(2)

    st.dataframe(
        display_stats,
        use_container_width=True,
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