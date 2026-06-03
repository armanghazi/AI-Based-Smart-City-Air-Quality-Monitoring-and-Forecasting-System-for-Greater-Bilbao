import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from config import DATA_FILE

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Urban Air Quality — Greater Bilbao",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

WHO_ANNUAL = {"PM2.5": 5.0, "PM10": 15.0, "NO2": 10.0}
WHO_SO2_DAILY = 40.0
CORE_POLLUTANTS = ["PM2.5", "PM10", "NO2"]

POLLUTANT_UNITS = {
    "NO2": "µg/m³", "PM10": "µg/m³",
    "PM2.5": "µg/m³", "SO2": "µg/m³"
}

POLLUTANT_COLOR = {
    "NO2": "#e74c3c", "PM10": "#e67e22",
    "PM2.5": "#9b59b6", "SO2": "#3498db"
}

RISK_COLORS = {
    "Below WHO guideline": "#2ecc71",
    "1–2× WHO guideline": "#f39c12",
    ">2× WHO guideline": "#e74c3c"
}

RISK_ORDER = ["Below WHO guideline", "1–2× WHO guideline", ">2× WHO guideline"]

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def classify_core_risk(score):
    if score < 100:
        return "Below WHO guideline"
    elif score < 200:
        return "1–2× WHO guideline"
    return ">2× WHO guideline"

def risk_color(score):
    if score < 100:
        return "#2ecc71"
    elif score < 200:
        return "#f39c12"
    return "#e74c3c"

def short_term_flag(rate):
    if rate == 0:
        return "No exceedance"
    elif rate < 0.05:
        return "Occasional"
    return "Frequent"

def who_ratio_label(val, pollutant):
    limit = WHO_ANNUAL.get(pollutant)
    if not limit:
        return "—"
    ratio = val / limit
    return f"{ratio:.1f}×"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["YearMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df

df = load_data()
all_stations = sorted(df["station"].unique().tolist())
all_years = sorted(df["Year"].dropna().unique().tolist())

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🌍 Urban Air Quality")
    st.markdown("Greater Bilbao · WHO 2021 Guidelines")
    st.divider()

    st.markdown("### Filters")

    year_options = ["All years"] + [str(y) for y in all_years]
    selected_year_str = st.selectbox("Year", year_options, index=0)
    selected_year = None if selected_year_str == "All years" else int(selected_year_str)

    selected_stations = st.multiselect(
        "Stations (leave empty = all)",
        options=all_stations,
        default=[]
    )

    st.divider()
    st.markdown("### Display")
    map_mode = st.radio("Map layer", ["Risk score", "Heatmap — PM2.5", "Heatmap — NO2", "Heatmap — PM10"])
    show_who_lines = st.toggle("Show WHO guideline lines in charts", value=True)

    st.divider()
    st.caption("Data: 7 monitoring stations, 2015–2026 (~27k daily records)")

# --------------------------------------------------
# FILTER BASE DATA
# --------------------------------------------------

base_df = df.copy()
if selected_year:
    base_df = base_df[base_df["Year"] == selected_year]
if selected_stations:
    base_df = base_df[base_df["station"].isin(selected_stations)]

if base_df.empty:
    st.warning("No data for selected filters.")
    st.stop()

# --------------------------------------------------
# DAILY AGGREGATION
# --------------------------------------------------

daily_df = (
    base_df
    .groupby(["station", "Town", "Latitude", "Longitude", "Date", "Year", "Month", "YearMonth"], as_index=False)
    .agg({"PM2.5": "mean", "PM10": "mean", "NO2": "mean", "SO2": "mean"})
)

# --------------------------------------------------
# STATION-YEAR ANNUAL TABLE
# --------------------------------------------------

station_year = (
    daily_df
    .groupby(["station", "Town", "Latitude", "Longitude", "Year"], as_index=False)
    .agg({"PM2.5": "mean", "PM10": "mean", "NO2": "mean", "SO2": "mean", "Date": "nunique"})
    .rename(columns={"Date": "ValidDays"})
)

if selected_year:
    total_days = daily_df["Date"].dt.normalize().nunique()
    min_valid = max(30, int(total_days * 0.6))
else:
    min_valid = 200

station_year = station_year[station_year["ValidDays"] >= min_valid].copy()

if station_year.empty:
    st.warning("Not enough data coverage for the selected period. Try 'All years'.")
    st.stop()

# WHO ratios & core risk
for p, col in [("PM2.5", "PM25_ratio"), ("PM10", "PM10_ratio"), ("NO2", "NO2_ratio")]:
    station_year[col] = (station_year[p] / WHO_ANNUAL[p]).clip(upper=5)

station_year["CoreRiskScore"] = 100 * (
    station_year["PM25_ratio"] + station_year["PM10_ratio"] + station_year["NO2_ratio"]
) / 3
station_year["CoreRiskLevel"] = station_year["CoreRiskScore"].apply(classify_core_risk)

# SO2 exceedance
daily_df["SO2_Exceed"] = daily_df["SO2"] > WHO_SO2_DAILY
so2_stats = (
    daily_df
    .groupby(["station", "Town", "Latitude", "Longitude", "Year"], as_index=False)
    .agg(SO2_HighDays=("SO2_Exceed", "sum"), SO2_ExceedanceRate=("SO2_Exceed", "mean"), SO2_AnnualMean=("SO2", "mean"))
)
station_year = station_year.merge(so2_stats, on=["station", "Town", "Latitude", "Longitude", "Year"], how="left")
station_year["SO2_HighDays"] = station_year["SO2_HighDays"].fillna(0).astype(int)
station_year["SO2_ExceedanceRate"] = station_year["SO2_ExceedanceRate"].fillna(0)
station_year["SO2_AnnualMean"] = station_year["SO2_AnnualMean"].fillna(0)
station_year["SO2_PressureLevel"] = station_year["SO2_ExceedanceRate"].apply(short_term_flag)

# Final station risk (average across years if "All")
if not selected_year:
    station_risk = (
        station_year
        .groupby(["station", "Town", "Latitude", "Longitude"], as_index=False)
        .agg(
            ValidYears=("Year", "nunique"),
            AvgValidDays=("ValidDays", "mean"),
            **{p: (p, "mean") for p in CORE_POLLUTANTS},
            SO2_AnnualMean=("SO2_AnnualMean", "mean"),
            CoreRiskScore=("CoreRiskScore", "mean"),
            SO2_HighDays=("SO2_HighDays", "mean"),
            SO2_ExceedanceRate=("SO2_ExceedanceRate", "mean")
        )
    )
    station_risk["CoreRiskLevel"] = station_risk["CoreRiskScore"].apply(classify_core_risk)
    station_risk["SO2_PressureLevel"] = station_risk["SO2_ExceedanceRate"].apply(short_term_flag)
else:
    station_risk = station_year.copy()
    station_risk["ValidYears"] = 1
    station_risk["AvgValidDays"] = station_risk["ValidDays"]

ranking = station_risk.sort_values("CoreRiskScore", ascending=False).reset_index(drop=True)

# --------------------------------------------------
# HEADER & KPIs
# --------------------------------------------------

period_label = selected_year_str if selected_year else "2015–2026 (all years)"
st.title("🌍 Urban Air Quality — Greater Bilbao")
st.caption(f"Period: **{period_label}** · Stations: **{len(ranking)}** · WHO 2021 annual guidelines")

avg_risk = ranking["CoreRiskScore"].mean()
worst = ranking.iloc[0]
best = ranking.sort_values("CoreRiskScore").iloc[0]
so2_stations = int((ranking["SO2_ExceedanceRate"] > 0).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean Core Risk Score", f"{avg_risk:.1f}", help="Average across stations (100 = 1× WHO limit)")
c2.metric("Worst Station", worst["station"], f"Score {worst['CoreRiskScore']:.1f}")
c3.metric("Best Station", best["station"], f"Score {best['CoreRiskScore']:.1f}")
c4.metric("Avg NO₂", f"{ranking['NO2'].mean():.1f} µg/m³", f"WHO limit: {WHO_ANNUAL['NO2']} µg/m³")
c5.metric("SO₂ exceedance stations", f"{so2_stations} / {len(ranking)}")

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
                heat_data,
                radius=40,
                blur=25,
                gradient={0.2: "#2ecc71", 0.5: "#f39c12", 0.8: "#e74c3c", 1.0: "#7b241c"}
            ).add_to(m)

        # Always show station markers
        for _, row in ranking.iterrows():
            color = risk_color(row["CoreRiskScore"])
            score = row["CoreRiskScore"]
            radius = 10 + (score / 50)  # size encodes severity

            popup_lines = [
                f"<b style='font-size:14px'>{row['station']}</b>",
                f"<i>{row['Town']}</i>",
                "<hr style='margin:4px 0'>",
                f"<b>Core Risk Score:</b> {score:.1f} ({row['CoreRiskLevel']})",
                "<hr style='margin:4px 0'>",
                f"PM2.5: <b>{row['PM2.5']:.1f}</b> µg/m³ ({who_ratio_label(row['PM2.5'], 'PM2.5')} WHO)",
                f"PM10: <b>{row['PM10']:.1f}</b> µg/m³ ({who_ratio_label(row['PM10'], 'PM10')} WHO)",
                f"NO₂: <b>{row['NO2']:.1f}</b> µg/m³ ({who_ratio_label(row['NO2'], 'NO2')} WHO)",
                f"SO₂: <b>{row['SO2_AnnualMean']:.1f}</b> µg/m³",
                "<hr style='margin:4px 0'>",
                f"SO₂ exceedance: <b>{row['SO2_ExceedanceRate']*100:.1f}%</b> of days ({row['SO2_PressureLevel']})",
            ]

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=radius,
                popup=folium.Popup("<br>".join(popup_lines), max_width=260),
                tooltip=f"<b>{row['station']}</b> · Score {score:.0f}",
                color="white",
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.88
            ).add_to(m)

            # Station label
            folium.Marker(
                location=[row["Latitude"] + 0.003, row["Longitude"]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px;font-weight:600;color:#2c3e50;white-space:nowrap">{row["station"].split("_")[0]}</div>',
                    icon_size=(120, 20),
                    icon_anchor=(60, 0)
                )
            ).add_to(m)

        st_folium(m, width=None, height=580, returned_objects=[])

    with col_legend:
        st.markdown("#### Legend")
        for level, color in RISK_COLORS.items():
            n = int((ranking["CoreRiskLevel"] == level).sum())
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
                f'<div style="width:18px;height:18px;border-radius:50%;background:{color};flex-shrink:0"></div>'
                f'<span style="font-size:13px">{level}<br><b>{n} station{"s" if n!=1 else ""}</b></span></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("**Score guide**")
        st.markdown("""
<div style='font-size:12px;line-height:1.8'>
• <b>&lt;100</b>: below WHO<br>
• <b>100–200</b>: 1–2× WHO<br>
• <b>&gt;200</b>: &gt;2× WHO<br><br>
Circle <i>size</i> = severity
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**WHO 2021 annual limits**")
        for p, v in WHO_ANNUAL.items():
            st.markdown(f"<div style='font-size:12px'>{p}: <b>{v} µg/m³</b></div>", unsafe_allow_html=True)


# ==================== TAB 2: RISK BREAKDOWN ====================
with tab_risk:
    c_left, c_right = st.columns(2)

    with c_left:
        # Bar chart: core risk score per station
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
            text="CoreRiskScore"
        )
        fig_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        if show_who_lines:
            fig_bar.add_vline(x=100, line_dash="dash", line_color="#f39c12",
                              annotation_text="1× WHO", annotation_position="top")
            fig_bar.add_vline(x=200, line_dash="dash", line_color="#e74c3c",
                              annotation_text="2× WHO", annotation_position="top")
        fig_bar.update_layout(showlegend=True, height=380, margin=dict(l=10, r=60, t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        # Radar / spider chart: pollutant ratios per station
        cats = ["PM2.5 ratio", "PM10 ratio", "NO₂ ratio"]
        fig_radar = go.Figure()

        for _, row in ranking.iterrows():
            vals = [row["PM25_ratio"] if "PM25_ratio" in row.index else row["PM2.5"]/WHO_ANNUAL["PM2.5"],
                    row["PM10_ratio"] if "PM10_ratio" in row.index else row["PM10"]/WHO_ANNUAL["PM10"],
                    row["NO2_ratio"] if "NO2_ratio" in row.index else row["NO2"]/WHO_ANNUAL["NO2"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                name=row["station"].split("_")[0],
                opacity=0.5,
                line=dict(width=1.5)
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title="Pollutant Ratios vs WHO Limit (1.0 = WHO limit)",
            showlegend=True,
            height=380,
            margin=dict(l=30, r=30, t=50, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Grouped bar: individual pollutant means
    st.markdown("#### Individual Pollutant Annual Means vs WHO Guideline")

    pollutant_rows = []
    for _, row in ranking.iterrows():
        for p in CORE_POLLUTANTS:
            pollutant_rows.append({
                "station": row["station"].split("_")[0],
                "Pollutant": p,
                "Concentration": row[p],
                "WHO_Limit": WHO_ANNUAL[p]
            })
    pdf = pd.DataFrame(pollutant_rows)

    fig_grouped = px.bar(
        pdf,
        x="station",
        y="Concentration",
        color="Pollutant",
        barmode="group",
        color_discrete_map=POLLUTANT_COLOR,
        labels={"Concentration": "µg/m³", "station": ""},
        title="Annual mean concentrations by station and pollutant"
    )

    if show_who_lines:
        for p, lim in WHO_ANNUAL.items():
            fig_grouped.add_hline(
                y=lim,
                line_dash="dot",
                line_color=POLLUTANT_COLOR[p],
                annotation_text=f"WHO {p}",
                annotation_font_size=10
            )

    fig_grouped.update_layout(height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig_grouped, use_container_width=True)

    # SO2 exceedance
    st.markdown("#### SO₂ Short-Term Pressure")
    fig_so2 = px.bar(
        ranking.sort_values("SO2_ExceedanceRate", ascending=False),
        x="station",
        y="SO2_ExceedanceRate",
        color="SO2_PressureLevel",
        color_discrete_map={"No exceedance": "#2ecc71", "Occasional": "#f39c12", "Frequent": "#e74c3c"},
        labels={"SO2_ExceedanceRate": "Exceedance rate (fraction)", "station": ""},
        title="Fraction of days exceeding WHO SO₂ 24h guideline (40 µg/m³)",
        text_auto=".1%"
    )
    fig_so2.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig_so2, use_container_width=True)


# ==================== TAB 3: TRENDS ====================
with tab_trend:
    st.markdown("### Annual trends — all stations")

    # Yearly means per station
    yearly = (
        df.groupby(["Year", "station"], as_index=False)
        .agg({"PM2.5": "mean", "PM10": "mean", "NO2": "mean", "SO2": "mean"})
    )
    if selected_stations:
        yearly = yearly[yearly["station"].isin(selected_stations)]

    pollutant_choice = st.selectbox("Pollutant", CORE_POLLUTANTS + ["SO2"], index=0, key="trend_poll")

    fig_trend = px.line(
        yearly,
        x="Year",
        y=pollutant_choice,
        color="station",
        markers=True,
        title=f"{pollutant_choice} annual mean per station (µg/m³)",
        labels={pollutant_choice: f"{pollutant_choice} µg/m³"}
    )

    if show_who_lines and pollutant_choice in WHO_ANNUAL:
        fig_trend.add_hline(
            y=WHO_ANNUAL[pollutant_choice],
            line_dash="dash",
            line_color="red",
            annotation_text=f"WHO limit {WHO_ANNUAL[pollutant_choice]} µg/m³",
            annotation_position="bottom right"
        )

    fig_trend.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Area chart: city-wide monthly average
    st.markdown("### City-wide monthly average (all stations combined)")

    monthly_all = (
        df.groupby("YearMonth", as_index=False)
        .agg({"PM2.5": "mean", "PM10": "mean", "NO2": "mean", "SO2": "mean"})
    )
    monthly_all = monthly_all.sort_values("YearMonth")

    fig_area = go.Figure()
    for p in CORE_POLLUTANTS:
        fig_area.add_trace(go.Scatter(
            x=monthly_all["YearMonth"],
            y=monthly_all[p].rolling(3, center=True, min_periods=1).mean(),
            name=p,
            fill="tozeroy",
            line=dict(color=POLLUTANT_COLOR[p], width=1.5),
            opacity=0.4
        ))

    if show_who_lines:
        for p, lim in WHO_ANNUAL.items():
            fig_area.add_hline(y=lim, line_dash="dot", line_color=POLLUTANT_COLOR[p],
                               annotation_text=f"WHO {p}", annotation_font_size=9, opacity=0.6)

    fig_area.update_layout(
        title="3-month rolling average — all stations (µg/m³)",
        xaxis_title="Date",
        yaxis_title="µg/m³",
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig_area, use_container_width=True)


# ==================== TAB 4: SEASONAL ====================
with tab_seasonal:
    st.markdown("### Monthly seasonality")

    monthly_station = (
        daily_df
        .groupby(["station", "Month"], as_index=False)
        .agg({"PM2.5": "mean", "PM10": "mean", "NO2": "mean", "SO2": "mean"})
    )

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly_station["MonthName"] = monthly_station["Month"].map(month_names)

    poll_s = st.selectbox("Pollutant", CORE_POLLUTANTS + ["SO2"], index=2, key="seas_poll")

    fig_season = px.line(
        monthly_station,
        x="Month",
        y=poll_s,
        color="station",
        markers=True,
        title=f"{poll_s} monthly mean by station",
        labels={poll_s: f"{poll_s} µg/m³", "Month": "Month"}
    )
    fig_season.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=list(month_names.values())
    )
    if show_who_lines and poll_s in WHO_ANNUAL:
        fig_season.add_hline(y=WHO_ANNUAL[poll_s], line_dash="dash", line_color="red",
                             annotation_text="WHO limit")
    fig_season.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig_season, use_container_width=True)

    # Heatmap: station × month
    st.markdown("### Station × Month heatmap")

    pivot = monthly_station.pivot(index="station", columns="Month", values=poll_s)
    pivot.columns = [month_names[c] for c in pivot.columns]

    fig_hm = px.imshow(
        pivot,
        color_continuous_scale=["#2ecc71", "#f9ca24", "#e74c3c"],
        aspect="auto",
        title=f"{poll_s} µg/m³ — station × month",
        labels={"color": f"{poll_s} µg/m³"}
    )
    fig_hm.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_hm, use_container_width=True)


# ==================== TAB 5: TABLE ====================
with tab_table:
    st.markdown("### Station risk summary table")

    display_cols = [
        "station", "Town", "CoreRiskScore", "CoreRiskLevel",
        "PM2.5", "PM10", "NO2", "SO2_AnnualMean",
        "SO2_ExceedanceRate", "SO2_PressureLevel"
    ]
    if selected_year:
        display_cols = ["Year", "ValidDays"] + display_cols
    else:
        display_cols = ["ValidYears", "AvgValidDays"] + display_cols

    display_df = ranking[[c for c in display_cols if c in ranking.columns]].copy()

    # Format floats
    float_cols = ["CoreRiskScore", "PM2.5", "PM10", "NO2", "SO2_AnnualMean", "SO2_ExceedanceRate", "AvgValidDays"]
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "CoreRiskScore": st.column_config.ProgressColumn(
                "Core Risk Score",
                min_value=0,
                max_value=display_df["CoreRiskScore"].max() * 1.1,
                format="%.1f"
            ),
            "SO2_ExceedanceRate": st.column_config.ProgressColumn(
                "SO₂ Exceedance Rate",
                min_value=0,
                max_value=1,
                format="%.1%"
            )
        }
    )

    st.markdown("---")
    st.markdown("### Daily raw data explorer")

    station_sel = st.selectbox("Select station", options=all_stations)
    raw = daily_df[daily_df["station"] == station_sel][["Date", "PM2.5", "PM10", "NO2", "SO2"]].copy()
    raw = raw.sort_values("Date")

    fig_raw = go.Figure()
    for p in CORE_POLLUTANTS:
        fig_raw.add_trace(go.Scatter(x=raw["Date"], y=raw[p], name=p,
                                     line=dict(color=POLLUTANT_COLOR[p], width=1),
                                     opacity=0.8))
    if show_who_lines:
        for p, lim in WHO_ANNUAL.items():
            fig_raw.add_hline(y=lim, line_dash="dot", line_color=POLLUTANT_COLOR[p],
                              annotation_text=f"WHO {p}", annotation_font_size=9)

    fig_raw.update_layout(
        title=f"Daily concentrations — {station_sel}",
        xaxis_title="Date",
        yaxis_title="µg/m³",
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig_raw, use_container_width=True)

    with st.expander("Show raw daily data"):
        st.dataframe(raw, use_container_width=True, hide_index=True)