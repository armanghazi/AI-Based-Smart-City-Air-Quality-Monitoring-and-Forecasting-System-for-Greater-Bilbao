import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from config import (
    load_data, WHO_ANNUAL, WHO_SO2_DAILY, CORE_POLLUTANTS,
    POLLUTANT_COLOR, ZONE_META, RISK_COLORS,
    classify_core_risk, risk_color,
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Smart City Decision Support",
    page_icon="🏛️",
    layout="wide",
)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]


def model_path(pollutant: str) -> Path:
    prefix = pollutant.replace(".", "").lower()
    return MODELS_DIR / f"xgb_{prefix}_forecast.joblib"


@st.cache_resource
def load_model(pollutant: str):
    path = model_path(pollutant)
    return joblib.load(path) if path.exists() else None


# --------------------------------------------------
# FEATURE PREPARATION (same logic as 5_Forecasting)
# --------------------------------------------------

def prepare_features(df: pd.DataFrame, required_features: list,
                     station_codes: dict | None = None) -> pd.DataFrame:
    """Build every feature the models need (mirrors page 5).
    station_codes: global {station -> code} mapping from the full frame.
    """
    df = df.copy()

    if "wind_u" not in df.columns and "Wind_X" in df.columns:
        df["wind_u"] = df["Wind_X"]
    if "wind_v" not in df.columns and "Wind_Y" in df.columns:
        df["wind_v"] = df["Wind_Y"]

    df["year"]         = df["Date"].dt.year
    df["month"]        = df["Date"].dt.month
    df["day"]          = df["Date"].dt.day
    df["day_of_year"]  = df["Date"].dt.dayofyear
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]  = df["Date"].dt.dayofweek
    df["is_weekend"]   = (df["Date"].dt.weekday >= 5).astype(int)

    # Season → int regardless of incoming dtype (object / string[pyarrow] / numeric)
    season_to_int = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    season_str = df["season"].astype(str).str.capitalize()
    mapped = season_str.map(season_to_int)
    month_season = df["Date"].dt.month.map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    df["season"] = mapped.fillna(month_season).fillna(0).astype(int)

 # station_code MUST use the global training mapping, never a per-subset one.
    # On a single-station frame .cat.codes collapses to 0 -> silent mismatch
    # with the codes the model saw at training (built over all 7 stations).
    if station_codes is not None:
        df["station_code"] = df["station"].map(station_codes).fillna(-1).astype(int)
    else:
        df["station_code"] = df["station"].astype("category").cat.codes

    for pollutant in POLLUTANTS:
        prefix = pollutant.replace(".", "")
        col = f"{prefix}_roll_mean_14"
        if col not in df.columns:
            df[col] = (
                df.groupby("station")[pollutant]
                .transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
            )

    df["wind_x_precip"] = df["WindSpeed"] * df["Precipitation"]
    df["temp_x_humid"]  = df["Temperature"] * df["Humidity"]

    for feat in required_features:
        if feat not in df.columns:
            df[feat] = 0.0
        df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0)

    return df


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_data()
latest_date = df["Date"].max()

# Stable station_code mapping — reproduces training codes.
# (.astype('category').cat.codes assigns codes in sorted order over the full set)
STATION_CODES = {s: i for i, s in enumerate(sorted(df["station"].unique()))}

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏛️ Smart City Decision Support")
st.markdown(
    "Actionable air-quality intelligence for urban decision-makers · "
    f"Latest data: **{latest_date.strftime('%d %b %Y')}**"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🏛️ Analysis Settings")
    st.divider()

    window_label = st.radio(
        "Risk assessment window",
        ["Last 30 days", "Last 90 days", "Last 365 days"],
        index=2,
    )
    window_days = {"Last 30 days": 30, "Last 90 days": 90,
                   "Last 365 days": 365}[window_label]

    st.divider()
    st.markdown("### 🗺️ Zone Legend")
    for z, meta in ZONE_META.items():
        st.markdown(f"{meta['icon']} **{z}**")

# Recent window for risk assessment
window_start = latest_date - pd.Timedelta(days=window_days)
recent = df[df["Date"] >= window_start].copy()

# ==================================================
# SECTION 1 — CITY-WIDE STATUS (KPI row)
# ==================================================

st.markdown(f"## 🚦 City-Wide Status — {window_label}")

# Mean per station/pollutant over the window
station_means = (
    recent.groupby(["station", "Zone"])[POLLUTANTS]
    .mean()
    .reset_index()
)

# Composite risk score per station: mean(value / WHO limit) over core pollutants × 100
def composite_score(row) -> float:
    ratios = [row[p] / WHO_ANNUAL[p] for p in CORE_POLLUTANTS]
    return float(np.mean(ratios) * 100)

station_means["RiskScore"] = station_means.apply(composite_score, axis=1)
station_means["RiskLevel"] = station_means["RiskScore"].apply(classify_core_risk)
station_means = station_means.sort_values("RiskScore", ascending=False)

n_above = (station_means["RiskScore"] >= 100).sum()
worst_station = station_means.iloc[0]

# Worst pollutant city-wide (highest mean WHO ratio)
city_ratios = {
    p: recent[p].mean() / WHO_ANNUAL[p] for p in CORE_POLLUTANTS
}
worst_pollutant = max(city_ratios, key=city_ratios.get)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Stations above WHO", f"{n_above} / {len(station_means)}")
k2.metric("Highest-risk station", worst_station["station"],
          f"{worst_station['RiskScore']:.0f} risk score",
          delta_color="inverse")
k3.metric("Most critical pollutant", worst_pollutant,
          f"{city_ratios[worst_pollutant]:.1f}× WHO limit",
          delta_color="inverse")
k4.metric("Observations analyzed", f"{len(recent):,}")

st.divider()

# ==================================================
# SECTION 2 — TOMORROW'S FORECAST ALERTS
# ==================================================

st.markdown("## 🔮 Next-Day Forecast Alerts")
st.caption(
    "XGBoost forecasts from the latest available data for every station and "
    "pollutant. Limits: WHO 2021 annual guidelines (SO₂: 24-hour guideline)."
)

# Load all 4 models once
bundles = {p: load_model(p) for p in POLLUTANTS}
missing_models = [p for p, b in bundles.items() if b is None]

if missing_models:
    st.warning(f"Models not found for: {', '.join(missing_models)}")

alerts = []
for station in sorted(df["station"].unique()):
    sdf = df[df["station"] == station].sort_values("Date")

    for pollutant in POLLUTANTS:
        bundle = bundles.get(pollutant)
        if bundle is None:
            continue

        feats = bundle["features"]
        prep = prepare_features(sdf, feats, station_codes=STATION_CODES)
        valid = prep.dropna(subset=feats)
        if valid.empty:
            continue

        X_last = valid[feats].iloc[[-1]]
        pred = float(bundle["model"].predict(X_last)[0])
        pred = max(pred, 0.0)

        limit = WHO_SO2_DAILY if pollutant == "SO2" else WHO_ANNUAL.get(pollutant)
        alerts.append({
            "Station":    station,
            "Zone":       sdf["Zone"].iloc[-1],
            "Pollutant":  pollutant,
            "Forecast":   round(pred, 1),
            "WHO limit":  limit,
            "Ratio":      round(pred / limit, 2) if limit else None,
            "Status":     "⚠️ Above WHO" if limit and pred > limit else "✅ OK",
        })

alerts_df = pd.DataFrame(alerts)

if not alerts_df.empty:
    exceed = alerts_df[alerts_df["Status"].str.contains("Above")]

    if exceed.empty:
        st.success("✅ No WHO exceedances forecast for tomorrow at any station.")
    else:
        st.error(
            f"⚠️ {len(exceed)} forecast exceedance(s) tomorrow — "
            f"stations: {', '.join(exceed['Station'].unique())}"
        )

    # Heatmap: forecast ratio per station × pollutant
    pivot = alerts_df.pivot(index="Station", columns="Pollutant", values="Ratio")
    pivot = pivot[POLLUTANTS]

    fig_hm = px.imshow(
        pivot,
        color_continuous_scale=["#2ecc71", "#f9ca24", "#e74c3c"],
        zmin=0, zmax=2,
        text_auto=".2f",
        aspect="auto",
        title="Tomorrow's forecast as ratio of WHO limit (>1 = exceedance)",
    )
    fig_hm.update_layout(height=340, margin=dict(t=50, b=10))
    st.plotly_chart(fig_hm, width="stretch")

    with st.expander("📋 Full forecast table"):
        st.dataframe(
            alerts_df.sort_values("Ratio", ascending=False),
            width="stretch", hide_index=True,
        )

st.divider()

# ==================================================
# SECTION 3 — GeoAI RISK MAP (next-day forecast)
# ==================================================

st.markdown("## 🗺️ GeoAI Risk Map — Tomorrow's Forecast")
st.caption(
    "Marker colour = worst-case forecast as a ratio of the WHO limit "
    "(max across the 4 pollutants). >1 = exceedance."
)

if alerts_df.empty:
    st.info("No forecasts available to map.")
else:
    # Worst-case ratio per station (max over pollutants — keeps SO2's 24h limit in play)
    station_risk = (
        alerts_df.groupby("Station", as_index=False)["Ratio"].max()
        .rename(columns={"Ratio": "MaxRatio"})
    )

    # Coordinates + zone, one row per station (latest known)
    coords = (
        df.sort_values("Date")
        .groupby("station", as_index=False)
        .agg(Latitude=("Latitude", "last"),
             Longitude=("Longitude", "last"),
             Zone=("Zone", "last"))
        .rename(columns={"station": "Station"})
    )
    map_df = station_risk.merge(coords, on="Station", how="left")

    # Tier label from ratio (consistent with the WHO-ratio logic, not raw values)
    def risk_tier(r: float) -> str:
        if r > 2:   return "🔴 High"
        if r > 1:   return "🟡 Moderate"
        return "🟢 Low"
    map_df["Tier"] = map_df["MaxRatio"].apply(risk_tier)

    fig_map = px.scatter_mapbox(
        map_df,
        lat="Latitude", lon="Longitude",
        color="MaxRatio",
        color_continuous_scale=["#2ecc71", "#f9ca24", "#e74c3c"],
        range_color=[0, 2],
        hover_name="Station",
        hover_data={"Zone": True, "Tier": True, "MaxRatio": ":.2f",
                    "Latitude": False, "Longitude": False},
        zoom=10,
        center=dict(lat=map_df["Latitude"].mean(),
                    lon=map_df["Longitude"].mean()),
        height=460,
    )
    fig_map.update_traces(marker=dict(size=18))
    fig_map.update_layout(
        mapbox_style="open-street-map",          # tokenless, Streamlit-Cloud safe
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="× WHO"),
    )
    st.plotly_chart(fig_map, width="stretch")

st.divider()

# ==================================================
# SECTION 4 — STATION RISK PRIORITIZATION
# ==================================================

st.markdown(f"## 📊 Station Risk Prioritization — {window_label}")
st.caption(
    "Composite risk = mean of (concentration ÷ WHO limit) across PM2.5, PM10, "
    "NO₂ · ×100. Below 100 = within guidelines."
)

col_t, col_c = st.columns([3, 2])

with col_t:
    display = station_means.copy()
    display["Rank"] = range(1, len(display) + 1)
    display = display[["Rank", "station", "Zone", "PM2.5", "PM10",
                       "NO2", "SO2", "RiskScore", "RiskLevel"]]
    display[POLLUTANTS] = display[POLLUTANTS].round(1)
    display["RiskScore"] = display["RiskScore"].round(0).astype(int)

    st.dataframe(
        display,
        width="stretch", hide_index=True,
        column_config={
            "RiskScore": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=300, format="%d"
            )
        },
    )

with col_c:
    fig_rank = go.Figure(go.Bar(
        x=station_means["RiskScore"],
        y=station_means["station"],
        orientation="h",
        marker_color=[risk_color(s) for s in station_means["RiskScore"]],
        text=station_means["RiskScore"].round(0).astype(int),
        textposition="outside",
    ))
    fig_rank.add_vline(x=100, line_dash="dash", line_color="#555",
                       annotation_text="WHO")
    fig_rank.update_layout(
        height=340, margin=dict(l=10, r=40, t=30, b=10),
        xaxis_title="Composite risk score",
        yaxis=dict(autorange="reversed"),
        title="Priority ranking",
    )
    st.plotly_chart(fig_rank, width="stretch")

st.divider()

# ==================================================
# SECTION 4 — ZONE-LEVEL RECOMMENDATIONS
# ==================================================

st.markdown("## 🗺️ Zone Recommendations")

ZONE_ACTIONS = {
    "Urban": (
        "Traffic management is the primary lever: low-emission zones, "
        "public-transport incentives, and rush-hour restrictions target the "
        "dominant NO₂ signal."
    ),
    "Industrial": (
        "Coordinate with industrial operators on emission controls and "
        "monitor PM episodes; prioritize continuous stack monitoring."
    ),
    "Port": (
        "Shore-power (cold ironing) for docked vessels and cleaner marine "
        "fuels address the SO₂ contribution from port activity."
    ),
    "Coastal": (
        "Favourable dispersion keeps levels lower; maintain monitoring and "
        "protect green buffers."
    ),
    "Refinery": (
        "Episodic SO₂/PM peaks require event-based alerting rather than "
        "annual averages; coordinate maintenance windows with forecast "
        "conditions (low-wind days are highest risk)."
    ),
}

zone_means = (
    recent.groupby("Zone")[POLLUTANTS].mean().round(1)
    .reindex([z for z in ZONE_META if z in recent["Zone"].unique()])
)

for zone, meta in ZONE_META.items():
    if zone not in zone_means.index:
        continue
    zrow = zone_means.loc[zone]
    key_p = meta["key_pollutant"]
    limit = WHO_SO2_DAILY if key_p == "SO2" else WHO_ANNUAL.get(key_p)
    ratio = zrow[key_p] / limit if limit else None

    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"### {meta['icon']} {zone}")
            st.metric(
                f"Key pollutant: {key_p}",
                f"{zrow[key_p]:.1f} µg/m³",
                f"{ratio:.1f}× WHO" if ratio else None,
                delta_color="inverse" if ratio and ratio > 1 else "normal",
            )
        with c2:
            st.markdown(f"**Profile:** {meta['description']}")
            st.markdown(f"**Recommended action:** {ZONE_ACTIONS.get(zone, '—')}")

st.divider()

# ==================================================
# SECTION 5 — EXPORT
# ==================================================

st.markdown("## 📥 Export for Reporting")

e1, e2 = st.columns(2)

with e1:
    csv_priority = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Station risk ranking (CSV)",
        data=csv_priority,
        file_name=f"risk_ranking_{latest_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        width="stretch",
    )

with e2:
    if not alerts_df.empty:
        csv_alerts = alerts_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Tomorrow's forecast alerts (CSV)",
            data=csv_alerts,
            file_name=f"forecast_alerts_{latest_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width="stretch",
        )

st.caption(
    "Data: Basque Government air-quality network + Open-Meteo (CC BY 4.0) · "
    "Forecasts: XGBoost next-day models (see Forecasting page for validation)."
)