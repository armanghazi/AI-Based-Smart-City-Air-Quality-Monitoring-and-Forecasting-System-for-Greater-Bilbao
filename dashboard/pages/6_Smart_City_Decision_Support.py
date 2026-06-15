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
from forecast_utils import prepare_features
from spatial_utils import idw_grid
from gauge_component import render_gauge_row
from aqi import overall_aqi, compute_aqi_category, AQI_POLLUTANTS, AQI_CATEGORIES
from aqi_components import (render_aqi_donut, render_station_aqi_cards,
                            render_aqi_calendar)


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
# LOAD DATA
# --------------------------------------------------

df = load_data()
latest_date = df["Date"].max()
forecast_date = latest_date + pd.Timedelta(days=1)

# Stable station_code mapping — reproduces training codes.
# (.astype('category').cat.codes assigns codes in sorted order over the full set)
STATION_CODES = {s: i for i, s in enumerate(sorted(df["station"].unique()))}

# Latest-day city-wide mean per pollutant
latest_day = df[df["Date"] == df["Date"].max()]
current_values = {
    p: float(latest_day[p].mean()) for p in ["PM2.5", "PM10", "NO2", "SO2"]
}

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

tab_status, tab_forecast, tab_action = st.tabs([
    "🚦 Current Status",
    "🔮 Forecast & Map",
    "🎯 Decisions & Actions",
])
with tab_status:
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

    latest_day = df[df["Date"] == df["Date"].max()]
    current_values = {p: float(latest_day[p].mean())
                      for p in ["PM2.5", "PM10", "NO2", "SO2"]}
    fig_gauges = render_gauge_row(current_values, WHO_ANNUAL, WHO_SO2_DAILY)
    st.plotly_chart(fig_gauges, width="stretch")

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

    col_d, col_c = st.columns([1, 2])
    with col_d:
        render_aqi_donut(current_values, title="City-wide AQI Today")
        st.caption(
            "European/ICA standard — 6 levels from Good to Extremely poor. "
            f"International reference: US EPA AQI ≈ {overall_aqi(current_values).get('epa_aqi', '—')} "
            f"({overall_aqi(current_values).get('epa_label', '—')}). "
            "⚠️ Daily-mean approximation — official indices use shorter windows."
        )
    with col_c:
        st.markdown("##### Air quality by station")
        render_station_aqi_cards(df, n_cols=4)

    st.divider()
    st.markdown("#### 📅 Pollution Calendar")
    cc1, cc2 = st.columns(2)
    with cc1:
        cal_station = st.selectbox("Station", sorted(df["station"].unique()),
                                   key="cal_st")
    with cc2:
        cal_year = st.selectbox("Year",
                                sorted(df["Date"].dt.year.unique(), reverse=True),
                                key="cal_yr")
    render_aqi_calendar(df, cal_station, cal_year)

    with st.expander("ℹ️ Why two different air quality indicators?"):
        st.markdown("""
**AQI (European/ICA)** answers: *"Is the air safe to go outside today?"*
It uses short-term thresholds calibrated for daily public communication.
Good = safe for everyone; Extremely poor = health alert.

**WHO Risk Score** answers: *"How far are we from the ideal long-term target?"*
It compares annual mean concentrations against the WHO 2021 annual guidelines
(PM2.5=5, PM10=15, NO2=10 µg/m³) — which are very strict.

**Why can they differ?** PM2.5 = 12 µg/m³ →
AQI says "Fairly good" (within daily safe range) but
WHO Risk shows 2.4× the annual limit (above the strict ideal).
Both are correct — they answer different questions.

Most European cities, including healthy ones, routinely exceed WHO annual
targets. The WHO score tracks progress toward an ambitious public-health goal,
not a daily alert threshold.

*US EPA reference:* The EPA AQI (0–500 continuous scale) is shown alongside
the European index as an internationally familiar reference point.
        """)

    # Latest-day city-wide mean per pollutant (used by existing AQI block below)
    latest_day = df[df["Date"] == df["Date"].max()]
    current_values = {
        p: float(latest_day[p].mean()) for p in ["PM2.5", "PM10", "NO2", "SO2"]
    }


    st.markdown("### 🌍 Air Quality Index (European / Spanish ICA)")

    # Latest-day city-wide mean per pollutant
    latest_day = df[df["Date"] == df["Date"].max()]
    city_values = {p: float(latest_day[p].mean()) for p in AQI_POLLUTANTS}

    aqi = overall_aqi(city_values)

    if aqi:
        st.markdown(
            f"""
            <div style="
                background: {aqi['color']};
                border-radius: 16px;
                padding: 24px;
                text-align: center;
                color: white;
                text-shadow: 0 1px 3px rgba(0,0,0,0.4);
            ">
                <div style="font-size: 1.1rem; opacity: 0.9;">City-wide Air Quality</div>
                <div style="font-size: 2.6rem; font-weight: 800; margin: 4px 0;">
                    {aqi['label']}
                </div>
                <div style="font-size: 0.95rem;">
                    Driven by {aqi['driver']} · {aqi['value']} µg/m³
                </div>
                <div style="font-size: 0.85rem; margin-top: 8px; opacity: 0.95;">
                    {aqi['advice']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "European Air Quality Index (Spanish ICA methodology) · daily-mean "
        "approximation · overall level = worst pollutant."
    )


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

with tab_forecast:
    # ==================================================
    # SECTION 2 — TOMORROW'S FORECAST ALERTS
    # ==================================================

    st.markdown(f"## 🔮 Forecast Alerts — {forecast_date.strftime('%d %b %Y')}")
    st.caption(
        f"XGBoost next-day forecast based on latest available data "
        f"({latest_date.strftime('%d %b %Y')}). "
        "Limits: WHO 2021 annual guidelines (SO₂: 24-hour guideline)."
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
            prep_filled = prep.copy()
            prep_filled[feats] = prep_filled[feats].ffill()
            valid = prep_filled.dropna(subset=feats)
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
            st.success(
                f"✅ No WHO exceedances forecast for "
                f"{forecast_date.strftime('%d %b %Y')} at any station."
            )
        else:
            st.error(
                f"⚠️ {len(exceed)} forecast exceedance(s) for "
                f"{forecast_date.strftime('%d %b %Y')} — "
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
            title=f"Forecast for {forecast_date.strftime('%d %b %Y')} — ratio of WHO limit (>1 = exceedance)",        )
       
        fig_hm.update_xaxes(side="top", ticklabelstandoff=40) 
        fig_hm.update_layout(
            height=400, 
            margin=dict(t=120, b=10)              )

        st.plotly_chart(fig_hm, use_container_width=True)  

        with st.expander("📋 Full forecast table"):
            st.dataframe(
                alerts_df.sort_values("Ratio", ascending=False),
                width="stretch", hide_index=True,
            )

    st.divider()

    # ==================================================
    # SECTION 3 — GeoAI RISK MAP (next-day forecast)
    # ==================================================

    st.markdown(f"## 🗺️ GeoAI Risk Map — Forecast for {forecast_date.strftime('%d %b %Y')}")
    st.caption(
            f"Based on latest available data ({latest_date.strftime('%d %b %Y')}). "
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

        # ==================================================
        # IDW INTERPOLATION SURFACE
        # ==================================================

        st.markdown("## 🌡️ Interpolated Forecast Surface (IDW)")
        st.caption(
            "⚠️ **Interpolated surface (IDW) from 7 station forecasts — "
            "colour = worst-case forecast ratio (× WHO limit), same scale as markers. "
            "Visual approximation only: terrain, local sources, and road density are not modelled.**"
        )

        _lats   = map_df["Latitude"].values
        _lons   = map_df["Longitude"].values
        _ratios = map_df["MaxRatio"].values

  
        _grid_lats, _grid_lons, _z_grid = idw_grid(_lats, _lons, _ratios)

        # Flatten the regular grid to lists for the Scattermapbox surface layer.
        _glat_mesh, _glon_mesh = np.meshgrid(_grid_lats, _grid_lons, indexing="ij")
        _flat_lats = _glat_mesh.ravel()
        _flat_lons = _glon_mesh.ravel()
        _flat_z    = _z_grid.ravel()

        # Shared colour scale — identical to the risk map above for consistency.
        _IDW_CS = [
            [0.00, "#2ecc71"],   # 0× — green
            [0.48, "#7fd957"],   # above 0× — light green 
            [0.50, "#f4d03f"],   #  1× WHO — yellow
            [0.65, "#e67e22"],   # ~1.3× — orange
            [0.85, "#e74c3c"],   # ~1.7× — red
            [1.00, "#c0392b"],   # 2× — dark red 
        ]

        fig_idw = go.Figure()

        # Layer 1: IDW value surface.
        # Each grid point is coloured by its interpolated MaxRatio value (same quantity
        # as the station markers), NOT by KDE density.  size=8px at zoom 10 covers one
        # grid step (~0.004°, ~3 px) with 2-3 px overlap so there are no visible gaps.
        fig_idw.add_trace(go.Scattermapbox(
            lat=_flat_lats,
            lon=_flat_lons,
            mode="markers",
            marker=dict(
                size=20,
                color=_flat_z,
                colorscale=_IDW_CS,
                cmin=0,
                cmax=2,
                opacity=0.65,
                showscale=True,
                colorbar=dict(title="× WHO"),
            ),
            hoverinfo="skip",
            name="IDW surface",
        ))

        # Layer 2: station markers on top showing actual forecast values.
        fig_idw.add_trace(go.Scattermapbox(
            lat=map_df["Latitude"],
            lon=map_df["Longitude"],
            mode="markers+text",
            marker=dict(
                size=16,
                color=map_df["MaxRatio"],
                colorscale=_IDW_CS,
                cmin=0,
                cmax=2,
                showscale=False,
                
            ),
            text=map_df["Station"],
            textposition="top center",
            customdata=map_df[["MaxRatio"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Max forecast: %{customdata[0]:.2f}× WHO<extra></extra>"
            ),
            name="Station forecasts",
        ))

        fig_idw.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(
                    lat=float(map_df["Latitude"].mean()),
                    lon=float(map_df["Longitude"].mean()),
                ),
                zoom=10,
            ),
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_idw, width="stretch")

    st.divider()

with tab_action:

    zone_means = (
    recent.groupby("Zone")[POLLUTANTS].mean().round(1)
    .reindex([z for z in ZONE_META if z in recent["Zone"].unique()])
)
    

     # ==================================================
    # SECTION 6 — SCENARIO SIMULATOR
    # ==================================================

    st.markdown("## 🧪 Scenario Simulator")
    st.caption(
        "Explore how the composite risk score responds to interventions. "
        "Two modes with different epistemic status — read the labels."
    )

    sim_mode = st.radio(
        "Simulation mode",
        ["Model-based counterfactual", "Policy elasticity assumption"],
        horizontal=True,
    )

    sim_station = st.selectbox(
        "Station to simulate", sorted(df["station"].unique()), key="sim_station"
    )

    sdf_sim = df[df["station"] == sim_station].sort_values("Date")

    # --- Baseline forecast for all 4 pollutants at this station ---
    baseline = {}
    for pollutant in POLLUTANTS:
        bundle = bundles.get(pollutant)
        if bundle is None:
            continue
        feats = bundle["features"]
        prep = prepare_features(sdf, feats, station_codes=STATION_CODES)
        prep_filled = prep.copy()
        prep_filled[feats] = prep_filled[feats].ffill()
        valid = prep_filled.dropna(subset=feats)
        if valid.empty:
            continue
        X_last = valid[feats].iloc[[-1]]
        baseline[pollutant] = max(float(bundle["model"].predict(X_last)[0]), 0.0)


    def composite_from_preds(pred_dict: dict) -> float:
        """Composite risk score from a {pollutant: value} dict — core pollutants only."""
        ratios = [pred_dict[p] / WHO_ANNUAL[p] for p in CORE_POLLUTANTS if p in pred_dict]
        return float(np.mean(ratios) * 100) if ratios else 0.0


    baseline_score = composite_from_preds(baseline)

    # ----------------------------------------------------------
    if sim_mode == "Model-based counterfactual":
        st.info(
            "🟢 **Honest what-if:** perturbs features the model actually uses "
            "(weather + today's levels), then re-runs `model.predict()`. "
            "This is a real model output."
        )

        c1, c2, c3 = st.columns(3)
        wind_mult   = c1.slider("Wind speed ×", 0.5, 2.0, 1.0, 0.1)
        precip_mult = c2.slider("Precipitation ×", 0.0, 3.0, 1.0, 0.1)
        today_mult  = c3.slider("Today's pollutant level ×", 0.5, 1.5, 1.0, 0.05)

        scenario = {}
        for pollutant in POLLUTANTS:
            bundle = bundles.get(pollutant)
            if bundle is None:
                continue
            feats = bundle["features"]
            prep = prepare_features(sdf_sim, feats, station_codes=STATION_CODES)
            prep_filled = prep.copy()
            prep_filled[feats] = prep_filled[feats].ffill()
            valid = prep_filled.dropna(subset=feats)
            if valid.empty:
                continue

            row = valid.iloc[[-1]].copy()
            # Scale all wind representations together: preserves direction, changes magnitude.
            # Models use WindSpeed, wind_u, and wind_v as separate features; all must move.
            for _wcol in ("WindSpeed", "wind_u", "wind_v"):
                if _wcol in row.columns:
                    row[_wcol] = row[_wcol] * wind_mult
            if "Precipitation" in row.columns:
                row["Precipitation"] = row["Precipitation"] * precip_mult
            if pollutant in row.columns:
                row[pollutant] = row[pollutant] * today_mult

            X = row[feats]
            scenario[pollutant] = max(float(bundle["model"].predict(X)[0]), 0.0)

        scenario_score = composite_from_preds(scenario)

    else:  # Policy elasticity assumption
        st.warning(
            "🟡 **Policy elasticity — NOT a model prediction.** The model has no "
            "traffic/emission features, so this applies an *assumed* linear "
            "response (e.g. −20% traffic → −20% NO₂). Use for illustrative "
            "policy framing only; the elasticity is a stated assumption."
        )

        c1, c2 = st.columns(2)
        traffic_cut    = c1.slider("Traffic reduction (%)", 0, 50, 0, 5)
        emission_cut   = c2.slider("Industrial emission reduction (%)", 0, 50, 0, 5)

        # Stated, explicit elasticities — not 1:1, to avoid the "where did 20→20 come from" trap
        NO2_ELASTICITY  = 0.7   # assumption: NO2 ~70% traffic-attributable
        PM_ELASTICITY   = 0.4   # assumption: PM partly traffic, partly background
        SO2_ELASTICITY  = 0.6   # assumption: SO2 ~industrial-attributable

        scenario = dict(baseline)
        if "NO2" in scenario:
            scenario["NO2"]   *= (1 - NO2_ELASTICITY  * traffic_cut  / 100)
        if "PM2.5" in scenario:
            scenario["PM2.5"] *= (1 - PM_ELASTICITY   * traffic_cut  / 100)
        if "PM10" in scenario:
            scenario["PM10"]  *= (1 - PM_ELASTICITY   * traffic_cut  / 100)
        if "SO2" in scenario:
            scenario["SO2"]   *= (1 - SO2_ELASTICITY  * emission_cut / 100)

        scenario_score = composite_from_preds(scenario)

    # ----------------------------------------------------------
    # Results
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline risk score", f"{baseline_score:.0f}")
    m2.metric("Scenario risk score", f"{scenario_score:.0f}")
    delta = (scenario_score - baseline_score) / baseline_score * 100 if baseline_score else 0
    m3.metric("Change", f"{delta:+.1f}%", delta_color="inverse")

    # Per-pollutant comparison
    comp = pd.DataFrame({
        "Pollutant": [p for p in POLLUTANTS if p in baseline],
        "Baseline":  [round(baseline[p], 1) for p in POLLUTANTS if p in baseline],
        "Scenario":  [round(scenario.get(p, baseline[p]), 1) for p in POLLUTANTS if p in baseline],
    })
    st.dataframe(comp, width="stretch", hide_index=True)

    st.divider()

    # ==================================================
    # SECTION 4.75 — EXECUTIVE SUMMARY
    # ==================================================

    st.markdown("## 📋 Executive Summary")

    # Pull together values already computed above — no new calculations.
    worst_zone_line = ""
    if not zone_means.empty:
        # Zone whose key pollutant is furthest above its WHO limit
        zone_ratios = {}
        for zone, meta in ZONE_META.items():
            if zone not in zone_means.index:
                continue
            key_p = meta["key_pollutant"]
            lim = WHO_SO2_DAILY if key_p == "SO2" else WHO_ANNUAL.get(key_p)
            if lim:
                zone_ratios[zone] = zone_means.loc[zone, key_p] / lim
        if zone_ratios:
            top_zone = max(zone_ratios, key=zone_ratios.get)
            worst_zone_line = (
                f"The **{top_zone}** zone shows the highest pressure on its key "
                f"pollutant ({ZONE_META[top_zone]['key_pollutant']}, "
                f"{zone_ratios[top_zone]:.1f}× WHO)."
            )

    # Forecast exceedance line (from Section 2)
    if not alerts_df.empty and not exceed.empty:
        forecast_line = (
            f"Forecast for {forecast_date.strftime('%d %b %Y')} flags "
            f"**{len(exceed)} WHO exceedance(s)** "
            f"at: {', '.join(exceed['Station'].unique())}."
        )
    elif not alerts_df.empty:
        forecast_line = (
            f"No WHO exceedances forecast for "
            f"{forecast_date.strftime('%d %b %Y')}."
        )
    else:
        forecast_line = "Forecast data unavailable."

    summary = f"""
    Over the **{window_label.lower()}**, **{n_above} of {len(station_means)}**
    monitoring stations exceeded WHO annual guidelines on the composite risk
    index. The highest-risk station is **{worst_station['station']}**
    ({worst_station['Zone']} zone) with a composite score of
    **{worst_station['RiskScore']:.0f}**, and the most critical pollutant
    city-wide is **{worst_pollutant}** at
    **{city_ratios[worst_pollutant]:.1f}× the WHO limit**.

    {worst_zone_line}

    {forecast_line}
    """

    with st.container(border=True):
        st.markdown(summary)
        st.caption(
            "Auto-generated from the current analysis window and latest forecasts. "
            f"Window: {window_label} · Latest data: {latest_date.strftime('%d %b %Y')}."
        )

    st.divider()

    # ==================================================
    # SECTION 5 — ZONE-LEVEL RECOMMENDATIONS
    # ==================================================

    st.markdown("## 🗺️ Zone Recommendations")

    # Zone determines the TYPE of action; ratio determines the TIER (urgency/intensity).
    ZONE_ACTION_TIERS = {
        "Urban": {
            "low":  "Maintain routine traffic monitoring; no intervention needed.",
            "mid":  "Promote public transport and soft mobility during peak hours; "
                    "monitor the NO₂ weekly cycle.",
            "high": "Activate low-emission-zone enforcement and rush-hour restrictions; "
                    "issue public NO₂ advisories.",
        },
        "Industrial": {
            "low":  "Routine stack monitoring; no escalation.",
            "mid":  "Increase inspection frequency; coordinate with operators on PM episodes.",
            "high": "Enforce temporary emission controls; mandatory continuous stack "
                    "monitoring; investigate PM source episodes.",
        },
        "Port": {
            "low":  "Standard port-activity monitoring.",
            "mid":  "Encourage cleaner marine fuels for docked vessels.",
            "high": "Prioritize shore-power (cold ironing) for berthed ships; "
                    "restrict high-sulphur fuel use during episodes.",
        },
        "Coastal": {
            "low":  "Favourable dispersion; maintain baseline monitoring.",
            "mid":  "Watch for marine-PM10 transport events; protect green buffers.",
            "high": "Unusual for this zone — verify sensors, then treat as a "
                    "transport/dust episode.",
        },
        "Refinery": {
            "low":  "Routine SO₂ monitoring; refinery operating normally.",
            "mid":  "Increase SO₂ sampling cadence; flag upcoming low-wind days.",
            "high": "Event-based alerting: coordinate refinery maintenance windows "
                    "with forecast; notify sensitive population near Petronor.",
        },
    }


    def action_tier(ratio: float) -> str:
        """Map a WHO ratio to an urgency tier."""
        if ratio is None:
            return "low"
        if ratio > 2:
            return "high"
        if ratio > 1:
            return "mid"
        return "low"


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

                tier = action_tier(ratio)   # ratio already computed just above
                tier_badge = {"low": "🟢 Routine", "mid": "🟡 Elevated",
                            "high": "🔴 Action required"}[tier]
                action_text = ZONE_ACTION_TIERS.get(zone, {}).get(tier, "—")

                st.markdown(f"**Status:** {tier_badge}")
                st.markdown(f"**Recommended action:** {action_text}")
    st.divider()

   
    # ==================================================
    # SECTION 8 — EXPORT
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

