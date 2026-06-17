import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

from config import WHO_ANNUAL, WHO_SO2_DAILY, POLLUTANT_COLOR

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Scope & Limitations",
    page_icon="📋",
    layout="wide",
)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# EU Directive 2008/50/EC — legally binding annual limit values in Spain
EU_ANNUAL: dict[str, float] = {
    "PM2.5": 25.0,
    "PM10":  40.0,
    "NO2":   40.0,
}
EU_DAILY: dict[str, float] = {
    "PM10": 50.0,   # max 35 exceedance days/year allowed
    "SO2":  125.0,  # 24-hour limit
}

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]


# --------------------------------------------------
# LOAD MODEL METRICS (live from bundles — never hardcoded)
# --------------------------------------------------

@st.cache_resource
def load_metrics() -> dict:
    """Read R², RMSE, MAE directly from saved model bundles."""
    metrics = {}
    for p in POLLUTANTS:
        prefix = p.replace(".", "").lower()
        path   = MODELS_DIR / f"xgb_{prefix}_forecast.joblib"
        if path.exists():
            bundle       = joblib.load(path)
            metrics[p]   = bundle.get("metrics", {})
        else:
            metrics[p] = {}
    return metrics


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📋 Scope & Limitations")
st.markdown(
    "Understanding what this platform **can** and **cannot** do "
    "is essential for responsible use in decision-making."
)
st.divider()

# ==================================================
# SECTION 1 — WHAT THIS PLATFORM DOES
# ==================================================

st.markdown("## ✅ What This Platform Does")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        - **Monitors** daily air quality across 7 stations in Greater Bilbao (2015–present)
        - **Identifies** spatial pollution patterns by environmental zone
        - **Analyses** long-term trends including COVID-19 impact
        - **Benchmarks** concentrations against WHO 2021 guidelines
        - **Forecasts** next-day pollutant levels using XGBoost ML models
        - **Updates** automatically every day via a GitHub Actions pipeline
        - **Supports** zone-level decision-making for urban managers
        """
    )

with col2:
    st.info(
        "**One-sentence summary:**\n\n"
        "This platform monitors air pollution across Greater Bilbao, "
        "forecasts tomorrow's levels 24 hours in advance, and identifies "
        "which zones require priority action — updated daily, automatically."
    )

st.divider()

# ==================================================
# SECTION 2 — WHAT THIS PLATFORM DOES NOT DO
# ==================================================

st.markdown("## ❌ What This Platform Does NOT Do")

c1, c2, c3 = st.columns(3)

with c1:
    st.error(
        "**Not for legal compliance**\n\n"
        "Uses WHO 2021 guidelines, not EU Directive 2008/50/EC "
        "limit values. Cannot be used as a substitute for official "
        "compliance reporting to Spanish authorities."
    )

with c2:
    st.error(
        "**Not for real-time emergencies**\n\n"
        "Data has a ~24-hour lag (D-1). Not suitable for immediate "
        "emergency response or real-time pollution events."
    )

with c3:
    st.error(
        "**Not spatially continuous**\n\n"
        "Point measurements only. No spatial interpolation between "
        "stations. Areas between monitoring points are not covered."
    )

st.divider()

# ==================================================
# SECTION 3 — GEOGRAPHIC & TEMPORAL SCOPE
# ==================================================

st.markdown("## 🗺️ Geographic & Temporal Scope")

col_geo, col_time = st.columns(2)

with col_geo:
    st.markdown("### Geographic Coverage")

    # Station table
    stations_data = {
        "Station":   ["ALGORTA_BBIZI2", "SANTURCE",  "BASAURI",    "BARAKALDO",  "ERANDIO",  "MAZARREDO", "MUSKIZ"],
        "Town":      ["Getxo",          "Santurtzi", "Basauri",    "Barakaldo",  "Erandio",  "Bilbao",    "Muskiz"],
        "Zone":      ["Coastal",        "Port",      "Industrial", "Industrial", "Urban",    "Urban",     "Refinery"],
        "PM10 sensor": ["✅", "✅ none", "✅", "✅", "✅", "✅", "✅"],
    }
    st.dataframe(
        pd.DataFrame(stations_data),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "SANTURCE has no PM10 sensor — PM10 values are always NaN for this station."
    )

with col_time:
    st.markdown("### Temporal Coverage")
    st.markdown(
        """
        | Parameter | Value |
        |---|---|
        | **Start date** | 2015-01-01 |
        | **Resolution** | Daily means |
        | **Update frequency** | Every day at ~06:00 UTC |
        | **Data lag** | D-1 (yesterday's data available today) |
        | **Missing data policy** | NaN stored as-is — never interpolated |
        | **Forecast horizon** | Next day only |
        """
    )

st.divider()

# ==================================================
# SECTION 4 — POLLUTANTS COVERED
# ==================================================

st.markdown("## 🧪 Pollutants Covered")

poll_data = {
    "Pollutant": ["PM2.5", "PM10", "NO₂", "SO₂"],
    "Monitored": ["✅", "✅", "✅", "✅"],
    "Forecasted": ["✅", "✅", "✅", "✅"],
    "WHO limit": ["5 µg/m³ (annual)", "15 µg/m³ (annual)",
                  "10 µg/m³ (annual)", "40 µg/m³ (24-hour)"],
    "EU limit":  ["25 µg/m³ (annual)", "40 µg/m³ (annual)",
                  "40 µg/m³ (annual)", "125 µg/m³ (24-hour)"],
    "Not covered": ["—", "—", "—", "—"],
}

st.dataframe(pd.DataFrame(poll_data), hide_index=True, use_container_width=True)

st.warning(
    "**Pollutants NOT monitored:** O₃ (ozone), CO (carbon monoxide), "
    "benzene, NO (nitric oxide), heavy metals. "
    "These may be relevant for full air quality assessment."
)

st.divider()

# ==================================================
# SECTION 5 — STANDARDS USED
# ==================================================

st.markdown("## ⚖️ Air Quality Standards")

st.markdown(
    "This platform uses **WHO 2021 guidelines** as the primary benchmark. "
    "These are stricter than legally binding EU limits. "
    "For official compliance reporting in Spain, EU Directive 2008/50/EC applies."
)

# Comparison table
compare_data = {
    "Pollutant":   ["PM2.5", "PM10", "NO₂", "SO₂"],
    "WHO 2021":    ["5 µg/m³",  "15 µg/m³", "10 µg/m³", "40 µg/m³ (24h)"],
    "EU Directive":["25 µg/m³", "40 µg/m³", "40 µg/m³", "125 µg/m³ (24h)"],
    "Ratio WHO/EU":["5×",       "2.7×",      "4×",        "3.1×"],
    "Legal basis": ["Guidance only", "Guidance only", "Guidance only", "Guidance only"],
}

st.dataframe(pd.DataFrame(compare_data), hide_index=True, use_container_width=True)

st.caption(
    "WHO guidelines are typically 2–5× stricter than EU legal limits. "
    "Exceeding WHO limits does NOT mean a legal violation — "
    "it means concentrations are above the health-optimal level."
)

st.divider()

# ==================================================
# SECTION 6 — MODEL PERFORMANCE
# ==================================================

st.markdown("## 🤖 Forecast Model Performance")
st.markdown(
    "XGBoost models trained on 2015–2022, validated on 2023, "
    "tested on 2024–2026 (strict time-based split — no data leakage)."
)

metrics = load_metrics()

cols = st.columns(4)
for col, pollutant in zip(cols, POLLUTANTS):
    m = metrics.get(pollutant, {})
    r2   = m.get("R2",   None)
    rmse = m.get("RMSE", None)
    mae  = m.get("MAE",  None)

    with col:
        color = POLLUTANT_COLOR.get(pollutant, "#888")
        st.markdown(
            f"""
            <div style="border-left:4px solid {color};
                        padding:12px; border-radius:6px;
                        background:{color}11;">
            <div style="font-weight:700; font-size:1.1rem">{pollutant}</div>
            <div style="font-size:0.85rem; margin-top:6px">
                R² = <b>{f"{r2:.3f}" if r2 else "N/A"}</b><br>
                RMSE = <b>{f"{rmse:.2f} µg/m³" if rmse else "N/A"}</b><br>
                MAE = <b>{f"{mae:.2f} µg/m³" if mae else "N/A"}</b>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
st.info(
    "**What these numbers mean:**\n\n"
    "- R² of 0.48 for PM2.5 means the model explains 48% of day-to-day variation. "
    "The remaining 52% is driven by unpredictable events (dust storms, industrial episodes).\n"
    "- SO₂ has the lowest R² (0.39) because its emissions are episodic "
    "(industrial/port events), not regular like traffic-driven NO₂.\n"
    "- Accuracy degrades for multi-day forecasts — only next-day predictions are published."
)

st.divider()

# ==================================================
# SECTION 7 — DATA SOURCES
# ==================================================

st.markdown("## 📡 Data Sources")

col_aq, col_met = st.columns(2)

with col_aq:
    st.markdown("### Air Quality")
    st.markdown(
        """
        **Source:** Basque Government Open Data  
        *(Red de Control de Calidad del Aire — RVCA)*

        **URL:** opendata.euskadi.eus  
        **License:** CC BY 4.0  
        **Coverage:** 60+ stations across the Basque Country  
        **Resolution:** Daily means (hourly data aggregated)  
        **Lag:** ~24 hours  
        **Known issues:**
        - Data quality depends on Basque Government maintenance
        - Occasional sensor gaps (stored as NaN, never interpolated)
        - Data quality depends on Basque Government maintenance
        """
    )

with col_met:
    st.markdown("### Meteorology")
    st.markdown(
        """
        **Source:** Open-Meteo ERA5 Archive API  
        **URL:** archive-api.open-meteo.com  
        **License:** CC BY 4.0  
        **Variables:** Temperature, Humidity, Precipitation,
        Wind Speed, Wind Direction, Wind U/V components  
        **Resolution:** Daily  
        **Lag:** ~24 hours  
        **Known issues:**
        - ERA5 reanalysis — not direct sensor measurement
        - Spatial resolution ~9km — may not capture local microclimates
        """
    )

st.divider()

# ==================================================
# SECTION 8 — RECOMMENDED & NOT RECOMMENDED USE CASES
# ==================================================

st.markdown("## 🎯 Use Cases")

col_yes, col_no = st.columns(2)

with col_yes:
    st.success(
        "**✅ Recommended for:**\n\n"
        "- Daily air quality monitoring by municipal staff\n"
        "- Early warning before pollution episodes\n"
        "- Long-term trend analysis (annual/seasonal)\n"
        "- Environmental zone comparison\n"
        "- Public communication and awareness\n"
        "- Research and academic analysis\n"
        "- Smart City dashboard integration\n"
        "- Environmental consultancy reporting (indicative)"
    )

with col_no:
    st.error(
        "**❌ Not recommended for:**\n\n"
        "- Legal compliance reporting to Spanish authorities\n"
        "- Real-time emergency response\n"
        "- Health risk assessment for individuals\n"
        "- Regulatory enforcement decisions\n"
        "- Areas outside Greater Bilbao\n"
        "- Pollutants not covered (O₃, CO, benzene)\n"
        "- Multi-day forecasts beyond tomorrow\n"
        "- Replacing official monitoring networks"
    )

st.divider()

# ==================================================
# SECTION 9 — KNOWN LIMITATIONS SUMMARY
# ==================================================

st.markdown("## ⚠️ Known Limitations Summary")

limitations = [
    ("Spatial coverage",    "7 stations only — large areas between stations have no direct measurement"),
    ("No spatial model",    "Point-wise forecasts only — no interpolated pollution surface maps"),
    ("Forecast horizon",    "Next-day only — accuracy degrades sharply beyond 24 hours"),
    ("Standard",            "WHO guidelines used — stricter than legally binding EU limits"),
    ("Missing pollutants",  "O₃, CO, benzene, heavy metals not included"),
    ("Data lag",            "~24 hours — not suitable for real-time response"),
    ("ERA5 meteorology",    "Reanalysis data — may differ from local measurements"),
    ("No causal inference", "Correlations shown — platform cannot prove causality"),
    ("Single city",         "Validated for Greater Bilbao only — not transferable without revalidation"),
    ("Model drift",         "Models trained on 2015–2022 — performance may degrade over time without retraining"),
]

for title, description in limitations:
    st.markdown(f"**{title}:** {description}")

st.divider()

# ==================================================
# FOOTER
# ==================================================

st.markdown("## 📬 Contact & Feedback")
st.markdown(
    "For questions, data requests, or collaboration enquiries:\n\n"
    "**Arman Ghaziaskari Naeini**  \n"
    "GIS & Remote Sensing Specialist | Spatial Data Scientist | GeoAI Enthusiast  \n"
    "Bilbao, Basque Country, Spain  \n\n"
    "🔗 [Portfolio](https://armanghazi.github.io/portfolio/projects)  \n"
    "🌐 [Dashboard](https://geoai-dashboard.streamlit.app/)"
)

st.caption(
    "Platform version: Phase C · Last pipeline update: automated daily · "
    "Data: Basque Government (CC BY 4.0) + Open-Meteo (CC BY 4.0)"
)