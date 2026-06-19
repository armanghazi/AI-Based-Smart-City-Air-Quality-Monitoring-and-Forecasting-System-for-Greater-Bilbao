# 🌍 GeoAI Smart City Platform — Air Quality Intelligence for Greater Bilbao

**Status:** Live on Streamlit Cloud · actively developed
**🔗 Live Dashboard:** https://geoai-dashboard.streamlit.app/

## Overview

An end-to-end **GeoAI and Spatial Data Science platform** that monitors, analyzes, visualizes, and forecasts urban air quality across the Greater Bilbao Metropolitan Area (Bizkaia, Basque Country, Spain).

The platform integrates Environmental Data Science, GIS, Machine Learning, and Interactive Dashboards to support Smart City decision-making and sustainable urban planning. As of mid-2026 it is no longer a static snapshot: an **automated daily pipeline** keeps the data live, **validated XGBoost models** forecast next-day pollution, **spatial interpolation masked to the metropolitan boundary** produces continuous surfaces, and a **conversational AI assistant** answers questions about both the data and the methodology directly from the dashboard.

---

## Project Objectives

- Monitor air pollution patterns across Greater Bilbao (2015–2026, updated daily).
- Identify spatial hotspots and pollution clusters by environmental zone.
- Analyze long-term temporal trends, including the COVID-19 impact.
- Assess urban environmental risk against **WHO 2021 guidelines** and **EU regulatory limits**.
- **Forecast next-day air quality with validated ML models (XGBoost).**
- Explain model behavior with **SHAP** to ensure physically meaningful predictions.
- Produce **continuous spatial surfaces** (IDW interpolation masked to the comarca boundary) from sparse station data.
- Provide decision-support tools, exportable reports, and a conversational AI assistant for Smart City stakeholders.

---

## Dataset

| Component       | Details                                                                          |
| --------------- | -------------------------------------------------------------------------------- |
| **Air quality** | Open Data Euskadi (Basque Government) · 7 stations · 2015–2026                   |
| **Meteorology** | Open-Meteo ERA5 Archive API (per-station, daily)                                 |
| **Records**     | ~29,300+ daily rows — **grows daily** via the automated pipeline                 |
| **Pollutants**  | PM2.5, PM10, NO₂, SO₂                                                            |
| **Weather**     | Temperature, Humidity, Precipitation, WindSpeed, WindDirection (+ wind_u/wind_v) |
| **Storage**     | Parquet (fast, compact, Streamlit-friendly)                                      |

**Missing-value handling — two distinct policies (intentional):**

- The **ML training snapshot** (`forecasting_dataset.parquet`) was cleaned with **MICE iterative imputation** per station rather than row deletion. This file is **frozen** — the pipeline never touches it and models are never retrained against shifting data.
- The **live dashboard source** (`air_quality_weather.parquet`) **preserves raw NaN values** — the daily pipeline appends source data faithfully and never interpolates.

### Environmental Zone Classification

| Station   | Zone            | Signature                                                 |
| --------- | --------------- | --------------------------------------------------------- |
| Mazarredo | 🏙️ Urban        | Highest NO₂ — traffic, urban canyon                       |
| Erandio   | 🏙️ Urban        | Traffic influence                                         |
| Basauri   | 🏭 Industrial   | High PM2.5 / PM10                                         |
| Barakaldo | 🏭 Industrial   | High PM, elevated NO₂                                     |
| Santurtzi | ⚓ Port         | Marine + traffic, elevated SO₂                            |
| Algorta   | 🌊 Coastal      | Best dispersion                                           |
| Muskiz    | 🛢️ **Refinery** | Petronor petrochemical profile — distinct SO₂/PM behavior |

> Muskiz was initially classed as Coastal; GIS analysis revealed its refinery-driven emission profile warranted a separate zone.

---

## Machine Learning — Forecasting

### Task

Next-day pollutant concentration forecasting: `target(t+1) = pollutant(t+1)` per station, using today's pollutant levels, meteorology, and lag/rolling features. **62 features per model**, verified directly from the joblib bundles.

### Validated Results (held-out test = 2024–2026)

| Pollutant | R²        | RMSE (µg/m³) | MAE (µg/m³) | Interpretation                               |
| --------- | --------- | ------------ | ----------- | -------------------------------------------- |
| **NO₂**   | **0.560** | 5.97         | 4.43        | Best — regular traffic weekly cycle          |
| **PM2.5** | 0.479     | 3.33         | 2.50        | Persistence + weather drivers                |
| **PM10**  | 0.460     | 6.26         | 4.45        | Dust events add variability                  |
| **SO₂**   | 0.390     | 1.87         | 1.23        | Hardest — episodic industrial/port emissions |

### Methodological Rigor

- **Strict time-based split** (train < 2023 · validation 2023 · test ≥ 2024). An early row-based split produced an inflated R² = 0.84 via leakage; this was identified and corrected. _When R² jumps suspiciously, leakage is suspected first._
- **Today's pollutant value is kept as a feature** — it is available at prediction time, hence valid (not leakage). Removing it drops PM2.5 R² from 0.48 to 0.34.
- **SHAP analysis** confirmed physically meaningful behavior: higher wind speed and precipitation consistently push predictions **down** (atmospheric dispersion, wet deposition).
- **Frozen by design:** feature engineering (lags, rolling means, targets) is recomputed at runtime in `config.load_data()`, so live daily data improves predictions without retraining.

### Benchmark Extensions (notebook 08)

| Model               | Type          | Scope          | R²        |
| ------------------- | ------------- | -------------- | --------- |
| **XGBoost**         | ML (trees)    | All stations   | **0.479** |
| MLP (128, 64)       | Deep learning | All stations   | 0.208     |
| SARIMA one-step     | Classical TS  | Mazarredo only | 0.463     |
| SARIMA long-horizon | Classical TS  | Mazarredo only | ≈ 0.00    |

**ARIMA/SARIMA/LSTM are benchmark-only, never production.**

---

## Automated Daily Data Pipeline (Phase B — complete)

- `scripts/daily_update.py` — fetches air-quality records from **Open Data Euskadi** and weather from **Open-Meteo**, appends to `air_quality_weather.parquet`.
- `.github/workflows/daily_update.yml` — **GitHub Actions cron** runs daily; the commit triggers a Streamlit Cloud auto-redeploy. No retraining.
- **Reliability:** idempotent append, D-0 protection (current incomplete day rejected), zero duplicates.

---

## Spatial GeoAI (Phase C — in progress)

- **IDW interpolation surfaces** masked to the Gran Bilbao comarca boundary (EPSG:25830).
- `dashboard/spatial_utils.py` — `build_mask()` and `mask_idw_grid()`, using Shapely 2.1.2+ `contains_xy`.
- **Spatial covariate analysis:** Distance to Petronor refinery vs. mean SO₂ → Pearson **r = −0.15** → SO₂ has distributed sources, not a single point source. Road density vs. NO₂ analysis in progress.

---

## Interactive Dashboard (10 pages)

| Page                                         | Status | Contents                                                                                                                                                                                                      |
| -------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 · Daily Briefing                           | ✅     | Automated daily summary, today's air quality across all stations                                                                                                                                              |
| 1 · Air Quality Monitoring                   | ✅     | Interactive GIS map, station comparison, WHO risk levels                                                                                                                                                      |
| 2 · Temporal Trends                          | ✅     | Long-term & seasonal trends, COVID-19 impact analysis                                                                                                                                                         |
| 3 · Urban Risk Index                         | ✅     | WHO + EU benchmarking, risk scoring, station ranking                                                                                                                                                          |
| 4 · Weather Drivers & Air Pollution Dynamics | ✅     | Correlation matrices, wind roses, lag analysis, feature ranking                                                                                                                                               |
| 5 · Forecasting                              | ✅     | Next-day backtest, recursive multi-day forecast with uncertainty warnings, pre-computed SHAP explainability                                                                                                   |
| 6 · Smart City Decision Support              | ✅     | GeoAI risk map (IDW surface masked to boundary), scenario simulator, recommended actions, PDF/CSV export                                                                                                      |
| 7 · Scope and Limitations                    | ✅     | Transparent documentation of what the platform can and cannot do                                                                                                                                              |
| 8 · Project Assistant                        | ✅     | **Conversational AI assistant** (Groq / Llama 3.3 70B) — answers questions about data AND methodology by querying the live dataset directly (parametric tool, no SQL injection risk)                          |
| 9 · Smart City Operations _(admin-only)_     | ✅     | **Enterprise-grade operational dashboard** — Data Quality Score (Freshness/Completeness/Integrity/Stability), active alerts, sensor health ranking, statistical outlier detection (>3σ), pipeline diagnostics |

**Cross-cutting features:** dual AQI display (EAQI/ICA + EPA), gauge indicators, calendar heatmaps, WHO + EU limit benchmarking, automated Daily Briefing, PDF report export, favourite-station selector, bilingual interface (EN/ES), admin authentication (Google OAuth via Streamlit native OIDC).

---

## Authentication & Access Control

- **Public pages (0–8):** accessible to all visitors without login.
- **Admin page (9):** protected by Google OAuth via Streamlit's native `st.login` / `st.user` (OIDC). Access is controlled by a whitelist in `st.secrets` — not hard-coded. The gate (`require_auth`) runs before any content renders, so URL-guessing does not bypass it.
- Admin is auto-redirected to the Operations Dashboard after login.

---

## Conversational AI Assistant (Page 8)

The Project Assistant answers questions about **both the live data and the project methodology**:

- **Data questions** (trends by day/month/year/season, station comparisons, WHO exceedance rates, specific date lookups) are answered by a **parametric query tool** that runs directly against the live parquet — no SQL injection risk, no hallucinated numbers.
- **Methodology questions** (why time-based split, why SO₂ separately, how SHAP works) are answered from a grounded project knowledge block.
- Backend: **Groq API** (Llama 3.3 70B) — free tier, fast inference.
- Responds in the user's language automatically (EN/ES/FA and others).
- Multi-row query results are accompanied by inline charts (line for trends, bar for comparisons).

---

## Smart City Operations Dashboard (Page 9 — admin)

A professional operational intelligence page modelled after enterprise observability systems (Datadog, Snowflake, AWS Data Quality):

**Data Quality Score (DQS)** — weighted composite (0–100):

| Component    | Formula                                | Weight |
| ------------ | -------------------------------------- | ------ |
| Freshness    | `100 − 20 × days_behind`               | 0.35   |
| Completeness | `100 × (1 − missing_ratio)`            | 0.30   |
| Integrity    | `100 − 5 × duplicates − 10 × invalids` | 0.20   |
| Stability    | `100 − 2 × outliers_last_7d (>3σ)`     | 0.15   |

**Sections:** System Status Overview · Active Alerts · Sensor Health Ranking · Statistical Outlier Detection (>3σ, honest label — not AI/ML) · Diagnostics (Freshness / Coverage Matrix / Completeness / Ingestion History / Integrity).

---

## Technologies

**GIS & Spatial:** QGIS · GeoPandas · Shapely 2.1.2 · osmnx · Folium · Contextily
**Data Science:** Python 3.14 · Pandas · NumPy · Scikit-Learn · statsmodels
**Machine Learning:** XGBoost (production) · MLP · SARIMA · SHAP (benchmarks)
**Dashboard:** Streamlit (Cloud) · Plotly · Streamlit-Folium
**AI/LLM:** Groq API (Llama 3.3 70B) · parametric query tool (pandas)
**Auth:** Streamlit native OIDC (`st.login` / `st.user`) · Google OAuth
**Data & Ops:** Parquet · Open Data Euskadi API · Open-Meteo API · GitHub Actions · pytest
**i18n:** Bilingual EN/ES via `i18n_auto.py`; RTL-ready for Persian (Vazirmatn)

---

## Project Architecture

```
project/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── forecasting_dataset.parquet      # ML-ready snapshot (62 features) — FROZEN
│       └── air_quality_weather.parquet      # dashboard source — live, pipeline appends daily
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb               # MICE imputation (training snapshot only)
│   ├── 03_eda.ipynb
│   ├── 04_gis_spatial_analysis.ipynb
│   ├── 05_weather_data.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_model_training.ipynb
│   └── 08_benchmark_extensions.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── weather_collector.py
│   └── spatial_analysis.py
│
├── scripts/
│   ├── daily_update.py                      # Euskadi API + Open-Meteo → parquet append
│   └── verify_pipeline.py
│
├── .github/workflows/
│   └── daily_update.yml                     # scheduled cron + auto-redeploy
│
├── models/                                  # production joblib bundles (frozen)
│   ├── xgb_pm25_forecast.joblib
│   ├── xgb_pm10_forecast.joblib
│   ├── xgb_no2_forecast.joblib
│   └── xgb_so2_forecast.joblib
│
├── dashboard/
│   ├── app.py                               # homepage + post-login admin redirect
│   ├── config.py                            # shared loader, zones, WHO + EU limits
│   ├── auth.py                              # OIDC gate: require_auth, current_role, logout_button
│   ├── forecast_utils.py                    # shared prepare_features
│   ├── spatial_utils.py                     # IDW mask (comarca boundary)
│   ├── aqi.py                               # EAQI/ICA + EPA dual index
│   ├── assistant_query.py                   # parametric query tool for the AI assistant
│   ├── i18n_auto.py                         # bilingual EN/ES + RTL support
│   ├── pdf_report.py                        # PDF export
│   ├── assets/                              # pre-computed SHAP plots
│   └── pages/
│       ├── 0_Daily_Briefing.py
│       ├── 1_Air_Quality_Monitoring.py
│       ├── 2_Temporal_Trends.py
│       ├── 3_Urban_Risk_Index.py
│       ├── 4_Weather_Drivers_&_Air_Pollution_Dynamics.py
│       ├── 5_Forecasting.py
│       ├── 6_Smart_City_Decision_Support.py
│       ├── 7_Scope_and_Limitations.py
│       ├── 8_Project_Assistant.py           # Groq AI assistant + parametric query tool
│       └── 9_Smart_City_Operations.py       # admin-only operations dashboard
│
├── GIS/boundaries/COMARCAS_5000_ETRS89.shp
├── tests/                                   # pytest suite
├── requirements.txt
└── README.md
```

---

## Key EDA Findings

- **PM2.5** — stable long-term decrease; urban/industrial stations highest; strong seasonality.
- **PM10** — greater variability; dust events; strong correlation with PM2.5.
- **NO₂** — clear traffic signature with weekly cycle; significant reduction over the decade; sharp COVID-19 dip.
- **SO₂** — low overall concentrations; episodic industrial/port/refinery origin; spatially **distributed** rather than single-point (refinery-distance correlation r = −0.15).

---

## Engineering & Reliability

- **Shared logic, no duplication:** `prepare_features` in `forecast_utils.py`; `aqi.py` is the single source of truth for AQI calculations; `auth.py` centralizes all access control.
- **Parametric query tool (`assistant_query.py`):** injection-proof by design — no SQL generated, only whitelisted pandas operations. Numbers in the AI assistant always match the dashboard.
- **Test suite (pytest):** model-contract tests + spatial-utils tests — all passing.
- **Streamlit Cloud is Linux (case-sensitive):** all pages use `sys.path.insert` + `pathlib.Path(__file__).parent.parent` for imports.
- **Python 3.14** `string[pyarrow]` dtypes — features coerced to plain numeric before XGBoost `predict`.
- `use_container_width` → `width="stretch"` migration complete across all pages.
- **Security:** secrets never hard-coded; `.gitignore` catches `secrets.toml` at any depth; one historical leak incident resolved via key rotation and history rewrite.

---

## Roadmap

**In development**

- Complete Phase C Task 2 (road density vs NO₂).
- Broader spatial-covariate conclusions to feed the dashboard narrative.

**Planned**

- Spatial forecast surfaces (per-zone / interpolated prediction maps).
- Land Use Regression (LUR) exploration.
- Satellite-derived indicators (NDVI) and DEM integration.
- Euskalmet API integration for official Basque meteorological data.

**Explicitly not doing**

- LSTM/GRU for production (MLP scored R² = 0.208 — benchmark-only).
- Retraining models to add `station_code` (lag/rolling features already carry each station's signature).

---

## Acknowledgements

This project was developed during the **AI & Data Tech** training pathway, made possible by the comprehensive support, guidance, and learning ecosystem provided by:

- **GAIA Cluster ICTA**
- **DEMA – Agencia de Empleo y Emprendimiento**
- **C2B**

### Special Thanks

- **Aitor Donado**: I would like to express my deepest gratitude to Aitor Donado for his exceptional instruction, valuable guidance, and continuous support throughout this course. His expertise and insights were instrumental in shaping the technical direction of this project.
- **GAIA Cluster ICTA**: Sincere thanks to GAIA for providing this incredible opportunity, driving innovation in Artificial Intelligence, Data Science, and Digital Transformation, and fostering an environment that inspired the development of applied **GeoAI** and **Smart City** solutions.

---

## Author

**Arman Ghaziaskari Naeini** _GIS & Remote Sensing Specialist | Spatial Data Scientist | GeoAI Enthusiast_ Bilbao, Spain

- **Portfolio:** [armanghazi.github.io/portfolio/projects](https://armanghazi.github.io/portfolio/projects)
- **Dashboard:** [geoai-dashboard.streamlit.app](https://geoai-dashboard.streamlit.app/)

---

## Data Sources & License

- **Air Quality:** Basque Government Open Data (**Open Data Euskadi**).
- **Meteorology:** [Open-Meteo](https://open-meteo.com/) ERA5 Archive (licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
