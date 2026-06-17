# 🌍 GeoAI Smart City Platform for Air Quality Monitoring and Forecasting in Greater Bilbao(Under Construction)

**Status:** Live on Streamlit Cloud · actively developed (Phase C — spatial GeoAI — in progress)

**🔗 Live Dashboard:** https://geoai-dashboard.streamlit.app/

## Overview

An end-to-end **GeoAI and Spatial Data Science platform** that monitors, analyzes, visualizes, and forecasts urban air quality across the Greater Bilbao Metropolitan Area (Bizkaia, Basque Country, Spain).

The platform integrates Environmental Data Science, GIS, Machine Learning, and Interactive Dashboards to support Smart City decision-making and sustainable urban planning. As of mid-2026 it is no longer a static snapshot: an **automated daily pipeline** keeps the data live, **validated XGBoost models** forecast next-day pollution, and **spatial interpolation masked to the metropolitan boundary** brings the analysis from point measurements toward continuous surfaces.

---

## Project Objectives

- Monitor air pollution patterns across Greater Bilbao (2015–2026, updated daily).
- Identify spatial hotspots and pollution clusters by environmental zone.
- Analyze long-term temporal trends, including the COVID-19 impact.
- Assess urban environmental risk against **WHO 2021 guidelines** and **EU regulatory limits**.
- **Forecast next-day air quality with validated ML models (XGBoost).**
- Explain model behavior with **SHAP** to ensure physically meaningful predictions.
- Produce **continuous spatial surfaces** (IDW interpolation masked to the comarca boundary) from sparse station data.
- Provide decision-support tools and exportable reports for Smart City stakeholders.

---

## Dataset

| Component       | Details                                                                          |
| --------------- | -------------------------------------------------------------------------------- |
| **Air quality** | Open Data Euskadi (Basque Government) · 7 stations · 2015–2026                   |
| **Meteorology** | Open-Meteo ERA5 Archive API (per-station, daily)                                 |
| **Records**     | 29,281 daily rows at the 2026-06-14 backfill — **grows daily** via the pipeline  |
| **Pollutants**  | PM2.5, PM10, NO₂, SO₂                                                            |
| **Weather**     | Temperature, Humidity, Precipitation, WindSpeed, WindDirection (+ wind_u/wind_v) |
| **Storage**     | Parquet (fast, compact, Streamlit-friendly)                                      |

**Missing-value handling — two distinct policies (intentional):**

- The **ML training snapshot** (`forecasting_dataset.parquet`) was cleaned with **MICE iterative imputation** per station (PM2.5 ≈ 33% missing) rather than row deletion. This file is **frozen** — the pipeline never touches it and models are never retrained against shifting data.
- The **live dashboard source** (`air_quality_weather.parquet`) **preserves raw NaN values** — the daily pipeline appends source data faithfully and never interpolates. Expected gaps as of the backfill: PM10 = 19 (SANTURCE has no PM10 sensor), NO2 = 2, PM2.5 = 5.

### Environmental Zone Classification

Five zones derived from spatial analysis and local domain knowledge:

| Station   | Zone            | Signature                                                 |
| --------- | --------------- | --------------------------------------------------------- |
| Mazarredo | 🏙️ Urban        | Highest NO₂ — traffic, urban canyon                       |
| Erandio   | 🏙️ Urban        | Traffic influence                                         |
| Basauri   | 🏭 Industrial   | High PM2.5 / PM10                                         |
| Barakaldo | 🏭 Industrial   | High PM, elevated NO₂                                     |
| Santurtzi | ⚓ Port         | Marine + traffic, elevated SO₂                            |
| Algorta   | 🌊 Coastal      | Best dispersion                                           |
| Muskiz    | 🛢️ **Refinery** | Petronor petrochemical profile — distinct SO₂/PM behavior |

> Note: Muskiz was initially classed as Coastal; GIS analysis revealed its refinery-driven emission profile warranted a separate class.

---

## Machine Learning — Forecasting

### Task

Next-day pollutant concentration forecasting: `target(t+1) = pollutant(t+1)` per station, using today's pollutant levels, meteorology, and lag/rolling features. **62 features per model**, verified directly from the joblib bundles and identical across all 4 pollutants.

### Validated Results (held-out test = 2024–2026)

| Pollutant | R²        | RMSE (µg/m³) | MAE (µg/m³) | Interpretation                               |
| --------- | --------- | ------------ | ----------- | -------------------------------------------- |
| **NO₂**   | **0.560** | 5.97         | 4.43        | Best — regular traffic weekly cycle          |
| **PM2.5** | 0.479     | 3.33         | 2.50        | Persistence + weather drivers                |
| **PM10**  | 0.460     | 6.26         | 4.45        | Dust events add variability                  |
| **SO₂**   | 0.390     | 1.87         | 1.23        | Hardest — episodic industrial/port emissions |

Production model: **XGBoost** (one model per pollutant, saved as joblib bundles with feature lists and metrics in `models/`). The `bundle["features"]` list is the single source of truth for column order at prediction time.

### Methodological Rigor

- **Strict time-based split** (train < 2023 · validation 2023 · test ≥ 2024). An early row-based split produced an inflated R² = 0.84 by mixing stations' time ranges; this leakage was identified and corrected — the honest figures above resulted. _When R² jumps suspiciously, leakage is suspected first._
- **Today's pollutant value is kept as a feature** — it is available at prediction time, hence valid (not leakage). Removing it drops PM2.5 R² from 0.48 to 0.34.
- **TimeSeriesSplit** used for hyperparameter search; tuning did not beat sensible baseline parameters (n_estimators=300, lr=0.05, max_depth=6), so the simpler model was retained.
- **SHAP analysis** confirmed physically meaningful behavior: higher wind speed and precipitation consistently push predictions **down** (atmospheric dispersion, wet deposition) — the model learned real mechanisms, not spurious correlations. Each pollutant's strongest predictor is its own current value.
- Pollutant-specific signatures: NO₂ shows strong `day_of_week` importance (traffic); SO₂ lacks temporal regularity (episodic origin) — explaining its lower predictability.
- **Frozen by design:** because feature engineering (lags, rolling means, targets) is recomputed at runtime in `config.load_data()`, the live daily data flows into the same features and improves predictions **without any retraining**.

### Benchmark Extensions (notebook 08)

A full methodological comparison on identical splits:

| Model               | Type          | Scope          | R²        |
| ------------------- | ------------- | -------------- | --------- |
| **XGBoost**         | ML (trees)    | All stations   | **0.479** |
| MLP (128, 64)       | Deep learning | All stations   | 0.208     |
| SARIMA one-step     | Classical TS  | Mazarredo only | 0.463     |
| SARIMA long-horizon | Classical TS  | Mazarredo only | ≈ 0.00    |

**Why XGBoost for production:** SARIMA one-step nearly matches it on a single regular station — confirming short-horizon forecasting is persistence-dominated — but would require 28 separate models (7 stations × 4 pollutants), ignores meteorological drivers, and collapses entirely on multi-day horizons. The MLP result is consistent with literature on tree-model dominance for medium-sized tabular data (Grinsztajn et al., 2022). **ARIMA/SARIMA/LSTM are benchmark-only, never production.**

---

## Automated Daily Data Pipeline (Phase B — complete)

The static snapshot is now a **live system**.

- `scripts/daily_update.py` — fetches the latest air-quality records from the **Open Data Euskadi** API and weather from **Open-Meteo**, then appends to `air_quality_weather.parquet`.
- `.github/workflows/daily_update.yml` — **GitHub Actions cron** runs the job daily (scheduled a few minutes off the top of the hour to avoid scheduler congestion); the commit triggers a Streamlit Cloud auto-redeploy. No retraining.
- `scripts/verify_pipeline.py` — local sanity check.

**Reliability properties (verified):** idempotent append (a second run adds 0 rows), **D-0 protection** (the current, incomplete day is rejected), zero duplicates, and a completed backfill (2015-01-01 → 2026-06-14, all 7 stations current).

---

## Spatial GeoAI (Phase C — in progress)

Closing the gap between the "GeoAI" name and point-wise ML.

- **IDW interpolation surfaces** rendered on the page-6 risk map, **masked to the Gran Bilbao comarca boundary** so the surface never bleeds outside the metropolitan area.
- `dashboard/spatial_utils.py` — `build_mask()` and `mask_idw_grid()`, built on the comarca shapefile (`GIS/boundaries/COMARCAS_5000_ETRS89.shp`, CRS **EPSG:25830**), using `shapely.contains_xy` (Shapely 2.1.2+). Hull-boundary stations (Algorta, Basauri, Muskiz) require a small polygon buffer to pass point-in-polygon tests.
- **Spatial covariate analysis (Task 2, ongoing):**
  - Distance to the Petronor refinery vs. mean SO₂ → Pearson **r = −0.15** → interpreted as SO₂ having **distributed** sources (traffic, local combustion) rather than a single dominant point source.
  - Road density (500 m buffer, via `osmnx`) vs. NO₂ → in progress.

---

## Interactive Dashboard (6 pages, all built)

| Page                                         | Status | Contents                                                                                                                                                             |
| -------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 · Air Quality Monitoring                   | ✅     | Interactive GIS map, station comparison, WHO risk levels                                                                                                             |
| 2 · Temporal Trends                          | ✅     | Long-term & seasonal trends, COVID-19 impact analysis                                                                                                                |
| 3 · Urban Risk Index                         | ✅     | WHO + EU benchmarking, risk scoring, station ranking                                                                                                                 |
| 4 · Weather Drivers & Air Pollution Dynamics | ✅     | Correlation matrices, wind roses, lag analysis, feature ranking                                                                                                      |
| 5 · Forecasting                              | ✅     | Next-day backtest (prediction vs actual on 2024+), recursive multi-day forecast with uncertainty warnings, model explainability (pre-computed SHAP)                  |
| 6 · Smart City Decision Support              | ✅     | GeoAI risk map (Plotly scatter_mapbox, no token) + IDW surface masked to boundary, scenario simulator (two modes), ratio×zone actions, executive summary, CSV export |

**Cross-cutting features:** dual AQI display, gauge indicators, calendar heatmaps, **WHO + EU limit** benchmarking with alert thresholds, an automated **Daily Briefing**, **PDF report export**, a **favourite-station** selector shared across all pages, and a redesigned homepage ("atmospheric instrument panel" identity).

> Page-6 tab order must stay status → forecast → action: later tabs reuse variables computed in earlier ones (Python runs top-to-bottom; tabs only control display, not execution order).

---

## Technologies

**GIS & Spatial:** QGIS · GeoPandas · Shapely 2.1.2 · osmnx · Folium · Contextily
**Data Science:** Python 3.14 · Pandas · NumPy · Scikit-Learn · statsmodels
**Machine Learning:** XGBoost (production) · MLP · SARIMA · SHAP (benchmarks)
**Dashboard:** Streamlit (Cloud) · Plotly (scatter_mapbox) · Streamlit-Folium
**Data & Ops:** Parquet · Open Data Euskadi API · Open-Meteo API · GitHub Actions (CI + scheduled pipeline) · pytest

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
│   ├── 04_gis_spatial_analysis.ipynb        # zone classification
│   ├── 05_weather_data.ipynb                # Open-Meteo integration
│   ├── 06_feature_engineering.ipynb         # lags, rolling, wind u/v, targets
│   ├── 07_model_training.ipynb              # comparison, tuning, SHAP, export
│   └── 08_benchmark_extensions.ipynb        # MLP & SARIMA benchmarks
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── weather_collector.py
│   └── spatial_analysis.py
│
├── scripts/
│   ├── daily_update.py                      # Euskadi API + Open-Meteo → parquet append
│   └── verify_pipeline.py                   # local sanity check
│
├── .github/workflows/
│   └── daily_update.yml                     # scheduled cron + auto-redeploy
│
├── models/                                  # production joblib bundles (frozen)
│   ├── xgb_pm25_forecast.joblib             # model + features (62) + metrics
│   ├── xgb_pm10_forecast.joblib
│   ├── xgb_no2_forecast.joblib
│   └── xgb_so2_forecast.joblib
│
├── dashboard/
│   ├── app.py                               # redesigned homepage
│   ├── config.py                            # shared loader, zones, WHO + EU limits
│   ├── forecast_utils.py                    # shared prepare_features (single source)
│   ├── spatial_utils.py                     # build_mask, mask_idw_grid (comarca boundary)
│   ├── pdf_report.py                        # PDF export
│   ├── assets/                              # pre-computed SHAP plots
│   └── pages/
│       ├── 1_Air_Quality_Monitoring.py
│       ├── 2_Temporal_Trends.py
│       ├── 3_Urban_Risk_Index.py
│       ├── 4_Weather_Drivers_&_Air_Pollution_Dynamics.py
│       ├── 5_Forecasting.py
│       └── 6_Smart_City_Decision_Support.py
│
├── GIS/boundaries/COMARCAS_5000_ETRS89.shp  # Gran Bilbao comarca (EPSG:25830)
├── maps/  ·  reports/
├── tests/                                   # pytest suite (model-contract + spatial-utils)
├── requirements.txt
└── README.md
```

---

## Key EDA Findings

- **PM2.5** — stable long-term decrease; urban/industrial stations highest; strong seasonality.
- **PM10** — greater variability; dust events; strong correlation with PM2.5.
- **NO₂** — clear traffic signature with weekly cycle; significant reduction over the decade; sharp COVID-19 dip.
- **SO₂** — low overall concentrations; industrial/port/refinery origin; behaves independently of other pollutants; spatially **distributed** rather than single-point (refinery-distance correlation r = −0.15).

---

## Engineering & Reliability

- **Shared logic, no duplication:** `prepare_features` lives only in `dashboard/forecast_utils.py`; pages 5, 6, and the tests all import it.
- **Test suite (pytest):** model-contract tests (bundles loadable, feature contract, orphan-feature tracking, station-code regression, end-to-end predict smoke test, stored-vs-documented metrics) plus spatial-utils tests (mask + masked IDW grid) — all passing.
- **Streamlit Cloud is Linux (case-sensitive):** page filenames must match nav paths exactly; this differs from local Windows and once caused a Cloud-only crash. Cloud pages also import via `sys.path.insert` (`pathlib.Path(__file__).parent.parent`) rather than package-relative imports.
- **Python 3.14** introduces `string[pyarrow]` dtypes — features are coerced to plain numeric before XGBoost `predict`.
- Migrated the deprecated `use_container_width=True` → `width="stretch"` across all pages.

---

## Roadmap

**In development**

- **Page 7 — Conversational Q&A:** an in-dashboard assistant that answers questions about both the data and the project methodology, grounded in a runtime data digest + project-knowledge context.
- Complete Phase C Task 2 (road density vs NO₂); broader spatial-covariate conclusions to feed the dashboard narrative.

**Planned**

- Spatial forecast surfaces (per-zone / interpolated prediction maps).
- Land Use Regression (LUR) exploration.
- Satellite-derived indicators (NDVI) and DEM integration.
- Euskalmet API integration for official Basque meteorological data.

**Explicitly not doing**

- LSTM/GRU for production (MLP scored R² = 0.208; tree models dominate this tabular, weather-driven, daily regime — benchmark-only if ever).
- Retraining models just to add `station_code` (shown predict-inert; lag/rolling features already carry each station's signature implicitly).

---

## Acknowledgements

Developed during the AI & Data Tech training pathway, with the support, guidance, and learning ecosystem provided by:

**GAIA Cluster ICTA** · **DEMA – Agencia de Empleo y Emprendimiento** · **C2B**

Special thanks to GAIA for promoting innovation in Artificial Intelligence, Data Science, and Digital Transformation, and for providing an environment that encouraged the development of applied GeoAI and Smart City solutions.

---

## Author

**Arman Ghaziaskari Naeini**
GIS & Remote Sensing Specialist | Spatial Data Scientist | GeoAI Enthusiast
Bilbao, Spain

**Portfolio:** https://armanghazi.github.io/portfolio/projects
**Dashboard:** https://geoai-dashboard.streamlit.app/

---

## Data Sources & License

- Air quality: Basque Government Open Data (Open Data Euskadi).
- Meteorology: [Open-Meteo](https://open-meteo.com/) ERA5 Archive (CC BY 4.0).
