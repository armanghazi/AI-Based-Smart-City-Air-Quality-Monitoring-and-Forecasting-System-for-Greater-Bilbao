# 🌍 GeoAI Smart City Platform for Air Quality Monitoring and Forecasting in Greater Bilbao (Under Development)

## Overview

An end-to-end **GeoAI and Spatial Data Science platform** that monitors, analyzes, visualizes, and forecasts urban air quality across the Greater Bilbao Metropolitan Area (Bizkaia, Basque Country, Spain).

The platform integrates Environmental Data Science, GIS, Machine Learning, and Interactive Dashboards to support Smart City decision-making and sustainable urban planning.

**🔗 Live Dashboard:** https://geoai-dashboard.streamlit.app/

---

## Project Objectives

- Monitor air pollution patterns across Greater Bilbao (2015–2026).
- Identify spatial hotspots and pollution clusters by environmental zone.
- Analyze long-term temporal trends, including the COVID-19 impact.
- Assess urban environmental risk against WHO 2021 guidelines.
- **Forecast next-day air quality with validated ML models (XGBoost).**
- Explain model behavior with **SHAP** to ensure physically meaningful predictions.
- Provide decision-support tools for Smart City stakeholders.

---

## Dataset

| Component       | Details                                                        |
| --------------- | -------------------------------------------------------------- |
| **Air quality** | Basque Government open data · 7 stations · 2015–2026           |
| **Meteorology** | Open-Meteo ERA5 Archive API (per-station, daily)               |
| **Records**     | ~29,008 daily observations after cleaning                      |
| **Pollutants**  | PM2.5, PM10, NO₂, SO₂                                          |
| **Weather**     | Temperature, Humidity, Precipitation, WindSpeed, WindDirection |
| **Storage**     | Parquet (fast, compact, Streamlit-friendly)                    |

Missing values (PM2.5 ≈ 33% missing) were recovered with **MICE iterative imputation** per station rather than row deletion.

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

Next-day pollutant concentration forecasting: `target(t+1) = pollutant(t+1)` per station, using today's pollutant levels, meteorology, lag/rolling features, and station encoding (~66 features).

### Validated Results (held-out test = 2024–2026)

| Pollutant | R²        | RMSE (µg/m³) | MAE (µg/m³) | Interpretation                               |
| --------- | --------- | ------------ | ----------- | -------------------------------------------- |
| **NO₂**   | **0.560** | 5.97         | 4.43        | Best — regular traffic weekly cycle          |
| **PM2.5** | 0.479     | 3.33         | 2.50        | Persistence + weather drivers                |
| **PM10**  | 0.460     | 6.26         | 4.45        | Dust events add variability                  |
| **SO₂**   | 0.390     | 1.87         | 1.23        | Hardest — episodic industrial/port emissions |

Production model: **XGBoost** (one model per pollutant, saved as joblib bundles with feature lists and metrics in `models/`).

### Methodological Rigor

- **Strict time-based split** (train < 2023 · validation 2023 · test ≥ 2024). An early row-based split produced an inflated R² = 0.84 by mixing stations' time ranges; this leakage was identified and corrected — the honest figures above resulted.
- **Today's pollutant value is kept as a feature** — it is available at prediction time, hence valid (not leakage). Removing it drops PM2.5 R² from 0.48 to 0.34.
- **TimeSeriesSplit** used for hyperparameter search; tuning did not beat sensible baseline parameters, so the simpler model was retained.
- **SHAP analysis** confirmed physically meaningful behavior: higher wind speed and precipitation consistently push predictions **down** (atmospheric dispersion, wet deposition) — the model learned real mechanisms, not spurious correlations.
- Pollutant-specific signatures: NO₂ shows strong `day_of_week` importance (traffic); SO₂ lacks temporal regularity (episodic origin) — explaining its lower predictability.

### Benchmark Extensions (notebook 08)

A full methodological comparison on identical splits:

| Model               | Type          | Scope          | R²        |
| ------------------- | ------------- | -------------- | --------- |
| **XGBoost**         | ML (trees)    | All stations   | **0.479** |
| MLP (128, 64)       | Deep learning | All stations   | 0.208     |
| SARIMA one-step     | Classical TS  | Mazarredo only | 0.463     |
| SARIMA long-horizon | Classical TS  | Mazarredo only | ≈ 0.00    |

**Why XGBoost for production:** SARIMA one-step nearly matches it on a single regular station — confirming short-horizon forecasting is persistence-dominated — but would require 28 separate models (7 stations × 4 pollutants), ignores meteorological drivers, and collapses entirely on multi-day horizons. The MLP result is consistent with literature on tree-model dominance for medium-sized tabular data (Grinsztajn et al., 2022).

---

## Interactive Dashboard (6 modules)

| Page                                         | Status     | Contents                                                                                                                                          |
| -------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 · Air Quality Monitoring                   | ✅         | Interactive GIS map, station comparison, WHO risk levels                                                                                          |
| 2 · Temporal Trends                          | ✅         | Long-term & seasonal trends, COVID-19 impact analysis                                                                                             |
| 3 · Urban Risk Index                         | ✅         | WHO benchmarking, risk scoring, station ranking                                                                                                   |
| 4 · Weather Drivers & Air Pollution Dynamics | ✅         | Correlation matrices, wind roses, lag analysis, feature ranking                                                                                   |
| 5 · Forecasting                              | ✅         | **Next-day backtest** (prediction vs actual on 2024+) + **recursive multi-day forecast** with explicit uncertainty warnings, WHO exceedance flags |
| 6 · Smart City Decision Support              | 🔄 Planned | Risk prioritization, exportable reports                                                                                                           |

---

## Technologies

**GIS & Spatial:** QGIS · GeoPandas · Shapely · Folium · Contextily
**Data Science:** Python · Pandas · NumPy · Scikit-Learn · statsmodels
**Machine Learning:** XGBoost · Random Forest · Gradient Boosting · MLP · SARIMA · SHAP
**Dashboard:** Streamlit · Plotly · Streamlit-Folium
**Data:** Parquet · Open-Meteo API · Basque Government Open Data

---

## Project Architecture

```
project/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── forecasting_dataset.parquet      # ML-ready (66 features)
│       └── air_quality_weather.parquet      # dashboard source
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb               # MICE imputation
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
├── models/                                  # production joblib bundles
│   ├── xgb_pm25_forecast.joblib             # model + features + metrics
│   ├── xgb_pm10_forecast.joblib
│   ├── xgb_no2_forecast.joblib
│   └── xgb_so2_forecast.joblib
│
├── dashboard/
│   ├── app.py
│   ├── config.py                            # shared loader, zones, WHO limits
│   └── pages/
│       ├── 1_Air_Quality_Monitoring.py
│       ├── 2_Temporal_Trends.py
│       ├── 3_Urban_Risk_Index.py
│       ├── 4_Weather_Drivers_&_Air_Pollution_Dynamics.py
│       ├── 5_Forecasting.py
│       └── 6_Smart_City_Decision_Support.py  (planned)
│
├── GIS/  ·  maps/  ·  reports/
├── requirements.txt
└── README.md
```

---

## Key EDA Findings

- **PM2.5** — stable long-term decrease; urban/industrial stations highest; strong seasonality.
- **PM10** — greater variability; dust events; strong correlation with PM2.5.
- **NO₂** — clear traffic signature with weekly cycle; significant reduction over the decade; sharp COVID-19 dip.
- **SO₂** — low overall concentrations; industrial/port/refinery origin; behaves independently of other pollutants.

---

## Future Development

- **Page 6 — Smart City Decision Support** (risk prioritization, exports)
- Spatial forecast maps (per-zone prediction surfaces)
- Spatial ML with DEM and satellite-derived NDVI
- Land Use Regression (LUR) exploration
- LSTM/GRU sequence models (future work — current daily-resolution regime favors gradient boosting)
- Euskalmet API integration for official Basque meteorological data

---

## Acknowledgements

This project was developed during the AI & Data Tech training pathway and has benefited from the support, guidance, and learning ecosystem provided by:

**GAIA Cluster ICTA**

**DEMA – Agencia de Empleo y Emprendimiento**

**C2B**

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

- Air quality: Basque Government Open Data (Open Data Euskadi)
- Meteorology: [Open-Meteo](https://open-meteo.com/) ERA5 Archive (CC BY 4.0)
