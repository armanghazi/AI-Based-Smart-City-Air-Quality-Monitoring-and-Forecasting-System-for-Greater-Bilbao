# 🌍 GeoAI Smart City Platform — Air Quality Intelligence for Greater Bilbao

<p align="center">
  <img src="dashboard/static/geoai_logo.svg" alt="GeoAI Logo" width="380"/>
</p>

**Status:** Live on Streamlit Cloud · actively developed  
**🔗 Live Dashboard:** https://geoai-dashboard.streamlit.app/

## Overview

An end-to-end **GeoAI and Spatial Data Science platform** that monitors, analyzes, visualizes, and forecasts urban air quality across the Greater Bilbao Metropolitan Area (Bizkaia, Basque Country, Spain).

The platform integrates Environmental Data Science, GIS, Machine Learning, and Interactive Dashboards to support Smart City decision-making and sustainable urban planning. As of mid-2026 it is no longer a static snapshot: an **automated daily pipeline** keeps the data live, **validated XGBoost models** forecast next-day pollution, **spatial GIS analysis across four notebooks** quantifies structural pollution drivers, **IDW interpolation masked to the metropolitan boundary** produces continuous surfaces, and a **conversational AI assistant** answers questions about both the data and the methodology directly from the dashboard.

---

## Project Objectives

- Monitor air pollution patterns across Greater Bilbao (2015–2026, updated daily).
- Identify spatial hotspots and pollution clusters by environmental zone.
- Analyze long-term temporal trends, including the COVID-19 impact.
- Assess urban environmental risk against **WHO 2021 guidelines**, **EU Directive 2008/50/EC**, and **EAQI/ICA** public index.
- **Forecast next-day air quality with validated ML models (XGBoost).**
- Explain model behavior with **SHAP** to ensure physically meaningful predictions.
- **Quantify structural spatial drivers** of air pollution through GIS buffer analysis, landmark distances, terrain analysis, and wind transport (Phase C — complete).
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

| Station   | Zone            | Signature                                                                       |
| --------- | --------------- | ------------------------------------------------------------------------------- |
| Mazarredo | 🏙️ Urban        | Highest NO₂ — traffic, urban canyon, 501 m from city centre                     |
| Erandio   | 🏙️ Urban        | Traffic influence, 1.3 km from AP-8 motorway                                    |
| Basauri   | 🏭 Industrial   | High PM2.5/PM10 — industrial land use 32% within 500 m                          |
| Barakaldo | 🏭 Industrial   | High PM, elevated NO₂ — 354 m from AP-8, highest road density in network        |
| Santurtzi | ⚓ Port         | Marine + traffic, elevated SO₂ — 784 m from Port of Bilbao                      |
| Algorta   | 🌊 Coastal      | Best dispersion — lowest road density, 2.6 km coast, NW sea breeze              |
| Muskiz    | 🛢️ Refinery     | Petronor petrochemical profile — paradoxically low PM due to coastal dispersion |

> Muskiz was initially classed as Coastal; GIS analysis revealed its refinery-driven emission profile warranted a separate zone. Despite having the highest industrial land use (34.6%), it records the network's lowest PM2.5 (6.5 µg/m³) due to coastal position and terrain complexity — the **MUSKIZ dispersion paradox**, confirmed by three independent GIS methods.

---

## Air Quality Standards — Three Layers

The platform uses three complementary standards, each answering a different question:

| Standard | Question | Values (NO₂) | Use in platform |
|----------|----------|--------------|-----------------|
| **EAQI / ICA** | Is the air safe to go outside today? | Good ≤ 40 µg/m³ | Public UI, alert banners, station cards, zone badges |
| **WHO 2021** | How far are we from the long-term health target? | Annual limit 10 µg/m³ | Risk Score, reference lines in charts, health analysis |
| **EU Directive 2008/50/EC** | Is the legal limit being broken? | Annual limit 40 µg/m³ | Daily Briefing alerts, Decision Support, PDF reports |

> EAQI is for public communication. WHO is for health science. EU Directive is for legal compliance. All three appear in the dashboard but in their appropriate context.

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
- **SHAP analysis** confirmed physically meaningful behavior: higher wind speed and precipitation consistently push predictions **down** (atmospheric dispersion, wet deposition). NO₂ shows strong day-of-week importance (traffic weekly cycle).
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
- `.github/workflows/daily_update.yml` — **GitHub Actions cron** runs daily (offset from top-of-hour to avoid scheduler congestion); the commit triggers a Streamlit Cloud auto-redeploy. No retraining.
- **Reliability:** idempotent append, D-0 protection (current incomplete day rejected), zero duplicates.

---

## Spatial GeoAI Analysis (Phase C — complete)

Four GIS notebooks deliver a **35-feature spatial context table** (`station_spatial_features_v3.csv`) for all 7 stations.

> **Statistical note:** All spatial correlations are computed across n = 7 stations and are exploratory and indicative only. Statistically significant inference requires ≥ 30 stations (standard LUR study design).

### Notebooks

| Notebook | Title               | Method                                                          | Key output                                   |
| -------- | ------------------- | --------------------------------------------------------------- | -------------------------------------------- |
| **10a**  | Buffer Analysis     | OSM land use + road density at 500m/1km/2km                     | green/industrial/residential %, road density |
| **10b**  | Distance Features   | Haversine distances to Port, Petronor, AP-8, coast, city centre | 6 landmarks × raw + log distance             |
| **10c**  | Elevation & Terrain | Copernicus GLO-30 DEM (30m)                                     | elevation, TRI, slope per station            |
| **10d**  | Wind Transport      | ERA5 direction × station cross-analysis                         | dispersion effect, directional signatures    |

### Top Spatial Findings

| Rank | Feature                    | Target | r         | Interpretation                                            |
| ---- | -------------------------- | ------ | --------- | --------------------------------------------------------- |
| 1    | Road density (1km buffer)  | NO₂    | **+0.83** | Traffic infrastructure = primary structural driver of NO₂ |
| 2    | Distance to city centre    | NO₂    | **−0.95** | Closest to centre = highest NO₂ — urban canyon effect     |
| 3    | Green cover (1km buffer)   | PM10   | **−0.66** | More vegetation = lower PM10 — dry deposition             |
| 4    | Terrain Relief Index (2km) | PM10   | **−0.63** | Complex terrain = stronger turbulent mixing               |
| 5    | Distance to AP-8 motorway  | PM2.5  | **−0.54** | BARAKALDO (354 m from AP-8) = highest PM2.5               |

### Wind Transport (ERA5 — single regional grid cell, ~31 km)

- Wind speed reduces NO₂ by **57%** from calm (< 10 m/s) to strong (> 25 m/s) conditions.
- Two dominant regimes: **NW/W (37.2% of days)** = Bay of Biscay sea air, NO₂ = 16.3 µg/m³ vs **S/SW (32.0%)** = inland recirculation, NO₂ = 21.4 µg/m³ (+31%).
- MUSKIZ under NE wind: SO₂ = **7.32 µg/m³** (Petronor trapped against terrain ridge, TRI = 343 m) — **1.93× higher** than under S wind.
- MAZARREDO under SE wind: NO₂ = **35.3 µg/m³** (+70% vs NW baseline) — Nervión valley channels SE flow.

---

## Risk Indicators

### Composite Risk Score

Used on the homepage and in **Page 6 — Smart City Decision Support**:

```
Risk Score = mean( concentration / WHO_annual_limit ) × 100
             across PM2.5, PM10, NO₂
```

- Score = 100 → station mean exactly at the WHO 2021 annual guideline
- Score > 100 → above WHO guideline
- Score > 200 → more than 2× the WHO limit
- SO₂ excluded (uses 24-hour guideline, different temporal basis)

### Structural Vulnerability Index (SVI)

A **time-invariant spatial index** that quantifies which stations are structurally predisposed to poor air quality — independent of daily weather conditions. Used in **Page 3 (GeoAI Spatial Analysis)** and **Page 6 (Spatial Intelligence tab)**.

**Formula:** z-score normalised composite of the three strongest spatial predictors:

| Feature                    | Direction                                 | r with NO₂ |
| -------------------------- | ----------------------------------------- | ---------- |
| Road density (1km buffer)  | Higher = more exposed                     | +0.83      |
| Distance to city centre    | Closer = more exposed (inverted)          | −0.77      |
| Terrain Relief Index (2km) | Higher TRI = better dispersion (inverted) | −0.63      |

```
SVI = rescale( mean( z_road_density, −z_dist_centre, −z_TRI ) ) → 0–100
```

**Validation:** SVI correlates with observed mean NO₂ at r = 0.77 across the network.

| Station   | SVI     | Structural reason                                                        |
| --------- | ------- | ------------------------------------------------------------------------ |
| BARAKALDO | **100** | Highest road density (21,267 m/km²) + 354 m from AP-8                    |
| MAZARREDO | 89.8    | Road density 19,060 m/km² + 501 m from city centre                       |
| ERANDIO   | 87.4    | 1,264 m from AP-8 + 18,631 m/km² road density                            |
| BASAURI   | 51.7    | Industrial land use 32% within 500 m                                     |
| ALGORTA   | 34.7    | Lowest road density (9,933 m/km²) + 2.6 km coast                         |
| SANTURCE  | 10.5    | TRI 445 m provides terrain-driven dispersion buffer                      |
| MUSKIZ    | **0**   | Coastal + TRI 343 m → structurally cleanest despite industrial proximity |

> **Important limitation:** SVI captures traffic and terrain drivers only. It does not represent port emissions (SANTURCE) or industrial point sources (MUSKIZ) — these require wind direction analysis rather than static distance metrics.

---

## Interactive Dashboard (10 pages)

| Page | Contents |
|------|----------|
| **0 · Daily Briefing** | Automated daily summary, D-1 air quality across all stations, EAQI badges per station, PDF export |
| **1 · Air Quality Monitoring** | Interactive Folium GIS map with AQI/SVI toggle, EAQI station cards, zone comparison, station ranking |
| **2 · Temporal Trends** | Long-term & seasonal trends, COVID-19 impact, GeoAI Pattern Discovery; WHO/EU/EAQI toggleable reference lines on all charts |
| **3 · GeoAI Spatial Analysis** | 4 tabs: Station DNA profiles · Spatial driver correlation matrix (12 features × 4 pollutants) · Terrain & dispersion (Copernicus DEM) · Wind transport analysis (Sections A–D) |
| **4 · Weather Drivers** | Correlation matrix (with Wind Transport Sections A–D integrated in Tab 1), wind roses (shared color scale), lag analysis, rolling mean features, forecast feature ranking (full dataset) |
| **5 · Forecasting** | Next-day backtest with EAQI forecast box, recursive multi-day forecast, WHO/EU/EAQI reference line toggles, multi-day table with WHO/EU/EAQI columns, SHAP explainability |
| **6 · Smart City Decision Support** | 4 tabs: Current Status · Forecast & Map · Decisions & Actions (model-based + policy elasticity simulation) · Spatial Intelligence; Risk chart with WHO/EU/EAQI toggle lines; station table with EU Directive + EAQI columns |
| **7 · GeoAI Methodology** | Full methodology reference: platform overview, GIS findings, ML architecture, benchmark comparison, 6 honest scientific caveats |
| **8 · Project Assistant** | Conversational AI (OpenRouter / Llama 3.3 70B) — answers data AND methodology questions; parametric query tool (no SQL injection) |
| **9 · Smart City Operations** _(admin-only)_ | Enterprise-grade operational dashboard — DQS (4 components), active alerts, sensor health, >3σ outlier detection |

**Cross-cutting features:** EAQI/ICA + EPA dual AQI display, gauge indicators, calendar heatmaps, WHO + EU + EAQI reference lines with toggles, PDF export, favourite-station selector, bilingual interface (EN/ES), admin authentication (Google OAuth OIDC).

**D-1 rule (strictly enforced):** No label anywhere says "Today", "Currently", "Live", or "Tomorrow". The pipeline rejects the current incomplete day. All labels show explicit dates: "Latest readings (D-1)" / "Next-day forecast — [date]".

---

## Authentication & Access Control

- **Public pages (0–8):** accessible to all visitors without login.
- **Admin page (9):** protected by Google OAuth OIDC (`auth.py`). Pages absent from `st.navigation` cannot render at all — this is genuine access control, not cosmetic.
- Admin is auto-redirected to the Operations Dashboard after login via `_post_login_routed` session flag.

---

## Conversational AI Assistant (Page 8)

The Project Assistant answers questions about **both the live data and the project methodology**:

- **Data questions** (trends, comparisons, WHO exceedance rates, specific dates) → parametric query tool runs directly against the live parquet — no SQL injection risk, no hallucinated numbers.
- **Methodology questions** (time-based split, SO₂ handling, SHAP, GIS findings, SVI) → answered from a grounded project knowledge block including all spatial analysis findings.
- **Spatial questions** (why is MUSKIZ clean, what is BARAKALDO's SVI, how does NE wind affect SO₂) → answered from GIS notebook findings embedded in the system prompt.
- Backend: **OpenRouter API** (Llama 3.3 70B). Responds in the user's language automatically (EN/ES/FA and others).

---

## Smart City Operations Dashboard (Page 9 — admin)

**Data Quality Score (DQS)** — weighted composite (0–100):

| Component    | Formula                                | Weight |
| ------------ | -------------------------------------- | ------ |
| Freshness    | `100 − 20 × days_behind`               | 0.35   |
| Completeness | `100 × (1 − missing_ratio)`            | 0.30   |
| Integrity    | `100 − 5 × duplicates − 10 × invalids` | 0.20   |
| Stability    | `100 − 2 × outliers_last_7d (>3σ)`     | 0.15   |

Outlier detection uses **>3σ rule** — explicitly labelled as a statistical method, not an AI/ML anomaly model.

---

## Technologies

**GIS & Spatial:** GeoPandas · Shapely 2.1.2 · OSMnx · Folium · rasterio · rasterstats  
**Data Science:** Python 3.14 · Pandas · NumPy · Scikit-Learn · statsmodels  
**Machine Learning:** XGBoost (production) · MLP · SARIMA · SHAP (benchmarks)  
**Dashboard:** Streamlit (Cloud) · Plotly ≤ 5.22.0 · streamlit-folium ≥ 0.27.2  
**AI/LLM:** OpenRouter API (Llama 3.3 70B) · parametric query tool (pandas)  
**Auth:** Google OAuth OIDC (`auth.py`, Authlib)  
**Data sources:** Open Data Euskadi API · Open-Meteo ERA5 · Copernicus GLO-30 DEM · OSM (via OSMnx) · Open Data Euskadi COMARCAS (EPSG:25830)  
**Ops:** Parquet · GitHub Actions (daily cron, offset scheduling) · pytest (11 tests)  
**i18n:** Bilingual EN/ES via `i18n_auto.py` (deep-translator)

---

## Key EDA Findings

- **PM2.5** — flat seasonality (unlike NO₂); urban/industrial stations highest; long-term structural decline driven by fleet electrification and Euro standards.
- **PM10** — greater variability; dust events; Industrial zone worst in summer (12.3 µg/m³) due to reduced wet deposition; Refinery best in summer (7.1 µg/m³) due to stronger NW sea breeze.
- **NO₂** — clear traffic signature with weekly cycle (weekday **+32%** vs weekend); significant reduction over the decade (**−41%** from 2015 to 2026); sharp COVID-19 dip (−22% in 2020, largest at high-road-density stations); winter paradox: strongest wind yet highest NO₂ due to lower boundary layer height.
- **SO₂** — low overall concentrations; episodic industrial/port/refinery origin; MUSKIZ SO₂ peaks under NE wind (**1.93×** S-wind baseline, Petronor terrain trapping); SANTURCE SO₂ driven by Port proximity (0.8 km), partially buffered by TRI = 445 m.

---

## Project Architecture

```
project/
│
├── data/
│   ├── raw/                                 # gitignored — original CSVs per station/year,
│   │                                         # fetched via Open Data Euskadi API (notebook 00)
│   └── processed/
│       ├── air_quality_weather.parquet       # dashboard source — live, pipeline appends daily
│       ├── forecasting_dataset.parquet       # ML-ready snapshot (62 features) — FROZEN
│       ├── station_spatial_features_v3.csv   # 7 stations × ~35 spatial features (GIS Phase C)
│       ├── weather_data.csv
│       └── archive/                          # superseded intermediate files, kept for history
│           ├── air_quality_bilbao_2012_2026.csv
│           ├── cleaned_air_quality_bilbao_2015_2026.csv
│           ├── final_air_quality.parquet
│           ├── station_spatial_features_v2.csv
│           └── station_spatial_features_long.csv
│
├── met/                                     # gitignored — raw per-station weather pulls
│                                             # (Open-Meteo ERA5), merged into the processed
│                                             # parquet above; see met/openmeteo.ipynb locally
│
├── notebooks/
│   ├── 00_api_exploration.ipynb
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb               # MICE imputation (training snapshot only)
│   ├── 03_EDA.ipynb
│   ├── 04_gis_spatial_analysis.ipynb        # zone classification
│   ├── 05_weather_data.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_model_training.ipynb
│   ├── 08_benchmark_extensions.ipynb
│   ├── 10a_spatial_buffer_analysis.ipynb    # OSM land use + road density at 3 buffer radii
│   ├── 10b_distance_features.ipynb          # Haversine distances to 6 landmarks
│   ├── 10c_elevation_terrain_features.ipynb # Copernicus GLO-30 DEM: elevation, TRI, slope
│   └── 10d_wind_direction_analysis.ipynb    # ERA5 wind transport: dispersion + directional signatures
│
├── scripts/
│   ├── daily_update.py                      # Euskadi API + Open-Meteo → parquet append
│   └── verify_pipeline.py
│
├── .github/workflows/
│   └── daily_update.yml                     # scheduled cron (offset) + auto-redeploy
│
├── models/                                  # production joblib bundles (frozen)
│   ├── xgb_pm25_forecast.joblib
│   ├── xgb_pm10_forecast.joblib
│   ├── xgb_no2_forecast.joblib
│   └── xgb_so2_forecast.joblib
│
├── GIS/
│   ├── boundaries/COMARCAS_5000_ETRS89.shp
│   ├── dem_N43_W003.tif                     # Copernicus GLO-30 tile (30m)
│   ├── dem_N43_W004.tif                     # Copernicus GLO-30 tile (30m)
│   └── dem_bilbao_utm30n.tif                # merged + reprojected to EPSG:25830
│
├── maps/                                    # static PNG exports from GIS notebooks
│                                             # (correlation charts, wind roses, station maps)
│
├── reports/                                 # markdown write-ups from EDA / spatial analysis
│
├── dashboard/
│   ├── app.py                               # st.navigation router — clustered layout
│   ├── config.py                            # shared loader, zones, WHO + EU + EAQI limits
│   ├── auth.py                              # OIDC gate: require_auth, is_admin, logout_button
│   ├── aqi.py                               # EAQI/ICA + EPA dual index (single source of truth)
│   ├── aqi_components.py                    # render_aqi_donut, render_station_aqi_cards, render_aqi_calendar
│   ├── assistant_core.py                    # shared AI backend: tools, TOOL_SPEC, get_reply, system prompt
│   ├── forecast_utils.py                    # shared prepare_features (62 features)
│   ├── gauge_component.py                   # WHO-referenced gauge row
│   ├── glossary.py                          # 36 plain-English definitions for tooltips
│   ├── i18n_auto.py                         # bilingual EN/ES — language_selector + tr()
│   ├── pdf_report.py                        # PDF export with _safe() Unicode sanitization
│   ├── spatial_utils.py                     # IDW + comarca boundary mask (EPSG:25830)
│   ├── weather_panel.py                     # weather_snapshot + weather_trend components
│   ├── assets/                              # pre-computed SHAP plots (PNG)
│   ├── static/                              # logo + branding assets
│   ├── .streamlit/                          # config.toml · secrets.toml (gitignored)
│   └── pages/
│       ├── 0_Daily_Briefing.py              # D-1 status snapshot, EAQI badges, PDF export
│       ├── 1_Air_Quality_Monitoring.py      # Folium map (AQI/SVI toggle), EAQI station cards
│       ├── 2_Temporal_Trends.py             # trends, COVID, seasonality; WHO/EU/EAQI toggle lines
│       ├── 3_GeoAI_Spatial_Analysis.py      # 4 tabs: DNA / Drivers / Terrain / Wind Transport
│       ├── 4_Weather_Drivers.py             # correlations, wind roses, lag, Wind Transport in Tab 1
│       ├── 5_Forecasting.py                 # backtest + EAQI box, multi-day table, SHAP
│       ├── 6_Smart_City_Decision_Support.py # 4 tabs, risk chart toggles, EU+EAQI table columns
│       ├── 7_Scope_and_Limitations.py       # GeoAI Methodology — 6 honest caveats
│       ├── 8_Project_Assistant.py           # AI chat (OpenRouter · Llama 3.3 70B)
│       └── 9_Smart_City_Operations.py       # admin-only: DQS, alerts, sensor health
│
├── tests/                                   # pytest suite (11 tests)
├── requirements.txt                         # streamlit-folium≥0.27.2, plotly≤5.22.0
├── LICENSE
└── README.md
```

> **Note:** `data/raw/` and `met/` are gitignored — they hold large per-station-per-year raw pulls that are fully reproducible from the public APIs (notebooks `00`–`01` and `met/openmeteo.ipynb`). Only processed, analysis-ready outputs are version-controlled.

---

## Engineering & Reliability

- **Shared logic, no duplication:** `prepare_features` in `forecast_utils.py`; `aqi.py` is the single source of truth for all AQI calculations; `auth.py` centralizes all access control; `assistant_core.py` holds all AI backend logic.
- **Reference lines architecture:** `_add_ref_lines(fig, poll)` helper pattern used consistently across pages 2, 4, 5, 6 — toggleable WHO (red dash), EU Directive (green dot), EAQI Good (teal longdash).
- **Test suite (pytest, 11 tests):** model-contract tests + spatial-utils tests — all passing.
- **Streamlit Cloud is Linux (case-sensitive):** all pages use `sys.path.insert` + `pathlib.Path(__file__).parent.parent` for imports.
- **Python 3.14** `string[pyarrow]` dtypes — features coerced to plain numeric before XGBoost `predict`.
- **Security:** secrets never hard-coded; `.gitignore` catches `secrets.toml` at any depth; Groq key leak resolved via `git reset --soft` + force-with-lease push.
- **st.navigation clustered layout:** pages absent from `st.navigation` cannot render at all — used for genuine Admin access control, not cosmetic hiding.

---

## Roadmap

**Completed**

- ✅ Phase A — Dashboard (10 pages, dual AQI, PDF export, AI assistant)
- ✅ Phase B — Automated daily pipeline (GitHub Actions CI/CD)
- ✅ Phase C — GIS spatial analysis (notebooks 10a–10d, SVI, 35-feature spatial table)
- ✅ Phase D — Dashboard refinement: EAQI integration across all pages, WHO/EU/EAQI toggle reference lines, AQI station cards, zone AQI badges, multi-standard forecast tables, OpenRouter migration

**Possible future work**

- Sentinel-2 NDVI integration (currently proxied by OSM green land-use fraction)
- Spatial forecast surfaces (per-zone prediction maps beyond IDW)
- Station-level wind data (currently single ERA5 grid cell ~31 km)
- Euskalmet API integration for official Basque meteorological data

**Explicitly not doing**

- LSTM/GRU for production (MLP scored R² = 0.208 — benchmark-only)
- Full LUR model (n = 7 stations is statistically insufficient; minimum ~30 required)
- Population Exposure Index (no validated population grid at required granularity)

---

## Acknowledgements

This project was developed during the **AI & Data Tech** training pathway, made possible by the comprehensive support, guidance, and learning ecosystem provided by:

- **GAIA Cluster ICTA**
- **DEMA – Agencia de Empleo y Emprendimiento**
- **C2B**

### Special Thanks

- **Aitor Donado** — exceptional instruction, valuable guidance, and continuous support throughout this course.
- **GAIA Cluster ICTA** — for driving innovation in Artificial Intelligence, Data Science, and Digital Transformation, and fostering an environment that inspired the development of applied **GeoAI** and **Smart City** solutions.

---

## Author

**Arman Ghaziaskari Naeini**  
_GIS & Remote Sensing Specialist | Spatial Data Scientist | GeoAI Engineer_  
Bilbao, Spain

- **Portfolio:** [armanghazi.github.io/portfolio](https://armanghazi.github.io/portfolio)
- **Dashboard:** [geoai-dashboard.streamlit.app](https://geoai-dashboard.streamlit.app/)

---

## Data Sources & License

- **Air Quality:** Basque Government Open Data — [Open Data Euskadi Air Quality API](https://opendata.euskadi.eus/api-air-quality/?api=air-quality) · CC BY 4.0
- **Meteorology:** [Open-Meteo](https://open-meteo.com/) ERA5 Archive · CC BY 4.0
- **GIS — DEM:** Copernicus GLO-30 Digital Elevation Model · CC BY 4.0
- **GIS — Land use:** OpenStreetMap via OSMnx · ODbL
- **GIS — Boundaries:** Open Data Euskadi COMARCAS shapefile (ETRS89/UTM 30N, EPSG:25830)
