# 🌍 GeoAI Smart City Platform for Air Quality Monitoring and Forecasting in Greater Bilbao (Under Development)

## Overview

This project is an end-to-end GeoAI and Spatial Data Science platform designed to monitor, analyze, visualize, and forecast urban air quality across the Greater Bilbao Metropolitan Area.

The platform integrates Environmental Data Science, Geographic Information Systems (GIS), Machine Learning, and Interactive Dashboards to support Smart City decision-making and sustainable urban planning.

The project combines:

- Spatial Data Analysis
- Air Quality Monitoring
- GeoAI & Machine Learning
- Interactive GIS Dashboards
- Urban Risk Assessment
- Environmental Intelligence

---

## Project Objectives

The main goals of the project are:

- Monitor air pollution patterns across Greater Bilbao.
- Identify spatial hotspots and pollution clusters.
- Analyze long-term temporal trends.
- Assess urban environmental risk using WHO guidelines.
- Forecast future air quality conditions using Machine Learning.
- Provide decision-support tools for Smart City stakeholders.

---

## Study Area

Greater Bilbao Metropolitan Area (Bizkaia, Basque Country, Spain)

Monitoring stations included:

- Algorta
- Barakaldo
- Basauri
- Erandio
- Mazarredo
- Muskiz
- Santurtzi

---

## Dataset

Air quality observations collected between:

**2015 – 2026**

Main pollutants:

| Pollutant | Description               |
| --------- | ------------------------- |
| PM2.5     | Fine particulate matter   |
| PM10      | Coarse particulate matter |
| NO₂       | Nitrogen dioxide          |
| SO₂       | Sulfur dioxide            |

Additional spatial attributes:

- Station
- Municipality
- Coordinates (Latitude / Longitude)
- Province
- Date

Final dataset:

- ~27,800 daily observations
- 7 monitoring stations
- Cleaned and validated data
- Stored in Parquet format

---

## Technologies

### GIS & Spatial Analysis

- QGIS
- GeoPandas
- Shapely
- Contextily
- Folium

### Data Science

- Python
- Pandas
- NumPy
- Scikit-Learn

### Machine Learning

- Random Forest
- XGBoost (planned)
- Spatial Feature Engineering

### Dashboard Development

- Streamlit
- Plotly
- Folium

---

## Project Architecture

```text
project/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_spatial_analysis.ipynb
│   └── 05_forecasting.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
│
├── dashboard/
│   ├── app.py
│   ├── config.py
│   └── pages/
│       ├── 1_Air_Quality_Monitoring.py
│       ├── 2_Temporal_Trends.py
│       ├── 3_Urban_Risk_Index.py
│       ├── 4_Forecasting.py
│       └── 5_Decision_Support.py
│
└── README.md
```

---

## Exploratory Data Analysis (EDA)

Key findings:

### PM2.5

- Stable long-term decrease.
- Higher concentrations in urban stations.
- Strong seasonality.

### PM10

- Greater variability.
- Influenced by dust events and meteorological conditions.
- Strong correlation with PM2.5.

### NO₂

- Strong urban traffic signature.
- Significant reduction over time.
- Highest levels in central urban stations.

### SO₂

- Low overall concentrations.
- Industrial and port-related influence.
- Independent behavior compared to other pollutants.

---

## Spatial Analysis Results

Three environmental zones were identified:

### 🏭 Industrial Corridor

Stations:

- Barakaldo
- Basauri

Characteristics:

- High PM2.5
- High PM10
- Elevated NO₂

---

### 🚗 Urban Core

Stations:

- Mazarredo
- Erandio

Characteristics:

- Highest NO₂ concentrations
- Strong traffic influence
- Urban canyon effects

---

### 🌊 Coastal Buffer Zone

Stations:

- Algorta
- Muskiz
- Santurtzi

Characteristics:

- Better atmospheric dispersion
- Lower NO₂ concentrations
- Marine influence on PM10

---

## Interactive Dashboard

Current modules:

### Page 1 — Air Quality Monitoring

- Interactive GIS Map
- Station-level pollutant visualization
- Spatial exploration

### Page 2 — Temporal Trends

- Long-term trends
- Annual evolution
- Seasonal analysis

### Page 3 — Urban Risk Index

- WHO Guideline comparison
- Pollution risk ranking
- Station benchmarking

### Page 4 — Forecasting (In Development)

- Random Forest models
- XGBoost models
- Time-series forecasting

### Page 5 — Smart City Decision Support (Planned)

- Urban environmental intelligence
- Risk prioritization
- Strategic planning support

---

## Future Development

Planned GeoAI enhancements:

- Spatial interpolation
- Land Use Regression (LUR)
- Random Forest Spatial Models
- XGBoost Spatial Prediction
- Satellite-derived indicators (NDVI)
- DEM integration
- Meteorological variables
- Smart City Environmental Risk Maps

Inspired by modern GeoAI ecosystems integrating advanced GIS technologies with cloud-based analytical platforms and next-generation spatial data infrastructures.

---

## Live Dashboard

Dashboard:

https://geoai-dashboard.streamlit.app/

---

## Portfolio

More projects:

https://armanghazi.github.io/portfolio/projects

---

## Author

Arman Ghaziaskari Naeini

GIS & Remote Sensing Specialist | Spatial Data Scientist | GeoAI Enthusiast

Bilbao, Spain
