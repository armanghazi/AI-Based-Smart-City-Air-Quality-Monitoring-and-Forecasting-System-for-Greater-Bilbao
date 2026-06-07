# 🌍 GeoAI Smart City Platform for Air Quality Monitoring and Forecasting in Greater Bilbao (Under Development)

## Overview

This project is an end-to-end GeoAI and Spatial Data Science platform designed to monitor, analyze, visualize, and forecast urban air quality across the Greater Bilbao Metropolitan Area (Basque Country, Spain).

The platform integrates:

* Geographic Information Systems (GIS)
* Environmental Data Science
* GeoAI & Machine Learning
* Spatial Analysis
* Meteorological Intelligence
* Urban Risk Assessment
* Interactive Dashboards
* Smart City Decision Support

The objective is to transform environmental monitoring data into actionable intelligence for sustainable urban planning and smart city governance.

---

## Project Objectives

The main goals of the project are:

* Monitor air pollution patterns across Greater Bilbao.
* Identify spatial hotspots and environmental risk zones.
* Analyze temporal trends and seasonal dynamics.
* Evaluate air quality against WHO guidelines.
* Understand meteorological drivers of pollution.
* Develop GeoAI forecasting models.
* Support evidence-based Smart City decision making.

---

## Study Area

Greater Bilbao Metropolitan Area

Bizkaia, Basque Country, Spain

Monitoring stations included:

| Station   | Environmental Class |
| --------- | ------------------- |
| Mazarredo | Urban               |
| Erandio   | Urban               |
| Basauri   | Industrial          |
| Barakaldo | Industrial          |
| Santurtzi | Port                |
| Algorta   | Coastal             |
| Muskiz    | Refinery            |

---

## Dataset

### Air Quality Dataset

Period:

**2015 – 2026**

Monitoring stations:

**7 Stations**

Total observations:

**29,000+ daily records**

Pollutants monitored:

| Pollutant | Description               |
| --------- | ------------------------- |
| PM2.5     | Fine Particulate Matter   |
| PM10      | Coarse Particulate Matter |
| NO₂       | Nitrogen Dioxide          |
| SO₂       | Sulfur Dioxide            |

Spatial attributes:

* Station
* Municipality
* Province
* Latitude
* Longitude

---

### Meteorological Dataset

Weather variables integrated into the platform:

| Variable       | Description                |
| -------------- | -------------------------- |
| Temperature    | Air Temperature            |
| Humidity       | Relative Humidity          |
| Precipitation  | Daily Rainfall             |
| Wind Speed     | Wind Intensity             |
| Wind Direction | Wind Bearing (degrees)     |
| Wind U         | East-West Wind Component   |
| Wind V         | North-South Wind Component |

Meteorological information is used to explain pollution dynamics and improve forecasting performance.

---

## Feature Engineering

The forecasting dataset includes:

### Temporal Features

* Year
* Month
* Day
* Day of Week
* Week of Year
* Day of Year
* Season

### Lag Features

Generated for:

* PM2.5
* PM10
* NO₂
* SO₂

Lag windows:

* 1 day
* 3 days
* 7 days
* 14 days
* 30 days
* 90 days
* 365 days

### Rolling Statistics

Generated rolling averages:

* 7 days
* 14 days
* 30 days
* 90 days
* 365 days

### Meteorological Features

* Temperature
* Humidity
* Precipitation
* Wind Speed
* Wind Direction
* Wind U Component
* Wind V Component

---

## Technologies

### GIS & Spatial Analysis

* QGIS
* GeoPandas
* Folium
* Shapely
* Contextily

### Data Science

* Python
* Pandas
* NumPy
* Scikit-Learn

### Machine Learning

* Random Forest
* XGBoost
* Gradient Boosting
* Spatial Feature Engineering

### Dashboard Development

* Streamlit
* Plotly
* Folium

### Data Storage

* Parquet
* GeoDataFrames

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
│   ├── 05_feature_engineering.ipynb
│   └── 06_forecasting.ipynb
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
│       ├── 4_Weather_Drivers.py
│       ├── 5_Forecasting.py
│       └── 6_Decision_Support.py
│
└── README.md
```

---

## Exploratory Data Analysis (EDA)

Main findings:

### PM2.5

* Strong urban-industrial signature.
* Seasonal variability.
* Higher concentrations during stable atmospheric conditions.

### PM10

* Influenced by industrial activity and marine aerosols.
* Higher variability than PM2.5.

### NO₂

* Strong traffic-related pollutant.
* Highest concentrations in urban environments.

### SO₂

* Associated with refinery and port activities.
* Lower concentrations overall but spatially differentiated.

---

## Spatial Analysis Results

GIS-based spatial assessment identified six environmental station typologies.

### 🚗 Urban Stations

#### Mazarredo

* Dense urban core
* Strong traffic influence
* Highest NO₂ levels
* Urban canyon effect

#### Erandio

* Metropolitan urban area
* Traffic and residential influence
* Transitional urban-industrial environment

---

### 🏭 Industrial Stations

#### Basauri

* Historical industrial zone
* Elevated PM2.5 and PM10
* Valley confinement effects

#### Barakaldo

* Industrial-logistics corridor
* High particulate concentrations
* Major transport hub

---

### ⚓ Port Station

#### Santurtzi

* Maritime activities
* Shipping emissions
* Port-related pollution dynamics

---

### 🌊 Coastal Station

#### Algorta

* Marine influence
* Strong atmospheric dispersion
* Sea-salt contribution to PM10

---

### 🛢️ Refinery Station

#### Muskiz

* Petronor refinery influence
* Energy sector emissions
* Coastal dispersion environment

---

## Interactive Dashboard

### Page 1 — Air Quality Monitoring

* Interactive GIS map
* Pollutant visualization
* Spatial exploration

### Page 2 — Temporal Trends

* Long-term trends
* Seasonal analysis
* Historical evolution

### Page 3 — Urban Risk Index

* WHO guideline assessment
* Station ranking
* Environmental risk scoring

### Page 4 — Weather Drivers & Air Pollution Dynamics

* Pollution-weather relationships
* Correlation analysis
* Wind and precipitation effects

### Page 5 — Forecasting

* Random Forest models
* XGBoost models
* Future pollutant prediction

### Page 6 — Smart City Decision Support

* Environmental intelligence
* Risk prioritization
* Urban planning support

---

## Future Development

Planned GeoAI enhancements:

* Spatial interpolation
* Land Use Regression (LUR)
* Random Forest Spatial Models
* XGBoost Spatial Prediction
* Weather-based forecasting
* NDVI integration
* DEM integration
* Satellite remote sensing
* Real-time API integration
* Smart City Environmental Intelligence System

The long-term vision is to create a GeoAI-powered Smart City platform combining GIS, environmental monitoring, machine learning, and decision-support capabilities.

---

## Live Dashboard

Dashboard:

https://geoai-dashboard.streamlit.app/

---

## Portfolio

More projects:

https://armanghazi.github.io/portfolio/projects

---

## Acknowledgements

This project was developed during the AI & Data Tech training pathway and has benefited from the support, guidance, and learning ecosystem provided by:

* GAIA Cluster ICTA
* DEMA – Agencia de Empleo y Emprendimiento
* C2B 

Special thanks to GAIA for promoting innovation in Artificial Intelligence, Data Science, and Digital Transformation, and for providing an environment that encouraged the development of applied GeoAI and Smart City solutions.

---

## Author

Arman Ghaziaskari Naeini

GIS & Remote Sensing Specialist | Spatial Data Scientist | GeoAI Enthusiast

Bilbao, Spain
