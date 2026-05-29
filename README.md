# 🌍 Smart City Air Quality Monitoring and Forecasting for Greater Bilbao

An end-to-end Data Science, GIS, and Machine Learning project for analyzing and forecasting urban air pollution in the metropolitan area of Bilbao, Spain.

---

## 📌 Project Overview

This project integrates historical air quality measurements from multiple monitoring stations across **Greater Bilbao (Gran Bilbao)** to build a Smart City analytics platform capable of:

- Monitoring air pollution trends.
- Comparing pollution levels across urban, coastal, and industrial zones.
- Detecting seasonal patterns.
- Forecasting pollutant concentrations.
- Visualizing results through interactive maps and dashboards.

The project combines:

- 🐍 Python
- 📊 Data Analysis
- 🗺️ GIS
- 🤖 Machine Learning
- 🌐 Streamlit Dashboard

---

## 🎯 Objectives

1. Consolidate and clean multi-source air quality data.
2. Perform exploratory spatial and temporal analysis.
3. Engineer features for forecasting models.
4. Train machine learning models to predict pollutant concentrations.
5. Build an interactive dashboard for visualization.

---

## 📍 Study Area

The study area covers **Greater Bilbao**, located in the Basque Country, Spain.

Selected monitoring stations include:

- ALGORTA (Getxo)
- BARAKALDO
- BASAURI
- ERANDIO
- LEIOA
- MUSKIZ
- SANTURTZI

These stations represent different environmental contexts:

- Urban areas
- Industrial zones
- Coastal locations

---

## 📂 Data Sources

### 1. Air Quality Data

- World Air Quality Index (WAQI)
- Calidad del Aire en Euskadi

### 2. Station Metadata

- Official station information from `estaciones2026.xlsx`

### 3. Temporal Coverage

- 2015–2026

### 4. Temporal Resolution

- Daily observations

---

## 🧪 Selected Pollutants

The following pollutants were retained for the core analysis:

- NO₂ (Nitrogen Dioxide)
- PM10
- PM2.5
- SO₂

### Excluded Variables

The following variables were excluded due to high missing values and inconsistent coverage across stations:

- O₃
- CO
- CO 8h
- VOCs (Benzene, Toluene, Xylenes, etc.)

---

## 🧾 Final Dataset Structure

| Column    | Description                 |
| --------- | --------------------------- |
| Date      | Observation date            |
| year      | Year                        |
| station   | Monitoring station          |
| NO2       | Nitrogen dioxide            |
| PM10      | Particulate matter < 10 μm  |
| PM2_5     | Particulate matter < 2.5 μm |
| SO2       | Sulfur dioxide              |
| Province  | Province                    |
| Town      | Municipality                |
| Address   | Station address             |
| Latitude  | Latitude                    |
| Longitude | Longitude                   |

---

## 🏗️ Project Architecture

```text
Raw CSV Files (150+ files)
        +
Station Metadata (Excel)
        │
        ▼
ETL Pipeline (Pandas)
        │
        ▼
Clean Unified Dataset
        │
        ├── Data Cleaning
        ├── Exploratory Data Analysis
        ├── GIS Mapping
        ├── Feature Engineering
        ├── Machine Learning
        └── Streamlit Dashboard
```

---

## 📁 Repository Structure

```text
smart-city-air-quality/
│
├── data/
│   ├── raw/
│   ├── metadata/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling.ipynb
│   └── 06_dashboard.ipynb
│
├── src/
│   ├── etl.py
│   ├── preprocessing.py
│   ├── features.py
│   └── modeling.py
│
├── dashboard/
│   └── app.py
│
├── reports/
│   ├── figures/
│   └── presentation.pptx
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🔄 Methodology

### 1. Data Acquisition

- Download historical CSV files for each station.
- Collect official metadata.

### 2. ETL Pipeline

- Merge all CSV files.
- Standardize column names.
- Add station metadata.

### 3. Data Cleaning

- Handle missing values.
- Remove low-quality variables.
- Convert data types.

### 4. Exploratory Data Analysis

- Annual trends.
- Monthly seasonality.
- Station comparisons.
- Correlation analysis.

### 5. GIS Analysis

- Station maps.
- Pollution heatmaps.
- Spatial comparisons.

### 6. Feature Engineering

- Month, season, day of week.
- Lag variables.
- Rolling averages.

### 7. Machine Learning

- Forecast next-day NO₂ or PM2.5.
- Compare multiple models.

### 8. Dashboard Development

- Interactive visualizations.
- Map-based exploration.

---

## 📊 Exploratory Data Analysis Highlights

The EDA investigates:

- Long-term pollution trends.
- Seasonal variability.
- Spatial differences among stations.
- Relationships among pollutants.

Typical insights include:

- Higher NO₂ concentrations in winter.
- Elevated PM10 levels near industrial areas.
- Cleaner air at coastal stations.
- Overall improvement in air quality over time.

---

## 🗺️ GIS Analysis

Spatial analysis includes:

- Monitoring station locations.
- Average pollutant concentrations on maps.
- Heatmaps of air pollution.
- Urban vs. industrial comparisons.

Tools:

- GeoPandas
- Folium
- Plotly

---

## 🤖 Machine Learning

### Target Variables

- Next-day NO₂
- Next-day PM2.5

### Models

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### Evaluation Metrics

- RMSE
- MAE
- R²

---

## 🌐 Interactive Dashboard

The Streamlit dashboard includes:

- Pollutant selector.
- Station selector.
- Time-series charts.
- Interactive map.
- Forecast results.

---

## 📈 Example Results

Potential findings:

- Muskiz exhibits the highest PM10 concentrations.
- NO₂ peaks during winter months.
- Coastal stations show lower average pollution.
- Forecast models achieve strong predictive performance.

---

## 🛠️ Technologies Used

### Data Processing

- Python
- Pandas
- NumPy

### Visualization

- Matplotlib
- Plotly
- Seaborn

### GIS

- GeoPandas
- Folium

### Machine Learning

- Scikit-learn
- XGBoost

### Dashboard

- Streamlit

---

## 🚀 Installation

```bash
git clone https://github.com/armanghazi/AI-Based-Smart-City-Air-Quality-Monitoring-and-Forecasting-System-for-Greater-Bilbao.git
cd AI-Based-Smart-City-Air-Quality-Monitoring-and-Forecasting-System-for-Greater-Bilbao
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run notebooks

```bash
jupyter lab
```

### Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📌 Key Skills Demonstrated

- ETL pipeline design
- Data cleaning
- Exploratory Data Analysis
- GIS and geospatial visualization
- Time series feature engineering
- Machine learning
- Dashboard development
- Technical documentation

---

## 📄 CV Description

> Developed an end-to-end Smart City analytics platform for Greater Bilbao, integrating multi-station air quality data with GIS and machine learning to analyze and forecast urban pollution patterns.

---

## 🔮 Future Improvements

- Integrate meteorological variables.
- Use hourly data.
- Multi-day forecasting.
- Cloud deployment.
- Real-time API integration.

---

## 👤 Author

**Arman Ghaziaskari Naeini**

- GIS & Remote Sensing Specialist
- Data Scientist
- Python Developer

Portfolio:

- [https://armanghazi.github.io/portfolio/](https://armanghazi.github.io/portfolio/)

---

## 🙏 Acknowledgements

This project was developed as part of the training program:

**Laborlan 2026 – IA & Data Tech: Inteligencia Artificial y Gestión de Proyectos Tecnológicos**

Organized by:

- GAIA
- DEMA
- C2B

---

## ⭐ If You Like This Project

If you find this project useful, please consider giving it a star ⭐ on GitHub.
