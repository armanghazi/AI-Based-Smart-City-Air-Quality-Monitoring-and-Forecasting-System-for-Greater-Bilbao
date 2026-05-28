# Exploratory Data Analysis (EDA) Report

## Smart City Air Quality Analysis — Greater Bilbao (2015–2026)

### Project Overview

This project analyzes long-term air quality patterns across the Greater Bilbao metropolitan area using data collected from multiple monitoring stations between 2015 and 2026. The objective is to identify spatial, temporal, and environmental relationships between major atmospheric pollutants within an urban and industrial Smart City context.

The analysis focuses on four key pollutants:

- PM2.5
- PM10
- NO2
- SO2

The dataset combines historical measurements from air quality monitoring stations distributed across urban, industrial, coastal, and residential areas of Bizkaia.

---

# 1. Dataset Description

The final curated dataset contains:

- 27,883 daily observations
- 7 monitoring stations
- Time range: 2015–2026
- Geographic metadata:
  - Province
  - Town
  - Address
  - Latitude
  - Longitude

Selected stations:

- ALGORTA_BBIZI2
- BARAKALDO
- BASAURI
- ERANDIO
- MAZARREDO
- MUSKIZ
- SANTURCE

Stations with excessive missing values and inconsistent temporal coverage were removed to improve data reliability and model robustness.

---

# 2. Pollutant Distribution Analysis

## PM2.5

PM2.5 displayed moderate variability with clear spatial differences among stations. Urban stations showed higher concentrations and wider distributions, while coastal stations presented more compact and stable patterns.

Seasonal behavior was evident, with higher concentrations during winter and lower values during summer. Long-term analysis revealed a gradual downward trend, suggesting improvements in air quality and urban emission control.

The relationship between PM2.5 and PM10 was consistently strong across all stations.

---

## PM10

PM10 exhibited the highest variability among the analyzed pollutants. Urban and industrial stations such as BARAKALDO and BASAURI showed elevated concentrations and larger dispersion ranges.

Several natural outliers were identified, likely associated with:

- Saharan dust episodes
- Wind resuspension
- Construction activities

PM10 demonstrated strong correlation with PM2.5 but weaker relationships with gaseous pollutants.

---

## NO2

NO2 showed a clear urban traffic signature. Central and high-traffic stations presented elevated concentrations with compact and predictable distributions.

Temporal analysis revealed a significant downward trend over the last decade, likely associated with:

- Reduced traffic emissions
- Urban mobility changes
- Environmental policies

NO2 correlations with PM pollutants remained moderate, confirming partially independent urban emission dynamics.

---

## SO2

SO2 concentrations were generally low throughout the monitoring network but displayed localized industrial variability.

Stations near industrial or port-related areas exhibited stronger fluctuations and more frequent outliers. Correlation analysis confirmed that SO2 behaves independently from most urban pollutants, indicating distinct emission sources.

---

# 3. Correlation Analysis

## Pearson Correlation

Strong positive correlations were observed between:

- PM2.5 and PM10

These pollutants likely share common sources such as:

- Traffic emissions
- Road dust
- Urban activities

NO2 displayed weaker correlations with particulate matter, reflecting its stronger dependence on urban traffic emissions.

SO2 presented almost no correlation with other pollutants, reinforcing its industrial and port-related origin.

---

## Spearman Correlation

Spearman analysis confirmed the robustness of the PM2.5–PM10 relationship even under non-linear conditions.

NO2 maintained structural independence, while SO2 continued to behave as a distinct pollutant with separate environmental dynamics.

---

# 4. Station-Level Environmental Behavior

Each station exhibited unique environmental characteristics:

- Urban stations showed strong and stable pollutant correlations.
- Coastal stations demonstrated weaker correlations due to stronger atmospheric dispersion.
- Industrial stations displayed independent SO2 dynamics and elevated particulate concentrations.

Particularly:

- MAZARREDO showed a clear urban traffic profile.
- MUSKIZ reflected industrial and refinery influence.
- ALGORTA demonstrated coastal ventilation effects.

---

# 5. Temporal Trends

Long-term analysis revealed several important patterns:

## PM2.5 and PM10

- Progressive decline over time
- Strong seasonality
- Winter concentration peaks

## NO2

- Significant long-term reduction
- Strong urban behavior
- Stable temporal structure

## SO2

- Relatively stable over time
- Localized industrial variability

---

# 6. COVID-19 Impact

The COVID-19 period produced visible environmental changes:

- NO2 concentrations dropped significantly during lockdown periods.
- PM2.5 and PM10 showed moderate reductions.
- SO2 slightly increased after the pandemic, potentially reflecting industrial and port activity recovery.

These findings demonstrate the strong relationship between human mobility, industrial activity, and urban air quality.

---

# 7. Key Findings

- PM2.5 and PM10 exhibit strong structural relationships across all stations.
- NO2 acts as a clear urban traffic indicator.
- SO2 behaves independently and reflects industrial emissions.
- Coastal stations benefit from stronger atmospheric ventilation.
- Urban and industrial zones show the highest pollution exposure.

---

# 8. Conclusion

The EDA phase revealed clear spatial and temporal pollution patterns across Greater Bilbao. The results highlight the importance of urban morphology, industrial activity, traffic density, and coastal meteorology in shaping local air quality dynamics.

This analysis establishes a strong foundation for:

- Spatial GIS analysis
- Smart City visualization
- Machine learning forecasting
- Environmental intelligence systems
