# Spatial GIS Analysis Report

## Smart City Air Quality Intelligence — Greater Bilbao

### Overview

This report presents the spatial analysis of air quality patterns across the Greater Bilbao metropolitan region using GIS methodologies and geospatial visualization techniques.

The objective is to identify geographic pollution structures, urban-environmental dynamics, and spatial typologies associated with industrial activity, transportation corridors, and coastal meteorology.

Spatial analysis was conducted using Python GIS libraries, including:

- GeoPandas
- Shapely
- Contextily
- Matplotlib
- Folium

---

# 1. Geographic Context

Greater Bilbao is structured around the Nervión-Ibaizabal estuary, historically shaped by:

- Heavy industry
- Port infrastructure
- Urban expansion
- Transportation corridors

This geography strongly influences pollutant accumulation and atmospheric dispersion.

The monitoring stations represent different environmental typologies:

- Urban core
- Industrial corridors
- Coastal environments
- Residential areas

---

# 2. Spatial Distribution of Pollutants

## Nitrogen Dioxide (NO2)

NO2 concentrations showed a clear urban spatial pattern.

### Hotspots

- MAZARREDO
- BASAURI
- BARAKALDO
- ERANDIO

### Interpretation

These stations are heavily influenced by:

- Road traffic
- Urban density
- Highway corridors
- Urban canyon effects

The highest concentrations were found in the Bilbao city center and along major transportation routes.

### Coldspots

- MUSKIZ
- ALGORTA

Coastal ventilation and lower traffic density contribute to cleaner atmospheric conditions in these areas.

---

## PM10

PM10 displayed strong spatial variability across the metropolitan area.

### Hotspots

- BASAURI
- BARAKALDO
- ALGORTA

### Interpretation

The elevated PM10 levels in BASAURI and BARAKALDO are associated with:

- Industrial activities
- Urban traffic
- Inland topographic confinement

The relatively high PM10 values observed in ALGORTA likely reflect marine aerosol contributions rather than combustion sources.

---

## PM2.5

PM2.5 concentrations showed a dominant urban-industrial signature.

### Hotspots

- BARAKALDO
- BASAURI

### Interpretation

These areas combine:

- Dense traffic flows
- Industrial emissions
- Freight transportation activity

The results indicate strong anthropogenic combustion influence.

---

## Sulfur Dioxide (SO2)

SO2 exhibited localized industrial patterns.

### Hotspots

- MAZARREDO
- MUSKIZ
- SANTURCE

### Interpretation

SO2 concentrations appear linked to:

- Port activity
- Industrial combustion
- Refinery operations

The influence of the Petronor refinery in Muskiz is particularly visible.

---

# 3. Integrated Spatial Typologies

By combining all pollutant layers, three major environmental zones were identified.

---

# Zone A — Industrial and Logistics Corridor

## Stations

- BASAURI
- BARAKALDO

## Characteristics

- High PM2.5
- High PM10
- Elevated NO2

## Interpretation

These areas function as industrial and freight transportation corridors where industrial activity and heavy vehicle traffic combine to create persistent particulate exposure.

---

# Zone B — Urban Core

## Stations

- MAZARREDO
- ERANDIO

## Characteristics

- Highest NO2 concentrations
- Moderate-to-high PM2.5
- Elevated SO2 variability

## Interpretation

This zone represents the dense urban center of Bilbao, strongly influenced by:

- Traffic emissions
- Urban morphology
- Reduced atmospheric dispersion

Topographic confinement along the estuary enhances pollutant accumulation.

---

# Zone C — Coastal and Peripheral Buffer

## Stations

- ALGORTA
- MUSKIZ
- SANTURCE

## Characteristics

- Lower NO2 concentrations
- Strong atmospheric ventilation
- Localized industrial signatures

## Interpretation

Coastal winds improve atmospheric cleaning and reduce pollutant accumulation. However, localized industrial and port-related activities still produce detectable SO2 and particulate impacts.

---

# 4. Spatial Environmental Insights

The GIS analysis demonstrates that air quality in Greater Bilbao is shaped by the interaction of:

- Urban density
- Industrial land use
- Transportation infrastructure
- Coastal meteorology
- Topographic confinement

The estuary acts as both an economic corridor and an environmental channel influencing pollutant transport and accumulation.

---

# 5. Smart City Perspective

This spatial analysis supports several Smart City applications:

- Environmental monitoring
- Pollution hotspot detection
- Sustainable mobility planning
- Industrial impact assessment
- Public health risk analysis

The integration of GIS and data science enables more intelligent environmental decision-making within metropolitan areas.

---

# 6. Conclusion

The spatial analysis revealed clear environmental structures across Greater Bilbao. Urban traffic dominates NO2 patterns, industrial corridors intensify particulate pollution, and coastal ventilation reduces pollutant accumulation in peripheral areas.

These findings provide a strong geospatial foundation for future stages of the project, including:

- Interactive GIS dashboards
- Machine learning forecasting
- Spatial clustering
- Smart City environmental intelligence systems
