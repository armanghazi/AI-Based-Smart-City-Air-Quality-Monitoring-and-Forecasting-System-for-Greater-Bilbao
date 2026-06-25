"""
dashboard/glossary.py

Central glossary of technical terms used across the dashboard.
Each entry has a plain-English, non-expert definition designed for
the `help=` tooltip parameter on Streamlit widgets.

Usage:
    from glossary import G
    st.metric("TRI 2km", "343 m", help=G["TRI"])

The wording deliberately avoids jargon-to-explain-jargon. A recruiter
or city official should understand each definition without prior
domain knowledge.
"""

# ──────────────────────────────────────────────────────────────────────────────
# GLOSSARY — plain-English definitions
# ──────────────────────────────────────────────────────────────────────────────

G = {
    # ── Spatial / GIS terms ───────────────────────────────────────────────────
    "TRI": (
        "Terrain Relief Index — a measure of how hilly the surrounding area is. "
        "Higher values mean more complex terrain, which helps mix and disperse "
        "polluted air. Flat areas trap pollution more easily."
    ),
    "SVI": (
        "Structural Vulnerability Index (0–100) — how exposed a station is to "
        "poor air quality based purely on its location: nearby roads, distance "
        "to the city centre, and terrain. Higher = structurally more at risk, "
        "regardless of the weather on any given day."
    ),
    "buffer": (
        "A circular zone drawn around each monitoring station (e.g. 1 km radius) "
        "used to measure what surrounds it — how many roads, how much greenery, "
        "how much industry."
    ),
    "road_density": (
        "Total length of roads within a given radius of the station, divided by "
        "the area. More roads nearby usually means more traffic emissions."
    ),
    "IDW": (
        "Inverse Distance Weighting — a method to estimate pollution at locations "
        "between monitoring stations. Nearby stations count more than distant ones. "
        "It is a visual estimate, not a measurement."
    ),
    "Haversine": (
        "A formula to calculate the straight-line distance between two points on "
        "the Earth's surface, accounting for the planet's curvature."
    ),
    "comarca": (
        "A Spanish administrative district. Here it refers to the Greater Bilbao "
        "metropolitan boundary used to clip the pollution maps."
    ),
    "EPSG_25830": (
        "A coordinate system (ETRS89 / UTM zone 30N) used for accurate distance "
        "and area calculations in the Basque region."
    ),

    # ── Air quality standards ─────────────────────────────────────────────────
    "WHO_annual": (
        "World Health Organization 2021 annual air quality guidelines — the "
        "yearly average pollution levels considered safe for long-term health. "
        "Stricter than EU legal limits."
    ),
    "EAQI": (
        "European Air Quality Index — the official EU rating of air quality, "
        "from 'Good' to 'Extremely Poor'. A location's rating equals its worst "
        "pollutant, not the average."
    ),
    "EPA_AQI": (
        "US Environmental Protection Agency Air Quality Index — an alternative "
        "0–500 scale shown as a secondary reference. The European and US systems "
        "answer slightly different questions."
    ),
    "risk_score": (
        "Composite Risk Score — the station's average pollution as a percentage "
        "of the WHO safe limit. 100 = exactly at the limit; above 200 = more than "
        "twice the recommended maximum. Combines PM2.5, PM10 and NO₂."
    ),
    "exceedance": (
        "A day when a pollutant rose above its guideline limit."
    ),

    # ── Pollutants ────────────────────────────────────────────────────────────
    "NO2": (
        "Nitrogen dioxide — a gas mainly from vehicle exhaust. A direct marker of "
        "traffic pollution."
    ),
    "PM25": (
        "Fine particulate matter under 2.5 micrometres — tiny particles that "
        "penetrate deep into the lungs. From combustion, traffic, and industry."
    ),
    "PM10": (
        "Coarse particulate matter under 10 micrometres — dust and larger "
        "particles from roads, construction, and natural sources."
    ),
    "SO2": (
        "Sulphur dioxide — a gas from burning fossil fuels, especially at "
        "refineries and ports. Episodic rather than constant."
    ),

    # ── Machine learning ──────────────────────────────────────────────────────
    "R2": (
        "R-squared — how well the model's predictions match reality, from 0 to 1. "
        "0.50 means the model explains about half of the day-to-day variation. "
        "Higher is better."
    ),
    "RMSE": (
        "Root Mean Squared Error — the typical size of the model's prediction "
        "error, in the same units as the pollutant (µg/m³). Lower is better."
    ),
    "MAE": (
        "Mean Absolute Error — the average gap between predicted and actual "
        "values, in µg/m³. Lower is better."
    ),
    "XGBoost": (
        "A machine-learning method that combines many simple decision rules into "
        "a strong predictor. The production forecasting model used here."
    ),
    "SHAP": (
        "A technique that explains why the model made a specific prediction — "
        "which factors pushed the forecast up or down. Confirms the model learned "
        "real cause-and-effect, not coincidence."
    ),
    "time_split": (
        "Training the model on older data and testing it on newer data it has "
        "never seen — the honest way to check if forecasts will work in the future."
    ),
    "leakage": (
        "When a model accidentally 'sees' the answer during training, giving "
        "falsely high accuracy. Avoided here by strict time-based testing."
    ),
    "feature": (
        "An input the model uses to make a prediction — e.g. today's pollution, "
        "wind speed, day of the week."
    ),
    "lag_feature": (
        "A past value used as an input — e.g. yesterday's pollution level helps "
        "predict tomorrow's."
    ),

    # ── Statistics ────────────────────────────────────────────────────────────
    "Pearson_r": (
        "Pearson correlation (−1 to +1) — how strongly two things move together. "
        "+0.83 = strong positive link; −0.77 = strong inverse link; near 0 = no link. "
        "Correlation is not proof of cause."
    ),
    "z_score": (
        "How far a value sits from the average, measured in standard deviations. "
        "Used to put different measurements on a comparable scale."
    ),
    "outlier_3sigma": (
        "A reading more than 3 standard deviations from normal — statistically "
        "unusual, flagged for review as a possible sensor error or real event."
    ),

    # ── Weather / dispersion ──────────────────────────────────────────────────
    "dispersion": (
        "How wind and air movement spread pollution out and dilute it. Strong "
        "wind disperses pollution; calm air lets it accumulate."
    ),
    "boundary_layer": (
        "The layer of air near the ground where pollution mixes. In winter it is "
        "lower and thinner, trapping pollution closer to the surface."
    ),
    "ERA5": (
        "A global weather dataset. Here all stations share one regional weather "
        "cell (~31 km), so wind reflects the whole area, not each exact street."
    ),
    "wind_regime": (
        "A recurring wind pattern. Greater Bilbao alternates between clean sea "
        "air from the northwest and more polluted recirculated air from the south."
    ),

    # ── Data engineering ──────────────────────────────────────────────────────
    "D1_data": (
        "All data is at least one day old (D-1). The pipeline waits for a complete "
        "day before showing it, so partial readings never create false alerts."
    ),
    "DQS": (
        "Data Quality Score (0–100) — an automated health check of the data: is it "
        "fresh, complete, consistent, and stable?"
    ),
    "parquet": (
        "A compact, fast file format for storing the dataset."
    ),
}


def g(term: str, fallback: str = "") -> str:
    """Safe glossary lookup — returns fallback if term is missing."""
    return G.get(term, fallback)