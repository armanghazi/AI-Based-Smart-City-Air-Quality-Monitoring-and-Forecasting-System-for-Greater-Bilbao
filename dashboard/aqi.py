"""
aqi.py — European / Spanish ICA Air Quality Index.

Shared module: import in dashboard pages (mirrors the forecast_utils pattern,
single source of truth — never duplicate this logic).

Standard: European Air Quality Index (EEA), which the Spanish national ICA
(MITECO) follows for the shared pollutants. Bands for PM2.5, PM10, NO2 are
based on WHO short-term exposure risk (HRAPIE); SO2 bands follow the EU Air
Quality Directive limit values.

NOTE ON RESOLUTION: the official index uses short-term concentrations
(daily for PM, hourly for NO2/SO2). This project's data is daily means, so the
result is a daily-mean-based AQI APPROXIMATION — label it as such in the UI.

The overall AQI of a location = the WORST (highest) category among its
pollutants — this is the official ICA/EAQI rule, not an average.
"""

from __future__ import annotations
import math

# ------------------------------------------------------------------
# CATEGORIES (1 = best, 6 = worst)
# ------------------------------------------------------------------

AQI_CATEGORIES = [
    {"level": 1, "label": "Good",            "label_es": "Buena",
     "color": "#50f0e6", "advice": "Air quality is good. Ideal for outdoor activity."},
    {"level": 2, "label": "Fairly good",     "label_es": "Razonablemente buena",
     "color": "#50ccaa", "advice": "Air quality is acceptable for most people."},
    {"level": 3, "label": "Moderate",        "label_es": "Regular",
     "color": "#f0e641", "advice": "Sensitive groups should consider limiting prolonged outdoor exertion."},
    {"level": 4, "label": "Poor",            "label_es": "Desfavorable",
     "color": "#ff5050", "advice": "Sensitive groups should reduce outdoor activity; others take care."},
    {"level": 5, "label": "Very poor",       "label_es": "Muy desfavorable",
     "color": "#960032", "advice": "Reduce outdoor activity. Sensitive groups should stay indoors."},
    {"level": 6, "label": "Extremely poor",  "label_es": "Extremadamente desfavorable",
     "color": "#7d2181", "advice": "Avoid outdoor activity. Health alert for everyone."},
]

# Quick lookup by level
CATEGORY_BY_LEVEL = {c["level"]: c for c in AQI_CATEGORIES}

# ------------------------------------------------------------------
# THRESHOLDS — upper bound (µg/m³) of each category, per pollutant.
# Index i corresponds to AQI level i+1. A value above the last finite
# bound falls into level 6.
# ------------------------------------------------------------------

AQI_THRESHOLDS = {
    # pollutant: [lvl1_max, lvl2_max, lvl3_max, lvl4_max, lvl5_max]  (lvl6 = above)
    "PM2.5": [10,  20,  25,  50,  75],
    "PM10":  [20,  40,  50, 100, 150],
    "NO2":   [40,  90, 120, 230, 340],
    "SO2":   [100, 200, 350, 500, 750],
}

AQI_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]


# ------------------------------------------------------------------
# CORE FUNCTIONS
# ------------------------------------------------------------------

def compute_aqi_category(pollutant: str, value: float) -> dict:
    """
    Map a concentration (µg/m³) to its AQI category for one pollutant.
    Returns the category dict (level, label, label_es, color, advice),
    plus the pollutant and value. Returns None if pollutant unknown or
    value missing.
    """
    if pollutant not in AQI_THRESHOLDS or value is None or math.isnan(value):
        return None

    bounds = AQI_THRESHOLDS[pollutant]
    level = 6  # default to worst; lowered if it fits a lower band
    for i, upper in enumerate(bounds):
        if value <= upper:
            level = i + 1
            break

    cat = dict(CATEGORY_BY_LEVEL[level])
    cat["pollutant"] = pollutant
    cat["value"] = round(float(value), 1)
    return cat


def overall_aqi(values: dict) -> dict:
    """
    Overall AQI of a location = the WORST category across its pollutants
    (official ICA/EAQI rule). 

    values: {"PM2.5": float, "PM10": float, "NO2": float, "SO2": float}
            (missing or None pollutants are skipped)

    Returns the worst category dict, with an added "driver" key naming the
    pollutant responsible. Returns None if no valid pollutant given.
    """
    worst = None
    driver = None

    for pol in AQI_POLLUTANTS:
        v = values.get(pol)
        if v is None:
            continue
        cat = compute_aqi_category(pol, v)
        if cat is None:
            continue
        if worst is None or cat["level"] > worst["level"]:
            worst = cat
            driver = pol

    if worst is None:
        return None

    result = dict(worst)
    result["driver"] = driver   # the pollutant that determined the AQI

    # EPA AQI — secondary international reference (worst pollutant, same driver)
    epa_val = compute_epa_aqi(driver, values.get(driver))
    result["epa_aqi"] = epa_val
    result["epa_label"] = epa_label(epa_val)
    return result


def aqi_color(pollutant: str, value: float) -> str:
    """Convenience: just the hex color for a single pollutant value."""
    cat = compute_aqi_category(pollutant, value)
    return cat["color"] if cat else "#cccccc"


# ------------------------------------------------------------------
# EPA AQI (US — secondary / international reference)
# ------------------------------------------------------------------

_EPA_BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    "PM2.5": [
        (0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500),
    ],
    "PM10": [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500),
    ],
    "NO2": [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500),
    ],
    "SO2": [
        (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
        (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500),
    ],
}

_EPA_LABELS = [
    (50,  "Good"),
    (100, "Moderate"),
    (150, "Unhealthy for Sensitive Groups"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
    (500, "Hazardous"),
]


def compute_epa_aqi(pollutant: str, value: float) -> int | None:
    """
    Compute the US EPA AQI for a single pollutant using linear interpolation.
    Returns a rounded int, or None if the pollutant is unknown or value is
    outside all defined breakpoint ranges.
    """
    breakpoints = _EPA_BREAKPOINTS.get(pollutant)
    if breakpoints is None or value is None:
        return None

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= value <= c_high:
            aqi = (i_high - i_low) / (c_high - c_low) * (value - c_low) + i_low
            return round(aqi)

    return None


def epa_label(aqi_value: int | None) -> str | None:
    """Map a numeric EPA AQI to its category label."""
    if aqi_value is None:
        return None
    for threshold, label in _EPA_LABELS:
        if aqi_value <= threshold:
            return label
    return "Hazardous"