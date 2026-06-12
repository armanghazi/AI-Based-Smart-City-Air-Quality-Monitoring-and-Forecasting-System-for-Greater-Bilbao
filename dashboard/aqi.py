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
    if pollutant not in AQI_THRESHOLDS or value is None:
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
    return result


def aqi_color(pollutant: str, value: float) -> str:
    """Convenience: just the hex color for a single pollutant value."""
    cat = compute_aqi_category(pollutant, value)
    return cat["color"] if cat else "#cccccc"