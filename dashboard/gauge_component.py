"""
Gauge row for the Smart City Decision Support page.

Drop this into 6_Smart_City_Decision_Support.py (or import it).
Shows one WHO-referenced gauge per pollutant for the latest day's
city-wide mean. SO2 uses the 24-hour WHO guideline; the others use annual.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_gauge_row(values: dict, who_limits: dict, so2_limit: float):
    """
    values:     {"PM2.5": float, "PM10": float, "NO2": float, "SO2": float}
                — current (e.g. latest-day city-mean) concentration µg/m³
    who_limits: annual WHO limits for PM2.5 / PM10 / NO2
    so2_limit:  24-hour WHO limit for SO2 (handled separately)
    Returns a Plotly figure with 4 gauges in one row.
    """
    pollutants = ["PM2.5", "PM10", "NO2", "SO2"]

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
        horizontal_spacing=0.06,
    )

    for i, pol in enumerate(pollutants):
        limit = so2_limit if pol == "SO2" else who_limits[pol]
        val = values.get(pol, 0.0)

        # Axis tops out at 4× the limit so the needle has room when polluted
        axis_max = limit * 4

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=round(val, 1),
                number={"suffix": " µg/m³", "font": {"size": 18}},
                delta={
                    "reference": limit,
                    "increasing": {"color": "#e74c3c"},
                    "decreasing": {"color": "#2ecc71"},
                    "suffix": " vs WHO",
                },
                title={"text": f"<b>{pol}</b>", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, axis_max], "tickwidth": 1},
                    "bar": {"color": "#2c3e50", "thickness": 0.25},
                    "steps": [
                        {"range": [0, limit],            "color": "#2ecc71"},
                        {"range": [limit, limit * 2],    "color": "#f39c12"},
                        {"range": [limit * 2, axis_max], "color": "#e74c3c"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.8,
                        "value": limit,
                    },
                },
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(
        height=260,
        margin=dict(t=50, b=10, l=20, r=20),
    )
    return fig