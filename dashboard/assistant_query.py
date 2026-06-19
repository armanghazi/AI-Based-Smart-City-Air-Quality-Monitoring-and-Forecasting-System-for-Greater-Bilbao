"""
assistant_query.py — parametric, injection-proof data tool for the Project Assistant.

Instead of letting the LLM write SQL, it returns a small set of validated parameters
(stations, pollutants, date range, grouping, aggregation). This module turns those into
a pandas query over the SAME load_data() the dashboard uses, so numbers always match the
pages. There is no SQL and no eval — only whitelisted parameters — so it cannot be injected.

Exposes:
  - TOOL_SPEC   : OpenAI/Groq function-calling schema
  - run_query() : executes a validated query and returns a compact JSON-able dict
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import load_data, WHO_ANNUAL, WHO_SO2_DAILY  # noqa: E402

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]

# group_by value -> column produced by config.load_data()
GROUP_COLS = {
    "none": None,
    "day": "Day",
    "month": "YearMonth",
    "year": "Year",
    "station": "station",
    "zone": "Zone",
    "season": "season",
}
AGGS = {"mean", "max", "min", "count", "exceedance_rate"}
MAX_ROWS = 60  # cap rows returned to the model (keeps us under Groq free-tier token limits)


def _limit_for(pollutant: str) -> float | None:
    """WHO reference: annual for PM/NO2, 24-hour guideline for SO2."""
    if pollutant == "SO2":
        return WHO_SO2_DAILY
    return WHO_ANNUAL.get(pollutant)


def _fmt_key(group_by: str, key) -> str:
    if group_by == "month" and isinstance(key, pd.Timestamp):
        return key.strftime("%Y-%m")
    if group_by == "day":
        return str(key)
    return str(key)


def _agg_value(series: pd.Series, agg: str, pollutant: str):
    s = series.dropna()  # NaN kept raw upstream; we skip it, never interpolate
    if agg == "count":
        return int(series.notna().sum())
    if s.empty:
        return None
    if agg == "mean":
        return round(float(s.mean()), 2)
    if agg == "max":
        return round(float(s.max()), 2)
    if agg == "min":
        return round(float(s.min()), 2)
    if agg == "exceedance_rate":
        limit = _limit_for(pollutant)
        if not limit:
            return None
        return round(float((s > limit).mean() * 100), 1)  # % of days over the guideline
    return None


def run_query(
    stations=None,
    pollutants=None,
    start_date=None,
    end_date=None,
    group_by="none",
    agg="mean",
    **_ignored,  # tolerate any unexpected keys the model might invent
) -> dict:
    """Run a validated aggregation over the air-quality dataset. Returns a compact dict."""
    df = load_data()
    valid_stations = set(df["station"].unique())

    # --- validate inputs (anything invalid is dropped, not executed) ---
    pols = [p for p in (pollutants or POLLUTANTS) if p in POLLUTANTS] or POLLUTANTS
    if agg not in AGGS:
        agg = "mean"
    if group_by not in GROUP_COLS:
        group_by = "none"

    if stations:
        sel = [s for s in stations if s in valid_stations]
        if not sel:
            return {"error": "Unknown station code(s).",
                    "valid_stations": sorted(valid_stations)}
        df = df[df["station"].isin(sel)]

    if start_date:
        d = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(d):
            df = df[df["Date"] >= d]
    if end_date:
        d = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(d):
            df = df[df["Date"] <= d]

    if df.empty:
        return {"error": "No rows match those filters.", "rows": []}

    unit_note = ("exceedance_rate = % of days above the WHO guideline "
                 "(annual for PM2.5/PM10/NO2; 24-hour for SO2)") if agg == "exceedance_rate" \
        else "values in ug/m3"

    gcol = GROUP_COLS[group_by]
    rows = []
    if gcol is None:
        row = {p: _agg_value(df[p], agg, p) for p in pols}
        rows.append(row)
    else:
        for key, g in df.groupby(gcol):
            row = {group_by: _fmt_key(group_by, key)}
            for p in pols:
                row[p] = _agg_value(g[p], agg, p)
            rows.append(row)

    truncated = False
    if len(rows) > MAX_ROWS:
        truncated = True
        rows = rows[-MAX_ROWS:]  # keep the most recent slice for time groupings

    out = {
        "agg": agg,
        "group_by": group_by,
        "pollutants": pols,
        "unit": unit_note,
        "n_groups": len(rows),
        "rows": rows,
    }
    if truncated:
        out["note"] = (f"Result truncated to the last {MAX_ROWS} groups. "
                       "Narrow the date range or use a coarser group_by (e.g. year).")
    return out


# OpenAI / Groq function-calling schema
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "query_air_quality",
        "description": (
            "Query the Greater Bilbao air-quality dataset (7 stations, daily, 2015-2026) for "
            "aggregated pollutant statistics. Use this for ANY question about specific values, "
            "time trends (by day/month/year/season), station or zone comparisons, or WHO "
            "exceedance. Never answer such questions from memory — always call this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "stations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Station codes (e.g. MAZARREDO, BASAURI). Omit for all stations.",
                },
                "pollutants": {
                    "type": "array",
                    "items": {"type": "string", "enum": POLLUTANTS},
                    "description": "Pollutants to include. Omit for all four.",
                },
                "start_date": {"type": "string", "description": "Inclusive lower bound, ISO YYYY-MM-DD."},
                "end_date": {"type": "string", "description": "Inclusive upper bound, ISO YYYY-MM-DD."},
                "group_by": {
                    "type": "string",
                    "enum": list(GROUP_COLS.keys()),
                    "description": "How to group results. 'none' = single overall figure.",
                },
                "agg": {
                    "type": "string",
                    "enum": sorted(AGGS),
                    "description": "Aggregation. 'exceedance_rate' = % of days above the WHO guideline.",
                },
            },
        },
    },
}