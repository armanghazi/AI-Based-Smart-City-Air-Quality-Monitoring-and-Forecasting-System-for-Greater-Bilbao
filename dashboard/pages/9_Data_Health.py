"""
9_Data_Health.py — Pipeline & data-health monitor (admin-only).

The honest version of an "admin" page: instead of fake role theater, it surfaces
REAL operational signals about the live data pipeline (Phase B):
  - data freshness (overall and per station — catches a single stalled sensor)
  - last write of the parquet file (proxy for the last pipeline commit/deploy)
  - calendar coverage and total records
  - per-station x pollutant completeness (raw NaN, never interpolated)
  - integrity check (duplicate Date+station rows — the pipeline guarantees zero)
  - recent daily ingestion (visualizes the cron appending day by day)

Access is gated by auth.require_auth(role="admin"); the gate runs BEFORE any content.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Streamlit Cloud import pattern (pages can't use package-relative imports) ---
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import load_data, DATA_FILE  # noqa: E402
from auth import require_auth, logout_button  # noqa: E402

st.set_page_config(page_title="Data Health", page_icon="🩺", layout="wide")

# ==================================================
# AUTH GATE — nothing below renders for non-admins
# ==================================================
user = require_auth(role="admin")
st.sidebar.success(f"Signed in: {user['name']}\n\nRole: {user['role']}")
logout_button()

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]


def freshness_status(days_behind: int) -> str:
    """Pipeline rejects D-0, so D-1 is the freshest possible; allow a small lag."""
    if days_behind <= 2:
        return "🟢 Healthy"
    if days_behind <= 5:
        return "🟡 Lagging"
    return "🔴 Stale"


# ==================================================
# LOAD
# ==================================================
st.title("🩺 Pipeline & Data Health")
st.caption("Operational view of the live dataset and the automated daily pipeline.")

if st.button("🔄 Reload data (clear cache)"):
    st.cache_data.clear()
    st.rerun()

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

today = pd.Timestamp(datetime.now().date())
latest = df["Date"].max()
earliest = df["Date"].min()
days_behind = int((today - latest).days)

# Last parquet write (proxy for last pipeline commit/deploy)
try:
    written = datetime.fromtimestamp(Path(DATA_FILE).stat().st_mtime)
    written_str = written.strftime("%d %b %Y %H:%M")
except Exception:
    written_str = "unavailable"

# Calendar coverage
n_days = df["Date"].dt.normalize().nunique()
expected_days = int((latest - earliest).days) + 1
coverage = 100 * n_days / expected_days if expected_days else 0

# ==================================================
# TOP METRICS
# ==================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest data point", latest.strftime("%d %b %Y"), f"{days_behind} day(s) behind")
c2.metric("Total records", f"{len(df):,}")
c3.metric("Calendar coverage", f"{coverage:.1f}%", f"{n_days:,} of {expected_days:,} days")
c4.metric("Data file written", written_str)

# Pipeline status banner
status = freshness_status(days_behind)
if status.startswith("🟢"):
    st.success(f"{status} — latest data is {latest.strftime('%d %b %Y')} "
               f"({days_behind} day(s) behind today; D-0 is rejected by design).")
elif status.startswith("🟡"):
    st.warning(f"{status} — data is {days_behind} days behind. Check the last GitHub Actions run.")
else:
    st.error(f"{status} — data is {days_behind} days behind. The pipeline may have failed.")

st.divider()

# ==================================================
# PER-STATION FRESHNESS (catches a single stalled sensor)
# ==================================================
st.subheader("Per-station freshness")
g = df.groupby("station")
latest_per = g["Date"].max()
fresh = pd.DataFrame({
    "Latest date": latest_per.dt.strftime("%d %b %Y"),
    "Days behind": (today - latest_per).dt.days.astype(int),
})
fresh["Status"] = fresh["Days behind"].apply(freshness_status)
fresh = fresh.sort_values("Days behind", ascending=False)
st.dataframe(fresh, width="stretch")

# ==================================================
# COMPLETENESS — per station x pollutant (raw NaN kept, never interpolated)
# ==================================================
st.subheader("Data completeness")
st.caption("Share of non-missing daily values. Gaps are kept raw — the pipeline never interpolates.")
comp = pd.DataFrame(
    {p: g[p].apply(lambda s: 100 * s.notna().mean()) for p in POLLUTANTS if p in df.columns}
).round(1)

fig = px.imshow(
    comp.values,
    x=list(comp.columns),
    y=list(comp.index),
    color_continuous_scale="RdYlGn",
    range_color=[0, 100],
    text_auto=True,
    aspect="auto",
    labels=dict(color="% present"),
)
fig.update_layout(height=60 + 32 * len(comp), margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")

# ==================================================
# INTEGRITY + RECENT INGESTION
# ==================================================
col_a, col_b = st.columns([1, 2])

with col_a:
    st.subheader("Integrity")
    dups = int(df.duplicated(subset=["Date", "station"]).sum())
    if dups == 0:
        st.success("✅ No duplicate (date, station) rows.")
    else:
        st.error(f"⚠️ {dups} duplicate (date, station) rows found.")
    total_nan = int(df[POLLUTANTS].isna().sum().sum())
    st.metric("Total missing pollutant values", f"{total_nan:,}")

with col_b:
    st.subheader("Recent daily ingestion")
    per_day = df.groupby(df["Date"].dt.normalize()).size().tail(30)
    bar = px.bar(
        x=per_day.index, y=per_day.values,
        labels={"x": "Date", "y": "Records appended"},
    )
    bar.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(bar, width="stretch")
    st.caption("Healthy days show one record per reporting station (≈7).")