"""
9_Smart_City_Operations.py — Smart City Operations Dashboard (admin-only).

A professional enterprise-grade operational intelligence page modelled after
Datadog / Snowflake / AWS Data Quality dashboards.

Sections:
  🔵 Overview      — System Status, Data Quality Score, Active Alerts, Forecast Reliability
  🟠 Operational   — Sensor Health Ranking, Statistical Outlier Detection (>3σ, honest label)
  🟣 Diagnostics   — Station Freshness, Coverage Matrix, Completeness, Ingestion, Integrity

Data Quality Score (DQS) is a weighted composite:
  Freshness (0.35) + Completeness (0.30) + Integrity (0.20) + Stability (0.15)
This mirrors enterprise observability standards while being fully computable from the
live parquet — no external monitoring agent required.

Access: admin-only via auth.require_auth(role="admin").
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import load_data, DATA_FILE, WHO_ANNUAL, WHO_SO2_DAILY, POLLUTANT_COLOR  # noqa: E402
from auth import require_auth, logout_button  # noqa: E402
from i18n_auto import language_selector, apply_lang_styles, tr  # noqa: E402

from forecast_utils import plotly_touch_config, PLOTLY_CONFIG

plotly_touch_config()  


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Smart City Operations",
    page_icon="🛰️",
    layout="wide",
)

# Auth gate — must come before any content
user = require_auth(role="admin")
language_selector()
apply_lang_styles()

st.sidebar.success(f"Signed in: {user['name']}\n\nRole: {user['role']}")
logout_button()

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]
SIGMA_THRESHOLD = 3.0   # outlier definition
NO_SENSOR_THRESHOLD = 0.10  # all-time coverage below this = not monitored


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if st.button(f"🔄 {tr('Reload data')}"):
    st.cache_data.clear()
    st.rerun()

try:
    df = load_data()
except Exception as exc:
    st.error(f"{tr('Could not load the dataset')}: {exc}")
    st.stop()

today = pd.Timestamp(datetime.now().date())
latest = df["Date"].max()
earliest = df["Date"].min()
days_behind = int((today - latest).days)

try:
    written = datetime.fromtimestamp(Path(DATA_FILE).stat().st_mtime)
    written_str = written.strftime("%d %b %Y %H:%M")
except Exception:
    written_str = tr("unavailable")

g = df.groupby("station")
all_stations = sorted(df["station"].unique())

# --------------------------------------------------
# DATA QUALITY SCORE — enterprise-grade weighted composite
# --------------------------------------------------
def freshness_score(days: int) -> float:
    """100 = current, 0 = 5+ days stale. Weight 0.35."""
    return max(0.0, 100 - 20 * days)


def completeness_score(df_: pd.DataFrame) -> float:
    """100 × (1 − missing_ratio) across measured pollutant columns. Weight 0.30."""
    measured = [
        p for p in POLLUTANTS
        if p in df_.columns and df_.groupby("station")[p].apply(lambda s: s.notna().mean()).mean() >= NO_SENSOR_THRESHOLD
    ]
    if not measured:
        return 100.0
    total = len(df_) * len(measured)
    missing = int(df_[measured].isna().sum().sum())
    return round(100 * (1 - missing / total), 1)


def integrity_score(df_: pd.DataFrame) -> float:
    """100 − 5 × duplicates − 10 × invalid rows. Weight 0.20."""
    dups = int(df_.duplicated(subset=["Date", "station"]).sum())
    invalids = int((df_[POLLUTANTS].lt(0)).any(axis=1).sum())
    return max(0.0, 100 - 5 * dups - 10 * invalids)


def stability_score(df_: pd.DataFrame) -> float:
    """100 − 2 × outlier rows in last 7 days (>3σ per station×pollutant). Weight 0.15."""
    cutoff = latest - pd.Timedelta(days=7)
    recent = df_[df_["Date"] >= cutoff]
    outlier_count = 0
    for stn, sg in df_.groupby("station"):
        for p in POLLUTANTS:
            if p not in sg.columns:
                continue
            s = sg[p].dropna()
            if len(s) < 10:
                continue
            mu, sigma = s.mean(), s.std()
            if sigma == 0:
                continue
            recent_stn = recent[recent["station"] == stn][p].dropna()
            outlier_count += int(((recent_stn - mu).abs() > SIGMA_THRESHOLD * sigma).sum())
    return max(0.0, 100 - 2 * outlier_count)


F = freshness_score(days_behind)
C = completeness_score(df)
I = integrity_score(df)
A = stability_score(df)
DQS = round(0.35 * F + 0.30 * C + 0.20 * I + 0.15 * A, 1)


def dqs_level(score: float) -> tuple[str, str]:
    if score >= 90:   return "🟢 Excellent", "#2ecc71"
    if score >= 75:   return "🟡 Good",      "#f39c12"
    if score >= 60:   return "🟠 Degraded",  "#e67e22"
    return "🔴 Critical", "#e74c3c"


def freshness_status(days: int) -> str:
    if days <= 2:  return "🟢 Healthy"
    if days <= 5:  return "🟡 Lagging"
    return "🔴 Stale"


# --------------------------------------------------
# ACTIVE ALERTS
# --------------------------------------------------
# Sensor outages: measured pollutants with gaps in last 30 days
cutoff_30 = latest - pd.Timedelta(days=30)
recent_30 = df[df["Date"] >= cutoff_30]
cov_all = pd.DataFrame({
    p: g[p].apply(lambda s: s.notna().mean())
    for p in POLLUTANTS if p in df.columns
})
alerts = []
for stn in all_stations:
    for p in POLLUTANTS:
        if p not in cov_all.columns:
            continue
        if cov_all.loc[stn, p] < NO_SENSOR_THRESHOLD:
            continue   # not monitored — skip
        recent_stn = recent_30[recent_30["station"] == stn][p]
        gaps = int(recent_stn.isna().sum())
        if gaps > 0:
            alerts.append(("error", f"⚠️ {stn} / {p}: {gaps} missing days in last 30 days"))

if days_behind > 2:
    alerts.append(("warning", f"🕐 Pipeline delayed: data is {days_behind} day(s) behind"))

# Statistical outliers in last 7 days
cutoff_7 = latest - pd.Timedelta(days=7)
for stn, sg in df.groupby("station"):
    for p in POLLUTANTS:
        if p not in sg.columns:
            continue
        s = sg[p].dropna()
        if len(s) < 10:
            continue
        mu, sigma = s.mean(), s.std()
        if sigma == 0:
            continue
        recent_stn = df[(df["station"] == stn) & (df["Date"] >= cutoff_7)][p].dropna()
        n_out = int(((recent_stn - mu).abs() > SIGMA_THRESHOLD * sigma).sum())
        if n_out > 0:
            alerts.append(("warning", f"📊 {stn} / {p}: {n_out} statistical outlier(s) in last 7 days (>{SIGMA_THRESHOLD:.0f}σ)"))

# --------------------------------------------------
# FORECAST RELIABILITY (from model bundle metrics)
# --------------------------------------------------
import joblib, glob

MODEL_DIR = Path(__file__).parent.parent / "models"
model_metrics: list[dict] = []
for path in sorted(glob.glob(str(MODEL_DIR / "*.joblib"))):
    try:
        bundle = joblib.load(path)
        if isinstance(bundle, dict) and "metrics" in bundle:
            m = bundle["metrics"]
            model_metrics.append({
                "pollutant": bundle.get("pollutant", Path(path).stem),
                "R2": round(m.get("r2", m.get("R2", float("nan"))), 3),
                "RMSE": round(m.get("rmse", m.get("RMSE", float("nan"))), 2),
            })
    except Exception:
        pass

avg_r2 = round(np.mean([m["R2"] for m in model_metrics if not np.isnan(m["R2"])]) * 100, 1) if model_metrics else None
forecast_label = f"{avg_r2}% avg R²" if avg_r2 else tr("N/A (models not loaded)")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title(f"🛰️ {tr('Smart City Operations Dashboard')}")
st.caption(tr("Operational intelligence for environmental data pipelines, sensors, and AI forecasts."))
st.divider()

# ==================================================
# 🔵 SECTION 1 — SYSTEM OVERVIEW
# ==================================================
st.subheader(f"🔵 {tr('System Status Overview')}")

dqs_label, dqs_color = dqs_level(DQS)
n_alerts = len(alerts)
pipeline_ok = days_behind <= 2

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    tr("System Status"),
    "🟢 Healthy" if pipeline_ok and n_alerts == 0 else "🟡 Attention",
    tr("Pipeline current") if pipeline_ok else f"{days_behind}d behind",
)
c2.metric(
    tr("Data Quality Score"),
    f"{DQS} / 100",
    dqs_label,
)
c3.metric(
    tr("Active Alerts"),
    str(n_alerts),
    "🟢 None" if n_alerts == 0 else f"⚠️ {n_alerts} issue(s)",
)
c4.metric(
    tr("Forecast Reliability"),
    forecast_label,
    tr("From XGBoost test metrics"),
)

# DQS breakdown gauge
st.markdown(f"##### {tr('Data Quality Score breakdown')}")
dqs_cols = st.columns(4)
for col, (label, score, weight) in zip(dqs_cols, [
    (tr("Freshness"),    F, 0.35),
    (tr("Completeness"), C, 0.30),
    (tr("Integrity"),    I, 0.20),
    (tr("Stability"),    A, 0.15),
]):
    _, color = dqs_level(score)
    col.metric(f"{label} (w={weight})", f"{score:.1f}")

with st.expander(f"ℹ️ {tr('How is the Data Quality Score calculated?')}"):
    st.markdown(f"""
**DQS = 0.35 F + 0.30 C + 0.20 I + 0.15 A**

| {tr('Component')} | {tr('Formula')} | {tr('Weight')} |
|---|---|---|
| **{tr('Freshness')}** | `100 − 20 × days_behind` | 0.35 |
| **{tr('Completeness')}** | `100 × (1 − missing_ratio)` | 0.30 |
| **{tr('Integrity')}** | `100 − 5 × duplicates − 10 × invalid_rows` | 0.20 |
| **{tr('Stability')}** | `100 − 2 × outliers_last_7d (>3σ)` | 0.15 |

| {tr('Score')} | {tr('Meaning')} |
|---|---|
| 90–100 | 🟢 Excellent |
| 75–89  | 🟡 Good |
| 60–74  | 🟠 Degraded |
| < 60   | 🔴 Critical |
""")

st.divider()

# ==================================================
# 🟠 SECTION 2 — ACTIVE ALERTS
# ==================================================
st.subheader(f"🟠 {tr('Active Alerts')}")
if not alerts:
    st.success(f"✅ {tr('No active alerts — all systems nominal.')}")
else:
    for kind, msg in alerts:
        if kind == "error":
            st.error(msg)
        else:
            st.warning(msg)

st.divider()

# ==================================================
# 🟠 SECTION 3 — SENSOR HEALTH RANKING
# ==================================================
st.subheader(f"🟠 {tr('Sensor Health Ranking')}")
st.caption(tr("Ranked by all-time completeness across monitored pollutants."))

health_rows = []
for stn in all_stations:
    measured_pols = [
        p for p in POLLUTANTS
        if p in cov_all.columns and cov_all.loc[stn, p] >= NO_SENSOR_THRESHOLD
    ]
    if not measured_pols:
        continue
    avg_cov = cov_all.loc[stn, measured_pols].mean() * 100
    recent_gaps = sum(
        int(recent_30[recent_30["station"] == stn][p].isna().sum())
        for p in measured_pols if p in recent_30.columns
    )
    health_rows.append({
        "Station":   stn,
        "Zone":      df[df["station"] == stn]["Zone"].iloc[0] if len(df[df["station"] == stn]) else "—",
        "Coverage":  f"{avg_cov:.1f}%",
        "30d Gaps":  recent_gaps,
        "Status":    "🟢 Good" if avg_cov >= 95 and recent_gaps == 0
                     else ("🟡 Minor gaps" if recent_gaps <= 5 else "🔴 Attention"),
    })

health_df = pd.DataFrame(health_rows).sort_values("Coverage", ascending=False)
st.dataframe(health_df, width="stretch")

st.divider()

# ==================================================
# 🟠 SECTION 4 — STATISTICAL OUTLIER DETECTION
# ==================================================
st.subheader(f"🟠 {tr('Statistical Outlier Detection')}")
st.caption(
    tr("Values exceeding ±3σ from each station's historical mean per pollutant. "
       "Rule-based statistical method — not an AI/ML anomaly model.")
)

outlier_rows = []
for stn, sg in df.groupby("station"):
    last_date = sg["Date"].max()
    for p in POLLUTANTS:
        if p not in sg.columns:
            continue
        s = sg[p].dropna()
        if len(s) < 30:
            continue
        mu, sigma = s.mean(), s.std()
        if sigma == 0:
            continue
        outliers = sg[((sg[p] - mu).abs() > SIGMA_THRESHOLD * sigma) & sg[p].notna()]
        if outliers.empty:
            continue
        last_out = outliers["Date"].max()
        outlier_rows.append({
            "Station":     stn,
            "Pollutant":   p,
            "Count":       len(outliers),
            "Last outlier": last_out.strftime("%d %b %Y"),
            "In last 7d":  int(((outliers["Date"] >= cutoff_7)).sum()),
            "Mean":        f"{mu:.1f}",
            "σ":           f"{sigma:.1f}",
        })

if outlier_rows:
    out_df = pd.DataFrame(outlier_rows).sort_values("In last 7d", ascending=False)
    st.dataframe(out_df, width="stretch")

    # Sparkline for the worst recent case
    if not out_df.empty:
        worst = out_df.iloc[0]
        stn_w, pol_w = worst["Station"], worst["Pollutant"]
        sg_w = df[df["station"] == stn_w].sort_values("Date").tail(90)
        mu_w = df[df["station"] == stn_w][pol_w].mean()
        sd_w = df[df["station"] == stn_w][pol_w].std()
        fig = px.line(sg_w, x="Date", y=pol_w,
                      title=f"{stn_w} / {pol_w} — {tr('last 90 days')}")
        fig.add_hline(y=mu_w + SIGMA_THRESHOLD * sd_w, line_dash="dash",
                      line_color="red", annotation_text=f"+{SIGMA_THRESHOLD:.0f}σ")
        fig.add_hline(y=mu_w - SIGMA_THRESHOLD * sd_w, line_dash="dash",
                      line_color="red", annotation_text=f"−{SIGMA_THRESHOLD:.0f}σ")
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
else:
    st.success(f"✅ {tr('No statistical outliers detected across all stations.')}")

st.divider()

# ==================================================
# 🟣 SECTION 5 — DIAGNOSTICS (expandable tabs)
# ==================================================
st.subheader(f"🟣 {tr('Diagnostics')}")

with st.expander(f"🛠 {tr('Detailed diagnostics')}", expanded=False):
    tab_fresh, tab_cov, tab_comp, tab_ingest, tab_int = st.tabs([
        f"📅 {tr('Freshness')}",
        f"📡 {tr('Coverage Matrix')}",
        f"📊 {tr('Completeness')}",
        f"📥 {tr('Ingestion History')}",
        f"🔍 {tr('Integrity')}",
    ])

    # ── Freshness ──
    with tab_fresh:
        latest_per = g["Date"].max()
        fresh_df = pd.DataFrame({
            "Zone":        g["Zone"].first(),
            "Latest date": latest_per.dt.strftime("%d %b %Y"),
            "Days behind": (today - latest_per).dt.days.astype(int),
        })
        fresh_df["Status"] = fresh_df["Days behind"].apply(freshness_status)
        st.dataframe(fresh_df.sort_values("Days behind", ascending=False), width="stretch")

    # ── Coverage Matrix ──
    with tab_cov:
        st.caption(tr("All-time completeness per pollutant. Under 10% = not monitored at this station."))
        def cov_cell(f):
            return tr("not monitored") if f < NO_SENSOR_THRESHOLD else f"{100*f:.1f}%"
        cov_disp = cov_all.map(cov_cell)
        cov_disp.insert(0, "Zone", g["Zone"].first())
        st.dataframe(cov_disp, width="stretch")

    # ── Completeness (recent gaps) ──
    with tab_comp:
        st.caption(tr("Missing values in the last 30 days for monitored pollutants — possible sensor outages."))
        gap_rows = {}
        for stn in all_stations:
            row = {}
            for p in POLLUTANTS:
                if p not in df.columns:
                    continue
                if cov_all.loc[stn, p] < NO_SENSOR_THRESHOLD:
                    row[p] = "—"
                elif stn in recent_30.groupby("station").groups:
                    row[p] = int(recent_30[recent_30["station"] == stn][p].isna().sum())
                else:
                    row[p] = 0
            gap_rows[stn] = row
        gaps_df = pd.DataFrame(gap_rows).T[[p for p in POLLUTANTS if p in cov_all.columns]]
        st.dataframe(gaps_df, width="stretch")
        flagged = [
            f"{s}/{p} ({gaps_df.loc[s,p]})"
            for s in gaps_df.index for p in gaps_df.columns
            if gaps_df.loc[s, p] != "—" and int(gaps_df.loc[s, p]) > 0
        ]
        if flagged:
            st.warning(f"⚠️ {tr('Possible sensor outage')}: " + ", ".join(flagged))
        else:
            st.success(f"✅ {tr('No recent gaps in any monitored series.')}")

    # ── Ingestion History ──
    with tab_ingest:
        per_day = df.groupby(df["Date"].dt.normalize()).size().tail(30)
        fig_bar = px.bar(
            x=per_day.index, y=per_day.values,
            labels={"x": tr("Date"), "y": tr("Records")},
            title=tr("Daily records ingested — last 30 days"),
        )
        fig_bar.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_bar, width="stretch")
        st.caption(tr("A healthy day shows one record per reporting station (≈7)."))

    # ...
    st.plotly_chart(fig, config=PLOTLY_CONFIG)

    # ── Integrity ──
    with tab_int:
        dups = int(df.duplicated(subset=["Date", "station"]).sum())
        invalids = int((df[POLLUTANTS].lt(0)).any(axis=1).sum())
        total_nan = int(df[POLLUTANTS].isna().sum().sum())
        if dups == 0:
            st.success(f"✅ {tr('No duplicate (date, station) rows.')}")
        else:
            st.error(f"⚠️ {dups} {tr('duplicate rows found.')}")
        if invalids == 0:
            st.success(f"✅ {tr('No physically invalid values (negative concentrations).')}")
        else:
            st.warning(f"⚠️ {invalids} {tr('rows with negative pollutant values.')}")
        st.metric(tr("Total missing pollutant values (all-time)"), f"{total_nan:,}")
        st.metric(tr("Data file last written"), written_str)