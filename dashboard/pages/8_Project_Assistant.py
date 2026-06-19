"""
7_Project_Assistant.py
In-dashboard conversational assistant for the GeoAI Air Quality project.

Answers questions about BOTH the live data and the project methodology, grounded in:
  (a) a runtime DATA DIGEST computed from config.load_data(), and
  (b) a static PROJECT KNOWLEDGE block.

Backend: Groq API (OpenAI-compatible chat completions).
Key is read from st.secrets["GROQ_API_KEY"] — add it in Streamlit Cloud > Settings > Secrets.
Provider-agnostic by design: to swap providers, only the client and MODEL name change.
Requires the 'groq' package (add `groq` to requirements.txt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# --- Streamlit Cloud import pattern: pages cannot use package-relative imports ---
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (  # noqa: E402
    load_data,
    WHO_ANNUAL,
    WHO_SO2_DAILY,
    CORE_POLLUTANTS,
    get_zone,
)


from i18n_auto import language_selector, apply_lang_styles, tr

# EU limits exist only in newer config versions — import defensively
try:
    from config import EU_ANNUAL, ALERT_LIMITS  # noqa: E402
except ImportError:
    EU_ANNUAL, ALERT_LIMITS = {}, {}

# AQI module (European/ICA primary + EPA reference) — single source of truth,
# the same logic the dashboard uses. Import defensively so the page still runs
# if an older config layout lacks aqi.py.
try:
    from aqi import overall_aqi  # noqa: E402
except ImportError:
    overall_aqi = None

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL = "llama-3.3-70b-versatile"   # Groq free-tier high-quota model (verify name on console.groq.com)
MAX_HISTORY = 8                     # prior turns sent to the model (keeps us under free-tier TPM)
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]

st.set_page_config(page_title="Project Assistant", page_icon="💬", layout="wide")
language_selector()
apply_lang_styles()

# --------------------------------------------------
# STATIC PROJECT KNOWLEDGE  (distilled from README / Architecture docs)
# --------------------------------------------------
PROJECT_TITLE = "GeoAI Smart City Air Quality Dashboard — Greater Bilbao"

PROJECT_KNOWLEDGE = """
PURPOSE: End-to-end GeoAI platform that monitors, analyzes, visualizes, and forecasts
urban air quality across Greater Bilbao (Bizkaia, Basque Country, Spain). Live on Streamlit Cloud.

DATA: 7 monitoring stations, 4 pollutants (PM2.5, PM10, NO2, SO2), daily resolution, 2015-2026.
Air quality from Open Data Euskadi; weather from Open-Meteo (ERA5). Stored as Parquet.
Two files: forecasting_dataset.parquet (frozen ML snapshot, MICE-imputed) and
air_quality_weather.parquet (live dashboard source, raw NaN kept, never interpolated).

ZONES: Mazarredo/Erandio = Urban (traffic, highest NO2); Basauri/Barakaldo = Industrial (high PM);
Santurtzi = Port (marine + traffic, SO2); Algorta = Coastal (best dispersion);
Muskiz = Refinery (Petronor petrochemical profile).

GUIDELINES: PM2.5/PM10/NO2 compared to WHO 2021 ANNUAL limits (5 / 15 / 10 µg/m3).
SO2 is handled SEPARATELY against the WHO 24-HOUR guideline (40 µg/m3) because its behavior is
episodic (industrial/port), not annual. Dashboard also benchmarks against EU regulatory limits.

AIR QUALITY INDEX (AQI): the dashboard shows a DUAL index. Primary = European Air Quality Index
(EEA), which the Spanish ICA (MITECO) follows; secondary reference = US EPA AQI. The overall AQI
of a station is the WORST (highest) pollutant category, NOT an average (official EAQI/ICA rule).
Because the data is daily means while the official index uses shorter windows, it is a
daily-mean-based APPROXIMATION. The logic lives in the shared aqi.py module (single source of truth).

FORECASTING (production = XGBoost, one model per pollutant, 62 features each):
- Task: next-day forecast, target(t+1) per station.
- Test-set results (held out 2024-2026): NO2 R2=0.560, PM2.5 R2=0.479, PM10 R2=0.460, SO2 R2=0.390.
- SO2's lower R2 is EXPECTED (episodic emissions), not a bug.
METHODOLOGY (rigor):
- Strict TIME-BASED split (train <2023, validation 2023, test >=2024). A row-based split once
  produced an inflated R2=0.84 via leakage; this was found and fixed. When R2 jumps, suspect leakage.
- Keeping TODAY's pollutant value as a feature is VALID (available at prediction time, not leakage).
  Removing PM2.5 drops its R2 from 0.48 to 0.34.
- SHAP confirmed physically meaningful behavior: higher wind speed and precipitation push predictions
  DOWN (dispersion, wet deposition). NO2 shows strong day-of-week (traffic) importance.
- Benchmarks (notebook 08): XGBoost 0.479 > SARIMA one-step 0.463 (single station only) > MLP 0.208.
  XGBoost chosen for production: 4 models cover all stations (vs 28 for SARIMA), uses weather signal,
  degrades gracefully on multi-day horizons. ARIMA/SARIMA/LSTM are benchmark-only, never production.
- Models are FROZEN: runtime feature engineering means live daily data improves predictions
  without retraining.

SPATIAL GeoAI (Phase C): IDW interpolation surfaces masked to the Gran Bilbao comarca boundary
(EPSG:25830). Covariate analysis: distance to Petronor refinery vs mean SO2 gave Pearson r = -0.15,
interpreted as SO2 having DISTRIBUTED sources (traffic, local combustion) rather than one point source.
Road-density (osmnx) vs NO2 analysis is in progress.

PIPELINE (Phase B, live): GitHub Actions cron runs scripts/daily_update.py to fetch new records from
Open Data Euskadi + Open-Meteo and append to the dashboard parquet (idempotent, current day rejected,
zero duplicates). The commit triggers a Streamlit Cloud redeploy. No retraining.

DASHBOARD PAGES: 1 Monitoring, 2 Temporal Trends, 3 Urban Risk Index, 4 Weather Drivers,
5 Forecasting (backtest + recursive forecast + SHAP), 6 Smart City Decision Support
(GeoAI risk map, IDW surface, scenario simulator, recommended actions, PDF/CSV export).
""".strip()

# Example questions shown as quick-start buttons
EXAMPLES = [
    "What is the yearly NO2 trend at MAZARREDO?",
    "Which station has the worst PM2.5 exceedance rate vs the WHO limit?",
    "Why is SO2 handled separately, and why is its R2 lower than the others?",
    "How were the forecasting models protected against data leakage?",
]


# --------------------------------------------------
# DATA DIGEST  (cached — recomputed only when the parquet changes)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def build_data_digest() -> tuple[str, str]:
    """Return (digest_text, freshness_label). Degrades gracefully if data is unavailable."""
    try:
        df = load_data()
    except Exception as exc:  # data not reachable (e.g. local run without parquet)
        return f"DATA DIGEST UNAVAILABLE ({exc}).", "unknown"

    lines: list[str] = []
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    n_stations = df["station"].nunique()
    lines.append(f"Coverage: {n_stations} stations, {len(df):,} daily rows, {dmin} to {dmax}.")
    lines.append(f"Most recent date in data: {dmax} (freshness indicator).")

    # Raw NaN counts (the dashboard parquet keeps gaps, never interpolates)
    nan_bits = [f"{p}={int(df[p].isna().sum())}" for p in POLLUTANTS if p in df.columns]
    lines.append("Missing values (kept raw, not interpolated): " + ", ".join(nan_bits) + ".")

    # Per-station latest value | period mean
    lines.append("\nPer-station (latest | period mean), µg/m3:")
    for stn, g in df.sort_values("Date").groupby("station"):
        last = g.iloc[-1]
        town = last.get("Town", stn)
        zone = get_zone(town)
        parts = []
        for p in POLLUTANTS:
            if p in g.columns:
                lv = last[p]
                lv_s = "NA" if pd.isna(lv) else f"{lv:.1f}"
                parts.append(f"{p} {lv_s}|{g[p].mean():.1f}")
        lines.append(f"  {stn} ({town}, {zone}): " + "; ".join(parts))

    # WHO exceedance summary
    lines.append("\nWHO exceedance (period mean vs guideline):")
    for p in CORE_POLLUTANTS:
        if p in df.columns:
            limit = WHO_ANNUAL[p]
            means = df.groupby("station")[p].mean().sort_values(ascending=False)
            top = means.index[0]
            lines.append(
                f"  {p} (WHO annual {limit}): highest mean {top} = {means.iloc[0]:.1f} "
                f"({means.iloc[0] / limit:.1f}x limit)."
            )
    if "SO2" in df.columns:
        pct_over = (
            df.groupby("station")["SO2"].apply(lambda s: (s > WHO_SO2_DAILY).mean() * 100)
            .sort_values(ascending=False)
        )
        lines.append(
            f"  SO2 (WHO 24h {WHO_SO2_DAILY}): % days over limit, highest {pct_over.index[0]} "
            f"= {pct_over.iloc[0]:.1f}%."
        )

    # Air Quality Index — reuses the dashboard's shared aqi module (EAQI/ICA + EPA).
    # Overall AQI per station = WORST pollutant category (official rule, not an average),
    # computed on the latest daily reading (a daily-mean approximation of the official index).
    if overall_aqi is not None:
        lines.append(
            "\nAir Quality Index (EAQI/ICA on latest reading; overall = WORST pollutant, "
            "not an average; daily-mean approximation):"
        )
        ranking = []
        for stn, g in df.sort_values("Date").groupby("station"):
            last = g.iloc[-1]
            vals = {
                p: (None if pd.isna(last[p]) else float(last[p]))
                for p in POLLUTANTS
                if p in g.columns
            }
            res = overall_aqi(vals)
            if res:
                ranking.append((stn, last.get("Town", stn), res))
        # Best -> worst: lower EAQI level first, EPA AQI breaks ties
        ranking.sort(key=lambda r: (r[2]["level"], r[2].get("epa_aqi") or 0))
        for stn, town, res in ranking:
            epa = res.get("epa_aqi")
            epa_s = f", EPA {epa} ({res.get('epa_label')})" if epa is not None else ""
            lines.append(
                f"  {stn} ({town}): EAQI {res['level']}/6 \"{res['label']}\", "
                f"driver {res['driver']}{epa_s}."
            )
        if ranking:
            lines.append(f"  => Best (cleanest) AQI: {ranking[0][0]}; worst: {ranking[-1][0]}.")

    return "\n".join(lines), str(dmax)


def build_system_prompt(digest: str) -> str:
    """Assemble the grounded system prompt: rules + project knowledge + live digest."""
    return (
        f"You are the Project Assistant for \"{PROJECT_TITLE}\". "
        "You help visitors understand both the air-quality DATA and the project's METHODOLOGY.\n\n"
        "GROUND RULES:\n"
        "- For ANY question about specific numbers, time trends (by day/month/year/season), "
        "station or zone comparisons, or WHO exceedance, you MUST call the query_air_quality "
        "tool and answer from its result. Never compute or guess these from memory.\n"
        "- Use exact station codes as they appear in the LIVE DATA DIGEST (e.g. MAZARREDO, SANTURCE).\n"
        "- For methodology / 'how does it work' questions, answer from the PROJECT KNOWLEDGE below.\n"
        "- If a question is outside both the tool's scope and the knowledge below, say you don't "
        "have that information. Never invent numbers, dates, or station values.\n"
        "- Be concise and precise; concentrations are in µg/m3. WHO limits: annual for "
        "PM2.5/PM10/NO2, 24-hour for SO2. Answer as a knowledgeable guide to this project.\n"
        "- Reply in the same language the user writes in. Keep technical terms and proper nouns "
        "unchanged (WHO, exceedance, data leakage, R2, XGBoost, SHAP, and station codes like "
        "MAZARREDO); translate only the surrounding explanation.\n\n"
        f"=== PROJECT KNOWLEDGE ===\n{PROJECT_KNOWLEDGE}\n\n"
        f"=== LIVE DATA DIGEST ===\n{digest}"
    )


# --------------------------------------------------
# GROQ CALL  (returns (ok, text); never raises to the UI)
# --------------------------------------------------
def get_assistant_reply(history: list[dict], digest: str) -> tuple[bool, str, list]:
    """Returns (ok, text, charts) where charts is a list of run_query result dicts."""
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return False, (
            "The assistant is offline because no `GROQ_API_KEY` is configured. "
            "Add it under Streamlit Cloud → Settings → Secrets to enable chat. "
            "Meanwhile, the project facts are available in the **Project facts** panel above."
        ), []
    try:
        import json
        from groq import Groq
        from assistant_query import run_query, TOOL_SPEC

        client = Groq(api_key=api_key)
        messages = [{"role": "system", "content": build_system_prompt(digest)}]
        # Only role/content go to the API (strip UI-only keys like "charts")
        messages += [{"role": m["role"], "content": m["content"]} for m in history[-MAX_HISTORY:]]

        tool_results: list = []  # collected for inline charts
        for _ in range(3):  # allow a couple of tool round-trips
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=900,
                tools=[TOOL_SPEC],
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            if not getattr(msg, "tool_calls", None):
                return True, (msg.content or "").strip(), tool_results

            # Record the assistant's tool-call turn
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            # Execute each requested query and feed the real result back
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = run_query(**args) if isinstance(args, dict) else {"error": "bad args"}
                except Exception as e:
                    result = {"error": f"query failed: {e}"}
                tool_results.append(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)[:4000],  # cap tokens fed back
                })

        # Safety net: if still requesting tools, force a plain text answer
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.3, max_tokens=900,
        )
        return True, (resp.choices[0].message.content or "").strip(), tool_results
    except Exception as exc:  # rate limit (429), network, bad key, etc.
        return False, (
            "The assistant could not answer right now (possibly a free-tier rate limit — "
            f"try again in a moment). Details: `{exc}`"
        ), []


def _result_to_chart(result: dict):
    """Turn a multi-row query result into (DataFrame, kind) for inline charting,
    or (None, None) if it isn't chartable (single row / no numeric columns)."""
    rows = result.get("rows") if isinstance(result, dict) else None
    gb = result.get("group_by") if isinstance(result, dict) else None
    if not rows or gb in (None, "none") or len(rows) < 2:
        return None, None
    cdf = pd.DataFrame(rows)
    if gb not in cdf.columns:
        return None, None
    cdf = cdf.set_index(gb)
    num = cdf.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if num.empty:
        return None, None
    kind = "line" if gb in ("day", "month", "year") else "bar"  # trend vs comparison
    return num, kind


def render_charts(charts: list) -> None:
    """Render any chartable query results as small inline charts."""
    for res in charts or []:
        cdf, kind = _result_to_chart(res)
        if cdf is None:
            continue
        if kind == "line":
            st.line_chart(cdf, height=240)
        else:
            st.bar_chart(cdf, height=240)



# --------------------------------------------------
# UI
# --------------------------------------------------
st.title(tr("💬 Project Assistant"))
st.caption(
    tr("Ask about the data or the methodology. Data questions (trends by day/month/year, "
       "station comparisons, WHO exceedance) are answered by querying the live dataset directly — "
       "the assistant does not guess numbers.")
)

digest_text, freshness = build_data_digest()
st.caption(f"{tr('Grounded on data through')} **{freshness}** · {tr('model')} `{MODEL}` {tr('via Groq')}")

with st.expander(tr("📋 Project facts (what the assistant is grounded on)")):
    st.code(digest_text, language="text")

# Quick-start buttons
st.markdown("##### " + tr("Quick start"))
cols = st.columns(2)
for i, q in enumerate(EXAMPLES):
    if cols[i % 2].button(tr(q), key=f"ex_{i}", width="stretch"):
        st.session_state.queued_prompt = q
        st.rerun()

# Chat state
if "assistant_msgs" not in st.session_state:
    st.session_state.assistant_msgs = []

# Render history
for m in st.session_state.assistant_msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# New input: typed prompt OR a queued quick-start question
typed = st.chat_input(tr("Ask about the data or the project…"))
prompt = typed or st.session_state.pop("queued_prompt", None)

if prompt:
    st.session_state.assistant_msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(tr("Thinking…")):
            ok, reply, charts = get_assistant_reply(st.session_state.assistant_msgs, digest_text)
        st.markdown(reply)

    st.session_state.assistant_msgs.append(
        {"role": "assistant", "content": reply, "charts": charts}
    )

# Reset
if st.session_state.assistant_msgs:
    if st.button(tr("Clear conversation")):
        st.session_state.assistant_msgs = []
        st.rerun()