"""
8_Project_Assistant.py
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

# --------------------------------------------------
# STATIC PROJECT KNOWLEDGE  (distilled from README / Architecture docs)
# --------------------------------------------------
PROJECT_TITLE = "GeoAI Smart City Air Quality Dashboard — Greater Bilbao"

PROJECT_KNOWLEDGE = """
PURPOSE: End-to-end GeoAI platform that monitors, analyzes, visualizes, and forecasts
urban air quality across Greater Bilbao (Bizkaia, Basque Country, Spain). Live on Streamlit Cloud.

DATA: 7 monitoring stations, 4 pollutants (PM2.5, PM10, NO2, SO2), daily resolution, 2015-2026.
~29,000 daily records. Air quality from Open Data Euskadi; weather from Open-Meteo (ERA5).
Stored as Parquet. D-1 constraint: pipeline rejects the current incomplete day.

ZONES: Mazarredo/Erandio = Urban (traffic, highest NO2); Basauri/Barakaldo = Industrial (high PM);
Santurtzi = Port (marine + traffic, SO2); Algorta = Coastal (best dispersion);
Muskiz = Refinery (Petronor petrochemical profile).

GUIDELINES: PM2.5/PM10/NO2 vs WHO 2021 ANNUAL limits (5 / 15 / 10 ug/m3).
SO2 vs WHO 24-HOUR guideline (40 ug/m3) — episodic industrial/port behavior.

AIR QUALITY INDEX: DUAL index. Primary = EAQI/ICA (European/Spanish). Secondary = US EPA AQI.
Overall AQI = WORST pollutant category (not an average — official EAQI/ICA rule).
Daily-mean approximation. Logic in shared aqi.py.

FORECASTING (XGBoost, one model per pollutant, 62 features, time-based split):
- NO2 R2=0.560 | PM2.5 R2=0.479 | PM10 R2=0.460 | SO2 R2=0.390 (held-out test 2024-2026)
- SO2 lower R2 is EXPECTED (episodic source), not a bug.
- Row-based split once gave fake R2=0.84 (leakage) — fixed to honest 0.479.
- Today's pollutant value is VALID as feature (available at prediction time).
- SHAP: wind speed/precipitation push DOWN (dispersion). NO2: day-of-week = traffic signal.
- ARIMA/SARIMA/LSTM are benchmark-only, never production.
- Models FROZEN: live daily data improves predictions without retraining.

GIS SPATIAL ANALYSIS (notebooks 10a-10d — COMPLETED):
NOTE: All spatial correlations across n=7 stations are exploratory/indicative only.

Top structural predictors of air pollution:
1. Road density 1km vs NO2:          r = +0.83 (strongest signal — traffic infrastructure)
2. Distance to city centre vs NO2:   r = -0.77 (closer = higher NO2 — urban canyon)
3. Green cover 1km vs PM10:          r = -0.66 (more green = lower PM10)
4. Terrain Relief Index 2km vs PM10: r = -0.63 (complex terrain = better dispersion)
5. Distance to AP-8 motorway vs PM2.5: r = -0.54 (BARAKALDO 354m from AP-8 = highest PM2.5)

Station structural profiles (from GIS analysis):
- BARAKALDO:  road density 21,267 m/km2 + 354m from AP-8 -> structural driver of elevated PM2.5/NO2
- MAZARREDO:  77% residential yet highest NO2 (25.8 ug/m3): road density 19,060 m/km2 + 501m from centre
- BASAURI:    industrial land use 32% within 500m (21% at 1km) -> local emission concentration
- MUSKIZ:     34.6% industrial (Petronor) yet LOWEST PM2.5 (6.5 ug/m3): TRI 343m + coastal NW breeze
- SANTURCE:   784m from Port of Bilbao + elevation 93m + TRI 445m -> port drives SO2 episodes
- ALGORTA:    lowest road density (9,933 m/km2) + 2.6km coast -> structurally cleanest station
- ERANDIO:    1,264m from AP-8 + 18,631 m/km2 road density -> traffic corridor exposure

Structural Vulnerability Index (SVI — composite: road density + city centre distance + TRI):
BARAKALDO=100 | MAZARREDO=89.8 | ERANDIO=87.4 | BASAURI=51.7 | ALGORTA=34.7 | SANTURCE=10.5 | MUSKIZ=0

MUSKIZ dispersion paradox (key GeoAI finding): Three independent GIS methods (buffer analysis,
distance features, DEM terrain) confirm that coastal position + TRI 343m + NW sea breeze override
Petronor proximity. Despite highest industrial land use (34.6%), MUSKIZ records lowest PM2.5 (6.5 ug/m3).

WIND TRANSPORT (notebook 10d — ERA5 single grid cell, ~31km, shared by all 7 stations):
- Wind speed dispersion: NO2 drops 57% from calm (<10 m/s) to strong (>25 m/s)
- Two regimes: NW/W (37.2% of days, sea air, NO2=16.3) vs S/SW (32.0%, inland, NO2=21.4, +31%)
- MUSKIZ NE wind: SO2=7.32 ug/m3 (Petronor trapped by terrain, 1.93x vs S-wind baseline)
- MAZARREDO SE wind: NO2=35.3 ug/m3 (+70% vs NW — Nervion valley channelling)
- ERA5 caveat: single grid = regional patterns only, not station micrometeorology

PIPELINE (Phase B, live): GitHub Actions cron daily_update.py -> append parquet -> Streamlit redeploy.

DASHBOARD PAGES:
0 Daily Briefing | 1 Air Quality Monitoring | 2 Temporal Trends
3 GeoAI Spatial Analysis (4 tabs: Station DNA / Spatial Drivers / Terrain & Dispersion / Wind Transport)
4 Weather Drivers (incl. wind transport section) | 5 Forecasting (backtest + recursive + SHAP)
6 Smart City Decision Support (4 tabs: Status / Forecast & Map / Decisions / Spatial Intelligence)
7 GeoAI Methodology (model metrics, spatial findings, honest caveats)
8 Project Assistant (this page) | 9 Admin Operations (Google OAuth, sensor health)
""".strip()

# Example questions shown as quick-start buttons
EXAMPLES = [
    "What is the latest air quality status across all stations?",
    "Why does MUSKIZ have low pollution despite being next to Petronor refinery?",
    "Which station has the worst structural vulnerability index (SVI)?",
    "How does wind direction affect SO2 at MUSKIZ?",
    "What is the yearly NO2 trend at MAZARREDO?",
    "Why is SO2 handled separately, and why is its R2 lower than the others?",
    "How were the forecasting models protected against data leakage?",
    "Which station benefits most from coastal dispersion?",
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
    lines.append("\nPer-station (latest | period mean), ug/m3:")
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
        "- For questions about a station's air-quality STATUS / AQI / ICA / índice, or a quality "
        "category (Good, Moderate, Poor...), call BOTH air_quality_status AND weather_summary. "
        "Report: (1) EAQI/ICA category (English + Spanish label) + WHO numbers/ratios; "
        "(2) weather: temperature, humidity, precipitation, wind speed (km/h) AND wind direction "
        "(degrees + cardinal); (3) the dispersion_verdict explaining how today's weather "
        "affects the air quality.\n"
        "- For questions specifically about weather, wind speed, wind direction, temperature, "
        "or 'why is pollution high/low', call weather_summary and explain the physical link.\n"
        "- Use exact station codes as they appear in the LIVE DATA DIGEST (e.g. MAZARREDO, SANTURCE).\n"
        "- For methodology / 'how does it work' questions, answer from the PROJECT KNOWLEDGE below.\n"
        "- If a question is outside both the tool's scope and the knowledge below, say you don't "
        "have that information. Never invent numbers, dates, or station values.\n"
        "- Be concise and precise; concentrations are in ug/m3. WHO limits: annual for "
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
        from assistant_query import (  # noqa: E402
            run_query, aqi_status, weather_summary, combined_status,
            TOOL_SPEC, AQI_TOOL_SPEC, WEATHER_TOOL_SPEC, COMBINED_TOOL_SPEC,
        )
        TOOL_DISPATCH = {
            "query_air_quality":  run_query,
            "air_quality_status": aqi_status,
            "weather_summary":    weather_summary,
            "combined_status":    combined_status,
        }

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
                max_tokens=600,   # reduced to stay under Groq free-tier TPM
                tools=[TOOL_SPEC, AQI_TOOL_SPEC, WEATHER_TOOL_SPEC, COMBINED_TOOL_SPEC],
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
            # Execute each requested tool and feed the real result back
            for tc in msg.tool_calls:
                fn = TOOL_DISPATCH.get(tc.function.name, run_query)
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = fn(**args) if isinstance(args, dict) else {"error": "bad args"}
                except Exception as e:
                    result = {"error": f"query failed: {e}"}
                # only query_air_quality results are chartable
                if tc.function.name == "query_air_quality":
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
            "The assistant could not respond (possibly a rate limit — "
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
st.title("💬 Project Assistant")
st.caption(
    "Ask about the data or the methodology. Data questions (trends by day/month/year, "
    "station comparisons, WHO exceedance) are answered by querying the live dataset directly — "
    "the assistant does not guess numbers."
)

digest_text, freshness = build_data_digest()
st.caption(f"Grounded on data through **{freshness}** · model `{MODEL}` via Groq")

with st.expander("📋 Project facts (what the assistant is grounded on)"):
    st.code(digest_text, language="text")

# ── Chat input — placed FIRST so it is immediately visible ──────────────
st.markdown("""
<style>
/* Make the chat input box more prominent */
[data-testid="stChatInput"] textarea {
    border: 2px solid #2563eb !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    background: #f0f4ff !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #1d4ed8 !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.15) !important;
    background: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# Chat state
if "assistant_msgs" not in st.session_state:
    st.session_state.assistant_msgs = []

# New input — at the top so the user sees it immediately
typed = st.chat_input("💬 Ask about the air quality, weather, or the methodology…")
prompt = typed or st.session_state.pop("queued_prompt", None)

# Quick-start buttons
st.markdown("##### ⚡ Quick start")
cols = st.columns(2)
for i, q in enumerate(EXAMPLES):
    if cols[i % 2].button(q, key=f"ex_{i}", width="stretch"):
        st.session_state.queued_prompt = q
        st.rerun()

st.divider()

# Render history
for m in st.session_state.assistant_msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("charts"):
            render_charts(m["charts"])

if prompt:
    st.session_state.assistant_msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ok, reply, charts = get_assistant_reply(st.session_state.assistant_msgs, digest_text)
        st.markdown(reply)
        render_charts(charts)
    st.session_state.assistant_msgs.append(
        {"role": "assistant", "content": reply, "charts": charts}
    )

# Reset
if st.session_state.assistant_msgs:
    if st.button("Clear conversation"):
        st.session_state.assistant_msgs = []
        st.rerun()