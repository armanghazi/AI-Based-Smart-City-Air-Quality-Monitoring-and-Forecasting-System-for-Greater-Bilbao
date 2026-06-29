import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    load_data, POLLUTANT_COLOR, WHO_ANNUAL, EU_ANNUAL,
    ZONE_META, get_zone,get_fav_station, center_tables
)

from aqi import AQI_THRESHOLDS

from i18n_auto import tr

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

# NOTE: st.set_page_config is intentionally absent — called once in app.py router.
center_tables()

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]

# Model filename per pollutant: xgb_pm25_forecast.joblib, etc.
def model_path(pollutant: str) -> Path:
    prefix = pollutant.replace(".", "").lower()   # PM2.5 -> pm25
    return MODELS_DIR / f"xgb_{prefix}_forecast.joblib"


# --------------------------------------------------
# MODEL LOADING (cached)
# --------------------------------------------------

@st.cache_resource
def load_model(pollutant: str):
    """Load a saved model bundle (model + features + metrics)."""
    path = model_path(pollutant)
    if not path.exists():
        return None
    return joblib.load(path)


# --------------------------------------------------
# FEATURE PREPARATION (shared — see dashboard/forecast_utils.py)
# --------------------------------------------------

from forecast_utils import prepare_features


# --------------------------------------------------
# RECURSIVE FORECAST
# --------------------------------------------------

def recursive_forecast(df_station, model, features, pollutant,
                       horizon=7, station_codes=None):
    """
    Predict `horizon` days ahead recursively.
    Future weather is approximated by the last known values (held constant) —
    this is a simplification; accuracy degrades as horizon grows.
    Returns a DataFrame: Date, prediction.

    station_codes: global {station -> code} mapping, threaded through to
    prepare_features so station_code stays consistent with training. Without
    it, prepare_features runs on a single-station frame and collapses
    station_code to 0 on every step.
    """
    work = df_station.sort_values("Date").copy().reset_index(drop=True)

    preds = []
    last_date = work["Date"].iloc[-1]

    for step in range(1, horizon + 1):
        # Use the most recent row as the basis for the next prediction
        latest = work.iloc[[-1]].copy()
        latest_prep = prepare_features(
            work, features, station_codes=station_codes
        ).iloc[[-1]]

        X = latest_prep[features]
        yhat = float(model.predict(X)[0])
        yhat = max(yhat, 0.0)   # pollution can't be negative

        next_date = last_date + pd.Timedelta(days=step)
        preds.append({"Date": next_date, "prediction": yhat})

        # Append the prediction as the new "actual" for the next step.
        # Weather columns carried forward (held constant).
        new_row = latest.copy()
        new_row["Date"] = next_date
        new_row[pollutant] = yhat   # feed prediction back in
        work = pd.concat([work, new_row], ignore_index=True)

    return pd.DataFrame(preds)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_data()

# Stable station_code mapping — reproduces training codes (sorted order).
STATION_CODES = {s: i for i, s in enumerate(sorted(df["station"].unique()))}

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(tr("🔮 Air Quality Forecasting"))
st.markdown(
    tr("Machine-learning forecasts for Greater Bilbao · XGBoost models trained "
       "on 2015–2023, validated on 2024–2026.")
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🔮 " + tr("Forecast Settings"))
    st.divider()

    sel_pollutant = st.selectbox(tr("Pollutant"), POLLUTANTS, index=0)


    _station_list = sorted(df["station"].unique().tolist())
    _fav = st.session_state.get("fav_station")
    _idx = _station_list.index(_fav) if _fav in _station_list else 0

    sel_station = st.selectbox(
        tr("Station"), _station_list, index=_idx
    )

    st.divider()
    st.markdown("### " + tr("Model Info"))

    bundle = load_model(sel_pollutant)
    if bundle:
        m = bundle["metrics"]
        st.metric(tr("Model R²"),  f"{m['R2']:.3f}")
        st.metric(tr("Model MAE"), f"{m['MAE']:.2f} µg/m³")
        st.metric(tr("Model RMSE"), f"{m['RMSE']:.2f} µg/m³")
    else:
        st.error(f"{tr('Model for')} {sel_pollutant} {tr('not found in')} {MODELS_DIR}")

    st.divider()
    st.markdown("### " + tr("Reference lines"))
    show_who  = st.toggle(tr("Show WHO guideline"),      value=True)
    show_eu   = st.toggle(tr("Show EU Directive limit"), value=False)
    show_eaqi = st.toggle(tr("Show EAQI Good ceiling"),  value=False)
    _who_lim  = WHO_ANNUAL.get(sel_pollutant)
    _eu_lim   = EU_ANNUAL.get(sel_pollutant)
    _aqi_good = AQI_THRESHOLDS.get(sel_pollutant, [None])[0]
    if _who_lim:
        st.markdown(f"🔴 WHO: {_who_lim} µg/m³")
    if _eu_lim:
        st.markdown(f"🟢 EU: {_eu_lim} µg/m³")
    if _aqi_good:
        st.markdown(f"🩵 EAQI Good: ≤{_aqi_good} µg/m³")

# --------------------------------------------------
# GUARD
# --------------------------------------------------

if bundle is None:
    st.error(
        f"{tr('No saved model for')} {sel_pollutant}. "
        f"{tr('Expected file:')} {model_path(sel_pollutant).name} {tr('in /models.')}"
    )
    st.stop()

model    = bundle["model"]
features = bundle["features"]


def _add_ref_lines(fig) -> None:
    """Add WHO (red), EU (green), EAQI Good (teal) lines based on toggles."""
    if show_who and _who_lim:
        fig.add_hline(
            y=_who_lim, line_dash="dash", line_color="#e74c3c", opacity=0.7,
            annotation_text=f"WHO {_who_lim} µg/m³",
            annotation_font_size=9, annotation_position="top right",
        )
    if show_eu and _eu_lim:
        fig.add_hline(
            y=_eu_lim, line_dash="dot", line_color="#27ae60", opacity=0.7,
            annotation_text=f"EU {_eu_lim} µg/m³",
            annotation_font_size=9, annotation_position="bottom right",
        )
    if show_eaqi and _aqi_good:
        fig.add_hline(
            y=_aqi_good, line_dash="longdash", line_color="#50ccaa", opacity=0.6,
            annotation_text=f"EAQI Good ≤{_aqi_good}",
            annotation_font_size=9, annotation_position="top left",
        )

# Filter to selected station
station_df = df[df["station"] == sel_station].sort_values("Date").copy()

if station_df.empty:
    st.warning(tr("No data for this station."))
    st.stop()

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    tr("📈 Next-Day Forecast (Backtest)"),
    tr("🔮 Multi-Day Forecast (Recursive)"),
    tr("🧠 Model Explainability"),
])

# ==================== TAB 1: NEXT-DAY BACKTEST ====================
with tab1:
    st.markdown(f"### {tr('Next-Day Forecast vs Actual')} — {sel_pollutant} {tr('at')} {sel_station}")
    st.caption(
        tr("The model predicts the next day's value using the latest available reading (D-1). "
           "Shown on the held-out test period (2024+) for honest evaluation.")
    )

    # Prepare features and predict on the test period
    prep = prepare_features(station_df, features, station_codes=STATION_CODES)
    test_period = prep[prep["Date"] >= "2024-01-01"].copy()
    test_period = test_period.dropna(subset=features)

    if test_period.empty:
        st.warning(tr("Not enough 2024+ data for this station to backtest."))
    else:
        X_test = test_period[features]
        test_period["prediction"] = model.predict(X_test)
        test_period["prediction"] = test_period["prediction"].clip(lower=0)

        # Actual next-day value for comparison
        target_col = bundle["target"]   # e.g. target_PM25
        if target_col in test_period.columns:
            test_period["actual"] = test_period[target_col]
        else:
            test_period["actual"] = (
                test_period[sel_pollutant].shift(-1)
            )

        plot_df = test_period.dropna(subset=["actual"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df["Date"], y=plot_df["actual"],
            name="Actual", mode="lines",
            line=dict(color="#2c3e50", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=plot_df["Date"], y=plot_df["prediction"],
            name="Predicted", mode="lines",
            line=dict(color=POLLUTANT_COLOR.get(sel_pollutant, "#e74c3c"),
                      width=1.5, dash="dot"),
        ))
        _add_ref_lines(fig)
        fig.update_layout(
            height=450, hovermode="x unified",
            xaxis_title="Date", yaxis_title=f"{sel_pollutant} (µg/m³)",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, width="stretch", key="fig_backtest")

        # Accuracy on this station
        from sklearn.metrics import mean_absolute_error, r2_score
        from aqi import compute_aqi_category
        mae = mean_absolute_error(plot_df["actual"], plot_df["prediction"])
        r2  = r2_score(plot_df["actual"], plot_df["prediction"])
        c1, c2 = st.columns(2)
        c1.metric(tr("Station MAE"), f"{mae:.2f} µg/m³")
        c2.metric(tr("Station R²"),  f"{r2:.3f}")

        # ── Next forecast value (latest row prediction) ──────────────────
        st.divider()
        latest_pred = float(test_period["prediction"].iloc[-1])
        latest_d1   = test_period["Date"].iloc[-1]
        next_date   = latest_d1 + pd.Timedelta(days=1)
        cat         = compute_aqi_category(sel_pollutant, latest_pred)
        aqi_color   = cat["color"] if cat else "#95a5a6"
        aqi_label   = cat["label"] if cat else "—"

        st.markdown(
            f"""
            <div style="border-radius:12px;padding:16px 20px;
                        background:linear-gradient(135deg,{aqi_color}22,{aqi_color}08);
                        border:2px solid {aqi_color};margin-top:4px">
                <div style="font-size:0.8rem;color:#888;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px">
                    {tr('Forecast for')} {next_date.strftime('%d %b %Y')} · {sel_station}
                </div>
                <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
                    <div style="font-size:2rem;font-weight:700;color:#0c1521;
                                font-family:'IBM Plex Mono',monospace">
                        {latest_pred:.1f}
                        <span style="font-size:1rem;font-weight:400;color:#888">µg/m³</span>
                    </div>
                    <div style="background:{aqi_color};color:white;border-radius:10px;
                                padding:6px 16px;font-weight:700;font-size:1rem">
                        EAQI {aqi_label}
                    </div>
                    <div style="font-size:0.8rem;color:#666">
                        {tr('Based on D-1 reading:')} {latest_d1.strftime('%d %b %Y')}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==================== TAB 2: RECURSIVE MULTI-DAY ====================
with tab2:
    st.markdown(f"### {tr('Multi-Day Forecast')} — {sel_pollutant} {tr('at')} {sel_station}")

    st.warning(
        tr("⚠️ **Recursive forecasting limitation:** each day's prediction feeds "
           "into the next, and future weather is held constant at the last known "
           "values. Accuracy decreases as the horizon grows — treat days 4+ as "
           "indicative trend, not precise values.")
    )

    horizon = st.slider(tr("Forecast horizon (days)"), 1, 14, 7)

    prep_station = prepare_features(station_df, features, station_codes=STATION_CODES)

    # Forward-fill lag/rolling NaNs caused by short sensor gaps (≤7 days).
    # Raw parquet values are NOT touched — only derived feature columns.
    prep_filled = prep_station.copy()
    prep_filled[features] = prep_filled[features].ffill()
    valid = prep_filled.dropna(subset=features)

    if valid.empty:
        st.warning(tr("Not enough complete data to forecast for this station."))
    else:
        fc = recursive_forecast(
            valid, model, features, sel_pollutant,
            horizon=horizon, station_codes=STATION_CODES
        )

        # Plot: recent history + forecast
        history = station_df.tail(30)[["Date", sel_pollutant]]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=history["Date"], y=history[sel_pollutant],
            name="Recent history", mode="lines+markers",
            line=dict(color="#2c3e50", width=1.5),
        ))
        fig2.add_trace(go.Scatter(
            x=fc["Date"], y=fc["prediction"],
            name="Forecast", mode="lines+markers",
            line=dict(color=POLLUTANT_COLOR.get(sel_pollutant, "#e74c3c"),
                      width=2, dash="dot"),
            marker=dict(size=7),
        ))
        _add_ref_lines(fig2)
        fig2.update_layout(
            height=450, hovermode="x unified",
            xaxis_title="Date", yaxis_title=f"{sel_pollutant} (µg/m³)",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig2,  width="stretch", key="fig_recursive")

        # Forecast table
        st.markdown("#### " + tr("Forecast values"))
        show = fc.copy()
        show["Date"] = show["Date"].dt.strftime("%a %d %b %Y")
        show["prediction"] = show["prediction"].round(1)
        show.columns = ["Date", f"{sel_pollutant} (µg/m³)"]

        # WHO status
        who_lim = WHO_ANNUAL.get(sel_pollutant)
        if who_lim:
            show[tr("WHO")] = fc["prediction"].apply(
                lambda v: tr("⚠️ Above") if v > who_lim else tr("✅ Below")
            )

        # EU Directive status
        eu_lim = EU_ANNUAL.get(sel_pollutant)
        if eu_lim:
            show[tr("EU Directive")] = fc["prediction"].apply(
                lambda v: tr("⚠️ Above") if v > eu_lim else tr("✅ Below")
            )

        # EAQI level
        from aqi import compute_aqi_category
        def _eaqi_label(v: float) -> str:
            cat = compute_aqi_category(sel_pollutant, v)
            return cat["label"] if cat else "—"
        show[tr("EAQI")] = fc["prediction"].apply(_eaqi_label)

        st.dataframe(show, width="stretch", hide_index=True)


# ==================== TAB 3: EXPLAINABILITY ====================
with tab3:
    st.markdown(f"### {tr('Why does the model predict this way?')} — {sel_pollutant}")
    st.caption(
        tr("SHAP (SHapley Additive exPlanations) shows how each feature pushes "
           "individual predictions up or down — beyond simple importance ranking.")
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        # Pre-computed SHAP beeswarm (generated in notebook 07)
        img_path = (
            Path(__file__).parent.parent / "assets" /
            f"shap_{sel_pollutant.replace('.', '').lower()}.png"
        )
        if img_path.exists():
            st.image(str(img_path), width="stretch")
            st.caption(
                tr("Each dot = one prediction · red = high feature value, "
                   "blue = low · right = pushes forecast up, left = down")
            )
        else:
            st.info(
                f"{tr('SHAP plot not found')} ({img_path.name}). "
                + tr("Generate it in notebook 07 and place it in dashboard/assets/.")
            )

    with col2:
        st.markdown("#### " + tr("Key physical insights"))
        st.markdown(
            """
- **Latest pollutant level (D-1)** is the strongest predictor — pollution
  persists day-to-day (atmospheric inertia).
- **Higher wind speed** pushes forecasts **down** → dispersion.
- **Precipitation** pushes forecasts **down** → wet deposition (washout).
- **Day of week** matters most for NO₂ → weekly traffic cycle.
- **SO₂** shows the weakest temporal structure → episodic
  industrial/port emissions, hence its lower R².
            """
        )
        st.success(
            tr("✅ SHAP confirms the model learned real atmospheric mechanisms — "
               "not spurious correlations.")
        )

    st.divider()

    # Live top-10 importance from the loaded model (instant, no SHAP needed)
    st.markdown(f"#### {tr('Top 10 features')} — {sel_pollutant} {tr('model (XGBoost importance)')}")

    imp = (
        pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig_imp = px.bar(
        imp[::-1],
        x="importance", y="feature",
        orientation="h",
        color_discrete_sequence=[POLLUTANT_COLOR.get(sel_pollutant, "#3498db")],
        labels={"importance": "Importance", "feature": ""},
    )
    fig_imp.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, width="stretch", key="fig_imp")

    # Model card
    with st.expander(tr("ℹ️ Model details")):
        st.markdown(
            f"""
| | |
|---|---|
| **Model** | XGBoost Regressor |
| **Task** | Next-day {sel_pollutant} forecast |
| **Features** | {len(features)} |
| **Training period** | {bundle.get('train_period', '2015–2023')} |
| **Test period** | {bundle.get('test_period', '2024–2026')} |
| **Test R²** | {bundle['metrics']['R2']:.3f} |
| **Test RMSE** | {bundle['metrics']['RMSE']:.2f} µg/m³ |
            """
        )