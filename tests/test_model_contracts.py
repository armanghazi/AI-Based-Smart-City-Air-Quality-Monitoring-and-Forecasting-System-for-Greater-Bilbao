"""Model-contract tests for the air-quality forecasting pipeline.

What these tests catch (and intentionally do NOT catch):
  Catch  — feature / dtype mismatches that would silently corrupt a prediction
  Catch  — stale or swapped model files
  Catch  — station-code collapse bug (per-subset .cat.codes)
  Catch  — prepare_features building columns no model consumes (drift)
  Ignore — model accuracy / generalisation (training notebooks own that)
  Ignore — Streamlit UI correctness (needs a browser)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.api.types as pat
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))

from forecast_utils import POLLUTANTS, prepare_features  # noqa: E402

# Expected R² values per model (from documented training runs).
# If a stored metric drifts more than R2_TOLERANCE from these, the test flags it —
# most likely a wrong/stale model file was swapped in.
EXPECTED_R2: dict[str, float] = {
    "NO2":   0.56,
    "PM2.5": 0.48,
    "PM10":  0.46,
    "SO2":   0.39,
}
R2_TOLERANCE = 0.05

# Columns that prepare_features creates but no current model consumes.
# Listed explicitly so the drift is tracked rather than silently allowed.
ALLOWED_UNUSED = {"wind_x_precip", "temp_x_humid", "station_code"}


# ---------------------------------------------------------------------------
# 1. Bundles loadable + required keys
# ---------------------------------------------------------------------------

def test_bundles_loadable(bundles):
    """Each joblib bundle loads and contains all expected keys."""
    required_keys = {"model", "features", "target", "pollutant", "metrics"}
    for p, bundle in bundles.items():
        assert isinstance(bundle, dict), f"{p}: bundle is not a dict"
        missing = required_keys - set(bundle.keys())
        assert not missing, f"{p}: bundle missing keys {missing}"
        assert len(bundle["features"]) > 0, f"{p}: features list is empty"


# ---------------------------------------------------------------------------
# 2. Feature contract — existence + numeric dtype after prepare_features
# ---------------------------------------------------------------------------

def test_feature_contract(raw_df, station_codes, bundles):
    """After prepare_features every model feature exists and is plain numeric.

    Catches: missing columns, object dtypes, pyarrow string dtypes, bool dtypes.
    All of these cause XGBoost predict() to fail or silently return NaN.
    """
    for p, bundle in bundles.items():
        feats = bundle["features"]
        prep = prepare_features(raw_df, feats, station_codes=station_codes)

        for feat in feats:
            assert feat in prep.columns, (
                f"{p}: feature '{feat}' missing from prepare_features output"
            )
            col = prep[feat]
            dtype = col.dtype

            assert str(dtype) not in ("object", "bool"), (
                f"{p}.{feat}: dtype is {dtype} — XGBoost will reject this"
            )
            assert pat.is_numeric_dtype(dtype), (
                f"{p}.{feat}: dtype {dtype} is not numeric — "
                f"possible pyarrow string or other extension type"
            )


# ---------------------------------------------------------------------------
# 3. No orphan features — every column prepare_features adds is consumed
# ---------------------------------------------------------------------------

def test_no_orphan_features(raw_df, station_codes, bundles):
    """prepare_features must not silently build columns that no model uses
    (outside the explicitly allowed set).

    If this test fails it means a new column was added to prepare_features
    without adding it to a model OR to ALLOWED_UNUSED. Fix one or the other.
    """
    all_model_features: set[str] = set()
    for b in bundles.values():
        all_model_features.update(b["features"])

    # Run prepare_features over the union of all features so every code-path fires.
    all_feats = sorted(all_model_features)
    baseline_cols = set(raw_df.columns)
    prep = prepare_features(raw_df, all_feats, station_codes=station_codes)
    new_cols = set(prep.columns) - baseline_cols

    orphans = new_cols - all_model_features - ALLOWED_UNUSED
    assert not orphans, (
        f"prepare_features creates columns consumed by no model "
        f"and absent from ALLOWED_UNUSED:\n  {sorted(orphans)}\n"
        f"Either add them to a model's feature list or to ALLOWED_UNUSED."
    )

    # Non-failing annotation: document the known drift in test output.
    known_drift = sorted(new_cols & ALLOWED_UNUSED)
    print(
        f"\n  [drift] {len(known_drift)} column(s) built by prepare_features "
        f"but in no current model: {known_drift}"
    )


# ---------------------------------------------------------------------------
# 4. Station-code mapping correctness (regression for .cat.codes collapse bug)
# ---------------------------------------------------------------------------

def test_station_code_mapping(raw_df, station_codes):
    """Global station→code mapping must be contiguous 0..n-1 and must NOT
    collapse when applied to a single-station frame.

    The regression target: using .astype('category').cat.codes on a frame that
    contains only one station always yields 0, regardless of training order.
    """
    stations = sorted(raw_df["station"].unique())
    n = len(stations)
    codes = list(station_codes.values())

    assert len(set(codes)) == n, f"Expected {n} unique codes, got {len(set(codes))}"
    assert set(codes) == set(range(n)), (
        f"Station codes are not contiguous 0..{n - 1}: {sorted(set(codes))}"
    )

    for st in stations:
        expected = station_codes[st]
        single = raw_df[raw_df["station"] == st].copy()

        # Correct path: global mapping
        global_code = int(single["station"].map(station_codes).iloc[0])
        assert global_code == expected, (
            f"'{st}': global mapping returned {global_code}, expected {expected}"
        )

        # Demonstrate the bug for non-first stations: per-subset cat.codes → 0
        if expected != 0:
            per_subset_code = int(
                single["station"].astype("category").cat.codes.iloc[0]
            )
            assert per_subset_code == 0, (
                f"'{st}': expected per-subset .cat.codes to collapse to 0 "
                f"(bug behaviour), got {per_subset_code}"
            )
            assert global_code != per_subset_code, (
                f"'{st}': global mapping and per-subset cat.codes agree — "
                f"regression target is no longer exercised"
            )


# ---------------------------------------------------------------------------
# 5. Predict smoke — pipeline runs end-to-end, output is finite and >= 0
# ---------------------------------------------------------------------------

def test_predict_smoke(raw_df, station_codes, bundles):
    """For every pollutant × station, the latest valid row produces a finite,
    non-negative prediction after the dashboard's clip(lower=0).

    No accuracy check — just that the pipeline doesn't crash or return NaN/inf.
    """
    for p, bundle in bundles.items():
        feats = bundle["features"]
        for st in sorted(raw_df["station"].unique()):
            sdf = raw_df[raw_df["station"] == st]
            prep = prepare_features(sdf, feats, station_codes=station_codes)
            valid = prep.dropna(subset=feats)
            if valid.empty:
                pytest.skip(f"{p}@{st}: no valid rows after dropna")
                continue

            X_last = valid[feats].iloc[[-1]]
            raw_pred = bundle["model"].predict(X_last)
            assert len(raw_pred) == 1, f"{p}@{st}: expected 1 prediction"

            clipped = max(float(raw_pred[0]), 0.0)
            assert math.isfinite(clipped), (
                f"{p}@{st}: prediction is not finite ({clipped})"
            )
            assert clipped >= 0.0  # tautological after clip, but explicit


# ---------------------------------------------------------------------------
# 6. Metrics sanity — stored R² must be near the documented training values
# ---------------------------------------------------------------------------

def test_metrics_sanity(bundles):
    """Stored bundle metrics must be within R2_TOLERANCE of the expected values.

    Catches a wrong or stale model file being swapped in without updating docs.
    Update EXPECTED_R2 at the top of this file when models are retrained.
    """
    for p, bundle in bundles.items():
        stored = bundle["metrics"]["R2"]
        expected = EXPECTED_R2[p]
        diff = abs(stored - expected)
        assert diff < R2_TOLERANCE, (
            f"{p}: stored R2={stored:.4f}, expected ~{expected:.2f}, "
            f"|diff|={diff:.4f} > tolerance={R2_TOLERANCE} — "
            f"wrong model file, or models were retrained without updating EXPECTED_R2"
        )
