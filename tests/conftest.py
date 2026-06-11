"""Shared fixtures for the model-contract test suite.

All expensive operations (loading data, loading bundles) are session-scoped
so the parquet and four joblib files are read once per pytest run.
"""
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import pytest

# Make dashboard/ importable so `import config` and `from forecast_utils import ...` work.
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))

# Streamlit emits "No runtime found" warnings when cache decorators run outside
# a Streamlit server — suppress them so test output stays readable.
warnings.filterwarnings("ignore", message="No runtime found")
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

from config import load_data  # noqa: E402 (must come after sys.path edit)

MODELS_DIR = Path(__file__).parent.parent / "models"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Full dataframe from config.load_data() — read once per session."""
    return load_data()


@pytest.fixture(scope="session")
def station_codes(raw_df: pd.DataFrame) -> dict:
    """Global station→code mapping identical to what the dashboards build."""
    return {s: i for i, s in enumerate(sorted(raw_df["station"].unique()))}


@pytest.fixture(scope="session")
def bundles() -> dict:
    """All four model bundles keyed by pollutant name."""
    return {
        p: joblib.load(MODELS_DIR / f"xgb_{p.replace('.', '').lower()}_forecast.joblib")
        for p in POLLUTANTS
    }
