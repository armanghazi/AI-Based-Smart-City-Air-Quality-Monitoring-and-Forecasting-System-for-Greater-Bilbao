"""
scripts/daily_update.py

Daily pipeline: fetch D-1 air quality + meteorology → append to parquet.
Runs automatically via GitHub Actions cron (06:00 UTC).

Usage:
    python scripts/daily_update.py                    # process D-1 (yesterday)
    python scripts/daily_update.py --date 2026-06-13  # backfill a specific date

Rules enforced:
  1. Only complete days: D-0 is rejected.
  2. Idempotent: (station, Date) pairs already in parquet are skipped.
  3. wind_u / wind_v decomposed from WindSpeed + WindDirection.
  4. Missing station data is logged and skipped — never filled with zeros.
  5. Never touches dashboard/ or models/.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import time

import pandas as pd
import requests

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------

ROOT         = Path(__file__).parent.parent
PARQUET_PATH = ROOT / "data" / "processed" / "air_quality_weather.parquet"
LOG_DIR      = ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "daily_update.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONSTANTS — verified from real API responses (00_api_exploration)
# ------------------------------------------------------------------

# Euskadi REST API — daily measurements per station
# Docs: https://opendata.euskadi.eus/api-air-quality/?api=air-quality
EUSKADI_API_URL = (
    "https://api.euskadi.eus/air-quality/measurements/daily"
    "/stations/{STATION_ID}/from/{DATE}/to/{DATE}"
)

# API station IDs (confirmed from https://api.euskadi.eus/air-quality/stations)
# Parquet station name → API numeric ID
STATIONS = {
    "ALGORTA_BBIZI2": "90",
    "SANTURCE":       "96",
    "BASAURI":        "58",
    "BARAKALDO":      "59",
    "ERANDIO":        "56",
    "MAZARREDO":      "60",
    "MUSKIZ":         "63",
}

# REST API measurement name → parquet column name
# Note: PM2.5 uses comma in API ("PM2,5")
EUSKADI_FIELD_MAP = {
    "NO2":   "NO2",
    "PM10":  "PM10",
    "PM2,5": "PM2.5",
    "SO2":   "SO2",
}

# Fallback: static JSON files (full year per station, newest-first)
EUSKADI_STATIC_URL = (
    "https://opendata.euskadi.eus/contenidos/ds_informes_estudios"
    "/calidad_aire_{YEAR}/es_def/adjuntos/datos_diarios/{STATION}.json"
)
# Static JSON field name → parquet column name (values are strings with comma decimals)
EUSKADI_STATIC_FIELD_MAP = {
    "NO2gm3":  "NO2",
    "PM10gm3": "PM10",
    "PM25gm3": "PM2.5",
    "SO2gm3":  "SO2",
}

# Required columns in the final parquet (must match existing schema exactly)
REQUIRED_COLUMNS = [
    "Date", "station", "Town", "Province", "Latitude", "Longitude",
    "NO2", "PM10", "PM2.5", "SO2",
    "Temperature", "Humidity", "Precipitation",
    "WindSpeed", "WindDirection", "wind_u", "wind_v",
]

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def load_station_meta() -> pd.DataFrame:
    """
    Read station metadata (Town, Province, Latitude, Longitude) from the
    existing parquet. Coordinates must come from the real data, never hardcoded.
    Returns a DataFrame indexed by station name.
    """
    df = pd.read_parquet(
        PARQUET_PATH,
        columns=["station", "Town", "Province", "Latitude", "Longitude"],
    )
    meta = df.drop_duplicates("station").set_index("station")
    log.info(f"Loaded metadata for {len(meta)} stations.")
    return meta


def parse_value(raw) -> float:
    """
    Convert a Euskadi API value to float.
    Handles: None, empty string, integer string "13", decimal string "0,23".
    Returns NaN for missing/unparseable values.
    """
    if raw is None or str(raw).strip() == "":
        return float("nan")
    return float(str(raw).replace(",", "."))


def decompose_wind(speed: float, direction_deg: float) -> tuple[float, float]:
    """
    Meteorological convention (matches notebooks/05_weather_data.ipynb):
        wind_u = -speed * sin(dir_rad)   positive = westerly component
        wind_v = -speed * cos(dir_rad)   positive = southerly component
    """
    rad = math.radians(direction_deg)
    return -speed * math.sin(rad), -speed * math.cos(rad)


# ------------------------------------------------------------------
# FETCH AIR QUALITY — Euskadi datos_diarios
# ------------------------------------------------------------------

def fetch_air_quality(target_date: date) -> pd.DataFrame:
    """
    Fetch daily air quality for all 7 stations from Euskadi REST API.
    Endpoint: /air-quality/measurements/daily/stations/{id}/from/{date}/to/{date}
    API date format: YYYY-MM-DD. Values are floats (no comma parsing needed).

    Returns DataFrame with columns: station, Date, NO2, PM10, PM2.5, SO2
    """
    date_str = target_date.strftime("%Y-%m-%d")   # REST API date format
    rows     = []

    for station, station_id in STATIONS.items():
        url = EUSKADI_API_URL.format(STATION_ID=station_id, DATE=date_str)

        # Try REST API first; fall back to static JSON if it times out.
        # Both endpoints are throttled from GitHub Actions IPs — one usually
        # succeeds on a given day even when the other does not.
        row = None

        # --- attempt 1: REST API ---
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            records = r.json()
            if records:
                measurements = records[0].get("station", [{}])[0].get("measurements", [])
                if measurements:
                    meas_map = {m["name"]: m["value"] for m in measurements}
                    row = {"station": station, "Date": pd.Timestamp(target_date)}
                    for api_name, col_name in EUSKADI_FIELD_MAP.items():
                        row[col_name] = float(meas_map.get(api_name, float("nan")))
                    log.info(f"[AQ] {station} (REST): NO2={row['NO2']:.0f}  PM10={row['PM10']:.0f}  PM2.5={row['PM2.5']:.0f}  SO2={row['SO2']:.0f}")
        except Exception as e:
            log.warning(f"[AQ] {station} REST failed: {e}. Trying static JSON fallback...")

        # --- attempt 2: static JSON fallback ---
        if row is None:
            fallback_url = EUSKADI_STATIC_URL.format(YEAR=target_date.year, STATION=station)
            date_key = target_date.strftime("%d/%m/%Y")
            try:
                r2 = requests.get(fallback_url, timeout=30)
                r2.raise_for_status()
                records2 = r2.json()
                match = next((rec for rec in records2 if rec.get("Date") == date_key), None)
                if match:
                    row = {"station": station, "Date": pd.Timestamp(target_date)}
                    for api_field, col_name in EUSKADI_STATIC_FIELD_MAP.items():
                        row[col_name] = parse_value(match.get(api_field))
                    log.info(f"[AQ] {station} (static): NO2={row['NO2']:.0f}  PM10={row['PM10']:.0f}  PM2.5={row['PM2.5']:.0f}  SO2={row['SO2']:.0f}")
                else:
                    log.warning(f"[AQ] {station} static JSON: no record for {date_key}.")
            except Exception as e2:
                log.warning(f"[AQ] {station} static JSON also failed: {e2}")

        if row is not None:
            rows.append(row)
        else:
            log.warning(f"[AQ] {station}: both endpoints failed — skipping station.")

    if not rows:
        log.warning(
            f"[AQ] No air quality data retrieved for {target_date}. "
            "Euskadi API may be unreachable. Skipping gracefully."
        )
        sys.exit(0)  # exit 0 = not a failure; parquet unchanged; retry tomorrow

    log.info(f"[AQ] Fetched {len(rows)}/{len(STATIONS)} stations for {target_date}.")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# FETCH METEOROLOGY — Open-Meteo archive
# ------------------------------------------------------------------

def fetch_meteorology(target_date: date, station_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch Open-Meteo ERA5 archive for all stations for target_date.
    Coordinates are read from the existing parquet — never hardcoded.

    D-1 is reliably available. Never call with D-0.

    Returns DataFrame with columns:
        station, Date, Temperature, Humidity, Precipitation,
        WindSpeed, WindDirection, wind_u, wind_v
    """
    rows = []

    for station, meta_row in station_meta.iterrows():
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":   meta_row["Latitude"],
            "longitude":  meta_row["Longitude"],
            "start_date": str(target_date),
            "end_date":   str(target_date),
            "daily": ",".join([
                "temperature_2m_mean",
                "relative_humidity_2m_mean",
                "precipitation_sum",
                "windspeed_10m_mean",
                "winddirection_10m_dominant",
            ]),
            "timezone": "Europe/Madrid",
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            d = r.json()["daily"]

            speed     = float(d["windspeed_10m_mean"][0])
            direction = float(d["winddirection_10m_dominant"][0])
            wind_u, wind_v = decompose_wind(speed, direction)

            rows.append({
                "station":       station,
                "Date":          pd.Timestamp(target_date),
                "Temperature":   float(d["temperature_2m_mean"][0]),
                "Humidity":      int(round(float(d["relative_humidity_2m_mean"][0]))),
                "Precipitation": float(d["precipitation_sum"][0]),
                "WindSpeed":     speed,
                "WindDirection": int(round(direction)),
                "wind_u":        wind_u,
                "wind_v":        wind_v,
            })
            log.info(
                f"[Met] {station}: "
                f"T={rows[-1]['Temperature']:.1f}°C  "
                f"WS={speed:.1f}km/h  Dir={direction:.0f}°  "
                f"wind_u={wind_u:.2f}  wind_v={wind_v:.2f}"
            )

        except Exception as e:
            log.warning(f"[Met] {station} failed: {e}")

    if not rows:
        raise ValueError(f"[Met] No meteorology data retrieved for {target_date}.")

    log.info(f"[Met] Fetched {len(rows)}/{len(station_meta)} stations for {target_date}.")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# MERGE & VALIDATE
# ------------------------------------------------------------------

def merge_and_validate(
    aq_df: pd.DataFrame,
    met_df: pd.DataFrame,
    station_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join air quality + meteorology on (station, Date).
    Attach Town / Province / Latitude / Longitude from station_meta.
    Raise ValueError if any required column is missing or all-NaN.
    Only stations present in BOTH sources are kept (inner join).
    """
    merged = pd.merge(aq_df, met_df, on=["station", "Date"], how="inner")

    # Attach static metadata from existing parquet
    merged = merged.merge(
        station_meta[["Town", "Province", "Latitude", "Longitude"]],
        left_on="station",
        right_index=True,
        how="left",
    )

    # Validate all required columns exist
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"[Merge] Missing columns after merge: {missing_cols}")

    # Validate no pollutant column is entirely NaN
    for col in ["NO2", "PM10", "PM2.5", "SO2"]:
        if merged[col].isna().all():
            raise ValueError(
                f"[Merge] Column '{col}' is entirely NaN — check Euskadi response."
            )

    # Log stations that dropped out of the inner join
    got     = set(merged["station"])
    missing = set(STATIONS) - got
    if missing:
        log.warning(f"[Merge] Stations missing after join (no data in both sources): {missing}")

    return merged[REQUIRED_COLUMNS]


# ------------------------------------------------------------------
# APPEND TO PARQUET  (idempotent)
# ------------------------------------------------------------------

def append_to_parquet(new_df: pd.DataFrame, parquet_path: Path) -> int:
    """
    Load existing parquet, skip (station, Date) pairs already present,
    append only new rows, save atomically (temp file → rename).

    Returns the number of rows actually added (0 = already up to date).
    """
    existing = pd.read_parquet(parquet_path)
    existing["Date"] = pd.to_datetime(existing["Date"])
    new_df["Date"]   = pd.to_datetime(new_df["Date"])

    # Build a set of already-present keys for fast lookup
    existing_keys = set(
        zip(existing["station"], existing["Date"].dt.date)
    )
    mask   = [
        (s, d.date()) not in existing_keys
        for s, d in zip(new_df["station"], new_df["Date"])
    ]
    to_add = new_df[mask]

    n_added = len(to_add)
    if n_added == 0:
        log.info("[Parquet] Nothing to add — all rows already present (idempotent).")
        return 0

    combined = (
        pd.concat([existing, to_add], ignore_index=True)
        .sort_values(["station", "Date"])
        .reset_index(drop=True)
    )

    # Atomic write: write to .tmp then rename so a crash never corrupts the file
    tmp = parquet_path.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp, index=False)
    tmp.replace(parquet_path)

    log.info(f"[Parquet] Added {n_added} rows. Total rows now: {len(combined):,}.")
    return n_added


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Daily air quality + meteorology pipeline.")
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to process (YYYY-MM-DD). Default: yesterday (D-1).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Determine target date
    target = (
        date.fromisoformat(args.date)
        if args.date
        else date.today() - timedelta(days=1)
    )

    # Rule 1: never process D-0 or future dates — partial day corrupts daily means
    if target >= date.today():
        log.error(
            f"Refusing to process {target}: only complete past days are allowed. "
            "D-0 data is partial and would corrupt daily means."
        )
        return 1

    log.info("=" * 55)
    log.info(f"Processing date: {target}")
    log.info("=" * 55)

    try:
        station_meta = load_station_meta()
        aq_df        = fetch_air_quality(target)
        met_df       = fetch_meteorology(target, station_meta)
        merged       = merge_and_validate(aq_df, met_df, station_meta)
        n_added      = append_to_parquet(merged, PARQUET_PATH)

        print(
            f"\n✅  {target}  |  "
            f"{len(merged)} stations merged  |  "
            f"{n_added} rows added to parquet"
        )
        return 0

    except Exception as e:
        log.exception(f"Pipeline failed: {e}")
        print(f"\n❌  Pipeline failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())