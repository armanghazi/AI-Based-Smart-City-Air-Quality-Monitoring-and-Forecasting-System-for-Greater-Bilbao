"""
scripts/verify_pipeline.py
Quick sanity check on the parquet after running daily_update.py.

Usage:
    python scripts/verify_pipeline.py
"""

import sys
from pathlib import Path
import pandas as pd

PARQUET_PATH = Path(__file__).parent.parent / "data" / "processed" / "air_quality_weather.parquet"

EXPECTED_STATIONS = {
    "ALGORTA_BBIZI2", "SANTURCE", "BASAURI",
    "BARAKALDO", "ERANDIO", "MAZARREDO", "MUSKIZ",
}


def main() -> int:
    if not PARQUET_PATH.exists():
        print(f"❌  Parquet not found: {PARQUET_PATH}")
        return 1

    df = pd.read_parquet(PARQUET_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    print("=" * 55)
    print("PIPELINE VERIFICATION")
    print("=" * 55)

    # 1. Basic counts
    print(f"\n  Total rows     : {len(df):,}")
    print(f"  Date range     : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Stations found : {sorted(df['station'].unique())}")

    # 2. Duplicate check — the most critical invariant
    dupes = df.duplicated(subset=["station", "Date"]).sum()
    if dupes > 0:
        print(f"\n❌  DUPLICATES: {dupes} duplicate (station, Date) pairs found!")
        print(df[df.duplicated(subset=["station", "Date"], keep=False)][["station", "Date"]].head())
        return 1
    print(f"\n✅  No duplicates")

    # 3. Latest date per station
    print("\n  Latest date per station:")
    latest = df.groupby("station")["Date"].max().sort_values(ascending=False)
    for station, dt in latest.items():
        print(f"    {station:<20} {dt.date()}")

    # 4. wind_u / wind_v sanity — should never be all zeros
    recent = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=7)]
    all_zero = (recent["wind_u"].abs() < 0.01) & (recent["wind_v"].abs() < 0.01)
    if all_zero.all():
        print("\n❌  wind_u and wind_v are ALL near-zero — decomposition broken!")
        return 1
    print(f"\n✅  wind_u range (last 7d): {recent['wind_u'].min():.2f} to {recent['wind_u'].max():.2f}")
    print(f"   wind_v range (last 7d): {recent['wind_v'].min():.2f} to {recent['wind_v'].max():.2f}")

    # 5. NaN counts on key columns
    key_cols = ["NO2", "PM10", "PM2.5", "SO2", "Temperature", "Humidity",
                "Precipitation", "WindSpeed", "wind_u", "wind_v"]
    nan_counts = df[key_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        print(f"\n  NaN counts (expected for some stations/pollutants):")
        for col, n in nan_counts.items():
            print(f"    {col:<15} {n:,}")
    else:
        print("\n  No NaN values")

    print("\n" + "=" * 55)
    print("✅  All checks passed.")
    print("=" * 55)
    return 0


if __name__ == "__main__":
    sys.exit(main())