import pandas as pd
import requests
import time
from pathlib import Path


STATIONS = {
    "ALGORTA":   (43.362056, -3.022782),
    "BARAKALDO": (43.298379, -2.987133),
    "BASAURI":   (43.241131, -2.883761),
    "ERANDIO":   (43.302653, -2.977240),
    "MAZARREDO": (43.267506, -2.935188),
    "MUSKIZ":    (43.320713, -3.112716),
    "SANTURTZI": (43.333012, -3.042560),
}

RENAME_MAP = {
    "temperature_2m_mean":             "Temperature",
    "relative_humidity_2m_mean":       "Humidity",
    "precipitation_sum":               "Precipitation",
    "wind_speed_10m_max":              "WindSpeed",
    "wind_direction_10m_dominant":     "WindDirection",
}


def fetch_weather(station_name: str, lat: float, lon: float,
                  start: str = "2015-01-01",
                  end:   str = "2026-05-31") -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/era5"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_mean,relative_humidity_2m_mean,"
        "precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant"
        "&timezone=Europe/Madrid"
    )
    resp = requests.get(url, timeout=15)
    data = resp.json()
    df = pd.DataFrame(data["daily"])
    df.rename(columns=RENAME_MAP, inplace=True)
    df["Date"]    = pd.to_datetime(df["time"])
    df["station"] = station_name
    return df.drop(columns=["time"])


def collect_all_weather(output_path: Path) -> pd.DataFrame:
    dfs = []
    for name, (lat, lon) in STATIONS.items():
        print(f"Fetching {name}...")
        try:
            df = fetch_weather(name, lat, lon)
            dfs.append(df)
            print(f"  ✅ {len(df)} rows")
        except Exception as e:
            print(f"  ❌ {e}")
        time.sleep(1.5)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    return combined