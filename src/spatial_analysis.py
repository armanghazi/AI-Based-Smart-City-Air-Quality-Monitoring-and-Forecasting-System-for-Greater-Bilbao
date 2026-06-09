import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def build_geodataframe(df: pd.DataFrame,
                       crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    geometry = gpd.points_from_xy(df["Longitude"], df["Latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs(crs, inplace=True)
    return gdf


def get_station_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """    Returns a GeoDataFrame with one row per station, containing average pollutant levels and geometry."""
    station_avg = (
        df.groupby(["station", "Latitude", "Longitude"])
        [["NO2", "PM2.5", "PM10", "SO2"]]
        .mean()
        .reset_index()
    )
    return build_geodataframe(station_avg)