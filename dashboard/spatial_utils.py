"""IDW spatial interpolation for 7-station point-forecast visualisation.

Kept separate from forecast_utils.py (which owns XGBoost feature engineering)
because spatial geometry is an orthogonal concern and is independently testable.
No scipy or geopandas dependency — pure numpy.
"""
import numpy as np

GRID_SIZE: int = 60    # cells per axis — smooth enough, cheap (~3 600 points total)
MARGIN: float  = 0.02  # degrees of lat/lon padding beyond the station bounding box
IDW_POWER: int = 2     # standard inverse-distance weighting exponent (p = 2)


def idw_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    grid_size: int = GRID_SIZE,
    margin: float  = MARGIN,
    power: int     = IDW_POWER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate n point values onto a regular lat/lon grid using IDW.

    IDW formula: z(x) = Σ(w_i · v_i) / Σ(w_i),  w_i = 1 / d_i^power

    This is honest distance-weighted smoothing between the 7 station points.
    It adds no information beyond the control points and makes no spatial-model
    assumptions.

    Parameters
    ----------
    lats, lons : array-like, shape (n,)   — control-point coordinates
    values     : array-like, shape (n,)   — values to interpolate (e.g. WHO ratios)
    grid_size  : int   — number of cells per axis
    margin     : float — degrees padding around the bounding box
    power      : int   — IDW exponent (2 is conventional)

    Returns
    -------
    grid_lats : shape (grid_size,)              — uniformly spaced latitudes
    grid_lons : shape (grid_size,)              — uniformly spaced longitudes
    z         : shape (grid_size, grid_size)    — z[i,j] at (grid_lats[i], grid_lons[j])
    """
    lats   = np.asarray(lats,   dtype=float)
    lons   = np.asarray(lons,   dtype=float)
    values = np.asarray(values, dtype=float)

    grid_lats = np.linspace(lats.min() - margin, lats.max() + margin, grid_size)
    grid_lons = np.linspace(lons.min() - margin, lons.max() + margin, grid_size)

    # Broadcast to (n_stations, grid_lat, grid_lon) without materialising a full copy.
    glat, glon = np.meshgrid(grid_lats, grid_lons, indexing="ij")
    dlat = glat[np.newaxis] - lats[:, np.newaxis, np.newaxis]
    dlon = glon[np.newaxis] - lons[:, np.newaxis, np.newaxis]
    dist_sq = dlat ** 2 + dlon ** 2  # squared Euclidean distance in lat/lon space

    # eps prevents 0/0 if a grid node coincides exactly with a station.
    # In practice the nearest node is ~half a grid step away, so this is
    # only a numerical safeguard.
    eps = 1e-18
    # weight_i = 1 / d_i^power = (d_i^2)^(-power/2)
    weights = 1.0 / np.maximum(dist_sq, eps) ** (power / 2)  # shape (n, gs, gs)

    z = (weights * values[:, np.newaxis, np.newaxis]).sum(axis=0) / weights.sum(axis=0)
    return grid_lats, grid_lons, z


# ---------------------------------------------------------------------------
# Phase C — AOI masking
# Requires: geopandas, shapely  (add to requirements.txt if not present)
# ---------------------------------------------------------------------------
from pathlib import Path
import geopandas as gpd
from shapely.geometry import MultiPoint, Point

_COMARCA_SHP = (
    Path(__file__).parent.parent   # project root
    / "GIS" / "boundaries"
    / "COMARCAS_5000_ETRS89.shp"
)
WGS84 = "EPSG:4326"


def _gran_bilbao_polygon():
    """Return the Gran Bilbao comarca as a single Shapely polygon in WGS84."""
    com = gpd.read_file(_COMARCA_SHP)
    gb  = com[com["COMARCA"] == "GRAN BILBAO"].to_crs(WGS84)
    return gb.union_all()


def build_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    use_boundary: bool = True,
):
    if not use_boundary:
        hull = MultiPoint(list(zip(np.asarray(lons), np.asarray(lats)))).convex_hull
        return hull
    # Use comarca boundary directly — hull with 7 stations is too angular
    return _gran_bilbao_polygon()


def mask_idw_grid(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    z: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    use_boundary: bool = True,
) -> np.ndarray:
    """
    Set IDW grid cells outside the station convex hull (+ optional comarca
    boundary) to NaN.  Signature mirrors idw_grid() outputs for easy chaining.

    Parameters
    ----------
    grid_lats, grid_lons : 1-D arrays returned by idw_grid()
    z                    : 2-D result array from idw_grid()  (shape: lat × lon)
    lats, lons           : station coordinates (same as passed to idw_grid)
    use_boundary         : passed to build_mask()

    Returns
    -------
    z_masked : copy of z with out-of-mask cells set to NaN
    """
    mask_poly = build_mask(lats, lons, use_boundary=use_boundary)
    glat, glon = np.meshgrid(grid_lats, grid_lons, indexing="ij")  # same as idw_grid

    # Vectorised containment — avoids Python loop over 3 600 cells
   
    from shapely import contains_xy
    from shapely.geometry import Point

    buffered = mask_poly.buffer(0.001)
    inside   = contains_xy(buffered, glon.ravel(), glat.ravel())

    z_masked = z.copy().astype(float)
    z_masked[~inside.reshape(z.shape)] = np.nan
    return z_masked