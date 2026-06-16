# tests/test_spatial_mask.py
import numpy as np
import pytest
from dashboard.spatial_utils import idw_grid, mask_idw_grid, build_mask

STATION_COORDS = {
    "ALGORTA_BBIZI2": (43.362056, -3.022782),
    "BARAKALDO":      (43.296,    -2.989),
    "BASAURI":        (43.239,    -2.885),
    "ERANDIO":        (43.307,    -2.973),
    "MAZARREDO":      (43.263,    -2.935),
    "MUSKIZ":         (43.323,    -3.113),
    "SANTURTZI":      (43.323,    -3.032),
}

LATS   = np.array([v[0] for v in STATION_COORDS.values()])
LONS   = np.array([v[1] for v in STATION_COORDS.values()])
VALUES = np.array([2.1, 3.4, 2.8, 4.1, 3.7, 2.2, 1.9])

def test_mask_clips_corners():
    """Some grid cells must be NaN after masking (corners outside hull)."""
    gl, glo, z = idw_grid(LATS, LONS, VALUES)
    z_masked   = mask_idw_grid(gl, glo, z, LATS, LONS)
    masked = int(np.isnan(z_masked).sum())
    total  = z.size
    print(f"\nMasked {masked}/{total} ({masked/total:.0%})")
    assert masked > 0, "No cells were masked — hull/boundary intersection failed"


def test_mask_keeps_station_cells():
    """Grid cells nearest to each station must NOT be NaN."""
    gl, glo, z = idw_grid(LATS, LONS, VALUES)
    z_masked   = mask_idw_grid(gl, glo, z, LATS, LONS)
    for lat, lon in zip(LATS, LONS):
        i = int(np.argmin(np.abs(gl  - lat)))
        j = int(np.argmin(np.abs(glo - lon)))
        assert not np.isnan(z_masked[i, j]), \
            f"Station at ({lat},{lon}) mapped to NaN cell [{i},{j}]"


def test_build_mask_is_valid_polygon():
    """build_mask must return a non-empty polygon in WGS84 bounds."""
    mask = build_mask(LATS, LONS, use_boundary=True)
    assert not mask.is_empty
    assert mask.geom_type in ("Polygon", "MultiPolygon")
    lon_min, lat_min, lon_max, lat_max = mask.bounds
    assert -4.0 < lon_min < -2.5
    assert  43.0 < lat_min < 43.5