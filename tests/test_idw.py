"""Sanity tests for the IDW spatial interpolation helper (spatial_utils.idw_grid).

Invariants verified:
  1. The output grid is entirely finite — no NaN or Inf.
  2. At the grid cell nearest to each control point, the IDW value closely
     matches that control point's value.  IDW is exact at control points;
     because our grid is discrete the nearest cell is at most half a grid
     step away, but the IDW weight ratio at that offset is ~600:1 (nearest
     station vs. all others), giving relative error < 0.2%.  We assert
     at 2% to stay robust against edge cases.
  3. Output shapes are correct.

No data files or Streamlit fixtures needed — purely geometric.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))
from spatial_utils import idw_grid, GRID_SIZE, MARGIN, IDW_POWER

# Approximate station coordinates for Greater Bilbao (7 stations).
# Geometry is what matters; values don't need to match live data.
LATS = np.array([43.263, 43.308, 43.237, 43.297, 43.334, 43.346, 43.359])
LONS = np.array([-2.935, -2.888, -2.886, -2.991, -3.031, -3.007, -3.105])
VALS = np.array([1.20,   0.80,   1.50,   0.60,   1.80,   0.40,   2.10])

# 2% relative tolerance — see module docstring for the derivation.
NEAREST_CELL_RTOL = 0.02


@pytest.fixture(scope="module")
def idw_result():
    return idw_grid(LATS, LONS, VALS)


def test_no_nan_or_inf(idw_result):
    """IDW grid must be entirely finite."""
    _, _, z = idw_result
    assert not np.any(np.isnan(z)), "IDW grid contains NaN"
    assert not np.any(np.isinf(z)), "IDW grid contains Inf"


def test_honours_control_points(idw_result):
    """IDW value at the grid cell nearest each station must match the
    station's control-point value within NEAREST_CELL_RTOL (2%)."""
    grid_lats, grid_lons, z = idw_result

    for k, (slat, slon, sval) in enumerate(zip(LATS, LONS, VALS)):
        i_lat = int(np.argmin(np.abs(grid_lats - slat)))
        i_lon = int(np.argmin(np.abs(grid_lons - slon)))
        cell_val = z[i_lat, i_lon]

        rel_err = abs(cell_val - sval) / max(abs(sval), 1e-9)
        assert rel_err < NEAREST_CELL_RTOL, (
            f"Station {k}: nearest-cell IDW={cell_val:.5f}, "
            f"control={sval:.5f}, rel_err={rel_err * 100:.3f}% "
            f"> {NEAREST_CELL_RTOL * 100:.0f}% tolerance"
        )


def test_grid_shape(idw_result):
    """Output arrays must have the shapes (GRID_SIZE,) and (GRID_SIZE, GRID_SIZE)."""
    grid_lats, grid_lons, z = idw_result
    assert grid_lats.shape == (GRID_SIZE,), f"grid_lats shape {grid_lats.shape}"
    assert grid_lons.shape == (GRID_SIZE,), f"grid_lons shape {grid_lons.shape}"
    assert z.shape == (GRID_SIZE, GRID_SIZE), f"z shape {z.shape}"


def test_values_in_range(idw_result):
    """Every interpolated value must lie within [min(VALS), max(VALS)].

    IDW is a convex combination of the control-point values, so it can
    never extrapolate beyond the observed range.
    """
    _, _, z = idw_result
    assert z.min() >= VALS.min() - 1e-9, (
        f"IDW surface undershot control-point minimum: {z.min():.5f} < {VALS.min():.5f}"
    )
    assert z.max() <= VALS.max() + 1e-9, (
        f"IDW surface overshot control-point maximum: {z.max():.5f} > {VALS.max():.5f}"
    )


def test_custom_power():
    """idw_grid is well-behaved for power values other than the default (2)."""
    for power in (1, 3):
        gl, glo, z = idw_grid(LATS, LONS, VALS, power=power)
        assert not np.any(np.isnan(z)), f"NaN with power={power}"
        assert not np.any(np.isinf(z)), f"Inf with power={power}"
