import numpy as np
import pandas as pd
import pytest

from src import copernicus_reanalysis as cr
from src.pipeline import prepare_weekly_series


def test_temperature_and_salinity_come_from_physics_dataset():
    # thetao/so don't exist in the BGC dataset; requesting them there was the original bug
    bgc, phy = cr.DATASETS["cmems_mod_bal_bgc_my_P1M-m"], cr.DATASETS["cmems_mod_bal_phy_my_P1M-m"]
    assert "thetao" not in bgc and "so" not in bgc
    assert set(phy) == {"thetao", "so"}


def _grid(depths, values_by_cell):
    xr = pytest.importorskip("xarray")
    times = pd.date_range("2000-01-01", periods=3, freq="MS")
    lats = np.array([cr.BOKNIS_LAT, cr.BOKNIS_LAT + 0.05])
    lons = np.array([cr.BOKNIS_LON, cr.BOKNIS_LON + 0.05])
    data = np.full((3, len(depths), 2, 2), np.nan)
    for (k, i, j), value in values_by_cell.items():
        data[:, k, i, j] = value
    return xr.DataArray(
        data,
        dims=("time", "depth", "latitude", "longitude"),
        coords={"time": times, "depth": depths, "latitude": lats, "longitude": lons},
    )


def test_nearest_valid_series_skips_land_cells():
    da = _grid([25.0], {(0, 1, 1): [1.0, 2.0, 3.0]})  # only the farther cell is sea
    series, _ = cr._nearest_valid_series(da, 25.0)
    assert series.tolist() == [1.0, 2.0, 3.0]


def test_nearest_valid_series_uses_shallower_bottom_level_within_tolerance():
    # nearest cell's bottom is at 21 m (25.5 m level is dry): use 21 m there
    da = _grid([1.0, 21.0, 25.5], {(0, 0, 0): 9.0, (1, 0, 0): 5.0, (2, 1, 1): 7.0})
    series, info = cr._nearest_valid_series(da, 25.0)
    assert (series == 5.0).all() and "21.0 m" in info


def test_nearest_valid_series_rejects_levels_too_far_from_target():
    da = _grid([1.0, 10.0], {(0, 0, 0): 9.0, (1, 0, 0): 5.0})
    assert cr._nearest_valid_series(da, 25.0) is None


def test_combine_keeps_one_column_per_variable_across_depths():
    dates = pd.date_range("2000-01-01", periods=2, freq="MS")
    frames = [
        pd.DataFrame({"Date": dates, "Depth_m": depth, var: value})
        for var in ("O2_umol_L", "Temp_C")
        for depth, value in ((1.0, 1.0), (25.0, 2.0))
    ]
    df = cr._combine(frames)
    assert sorted(df.columns) == ["Date", "Depth_m", "O2_umol_L", "Temp_C"]
    assert len(df) == 4 and df.notna().all().all()
    assert (df.loc[df["Depth_m"] == 25.0, "O2_umol_L"] == 2.0).all()


def test_fill_gaps_only_fills_missing_weeks_at_matching_depth():
    monthly = pd.DataFrame(
        {
            "Date": list(pd.date_range("2000-01-01", periods=6, freq="MS")) * 2,
            "Depth_m": [1.0] * 6 + [25.0] * 6,
            "O2_umol_L": [300.0] * 6 + [50.0] * 6,
        }
    )
    reanalysis = cr.interpolate_monthly_to_weekly(monthly)

    dates = pd.date_range("2000-01-03", periods=20, freq="W-MON")
    o2_25 = [100.0] * 3 + [np.nan] * 14 + [100.0] * 3  # interior gap > interpolation limit
    raw = pd.DataFrame(
        {
            "Date": list(dates) * 2,
            "Depth_m": [1] * 20 + [25] * 20,
            "O2_umol_L": [280.0] * 20 + o2_25,
        }
    )
    for col in ["Temp_C", "Salinity", "NO3", "NO2", "PO4", "Silicate", "Chl_a"]:
        raw[col] = 1.0

    weekly = prepare_weekly_series(raw)
    assert weekly.loc[weekly["Depth_m"] == 25, "O2_umol_L"].isna().any()

    filled = cr.fill_gaps(weekly, reanalysis)
    at_25 = filled[filled["Depth_m"] == 25]
    at_1 = filled[filled["Depth_m"] == 1]

    gap = at_25["O2_umol_L_source"] == 1
    assert gap.sum() > 0
    assert (at_25.loc[gap, "O2_umol_L"] == 50.0).all()  # 25 m values, not surface ones
    assert (at_25.loc[~gap, "O2_umol_L_source"] == 0).all()
    assert (at_1["O2_umol_L_source"] == 0).all()  # observed surface untouched
    assert not filled.columns.str.endswith("_reanalysis").any()


def test_load_returns_none_without_download(tmp_path, monkeypatch):
    monkeypatch.setenv("COPERNICUS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cr, "_MEMO", {})
    assert cr.load_or_fetch_reanalysis(allow_download=False) is None
