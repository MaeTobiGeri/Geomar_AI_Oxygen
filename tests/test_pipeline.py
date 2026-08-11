"""Tests for the weekly resampling/imputation logic in src/pipeline.py."""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import prepare_weekly_series


def _single_depth_frame(o2_values: list[float]) -> pd.DataFrame:
    """One depth, one row per week starting 2020-01-06 (a Monday), with the given
    O2_umol_L values (use np.nan for missing weeks). Every other numeric column is a
    constant 1.0 so the test can focus on O2_umol_L's gap behavior."""
    dates = pd.date_range("2020-01-06", periods=len(o2_values), freq="W-MON")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Depth_m": 25.0,
            "O2_umol_L": o2_values,
            "Temp_C": 1.0,
            "Salinity": 1.0,
            "NO3": 1.0,
            "NO2": 1.0,
            "PO4": 1.0,
            "Silicate": 1.0,
            "Chl_a": 1.0,
            "Air_Temp_C": 1.0,
            "Wind_Speed_ms": 1.0,
            "Wind_Dir_deg": 1.0,
            "Wind_U": 1.0,
            "Wind_V": 1.0,
        }
    )
    return frame


def test_short_gap_is_fully_interpolated_and_edges_are_filled():
    # leading NaN, 10.0, an 8-week interior gap (exactly at the limit), 20.0, trailing NaNs
    o2 = [np.nan, 10.0] + [np.nan] * 8 + [20.0] + [np.nan] * 3
    result = prepare_weekly_series(_single_depth_frame(o2))

    assert result["O2_umol_L"].notna().all()
    assert result["O2_umol_L"].iloc[0] == pytest.approx(10.0)  # leading edge -> bfilled
    assert result["O2_umol_L"].iloc[-1] == pytest.approx(20.0)  # trailing edge -> ffilled
    # linearly interpolated between 10.0 (index 1) and 20.0 (index 10) across the gap
    assert result["O2_umol_L"].iloc[5] == pytest.approx(10.0 + (20.0 - 10.0) * 4 / 9)


def test_interior_gap_longer_than_limit_is_not_fully_fabricated():
    # a 10-week interior gap: only 8 of the 10 NaNs should get interpolated
    o2 = [10.0] + [np.nan] * 10 + [20.0]
    result = prepare_weekly_series(_single_depth_frame(o2))

    assert result["O2_umol_L"].isna().sum() == 2


def test_time_idx_and_month_encoding():
    o2 = [10.0, 11.0, 12.0]
    result = prepare_weekly_series(_single_depth_frame(o2))

    assert result["Time_Idx"].tolist() == [0, 1, 2]
    first_month = result["Date"].iloc[0].month
    assert result["month_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * first_month / 12))
    assert result["month_cos"].iloc[0] == pytest.approx(np.cos(2 * np.pi * first_month / 12))
