"""Unit tests for feature engineering (src/features.py). Verifies that surface readings
are correctly reindexed, gradients are computed as Surface - Bottom, the O2 derivative
respects the INCLUDE_O2_DERIVATIVE flag, and candidate features are added as expected.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from src import features


def test_add_surface_readings():
    """Surface readings (1 m) are reindexed onto 25 m dates."""
    dates = pd.date_range("2020-01-06", periods=3, freq="W-MON")

    df_all = pd.DataFrame({
        "Date": dates.tolist() * 2,
        "Depth_m": [1.0, 1.0, 1.0, 25.0, 25.0, 25.0],
        "Temp_C": [10.0, 11.0, 12.0, 8.0, 8.5, 9.0],
        "O2_umol_L": [300.0, 310.0, 320.0, 250.0, 260.0, 270.0],
    })

    df_25m = df_all[df_all["Depth_m"] == 25.0].copy().reset_index(drop=True)

    result = features.add_surface_readings(df_25m, df_all)

    assert "Surface_Temp_C" in result.columns
    assert "Surface_O2_umol_L" in result.columns
    assert result["Surface_Temp_C"].tolist() == [10.0, 11.0, 12.0]
    assert result["Surface_O2_umol_L"].tolist() == [300.0, 310.0, 320.0]


def test_add_vertical_gradients():
    """Vertical gradients are Surface - Bottom."""
    df = pd.DataFrame({
        "Surface_Temp_C": [10.0, 11.0],
        "Temp_C": [8.0, 8.5],
        "Surface_O2_umol_L": [300.0, 310.0],
        "O2_umol_L": [250.0, 260.0],
    })

    result = features.add_vertical_gradients(df)

    assert "Vertical_Temp_Grad" in result.columns
    assert "Vertical_O2_Grad" in result.columns
    assert result["Vertical_Temp_Grad"].tolist() == [2.0, 2.5]
    assert result["Vertical_O2_Grad"].tolist() == [50.0, 50.0]


def test_add_o2_derivative():
    """O2_Derivative_1W is week-over-week change (diff periods=1)."""
    df = pd.DataFrame({
        "O2_umol_L": [250.0, 260.0, 255.0, 240.0],
    })

    result = features.add_o2_derivative(df)

    assert "O2_Derivative_1W" in result.columns
    # First value is NaN (no prior week), then diffs
    assert pd.isna(result["O2_Derivative_1W"].iloc[0])
    assert result["O2_Derivative_1W"].iloc[1] == 10.0  # 260 - 250
    assert result["O2_Derivative_1W"].iloc[2] == -5.0  # 255 - 260
    assert result["O2_Derivative_1W"].iloc[3] == -15.0  # 240 - 255


def test_add_wind_mixing_energy():
    """Wind_Mixing_Energy is wind speed squared."""
    df = pd.DataFrame({
        "Wind_Speed_ms": [5.0, 10.0, 0.0],
    })

    result = features.add_wind_mixing_energy(df)

    assert "Wind_Mixing_Energy" in result.columns
    assert result["Wind_Mixing_Energy"].tolist() == [25.0, 100.0, 0.0]


def test_add_chlorophyll_lags():
    """Chlorophyll lag features are 1-4 week shifts."""
    df = pd.DataFrame({
        "Chl_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })

    result = features.add_chlorophyll_lags(df)

    assert "Chl_a_lag_1W" in result.columns
    assert "Chl_a_lag_2W" in result.columns
    assert "Chl_a_lag_3W" in result.columns
    assert "Chl_a_lag_4W" in result.columns

    # 1-week lag: first value NaN, then 1.0, 2.0, ...
    assert pd.isna(result["Chl_a_lag_1W"].iloc[0])
    assert result["Chl_a_lag_1W"].iloc[1] == 1.0
    assert result["Chl_a_lag_1W"].iloc[2] == 2.0

    # 4-week lag: first 4 values NaN, then 1.0, 2.0
    assert pd.isna(result["Chl_a_lag_4W"].iloc[0])
    assert pd.isna(result["Chl_a_lag_4W"].iloc[3])
    assert result["Chl_a_lag_4W"].iloc[4] == 1.0
    assert result["Chl_a_lag_4W"].iloc[5] == 2.0


def test_engineer_features_includes_all_features():
    """engineer_features applies all transformations and includes candidate features."""
    dates = pd.date_range("2020-01-06", periods=5, freq="W-MON")

    df_all = pd.DataFrame({
        "Date": dates.tolist() * 2,
        "Depth_m": [1.0] * 5 + [25.0] * 5,
        "Temp_C": [10.0, 11.0, 12.0, 13.0, 14.0, 8.0, 8.5, 9.0, 9.5, 10.0],
        "O2_umol_L": [300.0, 310.0, 320.0, 330.0, 340.0, 250.0, 260.0, 255.0, 240.0, 230.0],
        "Chl_a": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
        "Wind_Speed_ms": [5.0, 6.0, 7.0, 8.0, 9.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "Wind_Dir_deg": [180.0, 190.0, 200.0, 210.0, 220.0, 180.0, 190.0, 200.0, 210.0, 220.0],
    })

    df_25m = df_all[df_all["Depth_m"] == 25.0].copy().reset_index(drop=True)

    result = features.engineer_features(df_25m, df_all)

    # Core engineered features
    assert "Surface_Temp_C" in result.columns
    assert "Surface_O2_umol_L" in result.columns
    assert "Vertical_Temp_Grad" in result.columns
    assert "Vertical_O2_Grad" in result.columns

    # Optional O2 derivative (depends on INCLUDE_O2_DERIVATIVE flag)
    if features.INCLUDE_O2_DERIVATIVE:
        assert "O2_Derivative_1W" in result.columns

    # Candidate features
    assert "Wind_Dir_deg" in result.columns  # Raw direction preserved
    assert "Wind_Mixing_Energy" in result.columns
    assert "Chl_a_lag_1W" in result.columns
    assert "Chl_a_lag_2W" in result.columns
    assert "Chl_a_lag_3W" in result.columns
    assert "Chl_a_lag_4W" in result.columns


def test_include_o2_derivative_flag():
    """O2_Derivative_1W is added only if INCLUDE_O2_DERIVATIVE is True."""
    dates = pd.date_range("2020-01-06", periods=3, freq="W-MON")

    df_all = pd.DataFrame({
        "Date": dates.tolist() * 2,
        "Depth_m": [1.0, 1.0, 1.0, 25.0, 25.0, 25.0],
        "Temp_C": [10.0, 11.0, 12.0, 8.0, 8.5, 9.0],
        "O2_umol_L": [300.0, 310.0, 320.0, 250.0, 260.0, 270.0],
        "Chl_a": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        "Wind_Speed_ms": [5.0, 6.0, 7.0, 5.0, 6.0, 7.0],
        "Wind_Dir_deg": [180.0, 190.0, 200.0, 180.0, 190.0, 200.0],
    })

    df_25m = df_all[df_all["Depth_m"] == 25.0].copy().reset_index(drop=True)

    # Test with flag enabled (current default)
    original_flag = features.INCLUDE_O2_DERIVATIVE
    features.INCLUDE_O2_DERIVATIVE = True
    result_with = features.engineer_features(df_25m, df_all)
    assert "O2_Derivative_1W" in result_with.columns

    # Test with flag disabled
    features.INCLUDE_O2_DERIVATIVE = False
    result_without = features.engineer_features(df_25m, df_all)
    assert "O2_Derivative_1W" not in result_without.columns

    # Restore original flag
    features.INCLUDE_O2_DERIVATIVE = original_flag
