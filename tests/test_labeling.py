"""Tests for the hypoxia tier/weight/episode logic in src/labeling.py."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.labeling import (
    THRESHOLDS,
    TIER_WEIGHTS,
    add_oxygen_deficit,
    add_sample_weight,
    identify_hypoxic_episodes,
    imbalance_report,
    select_target_series,
)


def _weekly_frame(o2_values: list[float], depth: float = 25.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-06", periods=len(o2_values), freq="W-MON")
    return pd.DataFrame({"Date": dates, "Depth_m": depth, "O2_umol_L": o2_values})


def test_select_target_series_keeps_only_25m():
    df = pd.concat([_weekly_frame([100.0, 100.0], depth=1.0), _weekly_frame([50.0, 50.0], depth=25.0)])

    result = select_target_series(df)

    assert (result["Depth_m"] == 25.0).all()
    assert len(result) == 2


def test_sample_weight_matches_tier():
    df = _weekly_frame([90.0, 70.0, 45.0, 10.0, np.nan])
    result = add_sample_weight(df)

    assert result["sample_weight"].tolist()[:4] == [
        TIER_WEIGHTS["normoxic"],
        TIER_WEIGHTS["watch"],
        TIER_WEIGHTS["hypoxic"],
        TIER_WEIGHTS["severe"],
    ]
    assert np.isnan(result["sample_weight"].iloc[4])  # unknown O2 -> not confidently normoxic


def test_oxygen_deficit_is_zero_above_hypoxic_line_and_nan_when_unknown():
    df = _weekly_frame([90.0, 40.0, np.nan])
    result = add_oxygen_deficit(df)

    assert result["oxygen_deficit"].iloc[0] == pytest.approx(0.0)
    assert result["oxygen_deficit"].iloc[1] == pytest.approx(THRESHOLDS["hypoxic"] - 40.0)
    assert np.isnan(result["oxygen_deficit"].iloc[2])


def test_imbalance_report_ignores_unknown_weeks():
    # 4 known weeks: one below each threshold boundary, plus one unknown week that
    # must not be counted in the denominator.
    df = _weekly_frame([90.0, 70.0, 45.0, 10.0, np.nan])

    report = imbalance_report(df)

    assert report["watch"] == pytest.approx(3 / 4)
    assert report["hypoxic"] == pytest.approx(2 / 4)
    assert report["severe"] == pytest.approx(1 / 4)


def test_identify_hypoxic_episodes_groups_consecutive_weeks():
    # normoxic, then a 3-week dip, normoxic, then a 1-week dip
    df = _weekly_frame([90.0, 40.0, 30.0, 20.0, 90.0, 50.0, 90.0])

    episodes = identify_hypoxic_episodes(df)

    assert len(episodes) == 2
    assert episodes["duration_weeks"].tolist() == [3, 1]
    assert episodes["min_o2"].tolist() == [20.0, 50.0]
