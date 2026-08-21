"""Tests for dataset construction, chronological splits, and event extraction."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.dataset import (
    split_train_validation,
    get_held_out_events,
)


def _synthetic_weekly_series(n_weeks: int = 100) -> pd.DataFrame:
    """Create synthetic weekly 25m series for testing dataset functions."""
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

    # Simulate some hypoxic periods (below 60 µmol/L)
    o2_values = np.random.uniform(70, 300, n_weeks)
    # Add 2 hypoxic episodes (adjust positions based on series length)
    if n_weeks >= 25:
        o2_values[20:25] = np.random.uniform(30, 55, 5)  # 5-week episode
    if n_weeks >= 53:
        o2_values[50:53] = np.random.uniform(40, 58, 3)  # 3-week episode
    elif n_weeks >= 43:
        # For shorter series, add second episode earlier
        o2_values[40:43] = np.random.uniform(40, 58, 3)  # 3-week episode

    df = pd.DataFrame({
        "Date": dates,
        "Depth_m": 25.0,
        "O2_umol_L": o2_values,
        "Time_Idx": range(n_weeks),
        "Temp_C": np.random.uniform(8, 12, n_weeks),
        "sample_weight": np.ones(n_weeks),  # Simplified weights
        "month_sin": np.sin(2 * np.pi * np.arange(n_weeks) / 52),
        "month_cos": np.cos(2 * np.pi * np.arange(n_weeks) / 52),
    })

    return df


def test_split_train_validation_is_chronological():
    """Train/val split should be chronological, not random."""
    df = _synthetic_weekly_series(n_weeks=100)

    train_df, val_df = split_train_validation(df, train_ratio=0.8)

    # Train should be first 80% by time
    assert len(train_df) == 80
    assert len(val_df) == 20

    # All train dates should be earlier than all val dates
    assert train_df["Date"].max() < val_df["Date"].min()

    # Check continuity: last train date + 1 week = first val date
    expected_val_start = train_df["Date"].max() + pd.Timedelta(weeks=1)
    assert val_df["Date"].min() == expected_val_start


def test_split_train_validation_preserves_order():
    """Split should preserve row order within each set."""
    df = _synthetic_weekly_series(n_weeks=50)

    train_df, val_df = split_train_validation(df, train_ratio=0.7)

    # Check Time_Idx is sequential
    assert train_df["Time_Idx"].tolist() == list(range(35))
    assert val_df["Time_Idx"].tolist() == list(range(35, 50))


def test_get_held_out_events_identifies_episodes():
    """get_held_out_events should find hypoxic episodes using Phase 4 logic."""
    df = _synthetic_weekly_series(n_weeks=100)

    events = get_held_out_events(df, min_weeks=2)

    # Should find at least 2 episodes (5-week and 3-week from synthetic data)
    assert len(events) >= 2

    # Check columns
    assert "start_date" in events.columns
    assert "end_date" in events.columns
    assert "duration_weeks" in events.columns
    assert "min_o2" in events.columns

    # All durations should be >= min_weeks
    assert (events["duration_weeks"] >= 2).all()


def test_get_held_out_events_filters_by_validation_period():
    """Event extraction can be filtered to validation period only."""
    df = _synthetic_weekly_series(n_weeks=100)

    # Split to get val start date
    train_df, val_df = split_train_validation(df, train_ratio=0.8)
    val_start_date = val_df["Date"].min()

    # Get events in validation period only
    val_events = get_held_out_events(df, min_weeks=1, val_start_date=val_start_date)

    # All events should start on or after val_start_date
    assert (val_events["start_date"] >= val_start_date).all()


def test_get_held_out_events_respects_min_weeks():
    """Short episodes below min_weeks threshold should be filtered out."""
    df = _synthetic_weekly_series(n_weeks=100)

    # Get events with different min_weeks
    events_2w = get_held_out_events(df, min_weeks=2)
    events_4w = get_held_out_events(df, min_weeks=4)

    # Stricter threshold should find fewer or equal events
    assert len(events_4w) <= len(events_2w)

    # 4-week threshold should filter out the 3-week episode
    if len(events_4w) > 0:
        assert (events_4w["duration_weeks"] >= 4).all()


def test_chronological_split_no_future_leakage():
    """Ensure validation set doesn't leak into training via Time_Idx."""
    df = _synthetic_weekly_series(n_weeks=60)

    train_df, val_df = split_train_validation(df, train_ratio=0.8)

    # Maximum Time_Idx in train should be less than minimum in val
    assert train_df["Time_Idx"].max() < val_df["Time_Idx"].min()

    # No overlap in Time_Idx
    train_indices = set(train_df["Time_Idx"])
    val_indices = set(val_df["Time_Idx"])
    assert len(train_indices & val_indices) == 0


# Note: create_training_dataset and create_dataloaders tests require pytorch-forecasting
# to be installed. These are integration tests that should be run once the package is available.
#
# Test structure for those (to be uncommented when pytorch-forecasting is installed):
#
# def test_create_training_dataset_includes_weight_column():
#     """TimeSeriesDataSet should be configured with weight parameter."""
#     df = _synthetic_weekly_series(n_weeks=100)
#     dataset = create_training_dataset(df)
#     assert dataset.weight == "sample_weight"
#
# def test_sanity_check_batch_weights_detects_weights():
#     """Sanity check should confirm weight tensor is present in batches."""
#     df = _synthetic_weekly_series(n_weeks=100)
#     train_df, val_df = split_train_validation(df)
#     train_dl, val_dl, _ = create_dataloaders(train_df, val_df, batch_size=16)
#
#     results = sanity_check_batch_weights(train_dl)
#     assert results["weight_present"] == True
#     assert results["weight_ratio"] is not None
