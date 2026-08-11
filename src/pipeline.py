"""Resample the harmonized ocean+weather data to a weekly grid and impute gaps.

Follows Documentation/SPEC.md section 4.
"""

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "Depth_m",
    "Temp_C",
    "Salinity",
    "NO3",
    "NO2",
    "PO4",
    "Silicate",
    "O2_umol_L",
    "Chl_a",
    "Air_Temp_C",
    "Wind_Speed_ms",
    "Wind_Dir_deg",
    "Wind_U",
    "Wind_V",
]

# Columns actually resampled/imputed per depth. Depth_m itself is excluded: within a
# depth group it's already a constant, not a series to interpolate.
IMPUTED_COLUMNS = [c for c in NUMERIC_COLUMNS if c != "Depth_m"]

INTERPOLATION_LIMIT_WEEKS = 8


def _resample_one_depth(depth_df: pd.DataFrame, weekly_dates: pd.DatetimeIndex, depth: float) -> pd.DataFrame:
    resampled = depth_df.resample("W-MON").mean().reindex(weekly_dates)
    resampled["Depth_m"] = depth

    # Interpolate only genuine short gaps (<= the limit, strictly between two real
    # observations); a longer interior gap is left as NaN rather than fabricated
    # (SPEC.md §4). limit_area="outside" below then fills only the series' leading/
    # trailing edges, never bleeding into an interior gap the interpolation skipped.
    resampled[IMPUTED_COLUMNS] = resampled[IMPUTED_COLUMNS].interpolate(
        method="linear", limit=INTERPOLATION_LIMIT_WEEKS, limit_area="inside"
    )
    resampled[IMPUTED_COLUMNS] = (
        resampled[IMPUTED_COLUMNS].bfill(limit_area="outside").ffill(limit_area="outside")
    )
    return resampled


def prepare_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` resampled to one row per depth per week, gaps imputed, with Time_Idx
    and cyclical month encodings added."""
    df = df.copy()
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.set_index("Date")
    weekly_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq="W-MON")

    depths = df["Depth_m"].dropna().unique()
    weekly = pd.concat(
        [_resample_one_depth(df[df["Depth_m"] == depth], weekly_dates, depth) for depth in depths]
    )
    weekly = weekly.reset_index().rename(columns={"index": "Date"})

    unique_dates = weekly["Date"].sort_values().unique()
    time_idx_by_date = {date: idx for idx, date in enumerate(unique_dates)}
    weekly["Time_Idx"] = weekly["Date"].map(time_idx_by_date)
    weekly["month_sin"] = np.sin(2 * np.pi * weekly["Date"].dt.month / 12)
    weekly["month_cos"] = np.cos(2 * np.pi * weekly["Date"].dt.month / 12)

    return weekly.sort_values(["Date", "Depth_m"]).reset_index(drop=True)
