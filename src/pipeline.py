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


def _add_calendar_columns(weekly: pd.DataFrame, sample_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Recompute the calendar/sampling columns from data_ingestion for EVERY week.

    data_ingestion computes them per raw sampling date, so after resampling they only
    exist in weeks that had a cruise (~13% of weeks) and the rest would be dropped as NaN
    even when every measurement is present (observed, interpolated or reanalysis).
    Season_* and Years_since_start are pure functions of the date. Days_since_prev and
    Segment_ID are carried from the most recent real sampling date at or before the week,
    so Days_since_prev becomes "days since the last actual measurement" (data freshness)."""
    weekly = weekly.copy()
    dates = weekly["Date"]

    if {"Season_sin", "Season_cos"} & set(weekly.columns):
        days_in_year = np.where(dates.dt.is_leap_year, 366, 365)
        weekly["Season_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / days_in_year)
        weekly["Season_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / days_in_year)

    if "Years_since_start" in weekly.columns:
        weekly["Years_since_start"] = (dates - sample_dates.min()).dt.days / 365.25

    carried = [c for c in ("Days_since_prev", "Segment_ID") if c in weekly.columns]
    if carried:
        samples = (
            pd.DataFrame({"Sample_Date": sample_dates})
            .drop_duplicates()
            .sort_values("Sample_Date")
            .reset_index(drop=True)
        )
        segment_lookup = weekly[["Date"]].drop_duplicates().sort_values("Date")
        segment_lookup = pd.merge_asof(
            segment_lookup, samples, left_on="Date", right_on="Sample_Date", direction="backward"
        )
        if "Segment_ID" in weekly.columns:
            # Segment of each sampling date, as assigned in data_ingestion
            segment_by_sample = (
                weekly.dropna(subset=["Segment_ID"]).groupby("Date")["Segment_ID"].first()
            )
        weekly = weekly.drop(columns=carried).merge(segment_lookup, on="Date", how="left")
        if "Days_since_prev" in carried:
            weekly["Days_since_prev"] = (weekly["Date"] - weekly["Sample_Date"]).dt.days.astype(float)
        if "Segment_ID" in carried:
            weekly["Segment_ID"] = (
                weekly["Date"].map(segment_by_sample).groupby(weekly["Depth_m"]).ffill().bfill()
            )
        weekly = weekly.drop(columns="Sample_Date")

    return weekly


def prepare_weekly_series(
    df: pd.DataFrame,
    use_reanalysis: bool = False,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return `df` resampled to one row per depth per week, gaps imputed, with Time_Idx
    and cyclical month encodings added.

    With `use_reanalysis`, weeks still missing after short-gap interpolation are filled
    from Copernicus reanalysis (1 m and 25 m, 1993+) and `<var>_source` columns mark
    which values came from the model (see src/copernicus_reanalysis.py)."""
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
    weekly = _add_calendar_columns(weekly, sample_dates=df.index)

    if use_reanalysis:
        from src import copernicus_reanalysis

        reanalysis = copernicus_reanalysis.load_or_fetch_reanalysis(allow_download=allow_download)
        if reanalysis is not None:
            weekly = copernicus_reanalysis.fill_gaps(weekly, reanalysis)
            filled = {
                c.removesuffix("_source"): int(weekly[c].sum())
                for c in weekly.columns
                if c.endswith("_source")
            }
            print(f"[Reanalysis] Weekly values filled from reanalysis: {filled}")

    unique_dates = weekly["Date"].sort_values().unique()
    time_idx_by_date = {date: idx for idx, date in enumerate(unique_dates)}
    weekly["Time_Idx"] = weekly["Date"].map(time_idx_by_date)
    weekly["month_sin"] = np.sin(2 * np.pi * weekly["Date"].dt.month / 12)
    weekly["month_cos"] = np.cos(2 * np.pi * weekly["Date"].dt.month / 12)

    return weekly.sort_values(["Date", "Depth_m"]).reset_index(drop=True)
