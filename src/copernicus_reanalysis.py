"""Fetch and process Copernicus Marine reanalysis data for gap-filling.

Uses BALTICSEA_MULTIYEAR_BGC_003_012 (Baltic Sea Biogeochemical Reanalysis)
to fill gaps in observed data at Boknis Eck station.

Dataset: cmems_mod_bal_bgc_my_P1M-m (monthly means)
Coverage: 1993-present
Variables: O2, Chl_a, NO3, PO4, Temp, Salinity
Resolution: ~1 nautical mile, 56 depth levels

Requires: copernicusmarine package
Install: pip install copernicusmarine
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Boknis Eck station coordinates
BOKNIS_LAT = 54.5295
BOKNIS_LON = 10.0393
BOKNIS_DEPTH = 25.0  # meters

# Copernicus dataset info
DATASET_ID = "cmems_mod_bal_bgc_my_P1M-m"
PRODUCT_ID = "BALTICSEA_MULTIYEAR_BGC_003_012"

# Cache directory for downloaded reanalysis data
CACHE_DIR = Path(__file__).resolve().parent.parent / ".copernicus_cache"

# Variable mappings: Copernicus name -> Our name
VARIABLE_MAPPINGS = {
    "o2": "O2_umol_L_reanalysis",
    "chl": "Chl_a_reanalysis",
    "no3": "NO3_reanalysis",
    "po4": "PO4_reanalysis",
    "thetao": "Temp_C_reanalysis",
    "so": "Salinity_reanalysis",
}


def fetch_copernicus_data(
    start_date: str = "1993-01-01",
    end_date: str = "2023-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch Copernicus reanalysis data for Boknis Eck at 25m depth.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        force_refresh: If True, re-download even if cache exists

    Returns:
        DataFrame with columns: Date, O2_umol_L_reanalysis, Chl_a_reanalysis,
                                NO3_reanalysis, PO4_reanalysis, Temp_C_reanalysis,
                                Salinity_reanalysis

    Notes:
        - Requires Copernicus Marine credentials (set via copernicusmarine login)
        - Data is cached locally to avoid repeated downloads
        - Monthly mean values are interpolated to weekly resolution
    """
    cache_file = CACHE_DIR / f"boknis_eck_reanalysis_{start_date}_{end_date}.csv"

    # Return cached data if available
    if cache_file.exists() and not force_refresh:
        print(f"Loading cached reanalysis data from: {cache_file}")
        return pd.read_csv(cache_file, parse_dates=["Date"])

    # Import copernicusmarine (only when actually fetching)
    try:
        import copernicusmarine
    except ImportError:
        raise ImportError(
            "copernicusmarine package not installed. "
            "Install with: pip install copernicusmarine"
        )

    print(f"Fetching Copernicus reanalysis data for {start_date} to {end_date}...")
    print(f"Location: Boknis Eck ({BOKNIS_LAT}°N, {BOKNIS_LON}°E) at {BOKNIS_DEPTH}m depth")

    # Fetch data using copernicusmarine API
    # Note: This requires valid Copernicus Marine credentials
    # Set up credentials: copernicusmarine login
    try:
        # Subset arguments for the API
        subset_args = {
            "dataset_id": DATASET_ID,
            "variables": list(VARIABLE_MAPPINGS.keys()),
            "minimum_longitude": BOKNIS_LON - 0.05,
            "maximum_longitude": BOKNIS_LON + 0.05,
            "minimum_latitude": BOKNIS_LAT - 0.05,
            "maximum_latitude": BOKNIS_LAT + 0.05,
            "minimum_depth": BOKNIS_DEPTH - 5,
            "maximum_depth": BOKNIS_DEPTH + 5,
            "start_datetime": start_date,
            "end_datetime": end_date,
        }

        # Download to temporary NetCDF file
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp_nc = CACHE_DIR / "temp_reanalysis.nc"

        copernicusmarine.subset(
            **subset_args,
            output_filename=str(temp_nc),
            output_directory=str(CACHE_DIR),
            force_download=True,
        )

        # Load NetCDF and convert to DataFrame
        import xarray as xr

        ds = xr.open_dataset(temp_nc)

        # Extract data at nearest point to Boknis Eck
        # Select nearest lat/lon/depth
        ds_boknis = ds.sel(
            latitude=BOKNIS_LAT,
            longitude=BOKNIS_LON,
            depth=BOKNIS_DEPTH,
            method="nearest",
        )

        # Convert to DataFrame
        df_list = []
        for copernicus_var, our_var in VARIABLE_MAPPINGS.items():
            if copernicus_var in ds_boknis:
                data = ds_boknis[copernicus_var].to_dataframe().reset_index()
                data = data.rename(columns={copernicus_var: our_var, "time": "Date"})
                data = data[["Date", our_var]]
                df_list.append(data)

        # Merge all variables
        df = df_list[0]
        for df_var in df_list[1:]:
            df = df.merge(df_var, on="Date", how="outer")

        # Clean up temp file
        temp_nc.unlink()

        # Convert Date to datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Save to cache
        df.to_csv(cache_file, index=False)
        print(f"Cached reanalysis data to: {cache_file}")

        return df

    except Exception as e:
        print(f"Error fetching Copernicus data: {e}")
        print("\nTroubleshooting:")
        print("1. Install: pip install copernicusmarine")
        print("2. Login: copernicusmarine login")
        print("3. Ensure valid Copernicus Marine credentials")
        raise


def interpolate_monthly_to_weekly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """Interpolate monthly reanalysis data to weekly resolution.

    Args:
        df_monthly: DataFrame with monthly data (Date column)

    Returns:
        DataFrame with weekly data (Monday-aligned)
    """
    # Set Date as index
    df_monthly = df_monthly.set_index("Date").sort_index()

    # Create weekly date range
    weekly_dates = pd.date_range(
        start=df_monthly.index.min(),
        end=df_monthly.index.max(),
        freq="W-MON",
    )

    # Resample to weekly using linear interpolation
    df_weekly = df_monthly.reindex(
        df_monthly.index.union(weekly_dates)
    ).interpolate(method="time").loc[weekly_dates]

    return df_weekly.reset_index().rename(columns={"index": "Date"})


def merge_with_observations(
    df_observed: pd.DataFrame,
    df_reanalysis: pd.DataFrame,
    variables_to_fill: list[str] = None,
) -> pd.DataFrame:
    """Merge observed data with reanalysis, using reanalysis to fill gaps.

    Args:
        df_observed: Observed data with Date column
        df_reanalysis: Reanalysis data with Date column and *_reanalysis columns
        variables_to_fill: List of variable names to fill (without _reanalysis suffix)
                          Default: ['O2_umol_L', 'Chl_a', 'NO3', 'PO4', 'Temp_C', 'Salinity']

    Returns:
        DataFrame with filled values and data_source columns indicating source of each value
    """
    if variables_to_fill is None:
        variables_to_fill = ["O2_umol_L", "Chl_a", "NO3", "PO4", "Temp_C", "Salinity"]

    df = df_observed.copy()

    # Merge reanalysis data
    df = df.merge(df_reanalysis, on="Date", how="left")

    # For each variable, fill NaNs with reanalysis and track source
    for var in variables_to_fill:
        reanalysis_var = f"{var}_reanalysis"

        if var in df.columns and reanalysis_var in df.columns:
            # Create source tracking column (0 = observed, 1 = reanalysis)
            source_col = f"{var}_source"
            df[source_col] = 0.0  # Default: observed

            # Identify where we're filling with reanalysis
            needs_fill = df[var].isna() & df[reanalysis_var].notna()

            # Fill gaps
            df.loc[needs_fill, var] = df.loc[needs_fill, reanalysis_var]
            df.loc[needs_fill, source_col] = 1.0  # Mark as reanalysis

            # Drop the reanalysis column (we've merged it)
            df = df.drop(columns=[reanalysis_var])

    return df


def load_or_fetch_reanalysis(
    start_date: str = "1993-01-01",
    end_date: str = "2023-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load reanalysis data from cache or fetch if not available.

    Convenience wrapper around fetch_copernicus_data() that handles caching
    and interpolation to weekly resolution.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        force_refresh: If True, re-download even if cache exists

    Returns:
        DataFrame with weekly reanalysis data
    """
    # Fetch monthly data
    df_monthly = fetch_copernicus_data(start_date, end_date, force_refresh)

    # Interpolate to weekly
    df_weekly = interpolate_monthly_to_weekly(df_monthly)

    return df_weekly


# Example usage
if __name__ == "__main__":
    # Fetch and display sample data
    print("Fetching Copernicus reanalysis data...")
    df = load_or_fetch_reanalysis()
    print(f"\nLoaded {len(df)} weeks of reanalysis data")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nLast few rows:\n{df.tail()}")
    print(f"\nMissing values:\n{df.isna().sum()}")
