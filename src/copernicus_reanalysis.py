"""Fetch and process Copernicus Marine reanalysis data for gap-filling.

The variables we fill come from TWO different Copernicus products:

- BALTICSEA_MULTIYEAR_BGC_003_012 (biogeochemistry), dataset cmems_mod_bal_bgc_my_P1M-m
  -> o2, chl, no3, po4
- BALTICSEA_MULTIYEAR_PHY_003_011 (physics), dataset cmems_mod_bal_phy_my_P1M-m
  -> thetao (temperature), so (salinity)

Temperature and salinity are NOT in the BGC dataset; asking it for `thetao` fails with
"The variable 'thetao' is neither a variable or a standard name in the dataset".

Data is extracted at the nearest valid model cell to Boknis Eck for the depths the
pipeline uses (1 m surface features, 25 m target), shifted to mid-month, interpolated
to the pipeline's weekly (W-MON) grid and cached as CSV. Once the cache exists no
credentials are needed.

Cache location: `.copernicus_cache/` in the repo, or the directory in the
COPERNICUS_CACHE_DIR environment variable (point it at Google Drive on Colab so the
download survives session resets).

Credentials: run `copernicusmarine login` once (stores ~/.copernicusmarine), or set
COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD.

Requires: copernicusmarine, xarray
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Boknis Eck station coordinates
BOKNIS_LAT = 54.5295
BOKNIS_LON = 10.0393
TARGET_DEPTHS = (1.0, 25.0)  # surface features + 25 m target series
BOX_HALF_WIDTH_DEG = 0.15  # search box for the nearest valid (sea, deep enough) cell
# The model's smoothed bathymetry is shallower than the real ~28 m station, so the level
# nearest 25 m is often dry near the coast. Accept the closest wet level within this
# tolerance (near-bottom water, which is what matters for hypoxia).
MAX_DEPTH_MISMATCH_M = 6.0

# Copernicus dataset id -> {Copernicus variable: our column name}
DATASETS = {
    "cmems_mod_bal_bgc_my_P1M-m": {
        "o2": "O2_umol_L",  # mmol m-3 == µmol/L
        "chl": "Chl_a",  # mg m-3 == µg/L
        "no3": "NO3",  # mmol m-3 == µmol/L
        "po4": "PO4",  # mmol m-3 == µmol/L
    },
    "cmems_mod_bal_phy_my_P1M-m": {
        "thetao": "Temp_C",  # °C
        "so": "Salinity",  # PSU
    },
}

REANALYSIS_VARIABLES = [col for mapping in DATASETS.values() for col in mapping.values()]

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".copernicus_cache"
CACHE_VERSION = "v3"  # v2 caches lack the 25 m series

# In-process memo so repeated pipeline calls (e.g. every Optuna trial) neither re-read
# the cache nor re-prompt for credentials after a failed download.
_MEMO: dict[tuple, pd.DataFrame | None] = {}


def cache_dir() -> Path:
    return Path(os.environ.get("COPERNICUS_CACHE_DIR", _DEFAULT_CACHE_DIR))


def _cache_file(start_date: str, end_date: str) -> Path:
    return cache_dir() / f"boknis_eck_reanalysis_{CACHE_VERSION}_{start_date}_{end_date}.csv"


def _nearest_valid_series(da, depth: float) -> tuple[pd.Series, str] | None:
    """Time series of `da` (time, depth, latitude, longitude) at the grid cell nearest
    Boknis Eck that has data at a model level within MAX_DEPTH_MISMATCH_M of `depth`,
    using that cell's wet level closest to `depth`. Returns (series, description of the
    chosen cell) or None if no cell qualifies."""
    levels = da["depth"].values
    wet = da.notnull().any("time").transpose("depth", "latitude", "longitude").values
    lat2d, lon2d = np.meshgrid(da["latitude"].values, da["longitude"].values, indexing="ij")
    dist = (lat2d - BOKNIS_LAT) ** 2 + ((lon2d - BOKNIS_LON) * np.cos(np.radians(BOKNIS_LAT))) ** 2

    best = None  # (distance, i, j, level index)
    for i, j in zip(*np.nonzero(wet.any(axis=0))):
        wet_levels = np.nonzero(wet[:, i, j])[0]
        k = wet_levels[np.argmin(np.abs(levels[wet_levels] - depth))]
        if abs(levels[k] - depth) <= MAX_DEPTH_MISMATCH_M and (best is None or dist[i, j] < best[0]):
            best = (dist[i, j], i, j, k)

    if best is None:
        return None
    _, i, j, k = best
    point = da.isel(latitude=i, longitude=j, depth=k)
    km = np.sqrt(best[0]) * 111.2
    info = f"cell {lat2d[i, j]:.3f}N {lon2d[i, j]:.3f}E ({km:.1f} km away), model level {levels[k]:.1f} m"
    return pd.Series(point.values, index=pd.to_datetime(point["time"].values)), info


def _deepest_wet_level(da) -> float:
    wet = da.notnull().any(["time", "latitude", "longitude"]).values
    return float(da["depth"].values[wet].max()) if wet.any() else float("nan")


def _subset_dataset(copernicusmarine, dataset_id: str, variables: list[str], start_date: str, end_date: str) -> Path:
    out_dir = cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"tmp_{dataset_id}.nc"
    if out_file.exists():
        out_file.unlink()

    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=BOKNIS_LON - BOX_HALF_WIDTH_DEG,
        maximum_longitude=BOKNIS_LON + BOX_HALF_WIDTH_DEG,
        minimum_latitude=BOKNIS_LAT - BOX_HALF_WIDTH_DEG,
        maximum_latitude=BOKNIS_LAT + BOX_HALF_WIDTH_DEG,
        minimum_depth=0.5,
        maximum_depth=max(TARGET_DEPTHS) + 5,
        start_datetime=start_date,
        end_datetime=end_date,
        output_filename=out_file.name,
        output_directory=str(out_dir),
    )
    return out_file


def _fetch_dataset(copernicusmarine, dataset_id: str, mapping: dict[str, str], start_date: str, end_date: str) -> pd.DataFrame | None:
    """Monthly long-format frame (Date, Depth_m, <our columns>) for one dataset, or None.
    Falls back to one request per variable if the combined request is rejected, so one
    unavailable variable doesn't lose the others."""
    import xarray as xr

    variable_groups = [list(mapping)]
    frames = []
    while variable_groups:
        variables = variable_groups.pop(0)
        try:
            nc_file = _subset_dataset(copernicusmarine, dataset_id, variables, start_date, end_date)
        except Exception as e:
            if len(variables) > 1:
                print(f"  {dataset_id}: combined request failed ({e}); retrying per variable")
                variable_groups.extend([v] for v in variables)
            else:
                print(f"  {dataset_id}: skipping '{variables[0]}' ({e})")
            continue

        with xr.open_dataset(nc_file) as ds:
            for var in variables:
                if var not in ds:
                    print(f"  {dataset_id}: '{var}' missing from download, skipping")
                    continue
                for depth in TARGET_DEPTHS:
                    da = ds[var].load()
                    found = _nearest_valid_series(da, depth)
                    if found is None:
                        print(
                            f"  {dataset_id}: no '{var}' cell within {MAX_DEPTH_MISMATCH_M} m of "
                            f"{depth} m (deepest wet level in box: {_deepest_wet_level(da):.1f} m)"
                        )
                        continue
                    series, info = found
                    print(f"  {dataset_id}: '{var}' @ {depth} m -> {info}")
                    frames.append(
                        pd.DataFrame({"Date": series.index, "Depth_m": depth, mapping[var]: series.values})
                    )
        nc_file.unlink(missing_ok=True)

    if not frames:
        return None
    df = frames[0]
    for frame in frames[1:]:
        df = df.merge(frame, on=["Date", "Depth_m"], how="outer")
    return df


def fetch_copernicus_data(start_date: str = "1993-01-01", end_date: str = "2023-12-31") -> pd.DataFrame:
    """Download monthly reanalysis for Boknis Eck at TARGET_DEPTHS from both datasets.

    Returns a long-format monthly frame: Date, Depth_m, O2_umol_L, Chl_a, NO3, PO4,
    Temp_C, Salinity (any variable that couldn't be fetched is absent)."""
    try:
        import copernicusmarine
    except ImportError as e:
        raise ImportError("copernicusmarine not installed. Install with: pip install copernicusmarine") from e

    print(f"Fetching Copernicus reanalysis for {start_date} to {end_date}...")
    print(f"Location: Boknis Eck ({BOKNIS_LAT}°N, {BOKNIS_LON}°E), depths {TARGET_DEPTHS} m")

    df = None
    for dataset_id, mapping in DATASETS.items():
        part = _fetch_dataset(copernicusmarine, dataset_id, mapping, start_date, end_date)
        if part is None:
            continue
        df = part if df is None else df.merge(part, on=["Date", "Depth_m"], how="outer")

    if df is None:
        raise RuntimeError("No reanalysis variables could be fetched (check credentials / dataset ids)")
    return df.sort_values(["Depth_m", "Date"]).reset_index(drop=True)


def interpolate_monthly_to_weekly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """Monthly means (stamped at month start) -> weekly W-MON values per depth.

    Each monthly mean is placed mid-month before linear time interpolation, so a
    month's value is centred on the month it represents."""
    weekly_frames = []
    for depth, group in df_monthly.groupby("Depth_m"):
        group = group.drop(columns="Depth_m").set_index("Date").sort_index()
        group.index = group.index + pd.Timedelta(days=14)
        weekly_dates = pd.date_range(group.index.min(), group.index.max(), freq="W-MON")
        weekly = (
            group.reindex(group.index.union(weekly_dates))
            .interpolate(method="time", limit_area="inside")
            .loc[weekly_dates]
        )
        weekly = weekly.rename_axis("Date").reset_index()
        weekly["Depth_m"] = depth
        weekly_frames.append(weekly)
    return pd.concat(weekly_frames, ignore_index=True)


def load_or_fetch_reanalysis(
    start_date: str = "1993-01-01",
    end_date: str = "2023-12-31",
    force_refresh: bool = False,
    allow_download: bool = True,
) -> pd.DataFrame | None:
    """Weekly reanalysis (Date, Depth_m, variables...) from cache, downloading if needed.

    Never raises: returns None (with a printed reason) if the data can't be obtained,
    and remembers that for the rest of the process so it won't prompt again."""
    key = (start_date, end_date)
    if key in _MEMO and not force_refresh:
        return _MEMO[key]

    cache_file = _cache_file(start_date, end_date)
    result = None
    if cache_file.exists() and not force_refresh:
        print(f"[Reanalysis] Loading cached data: {cache_file}")
        result = pd.read_csv(cache_file, parse_dates=["Date"])
        _describe(result)
    elif not allow_download:
        print(f"[Reanalysis] No cache at {cache_file} and downloads disabled; skipping gap-filling")
    else:
        try:
            weekly = interpolate_monthly_to_weekly(fetch_copernicus_data(start_date, end_date))
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            weekly.to_csv(cache_file, index=False)
            print(f"[Reanalysis] Cached {len(weekly)} weekly rows to: {cache_file}")
            _describe(weekly)
            result = weekly
        except Exception as e:
            print(f"[Reanalysis] Could not fetch data: {e}")
            print("[Reanalysis] Run `copernicusmarine login` once, then retry. Continuing without gap-filling.")

    _MEMO[key] = result
    return result


def _describe(df: pd.DataFrame) -> None:
    variables = [v for v in REANALYSIS_VARIABLES if v in df.columns]
    print(f"[Reanalysis] {len(df)} rows, columns: {df.columns.tolist()}")
    if "Depth_m" not in df.columns or not variables:
        print("[Reanalysis] WARNING: cache has no Depth_m/variable columns - delete it and re-download")
        return
    for depth, group in df.groupby("Depth_m"):
        counts = {v: int(group[v].notna().sum()) for v in variables}
        print(f"[Reanalysis]   {depth} m: {counts}")
    missing = [d for d in TARGET_DEPTHS if d not in set(df["Depth_m"])]
    if missing:
        print(f"[Reanalysis] WARNING: no data for depth(s) {missing} m")


def fill_gaps(weekly: pd.DataFrame, reanalysis: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs in the weekly per-depth series with reanalysis values on matching
    (Date, Depth_m) rows, adding `<var>_source` columns (0 = observed/interpolated,
    1 = reanalysis) for every reanalysis variable."""
    variables = [v for v in REANALYSIS_VARIABLES if v in reanalysis.columns and v in weekly.columns]
    if not variables:
        print(f"[Reanalysis] WARNING: no fillable variables in reanalysis columns {reanalysis.columns.tolist()}")
        return weekly
    rean = reanalysis[["Date", "Depth_m", *variables]].rename(columns={v: f"{v}_reanalysis" for v in variables})
    rean["Depth_m"] = rean["Depth_m"].astype(weekly["Depth_m"].dtype)

    df = weekly.merge(rean, on=["Date", "Depth_m"], how="left")
    for var in variables:
        filler = f"{var}_reanalysis"
        needs_fill = df[var].isna() & df[filler].notna()
        df[f"{var}_source"] = needs_fill.astype(float)
        df.loc[needs_fill, var] = df.loc[needs_fill, filler]
    return df.drop(columns=[f"{v}_reanalysis" for v in variables])


if __name__ == "__main__":
    df = load_or_fetch_reanalysis()
    if df is not None:
        print(f"\nLoaded {len(df)} weekly rows, {df['Date'].min()} to {df['Date'].max()}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nMissing values:\n{df.isna().sum()}")
