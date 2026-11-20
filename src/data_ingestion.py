
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Documentation" / "data"

OLD_OCEAN_COLUMNS = {
    "Date/Time": "Date",
    "Depth water [m]": "Depth_m",
    "Temp [°C]": "Temp_C",
    "Sal": "Salinity",
    "O2 [µmol/kg]": "O2_raw",
    "[NO3]- [µmol/l]": "NO3",
    "[NO2]- [µmol/l]": "NO2",
    "[PO4]3- [µmol/l]": "PO4",
    "SiO2 [µmol/l]": "Silicate",
    "Chl a [µg/l]": "Chl_a",
}

NEW_OCEAN_COLUMNS = {
    "Date/Time": "Date",
    "Depth water [m]": "Depth_m",
    "Temp [°C]": "Temp_C",
    "Sal": "Salinity",
    "O2 [µmol/l]": "O2_raw",
    "[NO3]- [µmol/l]": "NO3",
    "[NO2]- [µmol/l]": "NO2",
    "[PO4]3- [µmol/l]": "PO4",
    "Si(OH)4 [µmol/l]": "Silicate",
}

CHLOROPHYLL_COLUMNS = {
    "Date/Time": "Date",
    "Depth water [m]": "Depth_m",
    "Chl a [µg/l]": "Chl_a_supplement",
}


def _load_old_ocean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "BoknisEck_1957-2014.csv", sep=";", skiprows=31)
    df = df[list(OLD_OCEAN_COLUMNS.keys())].rename(columns=OLD_OCEAN_COLUMNS)
    df["O2_umol_L"] = df["O2_raw"] * 1.015
    return df.drop(columns="O2_raw")


def _load_new_ocean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "BoknisEck_2015-2023.csv", sep=";", skiprows=34)
    df = df[list(NEW_OCEAN_COLUMNS.keys())].rename(columns=NEW_OCEAN_COLUMNS)
    df["O2_umol_L"] = df["O2_raw"]
    return df.drop(columns="O2_raw")


def _load_chlorophyll_supplement() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "BoknisEck_chl_2015-2021.tab", sep="\t", skiprows=22)
    return df[list(CHLOROPHYLL_COLUMNS.keys())].rename(columns=CHLOROPHYLL_COLUMNS)


def _load_ocean_data() -> pd.DataFrame:
    ocean = pd.concat([_load_old_ocean_data(), _load_new_ocean_data()], ignore_index=True)
    ocean["Date"] = pd.to_datetime(ocean["Date"], format="ISO8601")

    chlorophyll = _load_chlorophyll_supplement()
    chlorophyll["Date"] = pd.to_datetime(chlorophyll["Date"], format="ISO8601")

    ocean = ocean.merge(chlorophyll, on=["Date", "Depth_m"], how="left")
    ocean["Chl_a"] = ocean["Chl_a"].fillna(ocean["Chl_a_supplement"])
    return ocean.drop(columns="Chl_a_supplement")


def load_and_clean_boknis_data(
    max_gap_days: int = 60,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    ocean = _load_ocean_data().sort_values("Date")
    ocean["Date"] = pd.to_datetime(ocean["Date"]).dt.as_unit("ns")
    combined = ocean

    # Copernicus reanalysis gap-filling happens on the weekly grid, per depth, in
    # pipeline.prepare_weekly_series(use_reanalysis=True) - raw sampling timestamps
    # never line up with the reanalysis dates, so it can't be merged here.
    if required_columns is not None:
        combined = combined.dropna(subset=required_columns)

    combined = combined.sort_values(["Date", "Depth_m"]).reset_index(drop=True)
    
    days_in_year = np.where(combined["Date"].dt.is_leap_year, 366, 365)
    day_of_year = combined["Date"].dt.dayofyear
    combined["Season_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
    combined["Season_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)

    combined["Years_since_start"] = (combined["Date"] - combined["Date"].min()).dt.days / 365.25

    dates = pd.Series(combined["Date"].unique()).sort_values().reset_index(drop=True)
    gap_days = dates.diff().dt.days
    lookup = pd.DataFrame(
        {
            "Date": dates,
            "Days_since_prev": gap_days.fillna(0),
            "Segment_ID": (gap_days > max_gap_days).cumsum(),
        }
    )
    combined = combined.merge(lookup, on="Date", how="left")

    return combined


def make_windows(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str], window: int = 6):
    X, y = [], []
    for _, g in df.groupby(["Segment_ID", "Depth_m"]):
        g = g.sort_values("Date")
        if len(g) <= window:
            continue
        feats = g[feature_cols].to_numpy()
        targs = g[target_cols].to_numpy()
        for i in range(len(g) - window):
            X.append(feats[i : i + window])
            y.append(targs[i + window])
    return np.array(X), np.array(y)
