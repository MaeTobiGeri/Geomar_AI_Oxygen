"""Load and harmonize the raw Boknis Eck ocean data and DWD weather data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from wetterdienst.provider.dwd.observation import DwdObservationRequest

DATA_DIR = Path(__file__).resolve().parent.parent / "Documentation" / "data"
WEATHER_CACHE_PATH = Path(__file__).resolve().parent.parent / ".weather_cache" / "schoenhagen_daily.csv"
DWD_STATION_ID = "05930"

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
    # This file reports oxygen in µmol/kg; the 2015-2023 file reports µmol/L directly.
    # Seawater density ~1.015 kg/L converts the two onto the same unit (SPEC.md §3).
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


def _fetch_schoenhagen_weather() -> pd.DataFrame:
    # DWD's "recent" period is a rolling window, so a cached fetch can go stale for the
    # most recent ~500 days. Good enough for this project: delete the cache file to refresh.
    if WEATHER_CACHE_PATH.exists():
        return pd.read_csv(WEATHER_CACHE_PATH, parse_dates=["Date"])

    request = DwdObservationRequest(
        parameters=[
            "hourly/wind/wind_speed",
            "hourly/wind/wind_direction",
            "hourly/temperature_air/temperature_air_mean_2m",
        ],
        periods=["historical", "recent"],
    ).filter_by_station_id(DWD_STATION_ID)

    raw = request.values.all().df.to_pandas()
    hourly = raw.pivot_table(index="date", columns="parameter", values="value", observed=True).reset_index()
    hourly = hourly.rename(
        columns={
            "date": "Date",
            "wind_speed": "Wind_Speed_ms",
            "wind_direction": "Wind_Dir_deg",
            "temperature_air_mean_2m": "Air_Temp_C",
        }
    )
    hourly["Date"] = pd.to_datetime(hourly["Date"]).dt.tz_localize(None)

    daily = hourly.set_index("Date").resample("D").mean().reset_index()

    # Speed + direction is unusable to a model as-is (direction wraps at 360°); east/north
    # wind components let the model see it as an ordinary continuous quantity.
    wind_dir_rad = np.radians(daily["Wind_Dir_deg"])
    daily["Wind_U"] = -daily["Wind_Speed_ms"] * np.sin(wind_dir_rad)
    daily["Wind_V"] = -daily["Wind_Speed_ms"] * np.cos(wind_dir_rad)

    daily = daily[["Date", "Air_Temp_C", "Wind_Speed_ms", "Wind_Dir_deg", "Wind_U", "Wind_V"]]

    WEATHER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(WEATHER_CACHE_PATH, index=False)
    return daily


def load_and_clean_boknis_data() -> pd.DataFrame:
    ocean = _load_ocean_data().sort_values("Date")
    weather = _fetch_schoenhagen_weather().sort_values("Date")

    # Ensure both Date columns have the same datetime dtype (fix for pandas version differences)
    ocean["Date"] = pd.to_datetime(ocean["Date"]).dt.as_unit("ns")
    weather["Date"] = pd.to_datetime(weather["Date"]).dt.as_unit("ns")

    combined = pd.merge_asof(ocean, weather, on="Date", direction="nearest", tolerance=pd.Timedelta("3 days"))
    return combined.sort_values(["Date", "Depth_m"]).reset_index(drop=True)
