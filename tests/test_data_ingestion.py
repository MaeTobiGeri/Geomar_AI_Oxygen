"""Tests for the raw-file loading and oxygen unit conversion in src/data_ingestion.py.

Uses tiny in-memory fixture files shaped like the real PANGAEA exports (same header-row
count, same column names) rather than the full multi-decade CSVs, so these run fast and
without touching the network. The weather fetch is not covered here - it's a live API call,
not something a unit test should exercise (Documentation/STYLE.md).
"""

import pandas as pd
import pytest

from src import data_ingestion


def _fixture_file(tmp_path, filename: str, header_row: str, data_row: str, skip_lines: int):
    padding = "\n".join(f"junk header line {i}" for i in range(skip_lines))
    (tmp_path / filename).write_text(f"{padding}\n{header_row}\n{data_row}\n")


def test_old_ocean_data_converts_oxygen_from_umol_per_kg(tmp_path, monkeypatch):
    _fixture_file(
        tmp_path,
        "BoknisEck_1957-2014.csv",
        header_row=(
            "Date/Time;Latitude;Longitude;Depth water [m];Cast;Sample label;Chl a [µg/l];"
            "[NO3]- [µmol/l];Flag (NO3);[NO2]- [µmol/l];Flag (NO2);O2 [µmol/kg];Flag (Oxygen);"
            "[PO4]3- [µmol/l];Flag (PO4);Sal;SiO2 [µmol/l];Flag (SiO2);Temp [°C]"
        ),
        data_row="1957-04-30T00:00:00;54.5295;10.0393;1;1;1;1.0;2.0;;0.1;;300.0;;0.5;;15.30;3.0;;7.70",
        skip_lines=31,
    )
    monkeypatch.setattr(data_ingestion, "DATA_DIR", tmp_path)

    df = data_ingestion._load_old_ocean_data()

    # 1957-2014 file reports oxygen in µmol/kg; SPEC.md §3 requires the 1.015 seawater
    # density conversion to bring it onto the same µmol/L scale as the 2015-2023 file.
    assert df["O2_umol_L"].iloc[0] == pytest.approx(300.0 * 1.015)
    assert "O2_raw" not in df.columns


def test_new_ocean_data_keeps_oxygen_already_in_umol_per_liter(tmp_path, monkeypatch):
    _fixture_file(
        tmp_path,
        "BoknisEck_2015-2023.csv",
        header_row=(
            "Date/Time;Latitude;Longitude;Depth water [m];Cast;Sample label;[NO3]- [µmol/l];"
            "Flag ((NO3));[NO2]- [µmol/l];Flag ((NO2));O2 [µmol/l];Flag ((Oxygen));"
            "[PO4]3- [µmol/l];Flag ((PO4));Sal;Flag ((Sal));Si(OH)4 [µmol/l];Flag ((SiO2));"
            "Temp [°C];Flag ((Temp))"
        ),
        data_row=(
            "2015-01-06T09:54:24;54.5295;10.0393;1;1;488;7.97;6;0.69;6;332.40;6;0.72;6;"
            "23.64;1;14.98;6;5.61;1"
        ),
        skip_lines=34,
    )
    monkeypatch.setattr(data_ingestion, "DATA_DIR", tmp_path)

    df = data_ingestion._load_new_ocean_data()

    assert df["O2_umol_L"].iloc[0] == pytest.approx(332.40)


def test_chlorophyll_supplement_fills_gaps_but_does_not_override(tmp_path, monkeypatch):
    _fixture_file(
        tmp_path,
        "BoknisEck_1957-2014.csv",
        header_row=(
            "Date/Time;Latitude;Longitude;Depth water [m];Cast;Sample label;Chl a [µg/l];"
            "[NO3]- [µmol/l];Flag (NO3);[NO2]- [µmol/l];Flag (NO2);O2 [µmol/kg];Flag (Oxygen);"
            "[PO4]3- [µmol/l];Flag (PO4);Sal;SiO2 [µmol/l];Flag (SiO2);Temp [°C]"
        ),
        data_row="2016-05-01T00:00:00;54.5295;10.0393;1;1;1;;2.0;;0.1;;300.0;;0.5;;15.30;3.0;;7.70",
        skip_lines=31,
    )
    _fixture_file(
        tmp_path,
        "BoknisEck_2015-2023.csv",
        header_row=(
            "Date/Time;Latitude;Longitude;Depth water [m];Cast;Sample label;[NO3]- [µmol/l];"
            "Flag ((NO3));[NO2]- [µmol/l];Flag ((NO2));O2 [µmol/l];Flag ((Oxygen));"
            "[PO4]3- [µmol/l];Flag ((PO4));Sal;Flag ((Sal));Si(OH)4 [µmol/l];Flag ((SiO2));"
            "Temp [°C];Flag ((Temp))"
        ),
        data_row=(
            "2016-06-01T00:00:00;54.5295;10.0393;1;1;488;7.97;6;0.69;6;332.40;6;0.72;6;"
            "23.64;1;14.98;6;5.61;1"
        ),
        skip_lines=34,
    )
    _fixture_file(
        tmp_path,
        "BoknisEck_chl_2015-2021.tab",
        header_row="Date/Time\tLatitude\tLongitude\tDepth water [m]\tCast\tSample label\tChl a [µg/l]",
        data_row="2016-05-01T00:00:00\t54.5295\t10.0393\t1\t1\t1\t2.5",
        skip_lines=22,
    )
    monkeypatch.setattr(data_ingestion, "DATA_DIR", tmp_path)

    df = data_ingestion._load_ocean_data()

    old_row = df[df["Date"] == "2016-05-01"].iloc[0]
    new_row = df[df["Date"] == "2016-06-01"].iloc[0]
    # Old-file row had no Chl_a of its own -> filled from the supplement.
    assert old_row["Chl_a"] == pytest.approx(2.5)
    # New-file row has no Chl_a column at all -> stays unset, not fabricated.
    assert pd.isna(new_row["Chl_a"])
