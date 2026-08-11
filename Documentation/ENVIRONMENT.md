# Environment

No environment currently exists in the project directory — the previous `.venv` was removed as
part of the documentation-only cleanup (see `Documentation/README.md`). This records what
previously worked, as a starting point for the rewrite (`BUILD_PLAN.md` Phase 1).

## Python version

`3.14.5` (previous implementation).

## Known-working dependency versions

Full previous `pip freeze` output is preserved at
[`reference/previous-implementation-pip-freeze.txt`](reference/previous-implementation-pip-freeze.txt).
The load-bearing packages, pinned to what was confirmed working previously:

| Package | Version | Role |
|---|---|---|
| `torch` | 2.12.0 | model backend |
| `pytorch-forecasting` | 1.7.0 | `TimeSeriesDataSet`, `TemporalFusionTransformer`, `QuantileLoss` — **the weighted-loss mechanism in `SPEC.md` §6.5 was verified against this exact version**; re-check if upgrading |
| `pytorch-lightning` / `lightning` | 2.6.5 | training loop |
| `pandas` | 3.0.3 | data wrangling |
| `numpy` | 2.4.6 | |
| `plotly` | 6.8.0 | dashboard charts |
| `matplotlib` | 3.10.9 | evaluation plots |
| `streamlit` | 1.58.0 | dashboard framework |
| `wetterdienst` | 0.121.1 | DWD weather API client |

## Build-plan action item

`BUILD_PLAN.md` Phase 1: create a `requirements.txt` (or `pyproject.toml`) from the start this time
— the previous implementation never had one, which meant dependency versions had to be
reconstructed from an installed `.venv` after the fact rather than being documented up front. Use
the table above as a starting point, but re-resolve rather than pinning blindly, since some
versions may have moved on since the previous build.
