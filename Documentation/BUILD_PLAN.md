# Build Plan: Extreme Oxygen (Hypoxia) Prediction

Ordered, concrete implementation checklist. Follows [`SPEC.md`](SPEC.md) section by section — each
phase below references the spec section it implements. This is a from-scratch rewrite: nothing in
the project directory currently exists except `Documentation/` (raw data, reference material, and
this doc set) — see `Documentation/README.md` for current status.

**Every phase below is also governed by [`STYLE.md`](STYLE.md)** — human-readable, as simple as
possible, no unnecessary checks, no speculative code, one responsibility per file. It's not repeated
in each phase; assume it applies throughout.

Suggested project layout (not mandatory, but concrete enough to build against directly):

```
Geomar_AI_Oxygen/
├── requirements.txt
├── src/
│   ├── data_ingestion.py   # raw file loading, weather fetch + cache (SPEC §2-3)
│   ├── pipeline.py         # resampling & imputation (SPEC §4)
│   ├── labeling.py         # thresholds, imbalance audit, weight column (SPEC §6)
│   ├── features.py         # feature engineering + forward selection (SPEC §5)
│   ├── dataset.py          # TimeSeriesDataSet construction incl. weight= (SPEC §6.5)
│   └── model.py            # model build + loss config (SPEC §7)
├── train.py                 # training entrypoint (SPEC §8)
├── evaluate.py               # evaluation suite (SPEC §9)
├── app.py                    # dashboard (SPEC §10)
└── tests/
```

## Phase 1 — Environment & project setup

- [ ] Create `requirements.txt` (or `pyproject.toml`) up front — this was skipped last time (`SPEC.md` §11) and cost a from-memory dependency reconstruction. Start from `Documentation/reference/previous-implementation-pip-freeze.txt` as a known-working baseline (see `Documentation/ENVIRONMENT.md`), but re-resolve rather than pinning blindly — some of those versions may have moved on.
- [ ] Set up a virtual environment, install dependencies, confirm `import torch, pytorch_forecasting, lightning, streamlit, wetterdienst` all succeed.
- [ ] Confirm network access to the DWD API works (`wetterdienst`) — the pipeline has a hard live-network dependency (`SPEC.md` §2/§11).
- [ ] Add a `.gitignore` (venv, `__pycache__`, model checkpoints/logs directory) if this becomes a git repo — not currently one.

## Phase 2 — Data ingestion (`src/data_ingestion.py`)

Implements `SPEC.md` §2–3.

- [ ] Load the three raw files from `Documentation/data/` with the correct separator/header-skip/column-mapping per `SPEC.md` §3's schema table.
- [ ] Apply the oxygen unit conversion correctly (µmol/kg → µmol/L via `×1.015` for the 1957–2014 file only) — get this wrong and every downstream number is silently off.
- [ ] Concatenate the two ocean series; left-join in supplementary chlorophyll on `(Date, Depth_m)`, filling nulls only.
- [ ] Implement the DWD weather fetch (station 05930, hourly wind + air temp, historical+recent periods), resample to daily means, vectorize wind to `Wind_U`/`Wind_V`.
- [ ] **Add a local cache for the weather fetch** (e.g. a parquet/CSV dump keyed by date range) — flagged as a gap in the previous implementation (`SPEC.md` §11); historical DWD data doesn't change, so re-fetching it every run is wasted network dependency.
- [ ] Join weather onto the ocean series via `merge_asof` (nearest match, ≤3 day tolerance).
- [ ] Unit test: load a small fixture slice of each raw file format and assert the harmonized schema/columns/units come out correct, especially the oxygen conversion.

## Phase 3 — Cleaning & resampling (`src/pipeline.py`)

Implements `SPEC.md` §4.

- [ ] Coerce numeric columns, resample to weekly (`W-MON`) per depth independently, interpolate gaps ≤8 weeks, back/forward-fill remaining edges.
- [ ] Add `Time_Idx` and `month_sin`/`month_cos`.
- [ ] **Run the interpolation-vs-known-events check from `SPEC.md` §4**: once Phase 4's event list exists, verify how many known hypoxic episodes fall inside an interpolated gap; shrink the interpolation window for the target variable specifically if this looks like a problem.
- [ ] Unit test: construct a small synthetic series with a known gap and assert interpolation/fill behavior matches spec (limit respected, edges ffilled/bfilled correctly).

## Phase 4 — Labeling & target definition (`src/labeling.py`)

Implements `SPEC.md` §6.

- [ ] **Imbalance audit** (`SPEC.md` §6.2): compute and report what fraction of weekly 25 m `O2_umol_L` observations fall below each candidate threshold (80/60/30 µmol/L starting proposal), and their distribution over time. Do this before finalizing anything else in this phase.
- [ ] Cross-check candidate hypoxic periods against literature-documented events (`Documentation/OPEN_QUESTIONS.md` #3) to build the held-out event-study set used in Phase 8/9.
- [ ] Decide and implement the target transform (`SPEC.md` §6.3) — plot raw `O2_umol_L` vs. candidate "oxygen deficit" distributions, pick a transform only if the data actually supports needing one.
- [ ] Implement the sample-weight column (`SPEC.md` §6.4) — start with a coarse tiered scheme, expose the tier boundaries and weight values as easily-tunable constants (not hardcoded inline), since these are meant to be tuned empirically per §6.4/Phase 7 below.
- [ ] Output: a dataframe with the target, the deficit/transformed target, and a `sample_weight` column ready to hand to dataset construction.

## Phase 5 — Feature engineering (`src/features.py`)

Implements `SPEC.md` §5.

- [ ] Port the existing engineered features: `Surface_Temp_C`/`Surface_O2_umol_L` (1 m reindexed onto 25 m dates), `Vertical_Temp_Grad`, `Vertical_O2_Grad`, `O2_Derivative_1W`.
- [ ] Evaluate whether `O2_Derivative_1W` should be dropped per the target-history caution in `SPEC.md` §5 — implement it behind a flag so it's easy to ablate rather than deciding up front.
- [ ] Add the candidate features from `Research.txt` worth testing: raw wind direction / a mixing-energy proxy, a lagged/rolling chlorophyll-a feature.
- [ ] Implement the forward feature-selection procedure (`SPEC.md` §5) as a reusable utility — test each candidate individually against a validation tail-metric (from Phase 9), keep the best, add the next-best in combination, stop when a feature stops helping. Run it once real training (Phase 7) is working, and record which features were selected and why (a short note in this repo, not just tribal knowledge).

## Phase 6 — Dataset construction (`src/dataset.py`)

Implements `SPEC.md` §6.5.

- [ ] Build the `TimeSeriesDataSet` with the selected features (Phase 5) and the `sample_weight` column (Phase 4) passed via `weight="sample_weight"`.
- [ ] **Before trusting this**: re-verify the `weight=` mechanism against whichever `pytorch-forecasting` version actually gets installed (`SPEC.md` §6.5 confirms it against `1.7.0` specifically — check `TimeSeriesDataSet.__init__`'s signature and `MultiHorizonMetric.update()`'s handling of the unpacked weight if the version differs).
- [ ] Confirm chronological train/validation split (not the paper's random-block split — `SPEC.md` §8), plus the separate held-out event set from Phase 4.
- [ ] Sanity check: pull one batch from the resulting dataloader and confirm the weight tensor is present and has the expected higher values on known-hypoxic rows.

## Phase 7 — Model & loss (`src/model.py`)

Implements `SPEC.md` §7.

- [ ] Build the `TemporalFusionTransformer` with `QuantileLoss` and the starting hyperparameters in `SPEC.md` §7.
- [ ] Confirm the weighted loss is actually being applied — e.g. temporarily set all `sample_weight` values to a single large multiplier on hypoxic rows and confirm training loss/gradients respond, before doing a full training run.
- [ ] Once a baseline weighted model trains successfully, re-tune hyperparameters against the *weighted* objective (`SPEC.md` §7) — don't assume the previous implementation's values still apply.

## Phase 8 — Training script (`train.py`)

Implements `SPEC.md` §8.

- [ ] CLI/script entrypoint: run Phases 2–7 end to end, train with early stopping (`val_loss`, patience per `SPEC.md` §8), save a checkpoint to a **fixed/named path** rather than PyTorch Lightning's default incrementing `version_N/` scheme (`SPEC.md` §11's pitfall).
- [ ] Log which features were used (Phase 5 output) and the weight-tier configuration (Phase 4) alongside the checkpoint, so a saved model is reproducible without re-reading source code.

## Phase 9 — Evaluation suite (`evaluate.py`)

Implements `SPEC.md` §9.

- [ ] Weighted correlation/RMSE per split.
- [ ] Threshold-sweep classification metrics (accuracy/precision/recall/F1 across a sweep, plus ROC-AUC) evaluating the regression output as a classifier — don't train a separate classifier.
- [ ] Event-study plots: observation vs. new weighted model vs. an unweighted baseline, for each event in Phase 4's held-out set.
- [ ] Persistence-baseline comparison, specifically on tail/event metrics, not just aggregate error.
- [ ] Compare results against the calibration numbers in `SPEC.md` §9 (F1 ≈ 0.2–0.4, AUC ≈ 0.7–0.85 as a plausible "working" outcome) — this is a sanity check on expectations, not a hard pass/fail gate.
- [ ] Empirically establish the reliable forecast horizon and record it — this becomes the app's horizon control bound (Phase 10).

## Phase 10 — Dashboard (`app.py`)

Implements `SPEC.md` §10.

- [ ] Load the fixed-path checkpoint from Phase 8 (not "most recent by file timestamp" — the pitfall in `SPEC.md` §10/§11).
- [ ] Threshold reference line + shaded hypoxic zone on the forecast chart, using Phase 4's tiers.
- [ ] Risk readout (e.g. "X% probability of hypoxic conditions by [date]") derived from the P10/P50/P90 quantile spread crossing the threshold.
- [ ] Forecast horizon control bounded to Phase 9's empirically-established reliable horizon, not an arbitrary large range.
- [ ] Manual smoke test in a browser: run the golden path (load → pick a date → view forecast) and at least one edge case (a date very early/late in the series) before considering this done.

## Phase 11 — Tests

- [ ] Unit tests for Phases 2–5 (data correctness, especially the oxygen unit conversion and the imputation-limit behavior — these are the two easiest places to introduce a silent, hard-to-notice bug).
- [ ] A smoke test equivalent to the previous implementation's `test_streamlit.py`/`test_forecast.py` pattern: exercise the app end-to-end and assert it doesn't crash.
- [ ] At least one regression test asserting the weighted loss actually changes model behavior vs. an unweighted baseline on a small fixture dataset (guards against silently regressing back to the previous implementation's failure mode).

## Phase 12 — Documentation upkeep

- [ ] As real decisions get made (final thresholds, weight values, selected features, chosen target transform, established reliable horizon), update `SPEC.md` in place rather than leaving it as "proposed starting points" — the point of this rewrite is that the docs stay the source of truth.
- [ ] Update `Documentation/README.md`'s status line once code exists, and add a short "how to run" section once `train.py`/`app.py` work end to end.
