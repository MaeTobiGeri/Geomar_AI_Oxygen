# Technical Specification: Extreme Oxygen (Hypoxia) Prediction

Self-contained build spec. Read [`CONCEPT.md`](CONCEPT.md) first for the *why*; this document is
the *what* — detailed enough that the project can be implemented from it without needing to see any
previous code (there isn't any — this is a from-scratch rewrite, see [`BUILD_PLAN.md`](BUILD_PLAN.md)
for the *how*/order).

Numbers and defaults below are **proposed starting points**, following the reference paper's own
practice of picking reasonable defaults and tuning them empirically once real data is in front of
you (`CONCEPT.md` §2.2) — not fixed requirements. Where a decision should be validated against data
before being trusted, that's called out explicitly.

## 1. Overview & scope

**Goal**: given oceanographic and weather driver variables at the Boknis Eck station, predict the
risk of dissolved oxygen (`O2`) at 25 m depth reaching dangerous-for-fish (hypoxic) levels in the
near future — as opposed to the previous implementation's goal of forecasting the general oxygen
value at any horizon.

**In scope**: data ingestion/cleaning pipeline, feature engineering, a weighted-loss model that
predicts oxygen (or oxygen deficit) with emphasis on the low tail, threshold-based evaluation, and a
dashboard that surfaces hypoxia risk.

**Out of scope for the first build** (candidates for later iteration, not blockers):
multi-station/multi-depth modeling, sub-weekly temporal resolution, a from-scratch classification
model (see `CONCEPT.md` §2.4 — derive alerts from the regression output instead), real-time
production deployment.

## 2. Data sources

| Source | Location | Format | Notes |
|---|---|---|---|
| Historical ocean series (1957–2014) | `Documentation/data/BoknisEck_1957-2014.csv` | `;`-separated, 31 header rows to skip | PANGAEA export |
| Recent ocean series (2015–2023) | `Documentation/data/BoknisEck_2015-2023.csv` | `;`-separated, 34 header rows to skip | PANGAEA export |
| Supplementary chlorophyll (2015–2021) | `Documentation/data/BoknisEck_chl_2015-2021.tab` | tab-separated, 22 header rows to skip | PANGAEA export |
| Weather | live pull, DWD station **05930 (Schönhagen)** via [`wetterdienst`](https://pypi.org/project/wetterdienst/) | API | Hourly wind speed, wind direction, 2 m air temperature. 

Reference material (not raw data, but informs decisions below): `Documentation/reference/Research.txt`
(literature review — feature rationale, station-specific physics, hypoxia threshold) and
`Documentation/reference/Chu-2025-IRANNA-SpaceWeather.pdf` (methodology source, see `CONCEPT.md`).

See `Documentation/OPEN_QUESTIONS.md` for data the user may be able to provide that would improve
on this list (higher-frequency oxygen data, a better advection/mixing proxy, a documented event
list, additional depths/stations, more recent nutrient/chlorophyll extracts).

## 3. Data cleaning & schema harmonization

Both ocean CSVs and the chlorophyll `.tab` file use different column names for the same
quantities; harmonize to this common schema:

| Canonical column | Meaning | Notes |
|---|---|---|
| `Date` | timestamp | parse as ISO8601 |
| `Depth_m` | sampling depth (m) | |
| `Temp_C` | water temperature | |
| `Salinity` | | |
| `O2_raw` → `O2_umol_L` | dissolved oxygen | **unit mismatch between files**: the 1957–2014 file reports µmol/**kg**, converted via `× 1.015`; the 2015–2023 file already reports µmol/**L** directly. Get this conversion right — it's a silent correctness bug if skipped. |
| `NO3`, `NO2`, `PO4`, `Silicate` | nutrients (µmol/L) | column names for these vary slightly between the two ocean files (e.g. `SiO2` vs `Si(OH)4` for silicate) — map both to `Silicate` |
| `Chl_a` | chlorophyll-a (µg/L) | present in both the old ocean file and the supplementary chlorophyll file; when merging, prefer the ocean file's value and fill gaps from the supplementary file (`Date`+`Depth_m` join) |

**Procedure**: load each file skipping its header block → rename to canonical schema → apply the
oxygen unit conversion → concatenate the old and new ocean series → merge in supplementary
chlorophyll (left join on `Date`+`Depth_m`, fill nulls only) → sort by `Date`, then `Depth_m`.

### Weather ingestion

Pull hourly wind speed, wind direction, and 2 m air temperature for DWD station 05930
(Schönhagen), for both `historical` and `recent` periods. Resample to daily means. Vectorize wind
into east/north components rather than keeping raw speed+direction — a network can't use a
360°-wrapping angle directly:

```
Wind_U = -Wind_Speed_ms * sin(radians(Wind_Dir_deg))
Wind_V = -Wind_Speed_ms * cos(radians(Wind_Dir_deg))
```

Join onto the ocean series with a nearest-match, tolerance-bounded merge (`merge_asof`, ≤3 day
tolerance) — the ocean data isn't sampled at a fixed daily cadence, so an exact-date join would drop
most rows.

## 4. Resampling & imputation

Ocean sampling is irregular (ship-based visits, not a fixed sensor cadence). Per depth level:

1. Force all numeric columns to numeric type (`errors='coerce'`) to clear string artifacts from the raw export.
2. Resample to a fixed **weekly grid** (Monday-anchored) spanning the full available date range, independently **per depth** (each depth is its own time series with its own gaps — don't resample across depths together).
3. Linearly interpolate gaps of up to **8 consecutive weeks**; beyond that, don't fabricate data — leave it for back/forward-fill only at the series edges.
4. Add a global integer `Time_Idx` (sequence position) and cyclical calendar encodings `month_sin`/`month_cos` (`sin`/`cos` of `2π × month / 12`).

**Known limitation carried forward from the previous implementation — checked (2026-08-11)**:
8-week linear interpolation across a variable that's specifically being used to detect brief
hypoxic dips risks smoothing over the very events being predicted, if a real dip happens to fall
inside an interpolated gap. Measured directly against the real pipeline output: of the 2,722
known (non-null) weekly 25 m observations, 2,277 (83.6%) are filled by interpolation rather than a
raw ship-based sample — expected, since weekly resolution is far finer than actual sampling
frequency, not a sign of long-gap fabrication. Of those interpolated weeks, 230 (8.5% of all known
weeks) fall below the 60 µmol/L hypoxic line. This is *not* evidence of long-gap fabrication —
every interpolated value is, by construction, a short linear fill between two real bracketing
observations within the 8-week limit (`limit_area="inside"`, see `src/pipeline.py`), not an
extrapolation across a season-long gap — but it does mean most individual "hypoxic weeks" in the
labeled series are short-interval estimates rather than direct measurements. **Decision: keep the
8-week limit as-is** — shrinking it further would mostly remove legitimate short interpolations
between closely-spaced real casts, not fix a fabrication problem, since the limit already prevents
interpolation across long gaps. Revisit only if a specific held-out event's timing/magnitude looks
wrong during Phase 9 evaluation.

## 5. Feature engineering

Work on the **25 m depth series** — this is the target depth (dangerous-for-fish conditions are a
bottom-water phenomenon at this station per `Research.txt`).

**Engineered features** (in addition to the raw harmonized columns):

- `Surface_Temp_C`, `Surface_O2_umol_L` — the 1 m-depth series reindexed onto the 25 m series' dates, giving each 25 m row a same-date surface reading.
- `Vertical_Temp_Grad = Surface_Temp_C − Temp_C`, `Vertical_O2_Grad = Surface_O2_umol_L − O2_umol_L` — stratification-strength proxies; `Research.txt` identifies stratification as a primary physical trigger for localized hypoxia.
- `O2_Derivative_1W` — week-over-week change in oxygen (kinematic/momentum feature).

**A note on `Vertical_O2_Grad` and `O2_Derivative_1W` specifically**: both are partially derived
from the target (`O2_umol_L`) itself. `CONCEPT.md` §2.3 flags "don't feed the target's own history
as an input" as a principle worth taking seriously — but these two aren't quite the same thing as an
autoregressive lag of the target: `Vertical_O2_Grad` is a same-timestep cross-depth physical
relationship (stratification strength), not the target's own past. `O2_Derivative_1W` *is* closer to
a disguised lag feature and deserves more scrutiny — validate during Phase 2 of the build plan
whether removing it changes tail-prediction accuracy before assuming it's safe to keep.

**Candidate additional features** (from `Research.txt`, not yet in the pipeline — worth testing via
forward selection rather than assuming they help):

- Raw wind direction (not just the U/V-vectorized speed) or a rolling wind-stress/mixing-energy proxy — `Research.txt` is explicit that wind-driven advection of low-oxygen water from the adjacent Kiel Bight is the *primary* physical driver of hypoxia at this specific station, ahead of local biological consumption.
- A lagged/rolling chlorophyll-a feature — oxygen depletion typically follows peak bloom by days to weeks, so the instantaneous Chl-a value may be less predictive than a 2–4 week rolling lag.

**Feature selection procedure** (adopt from the reference paper, `CONCEPT.md` §2.2): don't feed
every candidate feature to the model by default. Test candidates individually against
validation performance (using whatever weighted-loss/tail-metric setup Phase 3 below establishes),
keep the best, add the next-best in combination, stop once an added feature stops helping. This
matters more here than it did in the previous point-forecasting implementation, because a feature
useful for average-case accuracy isn't guaranteed to help tail/extreme-event accuracy specifically —
they should be evaluated against the tail metrics (§7 below), not aggregate error.

## 6. Target definition & sample weighting

### 6.1 Threshold tiers — confirmed (2026-08-11) against real data, see §6.2

Based on the ecological hypoxia line already established in `Research.txt` (~60 µmol/L), three
tiers, analogous to IRANNA's 1,000/2,000/3,000 nT strong/extreme/severe structure. Implemented in
`src/labeling.py`'s `THRESHOLDS`:

| Tier | Threshold | Rationale |
|---|---|---|
| Watch | < 80 µmol/L | Approaching the ecological hypoxia line |
| Hypoxic | < 60 µmol/L | The standard ecological hypoxia definition |
| Severe | < 30 µmol/L | Well below the hypoxia line — acute risk |

### 6.2 Measured imbalance (2026-08-11, real weekly 25 m series, 2,722 known weeks, 1957–2023)

Confirmed the expectation below by actually computing it, the same way the reference paper's
Figure 2 characterizes the `|SML|` distribution: **15.98% of weeks fall below 80 µmol/L, 11.94%
below 60, 6.10% below 30** — a far gentler imbalance than IRANNA's (their tiers hold 0.33% / 0.01%
/ 0.0006% of samples), confirming oxygen is a bounded physical quantity, not a heavy-tailed index.
The weight tiers in §6.4 are correspondingly gentler than the paper's ~1×/10×/40×/80×/160×
progression. The imbalance is also strongly non-stationary: hypoxic weeks are concentrated
July–October (the productive/stratified season, as `Research.txt` predicts) and were essentially
absent before 1970 (0% of weeks below 80 µmol/L in the 1950s/60s) but common from the 1980s onward
(19–29% of weeks per decade) — a real historical trend, not sampling noise, worth keeping in mind
for Phase 6/8's chronological split (early decades contain almost no positive examples).

### 6.3 Target transform — decided: none

Checked (2026-08-11) against the real data: the raw `O2_umol_L` distribution has skew ≈ −0.29
(near-symmetric already). The candidate **"oxygen deficit"** `max(0, 60 − O2_umol_L)`, restricted
to weeks that are actually hypoxic (deficit > 0, n=325), has skew ≈ 0.08 — already close to
symmetric. Applying `log1p` to that nonzero deficit made skew **worse**, not better (−1.09,
now skewed the other direction). **Decision: no transform.** The model's target stays plain
`O2_umol_L` (interpretable, quantile-native, directly usable by the app); `oxygen_deficit` is
computed separately by `src/labeling.py` purely for weighting/evaluation, untransformed. This
confirms §6.3's original prediction — oxygen isn't a multi-order-of-magnitude quantity the way
`|SML|` is, so it doesn't need the same compression.

### 6.4 Sample weighting — implemented in `src/labeling.py`

Assign each training sample a weight based on how severe its target value is: `wᵢ = 1.0` for
normoxic samples (above the "watch" tier), progressively larger for samples in and below each
tier. Given §6.2's measured imbalance is far gentler than IRANNA's, the starting weights are
correspondingly gentler than the paper's progression — `TIER_WEIGHTS` in `src/labeling.py`:
normoxic 1.0, watch 3.0, hypoxic 6.0, severe 12.0. **Still a starting point, not a derived
formula** — tune against tail-metric validation performance once Phase 7 training exists, the same
way the paper's Figure 3 shows hand-tuned discrete clusters rather than a smooth function. Weeks
with unknown `O2_umol_L` (an interior gap wider than §4's interpolation limit) get `NaN` weight,
not a default of 1.0 — treating "unknown" as "confidently normoxic" would silently bias training.

### 6.5 Implementation mechanism — verified against the installed dependency stack

**`pytorch-forecasting`'s `TimeSeriesDataSet` natively supports per-sample loss weighting** — pass
a `weight="column_name"` argument pointing at a dataframe column containing the per-row weight from
§6.4. Confirmed in the installed `pytorch_forecasting==1.7.0` source
(`_timeseries.py`: `weight: str | None = None` parameter; `_base_metrics.py`'s
`MultiHorizonMetric.update()`: `losses = losses * unsqueeze_like(weight, losses)` — the weight is
multiplied directly into the per-timestep loss before aggregation, for `QuantileLoss` and any other
`MultiHorizonMetric`-based loss). **This means the paper's WMSE technique doesn't need a custom loss
class** — compute the weight column during feature engineering (§6.4) and pass it straight into
`TimeSeriesDataSet`. Re-verify this API against whatever `pytorch-forecasting` version actually
gets installed for the rewrite, since this was checked against a specific installed version that
may have moved on.

## 7. Model architecture

**Recommendation: keep a `TemporalFusionTransformer`** (as the previous implementation used) rather
than switching to the reference paper's plain fully-connected architecture. Reasoning: the paper's
own architecture choice was secondary to its loss-reweighting insight (`CONCEPT.md` §2 — they
explicitly frame the FCNN as a simplicity/efficiency choice, not a claim that FCNNs are better
suited to this problem, and flag recurrent architectures as future work for their case). A TFT is
already quantile-native (P10/P50/P90 output, useful for a threshold-crossing-probability alert in
the app), already handles the multi-feature input this project actually has (vs. IRANNA's 2
forward-selected features), and §6.5 confirms the weighting mechanism attaches cleanly to it via
`TimeSeriesDataSet`. Revisit only if a weighted-TFT proves difficult to get working — a
fully-connected fallback closer to the paper's own architecture (3 hidden layers, sigmoid + batch
norm + dropout, non-recurrent flattened lookback window) is a reasonable second option.

Starting hyperparameters (previous implementation's values — expect to re-tune, see next
paragraph): `hidden_size=16`, `attention_head_size=1`, `dropout=0.1`, `hidden_continuous_size=8`,
`learning_rate=0.03`, Adam optimizer.

**Re-tune hyperparameters after switching to a weighted loss.** Reweighting the loss changes the
optimization landscape — hyperparameters tuned for a symmetric, unweighted objective aren't
guaranteed to still be good. The reference paper tuned its architecture (via Optuna/TPE) directly
against its weighted objective, not reused from an unweighted baseline; do the same here rather than
assuming the previous implementation's hyperparameters still apply.

## 8. Training procedure

- **Splitting**: chronological train/validation split (not the paper's random-block split — see
  `CONCEPT.md` §4 for why: their split is safe for an exogenous-input-only model where every sample
  is self-contained; a TFT's genuine encoder/decoder sequential context needs chronological splits
  to evaluate realistically).
- **Held-out event set**: separately from the chronological split, hold out specific known
  historical hypoxic episodes (identified via `Documentation/OPEN_QUESTIONS.md` #3, or inferred from
  threshold crossings cross-checked against `Research.txt`'s cited literature) as a dedicated
  out-of-sample event-study set — directly transferable from the paper regardless of the splitting
  method used for the rest of the data.
- **Regularization**: dropout (already in the TFT), early stopping on validation loss (previous
  implementation used patience=3 on `val_loss` — reasonable starting point), gradient clipping.
- **Loss**: weighted `QuantileLoss` via the `weight=` mechanism (§6.5).

## 9. Evaluation

Standard aggregate regression metrics (plain MAE/RMSE) are close to meaningless for this problem —
dominated by the easily-predicted quiet-time weeks. Use the reference paper's four-part evaluation
instead (`CONCEPT.md` §2.4):

1. **Weighted aggregate metrics** — weighted correlation and weighted RMSE, using the §6.4 sample
   weights, reported per split.
2. **Threshold-sweep classification metrics** — evaluate the regression output *as if* it were a
   classifier across a sweep of thresholds (accuracy/precision/recall/F1 at each), plus a single
   ROC curve/AUC. Don't build a separate classifier (`CONCEPT.md` §2.4/§4).
3. **Event studies** — for each held-out historical hypoxic episode, plot observation vs. new
   (weighted) model vs. an unweighted baseline, side by side, to demonstrate whether reweighting
   actually improved tail prediction (mirrors the paper's Figures 6–8).
4. **Persistence-baseline check** — also plot a trivial "predict = last known value" baseline
   against the tail/event metrics specifically (not just aggregate RMSE). Oxygen's strong
   week-to-week autocorrelation means persistence can look deceptively good in aggregate while being
   useless for anticipating an onset (`CONCEPT.md` §4) — confirm the new model beats it specifically
   on tail metrics.

**Calibrate expectations** (`CONCEPT.md` §2.5): the reference paper's best result, on a much larger
and cleaner dataset than this one, reaches F1 ≈ 0.3–0.4 at its extreme-event thresholds with
AUC ≈ 0.88. Given this project's much smaller, weekly-resampled, gap-interpolated dataset, treat
F1 in the 0.2–0.4 range and AUC ≈ 0.7–0.85 as a plausible "this worked" outcome, not a shortfall.

**Forecast horizon**: validate at horizons short enough to be honest — probably weeks, not years.
The previous implementation's app allowed forecasting up to 520 weeks out, which produced the
flatline failure mode documented in `reference/autoregressive-flatline-example.png`; §2.3 of
`CONCEPT.md` explains why the new, non-autoregressive approach shouldn't reproduce that specific
failure, but a realistic reliable horizon (e.g. "confident out to ~8–12 weeks") should still be
established empirically here rather than assumed away.

**Sanity check before blaming the model**: if a specific historical prediction looks wrong, check
whether it might be a single-point sampling representativeness issue (`CONCEPT.md` §4, the
SML-localized-station analogue) before concluding the weighting scheme needs retuning.

## 10. Application / dashboard requirements

Rebuild the interactive dashboard (previously a Streamlit app) with these requirements:

- A threshold reference line (and shaded hypoxic zone) on the oxygen forecast chart, using the
  tiers from §6.1.
- A simple risk readout — e.g. "X% probability of hypoxic conditions by [date]" — derived from the
  quantile spread (P10/P50/P90) crossing the threshold, consistent with the "derive the alert from
  the regression output" decision in §6/`CONCEPT.md` §2.4.
- A forecast horizon control bounded to whatever range §9 establishes as reliable — don't reproduce
  the previous implementation's 10-year slider range without evidence the new approach is trustworthy
  that far out.
- Load the most recent trained checkpoint. **Known pitfall from the previous implementation**: if
  using PyTorch Lightning's default versioned logging (`lightning_logs/version_N/`), picking "the
  latest checkpoint" by file creation time (`os.path.getctime`) rather than by version number is
  fragile — cloning/restoring the repo can change file timestamps and silently select the wrong
  checkpoint. Prefer a fixed/named checkpoint path, or explicitly sort by version number, not file
  timestamp.

## 11. Known pitfalls carried forward from the previous implementation

- **No `requirements.txt` was ever created** — the exact dependency versions that were known to
  work are preserved in `Documentation/reference/previous-implementation-pip-freeze.txt`. Start the
  rewrite by creating a `requirements.txt` (or `pyproject.toml`) from day one — see
  `Documentation/ENVIRONMENT.md`.
- **`lightning_logs/` grew to 1,147 version folders with only 7 actual checkpoints** — every
  training run, including incomplete ones, created a new versioned folder. Either disable versioned
  logging in favor of a fixed path, or add cleanup as a routine step.
- **The weather-fetch step has no local cache** — every full pipeline run re-downloads from DWD's
  live API, so the pipeline has a hard network dependency at both training and (if the app rebuilds
  data on load) serving time. Consider caching the fetched weather data to a local file, especially
  since historical DWD data for past periods doesn't change.
- **The 4-week-chunked, autoregressive-feedback forecasting approach in the previous app** is the
  direct cause of the flatline failure mode — don't reintroduce it; see §10 above and `CONCEPT.md`
  §2.3.
