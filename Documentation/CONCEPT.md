# Concept: From Point Forecasting to Extreme-Event (Hypoxia) Prediction

This document explains **why** the project is being rebuilt around a different goal. For **what**
to actually build, see [`SPEC.md`](SPEC.md); for **how**/**in what order**, see
[`BUILD_PLAN.md`](BUILD_PLAN.md). This file intentionally contains no implementation detail —
if a number or formula matters for building the system, it lives in `SPEC.md`, not here.

## 1. The pivot

The previous implementation of this project was a general-purpose regressor: given weeks of
history, predict future dissolved oxygen (`O2`) as a continuous quantile forecast. It was optimized
to be accurate *on average*, across the entire range of oxygen values ever observed.

The new goal is narrower and more actionable: **predict when oxygen is heading toward levels that
are dangerous for fish** — i.e. treat this as an **extreme/rare-event prediction problem**, not a
general point-forecasting problem. Getting the bottom of the oxygen distribution right matters far
more than shaving error off the routine mid-range fluctuations that dominate the dataset.

The project already has a working definition of the danger threshold, cited in
[`reference/Research.txt`](reference/Research.txt):

> Hypoxia is typically defined by dissolved oxygen concentrations falling below **2 mg/L**, or
> approximately **60 µmol/L**.

At the Boknis Eck station's 25 m depth, oxygen spends most of its time well above that line but
repeatedly dips toward and below it during the productive/stratified season (see
[`reference/autoregressive-flatline-example.png`](reference/autoregressive-flatline-example.png)
for the seasonal shape of the series). Those dips are exactly the rare, high-value events a plain
regression model tends to under-predict.

## 2. The reference paper — confirmed methodology

**Chu, Jia, McPherron, Li & Bortnik (2025), "Imbalanced Regression Artificial Neural Network Model
for Auroral Electrojet Indices (IRANNA): Can We Predict Strong Events?"**, *Space Weather*, 23,
e2024SW004236. DOI: [10.1029/2024SW004236](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024SW004236).
Open access; full text archived at
[`reference/Chu-2025-IRANNA-SpaceWeather.pdf`](reference/Chu-2025-IRANNA-SpaceWeather.pdf).

### 2.1 The problem they're solving

The target is the **SuperMAG SML index** (nT), a measure of westward auroral electrojet strength —
a proxy for substorm intensity. Its distribution is savagely imbalanced: **80% of `|SML|` values are
below 200 nT**, only **0.33% exceed 1,000 nT** ("strong" events), only **0.01% exceed 2,000 nT**
("extreme" events, i.e. super substorms), and a mere **0.0006% exceed 3,000 nT**. A model trained
with plain MSE learns to minimize error on the abundant quiet-time values and "regresses to the
mean" on the rare, large ones — a traditional MSE-based SML model the authors trained as a baseline
saturates and **cannot predict `|SML|` beyond ~1,000 nT at all**, even during observed 3,000+ nT
events.

### 2.2 Their fix, in one sentence

Keep it a regression problem, but **reweight the loss function per training sample based on how
extreme its target value is** — quiet/background samples get weight 1.0, rarer/larger-magnitude
samples get progressively larger, empirically-tuned weights (not a fixed formula — see their
Figure 3, a scatter of discrete hand-tuned clusters, not a smooth curve). This forces the optimizer
to pay attention to the tail instead of drowning it out in the mean. They call this technique
**"imbalanced regression,"** reused from their own earlier work on chorus and whistler-mode hiss
waves.

### 2.3 The detail that matters most for us

**The model deliberately excludes the target's own past values from its inputs.** It's
"solar-wind-driven," not "history-dependent" (their own terminology) — predicting purely from
external forcing variables (solar wind speed and a coupling function derived from IMF Bz), never
from the index's own recent history. They argue explicitly (§6.3 of the paper) that feeding a
model its own target's history tends to turn it into something that predicts *changes between
timesteps* rather than the actual driven response — degrading both interpretability and, relevant
to us, creating exactly the mechanism behind a bug in the previous implementation: forecasting
multiple weeks ahead by feeding the model's own prior median predictions back in as if they were
observations caused long-horizon forecasts to decay into a flat line
(`reference/autoregressive-flatline-example.png`). Predicting hypoxia risk directly from forcing
variables for a given future week — rather than chaining predictions into each other — removes the
mechanism that produces that failure mode. This isn't a side benefit of adopting IRANNA's approach;
it's the same architectural principle the paper is describing, applied to our domain.

### 2.4 How they evaluate it

Not with plain RMSE. Four complementary methods, all listed here because all four are worth
reusing (see `SPEC.md` §7 for how):

1. **Weighted correlation/RMSE** (WR/WRMSE) — the same formulas as R/RMSE, but computed with the
   same per-sample weights used in training, reported separately per data split.
2. **Threshold-sweep classification metrics** — the *regression* output is evaluated *as if* it
   were a binary classifier at a sweep of thresholds (precision/recall/F1/accuracy at each), plus a
   single ROC curve/AUC. They never train a separate classifier — the regression output alone, read
   against a threshold, *is* the classifier.
3. **Event studies** — real, named historical extreme events, held out entirely from
   training/validation, each plotted observation vs. model vs. a naive baseline to see whether peak
   timing and magnitude were captured.
4. **Virtual experiments** — synthetic, hand-constructed inputs fed through the trained model to
   sanity-check that its response scales physically sensibly with driver strength/duration. Useful
   for interrogating a trained model, not for training or scoring it.

### 2.5 What "success" looked like for them (sets our expectations)

Even their best model — trained on ~30 years of 1-minute-resolution, satellite-measured solar wind
data, much larger and cleaner than anything available here — only reaches **F1 ≈ 0.4** at the
"strong event" threshold and **≈ 0.3** at the "extreme event" threshold, with ROC-AUC = 0.879
(their own scale calls that "excellent," not "near-perfect"). This is useful for calibrating
ambition: successfully applying this technique doesn't mean near-perfect classification of hypoxic
weeks — it means moving from *cannot predict extremes at all* (the unweighted baseline, capped at
predicting ~1,000 nT regardless of true event size) to *meaningfully better than that*, on
imperfect real-world data.

## 3. Conceptual mapping

| IRANNA (space weather) | Boknis Eck oxygen |
|---|---|
| Target: SML index, strength of the auroral electrojet | Target: dissolved oxygen at 25 m, or a derived "oxygen deficit" below a hypoxia threshold |
| Extreme = super substorm, tiered at 1,000 / 2,000 / 3,000 nT | Extreme = hypoxic/near-hypoxic event, tiered around the ~60 µmol/L ecological hypoxia line |
| Savage imbalance: 0.33% / 0.01% / 0.0006% of samples in each tier | Almost certainly a gentler imbalance — oxygen is a bounded physical quantity, not a heavy-tailed index; the actual imbalance needs measuring on this dataset (`SPEC.md` §3) rather than assumed |
| Input: two solar wind driver variables, forward-selected from many candidates | Input: temperature, salinity, nutrients, chlorophyll, vertical stratification gradients, weather — already identified in `reference/Research.txt`; needs the same forward-selection discipline rather than throwing every engineered feature in by default |
| No autoregressive feedback of the target | Same principle — direct motivation for fixing the flatline bug (§2.3 above) |
| Weighted MSE loss, discrete empirically-tuned weight tiers | Same approach: don't guess a closed-form weighting function, tier by severity and tune against validation performance |
| Plain fully-connected NN — architecture choice was secondary to the loss-reweighting insight | Architecture choice for us is likewise secondary; `SPEC.md` recommends a specific option but the loss-reweighting technique is the actual point being adopted |

## 4. What doesn't carry over directly

- **Domain physics differ completely.** Solar wind → magnetosphere coupling has nothing in common
  with ocean biogeochemistry beyond the *shape* of the statistical problem. We're borrowing the
  loss-function and evaluation methodology, not any domain-specific feature or physical mechanism.
- **Timescales differ by orders of magnitude.** IRANNA operates on 1-minute data with a 6-hour
  lookback (substorms last 1–3 hr); Boknis Eck data is weekly-resampled with a much longer lookback.
  Their caution against comparing a driver-based model to a deceptively-strong short-lag persistence
  baseline (§6.3 of the paper) is, if anything, *more* relevant here: oxygen is strongly seasonally
  autocorrelated week-to-week, so a naive "predict = last week's value" baseline will look
  deceptively good on aggregate error metrics. Evaluation needs to specifically check tail/event
  performance against such a baseline, not just aggregate RMSE (`SPEC.md` §7).
- **Their leakage-safe splitting (random daily blocks) assumes a model with no genuine sequential
  memory** — every training sample is a self-contained 6-hour input window. A sequence model with
  real encoder/decoder context (which this project may still use — see `SPEC.md` §5) needs
  chronological splits instead; what *is* directly transferable is holding out specific known
  hypoxic episodes as a dedicated event-study test set, regardless of how the rest of the data is
  split.
- **Their "global index dominated by one local station" limitation has a plausible analogue, not a
  guaranteed one.** SML can be dominated by a single unrepresentative magnetometer reading; a
  single-depth, single-station oxygen measurement at Boknis Eck may or may not represent broader
  bay-wide hypoxia the same way. `reference/Research.txt` already notes that station-specific models
  generalize poorly to nearby sites. Worth remembering when a prediction looks "wrong" during
  evaluation — check whether it's a genuine model error or a single-point sampling
  representativeness issue before concluding the weighting scheme needs retuning.
