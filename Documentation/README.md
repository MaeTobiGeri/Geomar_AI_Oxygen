# Geomar AI Oxygen — Documentation

**Status (2026-08-11): documentation-only. There is no code in this repository right now.** The
project is being rebuilt from scratch around a new goal — see below — and this folder is the
complete specification for that rebuild. The previous implementation (a general oxygen-level
forecaster) was removed; its lessons learned are folded into the docs here rather than kept as
legacy code.

**New goal**: predict when dissolved oxygen at the **Boknis Eck** time-series station (Baltic Sea,
Kiel Bight) is heading toward levels dangerous for fish (hypoxia), rather than forecasting the
general oxygen level at any horizon. The approach adapts an "imbalanced regression" technique from
a space-weather paper on predicting rare extreme geomagnetic events, applied here to rare extreme
low-oxygen events.

## Where to start

| Document | Read this for |
|---|---|
| [`CONCEPT.md`](CONCEPT.md) | **Why** — the rationale for the pivot, the reference paper's methodology, and how it maps onto this domain |
| [`SPEC.md`](SPEC.md) | **What** — the complete technical specification: data pipeline, target definition, feature engineering, model, loss, training, evaluation, and dashboard requirements |
| [`BUILD_PLAN.md`](BUILD_PLAN.md) | **How / in what order** — a concrete, file-level implementation checklist covering the entire project from environment setup to a working dashboard |
| [`STYLE.md`](STYLE.md) | **How it should be written** — code style and project structure rules that apply to every file `BUILD_PLAN.md` produces: human-readable, as simple as possible, no unnecessary checks, no unneeded code, split by responsibility across files |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | Python version and dependency versions known to work, carried over from the previous implementation |
| [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) | Additional data that would help, flagged for the project owner to weigh in on — not a blocker to starting |

Read in that order: `CONCEPT.md` → `SPEC.md` → `BUILD_PLAN.md` → `STYLE.md`. `ENVIRONMENT.md` and
`OPEN_QUESTIONS.md` are reference material, not narrative.

## Supporting material

- [`data/`](data/) — the raw source files the pipeline is built from: `BoknisEck_1957-2014.csv`,
  `BoknisEck_2015-2023.csv` (ocean chemistry, PANGAEA exports), `BoknisEck_chl_2015-2021.tab`
  (supplementary chlorophyll). Live DWD weather data is fetched at build/train time, not stored
  here — see `SPEC.md` §2.
- [`reference/`](reference/) — source material the docs are built from, not documentation itself:
  `Research.txt` (a literature review on ML approaches to marine dissolved-oxygen prediction, cited
  throughout `SPEC.md`/`CONCEPT.md`), `Chu-2025-IRANNA-SpaceWeather.pdf` (the reference paper),
  `autoregressive-flatline-example.png` (a real failure mode from the previous implementation,
  referenced in `CONCEPT.md` §2.3 and `SPEC.md` §10–11), and
  `previous-implementation-pip-freeze.txt` (dependency versions, see `ENVIRONMENT.md`).

## Project summary (target state, per `SPEC.md`)

- **Target**: hypoxia risk for `O2_umol_L` (dissolved oxygen, µmol/L) at 25 m depth at Boknis Eck — a weighted-regression prediction of oxygen/oxygen-deficit, evaluated as a threshold-crossing alert.
- **Approach**: a weighted loss (analogous to the reference paper's WMSE) that emphasizes the low-oxygen tail of the distribution, rather than a symmetric loss optimized for average-case accuracy.
- **Model**: `TemporalFusionTransformer` from [`pytorch-forecasting`](https://pytorch-forecasting.readthedocs.io/) (recommended default — see `SPEC.md` §7 for the reasoning and the fallback option).
- **Inputs**: weekly-resampled water chemistry, engineered stratification/gradient features, calendar encoding, and DWD weather data — see `SPEC.md` §5 for the full list and the feature-selection procedure to apply.
- **Interface**: a rebuilt interactive dashboard with a hypoxia threshold line and a risk readout — see `SPEC.md` §10.

Nothing above is final — thresholds, weight values, selected features, and the exact target
transform are proposed starting points meant to be validated against the real data during the
build (`SPEC.md`'s framing, `BUILD_PLAN.md` Phases 4–7).
