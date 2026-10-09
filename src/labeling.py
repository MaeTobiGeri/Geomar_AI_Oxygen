"""Define hypoxia risk on the 25 m target series: severity tiers, sample weights, and the
oxygen deficit used for evaluation. Follows Documentation/SPEC.md section 6.

Threshold and weight values were set from an imbalance audit run against the real weekly
25 m series (2,722 non-null weeks, 1957-2023): 15.98% of weeks fall below 80 umol/L, 11.94%
below 60, 6.10% below 30 - confirming SPEC.md's expectation of a far gentler imbalance than
the reference paper's SML index, so the weight tiers below are correspondingly gentler than
IRANNA's ~160x. These are starting points to tune once real training exists (SPEC.md §6.4,
BUILD_PLAN.md Phase 7), not final values.

The oxygen deficit's distribution among already-hypoxic weeks came out close to symmetric
(skew 0.08), and a log1p transform made it worse (skew -1.09, the wrong direction) rather
than better - so, per SPEC.md §6.3, no transform is applied.
"""

import numpy as np
import pandas as pd

TARGET_DEPTH_M = 25.0

# umol/L. SPEC.md §6.1.
THRESHOLDS = {
    "watch": 80.0,
    "hypoxic": 60.0,
    "severe": 30.0,
}

# Sample weight by tier, used by src/dataset.py's TimeSeriesDataSet(weight=...) (SPEC.md §6.5).
TIER_WEIGHTS = {
    "normoxic": 1.0,
    "watch": 3.0,
    "hypoxic": 6.0,
    "severe": 12.0,
}


def select_target_series(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """The 25 m depth rows - Boknis Eck's target series for hypoxia risk (SPEC.md §5)."""
    return weekly_df[weekly_df["Depth_m"] == TARGET_DEPTH_M].reset_index(drop=True)


def add_oxygen_deficit(df: pd.DataFrame) -> pd.DataFrame:
    """How far below the hypoxia line each week is; 0 when not hypoxic, NaN when O2 is
    unknown (an interior gap wider than the pipeline's interpolation limit)."""
    df = df.copy()
    df["oxygen_deficit"] = (THRESHOLDS["hypoxic"] - df["O2_umol_L"]).clip(lower=0)
    return df


def add_sample_weight(df: pd.DataFrame, reanalysis_penalty: float = 0.5) -> pd.DataFrame:
    """Tiered loss weight by severity; NaN where O2 is unknown, so an unlabeled week isn't
    mistaken for a confidently-normoxic one downstream.

    Args:
        df: DataFrame with O2_umol_L column
        reanalysis_penalty: Multiplier for samples with reanalysis data (0.0-1.0).
                           Default 0.5 means reanalysis samples get 50% weight.
                           Set to 1.0 to treat reanalysis equal to observations.

    Returns:
        DataFrame with sample_weight column added, adjusted for reanalysis data source
    """
    df = df.copy()
    conditions = [
        df["O2_umol_L"].isna(),
        df["O2_umol_L"] < THRESHOLDS["severe"],
        df["O2_umol_L"] < THRESHOLDS["hypoxic"],
        df["O2_umol_L"] < THRESHOLDS["watch"],
    ]
    weights = [np.nan, TIER_WEIGHTS["severe"], TIER_WEIGHTS["hypoxic"], TIER_WEIGHTS["watch"]]
    df["sample_weight"] = np.select(conditions, weights, default=TIER_WEIGHTS["normoxic"])

    # Apply reanalysis penalty if data source columns exist
    # Average across all variables to get overall reanalysis fraction
    source_cols = [col for col in df.columns if col.endswith("_source")]
    if source_cols:
        # Calculate average data source (0=all observed, 1=all reanalysis, 0.5=mixed)
        df["reanalysis_fraction"] = df[source_cols].mean(axis=1)

        # Apply penalty: weight = base_weight * (1 - fraction * (1 - penalty))
        # If fraction=0 (observed): weight unchanged
        # If fraction=1 (reanalysis): weight *= penalty
        # If fraction=0.5 (mixed): weight *= (1 + penalty)/2
        penalty_factor = 1 - df["reanalysis_fraction"] * (1 - reanalysis_penalty)
        df["sample_weight"] = df["sample_weight"] * penalty_factor

    return df


def label_hypoxia_risk(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """The 25 m series with `oxygen_deficit` and `sample_weight` added, ready for
    src/dataset.py."""
    target = select_target_series(weekly_df)
    target = add_oxygen_deficit(target)
    target = add_sample_weight(target)
    return target


def imbalance_report(labeled_df: pd.DataFrame) -> dict[str, float]:
    """Fraction of non-null weeks in each severity tier (SPEC.md §6.2)."""
    known = labeled_df.dropna(subset=["O2_umol_L"])
    return {
        tier: (known["O2_umol_L"] < THRESHOLDS[tier]).mean()
        for tier in ("watch", "hypoxic", "severe")
    }


def identify_hypoxic_episodes(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive weeks below the hypoxic threshold into discrete episodes, as a
    starting point for the held-out event-study set (SPEC.md §8) until a literature-verified
    list is available (Documentation/OPEN_QUESTIONS.md #3)."""
    below = labeled_df["O2_umol_L"] < THRESHOLDS["hypoxic"]
    episode_id = (below != below.shift(fill_value=False)).cumsum()

    return (
        labeled_df[below]
        .assign(episode_id=episode_id[below])
        .groupby("episode_id")
        .agg(
            start_date=("Date", "min"),
            end_date=("Date", "max"),
            duration_weeks=("Date", "count"),
            min_o2=("O2_umol_L", "min"),
        )
        .reset_index(drop=True)
    )
