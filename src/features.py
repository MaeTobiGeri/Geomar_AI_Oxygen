"""Feature engineering for the 25 m target series: surface readings reindexed onto
25 m dates, vertical gradients (stratification proxies), kinematic features, and candidate
features from Research.txt for forward selection. Follows Documentation/SPEC.md section 5.

Wind-driven advection from the Kiel Bight is the PRIMARY driver of hypoxia at Boknis Eck
(Research.txt), so wind features beyond the existing U/V components are included as
candidates. Chlorophyll-a lag features capture the 2-4 week delay between bloom peak and
oxygen depletion (Research.txt).

The O2_Derivative_1W feature is implemented behind INCLUDE_O2_DERIVATIVE flag per
SPEC.md §5's caution about feeding the target's own history as input (though this is a
week-over-week change, not an autoregressive lag, it still deserves scrutiny during
ablation testing in Phase 7).
"""

import numpy as np
import pandas as pd

# Module-level constant: flag to include/exclude O2_Derivative_1W (SPEC.md §5).
INCLUDE_O2_DERIVATIVE = True


def add_surface_readings(df_25m: pd.DataFrame, df_all_depths: pd.DataFrame) -> pd.DataFrame:
    """Reindex 1 m surface readings onto the 25 m series' dates, giving each 25 m row
    same-date surface values for Temp_C and O2_umol_L (SPEC.md §5)."""
    df_1m = df_all_depths[df_all_depths["Depth_m"] == 1.0].set_index("Date")

    df_25m = df_25m.copy()
    df_25m = df_25m.set_index("Date")

    df_25m["Surface_Temp_C"] = df_1m["Temp_C"].reindex(df_25m.index)
    df_25m["Surface_O2_umol_L"] = df_1m["O2_umol_L"].reindex(df_25m.index)

    return df_25m.reset_index()


def add_vertical_gradients(df: pd.DataFrame) -> pd.DataFrame:
    """Stratification-strength proxies: Surface - Bottom differences for temperature and
    oxygen (SPEC.md §5). Research.txt identifies stratification as a primary physical
    trigger for localized hypoxia."""
    df = df.copy()
    df["Vertical_Temp_Grad"] = df["Surface_Temp_C"] - df["Temp_C"]
    df["Vertical_O2_Grad"] = df["Surface_O2_umol_L"] - df["O2_umol_L"]
    return df


def add_o2_derivative(df: pd.DataFrame) -> pd.DataFrame:
    """Week-over-week change in O2 (kinematic/momentum feature). This is closer to a
    disguised lag of the target than Vertical_O2_Grad is, per SPEC.md §5 - validate
    whether removing it changes tail-prediction accuracy before assuming it's safe."""
    df = df.copy()
    df["O2_Derivative_1W"] = df["O2_umol_L"].diff(periods=1)
    return df


def add_wind_direction_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve raw wind direction alongside the existing U/V vectorized components.
    Research.txt: westerly winds drive low-oxygen water into the bight, easterly winds
    ventilate it - the actual direction matters, not just the vectorized speed."""
    # Wind_Dir_deg already exists in the dataframe from data_ingestion.py's weather fetch
    # - no transformation needed, just ensuring it's present for forward selection
    return df


def add_wind_mixing_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Mixing-energy proxy: wind stress is proportional to wind speed squared.
    Research.txt: wind-driven mixing is a primary control on oxygen dynamics."""
    df = df.copy()
    df["Wind_Mixing_Energy"] = df["Wind_Speed_ms"] ** 2
    return df


def add_chlorophyll_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged chlorophyll features: oxygen depletion follows bloom peak by 2-4 weeks
    (Research.txt), so instantaneous Chl_a may be less predictive than lagged values."""
    df = df.copy()
    for weeks in [1, 2, 3, 4]:
        df[f"Chl_a_lag_{weeks}W"] = df["Chl_a"].shift(periods=weeks)
    return df


def engineer_features(df_25m: pd.DataFrame, df_all_depths: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering to the 25 m target series: surface readings,
    vertical gradients, optional O2 derivative, and candidate features for forward
    selection (wind direction/mixing, chlorophyll lags). Returns the 25 m dataframe
    with engineered features appended."""

    df = add_surface_readings(df_25m, df_all_depths)
    df = add_vertical_gradients(df)

    if INCLUDE_O2_DERIVATIVE:
        df = add_o2_derivative(df)

    # Candidate features for forward selection (SPEC.md §5)
    df = add_wind_direction_raw(df)
    df = add_wind_mixing_energy(df)
    df = add_chlorophyll_lags(df)

    return df


def forward_feature_selection(
    candidate_features: list[str],
    base_features: list[str],
    df: pd.DataFrame,
    target_col: str,
    validation_metric_fn,
    verbose: bool = True
) -> list[str]:
    """Forward feature selection: test candidates individually against validation
    tail-metric (from Phase 9), keep the best, add the next-best in combination, stop
    when a feature stops helping (SPEC.md §5, following the reference paper's procedure
    from Chu et al. 2025 §3.2).

    This is a skeleton to be run once real training (Phase 7) exists - it returns the
    input base_features unchanged for now, since validation_metric_fn requires a trained
    model to evaluate.

    Args:
        candidate_features: List of feature column names to test
        base_features: List of already-selected feature column names (starting point)
        df: Dataframe with all features and target
        target_col: Name of the target column
        validation_metric_fn: Function(selected_features, df, target_col) -> float
            that trains a model on selected_features and returns a validation metric
            (lower is better, e.g. weighted RMSE on hypoxic weeks)
        verbose: Print selection progress

    Returns:
        List of selected feature names (base_features + best candidates)
    """
    selected = base_features.copy()
    remaining = candidate_features.copy()

    if verbose:
        print(f"Forward selection starting with {len(selected)} base features")
        print(f"Testing {len(remaining)} candidates")

    # Baseline performance with just base_features
    # (Placeholder - requires Phase 7 training to implement)
    # best_metric = validation_metric_fn(selected, df, target_col)

    while remaining:
        metrics = {}

        # Test each remaining candidate in combination with selected features
        for candidate in remaining:
            test_features = selected + [candidate]
            # metric = validation_metric_fn(test_features, df, target_col)
            # metrics[candidate] = metric
            pass  # Placeholder

        if not metrics:
            break

        # Keep the candidate that improves the metric most
        # best_candidate = min(metrics, key=metrics.get)
        # improvement = best_metric - metrics[best_candidate]

        # if improvement <= 0:
        #     if verbose:
        #         print(f"No improvement from remaining candidates, stopping")
        #     break

        # selected.append(best_candidate)
        # remaining.remove(best_candidate)
        # best_metric = metrics[best_candidate]

        # if verbose:
        #     print(f"Added {best_candidate}, metric: {best_metric:.4f}")

        break  # Placeholder exit

    if verbose:
        print(f"Selection complete: {len(selected)} features")

    return selected
