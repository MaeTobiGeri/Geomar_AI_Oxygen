"""Outlier detection for hypoxia prediction data.

Implements Whisker's box-plot method from reference paper (AE-DeepAR):
- Detects outliers in INPUT features (sensor errors)
- PRESERVES extreme hypoxia values (target variable)
- Adds outlier flags as additional features

Based on reference model comparison and data quality requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5
) -> np.ndarray:
    """Detect outliers using Interquartile Range (IQR) method.

    Whisker's box-plot method:
    - Lower bound: Q1 - multiplier * IQR
    - Upper bound: Q3 + multiplier * IQR
    - Outliers are values outside these bounds

    Args:
        series: Pandas Series to check for outliers
        multiplier: IQR multiplier (default: 1.5 for standard box-plot)

    Returns:
        Boolean array where True indicates outlier
    """
    # Calculate quartiles
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate bounds
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Identify outliers
    outliers = (series < lower_bound) | (series > upper_bound)

    return outliers.values


def detect_physical_impossibilities(
    df: pd.DataFrame,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """Detect physically impossible values (sensor errors).

    Args:
        df: DataFrame with features
        feature_bounds: Dictionary mapping feature names to (min, max) bounds
                       If None, use default bounds for common ocean features

    Returns:
        Dictionary mapping feature names to boolean outlier arrays
    """
    if feature_bounds is None:
        # Default physical bounds for common ocean features
        feature_bounds = {
            'Temp_C': (-2.0, 40.0),  # Ocean temperature range
            'Sal_psu': (0.0, 45.0),   # Salinity range
            'Density_kg_m3': (1000.0, 1040.0),  # Density range
            'O2_umol_L': (0.0, 500.0),  # Oxygen (allow 0, but not negative)
            'Pressure_dbar': (0.0, 1000.0),  # Pressure range (for shallow water)
            # Weather features
            'Temp_2m_C': (-30.0, 45.0),
            'Wind_10m_ms': (0.0, 50.0),
            'Precip_mm': (0.0, 200.0),
        }

    outliers = {}

    for feature, (min_val, max_val) in feature_bounds.items():
        if feature in df.columns:
            series = df[feature]
            outliers[feature] = (series < min_val) | (series > max_val)

    return outliers


def flag_outliers(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'O2_umol_L',
    iqr_multiplier: float = 1.5,
    preserve_target: bool = True,
    physical_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """Flag outliers in feature data.

    IMPORTANT: Does NOT remove outliers, only flags them.
    Extreme hypoxia values in target are NEVER flagged.

    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names to check
        target_column: Target variable column name (default: 'O2_umol_L')
        iqr_multiplier: IQR multiplier for box-plot method (default: 1.5)
        preserve_target: If True, never flag target variable outliers (default: True)
        physical_bounds: Physical bounds for impossibility detection

    Returns:
        Tuple of (df_with_flags, outlier_stats)
    """
    df_flagged = df.copy()
    outlier_stats = {}

    print("\n" + "="*80)
    print("OUTLIER DETECTION (Whisker's Box-Plot Method)")
    print("="*80)

    # Check each feature
    for feature in feature_columns:
        if feature not in df.columns:
            continue

        # Skip target if preserve_target is True
        if preserve_target and feature == target_column:
            print(f"\nSkipping {feature} (target variable - preserving extreme values)")
            continue

        print(f"\nAnalyzing {feature}:")

        # IQR-based outlier detection
        iqr_outliers = detect_outliers_iqr(df[feature], multiplier=iqr_multiplier)
        n_iqr_outliers = iqr_outliers.sum()

        # Physical impossibility detection
        physical_outliers_dict = detect_physical_impossibilities(
            df[[feature]],
            feature_bounds=physical_bounds
        )
        physical_outliers = physical_outliers_dict.get(feature, np.zeros(len(df), dtype=bool))
        n_physical_outliers = physical_outliers.sum()

        # Combine outlier flags (OR operation)
        combined_outliers = iqr_outliers | physical_outliers

        # Add flag column to dataframe
        flag_column = f"{feature}_outlier_flag"
        df_flagged[flag_column] = combined_outliers.astype(int)

        # Statistics
        outlier_stats[feature] = {
            'iqr_outliers': n_iqr_outliers,
            'physical_outliers': n_physical_outliers,
            'total_outliers': combined_outliers.sum(),
            'percentage': (combined_outliers.sum() / len(df)) * 100,
            'flag_column': flag_column
        }

        print(f"  IQR outliers:      {n_iqr_outliers} ({n_iqr_outliers/len(df)*100:.2f}%)")
        print(f"  Physical outliers: {n_physical_outliers} ({n_physical_outliers/len(df)*100:.2f}%)")
        print(f"  Total outliers:    {combined_outliers.sum()} ({combined_outliers.sum()/len(df)*100:.2f}%)")

        if combined_outliers.sum() > 0:
            # Show outlier value ranges
            outlier_values = df.loc[combined_outliers, feature]
            print(f"  Outlier range:     [{outlier_values.min():.2f}, {outlier_values.max():.2f}]")
            print(f"  Normal range:      [{df.loc[~combined_outliers, feature].min():.2f}, {df.loc[~combined_outliers, feature].max():.2f}]")

    # Summary
    total_flagged = sum(stats['total_outliers'] for stats in outlier_stats.values())
    total_cells = len(df) * len(outlier_stats)

    print("\n" + "-"*80)
    print("SUMMARY:")
    print(f"  Features analyzed: {len(outlier_stats)}")
    print(f"  Total data points: {total_cells}")
    print(f"  Total outliers:    {total_flagged} ({total_flagged/total_cells*100:.2f}%)")
    print(f"  Outlier flag columns added: {len(outlier_stats)}")

    if preserve_target:
        print(f"\n  NOTE: Target variable '{target_column}' outliers preserved (extreme hypoxia events)")

    print("="*80)

    return df_flagged, outlier_stats


def handle_outliers(
    df: pd.DataFrame,
    outlier_stats: Dict,
    strategy: str = 'flag',
    replacement_method: str = 'median'
) -> pd.DataFrame:
    """Handle detected outliers using specified strategy.

    Args:
        df: DataFrame with outlier flags
        outlier_stats: Statistics from flag_outliers()
        strategy: How to handle outliers:
            - 'flag': Keep outliers, flags already added (default)
            - 'replace': Replace outliers with replacement_method
            - 'remove': Remove rows with outliers (NOT RECOMMENDED for hypoxia)
        replacement_method: Method for replacing outliers ('median', 'mean', 'interpolate')

    Returns:
        DataFrame with outliers handled
    """
    df_handled = df.copy()

    if strategy == 'flag':
        # Already flagged, nothing to do
        return df_handled

    elif strategy == 'replace':
        print(f"\nReplacing outliers using {replacement_method} method...")

        for feature, stats in outlier_stats.items():
            flag_col = stats['flag_column']

            if flag_col not in df.columns:
                continue

            outlier_mask = df[flag_col] == 1

            if outlier_mask.sum() == 0:
                continue

            # Calculate replacement value
            if replacement_method == 'median':
                replacement = df.loc[~outlier_mask, feature].median()
            elif replacement_method == 'mean':
                replacement = df.loc[~outlier_mask, feature].mean()
            elif replacement_method == 'interpolate':
                # Linear interpolation
                df_handled.loc[outlier_mask, feature] = np.nan
                df_handled[feature] = df_handled[feature].interpolate(method='linear')
                continue
            else:
                raise ValueError(f"Unknown replacement method: {replacement_method}")

            # Replace outliers
            df_handled.loc[outlier_mask, feature] = replacement

            print(f"  {feature}: Replaced {outlier_mask.sum()} outliers with {replacement:.2f}")

    elif strategy == 'remove':
        print("\nRemoving rows with outliers...")

        # Create mask for rows with any outlier
        outlier_mask = pd.Series(False, index=df.index)

        for feature, stats in outlier_stats.items():
            flag_col = stats['flag_column']
            if flag_col in df.columns:
                outlier_mask |= (df[flag_col] == 1)

        rows_before = len(df_handled)
        df_handled = df_handled[~outlier_mask].reset_index(drop=True)
        rows_after = len(df_handled)

        print(f"  Removed {rows_before - rows_after} rows ({(rows_before - rows_after)/rows_before*100:.2f}%)")
        print(f"  Remaining: {rows_after} rows")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df_handled


def get_outlier_summary_stats(outlier_stats: Dict) -> pd.DataFrame:
    """Create summary DataFrame of outlier statistics.

    Args:
        outlier_stats: Statistics from flag_outliers()

    Returns:
        DataFrame with summary statistics
    """
    summary_data = []

    for feature, stats in outlier_stats.items():
        summary_data.append({
            'Feature': feature,
            'IQR Outliers': stats['iqr_outliers'],
            'Physical Outliers': stats['physical_outliers'],
            'Total Outliers': stats['total_outliers'],
            'Percentage': f"{stats['percentage']:.2f}%",
            'Flag Column': stats['flag_column']
        })

    return pd.DataFrame(summary_data)
