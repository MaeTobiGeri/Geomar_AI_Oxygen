"""Uncertainty quantification and evaluation metrics for hypoxia prediction.

Implements metrics from reference paper (AE-DeepAR) including:
- PICP (Prediction Interval Coverage Probability)
- MAE, MSE, RMSE, MAPE
- Stratified metrics by hypoxia tier
- Calibration curves for uncertainty assessment

Based on BUILD_PLAN.md Phase 9 and reference model comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Hypoxia thresholds from labeling.py
THRESHOLDS = {
    "tier_1": 62.5,   # Severe hypoxia
    "tier_2": 93.75,  # Moderate hypoxia
    "tier_3": 125.0,  # Mild hypoxia
}


def calculate_picp(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    confidence_level: float = 0.90
) -> float:
    """Calculate Prediction Interval Coverage Probability (PICP).

    PICP measures how often the true value falls within the predicted interval.
    A well-calibrated model at 90% confidence should have PICP ≈ 0.90.

    Args:
        y_true: True values
        y_lower: Lower bound of prediction interval (e.g., P10)
        y_upper: Upper bound of prediction interval (e.g., P90)
        confidence_level: Expected coverage probability (default: 0.90 for P10-P90)

    Returns:
        PICP score (0 to 1, higher is better)
    """
    # Check which predictions cover the true value
    covered = (y_true >= y_lower) & (y_true <= y_upper)

    # Calculate coverage percentage
    picp = covered.mean()

    return picp


def calculate_mpiw(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    y_true_range: Optional[float] = None
) -> float:
    """Calculate Mean Prediction Interval Width (MPIW).

    MPIW measures the average width of prediction intervals.
    Narrower intervals are better (more precise predictions).

    Args:
        y_lower: Lower bound of prediction interval
        y_upper: Upper bound of prediction interval
        y_true_range: Range of true values for normalization (optional)

    Returns:
        MPIW score (lower is better)
    """
    widths = y_upper - y_lower
    mpiw = widths.mean()

    # Normalize by target range if provided
    if y_true_range is not None:
        mpiw = mpiw / y_true_range

    return mpiw


def calculate_calibration_curve(
    y_true: np.ndarray,
    predictions: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate calibration curve for predicted quantiles.

    For a well-calibrated model, predicted quantiles should match empirical quantiles.

    Args:
        y_true: True values
        predictions: Predicted quantile values (e.g., P50)
        n_bins: Number of bins for calibration curve

    Returns:
        Tuple of (bin_centers, empirical_coverage, predicted_coverage)
    """
    # Sort by predictions
    sorted_indices = np.argsort(predictions)
    sorted_true = y_true[sorted_indices]
    sorted_pred = predictions[sorted_indices]

    # Create bins
    bin_edges = np.linspace(0, len(predictions), n_bins + 1, dtype=int)

    bin_centers = []
    empirical_coverage = []
    predicted_coverage = []

    for i in range(n_bins):
        start_idx = bin_edges[i]
        end_idx = bin_edges[i + 1]

        if start_idx >= end_idx:
            continue

        bin_true = sorted_true[start_idx:end_idx]
        bin_pred = sorted_pred[start_idx:end_idx]

        # Empirical: how many true values are below bin mean
        bin_center = (start_idx + end_idx) / 2 / len(predictions)
        bin_centers.append(bin_center)

        # Predicted coverage (average predicted value in bin)
        predicted_coverage.append(bin_pred.mean())

        # Empirical coverage (fraction of true values below this bin)
        empirical_coverage.append(bin_true.mean())

    return (
        np.array(bin_centers),
        np.array(empirical_coverage),
        np.array(predicted_coverage)
    )


def calculate_standard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate standard regression metrics (MAE, MSE, RMSE, MAPE).

    Args:
        y_true: True values
        y_pred: Predicted values (typically P50/median)

    Returns:
        Dictionary of metric scores
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {
            'mae': np.nan,
            'mse': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'n_samples': 0
        }

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # MAPE (avoid division by zero)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'n_samples': len(y_true)
    }


def stratified_metrics_by_tier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None
) -> Dict[str, Dict]:
    """Calculate metrics stratified by hypoxia tier.

    Important for assessing model performance on extreme hypoxia events.

    Args:
        y_true: True oxygen values
        y_pred: Predicted values (P50)
        y_lower: Lower prediction bound (P10), optional
        y_upper: Upper prediction bound (P90), optional

    Returns:
        Dictionary mapping tier names to metric dictionaries
    """
    results = {}

    # Define tiers
    tiers = {
        'severe_hypoxia': y_true < THRESHOLDS['tier_1'],
        'moderate_hypoxia': (y_true >= THRESHOLDS['tier_1']) & (y_true < THRESHOLDS['tier_2']),
        'mild_hypoxia': (y_true >= THRESHOLDS['tier_2']) & (y_true < THRESHOLDS['tier_3']),
        'normoxic': y_true >= THRESHOLDS['tier_3']
    }

    for tier_name, tier_mask in tiers.items():
        if tier_mask.sum() == 0:
            results[tier_name] = {
                'n_samples': 0,
                'mae': np.nan,
                'rmse': np.nan,
                'picp': np.nan
            }
            continue

        # Standard metrics for this tier
        tier_metrics = calculate_standard_metrics(
            y_true[tier_mask],
            y_pred[tier_mask]
        )

        # PICP for this tier (if bounds provided)
        if y_lower is not None and y_upper is not None:
            picp = calculate_picp(
                y_true[tier_mask],
                y_lower[tier_mask],
                y_upper[tier_mask]
            )
            tier_metrics['picp'] = picp

        results[tier_name] = tier_metrics

    return results


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_p10: np.ndarray,
    y_pred_p50: np.ndarray,
    y_pred_p90: np.ndarray,
    confidence_levels: List[float] = [0.80, 0.90, 0.95]
) -> Dict:
    """Comprehensive evaluation of probabilistic predictions.

    Args:
        y_true: True oxygen values
        y_pred_p10: 10th percentile predictions
        y_pred_p50: 50th percentile (median) predictions
        y_pred_p90: 90th percentile predictions
        confidence_levels: Confidence levels to evaluate (default: [0.80, 0.90, 0.95])

    Returns:
        Dictionary with comprehensive metrics
    """
    results = {}

    # Standard metrics (on median prediction)
    results['standard_metrics'] = calculate_standard_metrics(y_true, y_pred_p50)

    # PICP at P10-P90 interval (80% confidence)
    results['picp_p10_p90'] = calculate_picp(y_true, y_pred_p10, y_pred_p90, confidence_level=0.80)

    # MPIW (prediction interval width)
    y_range = y_true.max() - y_true.min()
    results['mpiw'] = calculate_mpiw(y_pred_p10, y_pred_p90, y_true_range=y_range)
    results['mpiw_raw'] = calculate_mpiw(y_pred_p10, y_pred_p90)

    # Stratified metrics by hypoxia tier
    results['stratified'] = stratified_metrics_by_tier(
        y_true,
        y_pred_p50,
        y_pred_p10,
        y_pred_p90
    )

    # Calibration assessment
    calibration = calculate_calibration_curve(y_true, y_pred_p50, n_bins=10)
    results['calibration'] = {
        'bin_centers': calibration[0].tolist(),
        'empirical_coverage': calibration[1].tolist(),
        'predicted_coverage': calibration[2].tolist()
    }

    return results


def print_evaluation_report(results: Dict):
    """Print formatted evaluation report.

    Args:
        results: Results dictionary from evaluate_predictions()
    """
    print("\n" + "="*80)
    print("UNCERTAINTY QUANTIFICATION METRICS")
    print("="*80)

    # Standard metrics
    std_metrics = results['standard_metrics']
    print("\nStandard Metrics (on P50 predictions):")
    print(f"  MAE:         {std_metrics['mae']:.2f} µmol/L")
    print(f"  RMSE:        {std_metrics['rmse']:.2f} µmol/L")
    print(f"  MAPE:        {std_metrics['mape']:.2f}%")
    print(f"  N samples:   {std_metrics['n_samples']}")

    # Uncertainty metrics
    print("\nUncertainty Metrics:")
    print(f"  PICP (P10-P90): {results['picp_p10_p90']:.2%} (target: ~80%)")
    print(f"  MPIW (normalized): {results['mpiw']:.2%}")
    print(f"  MPIW (raw):        {results['mpiw_raw']:.2f} µmol/L")

    # Calibration assessment
    picp_diff = abs(results['picp_p10_p90'] - 0.80)
    if picp_diff < 0.05:
        calibration_status = "Well calibrated ✓"
    elif picp_diff < 0.10:
        calibration_status = "Moderately calibrated"
    else:
        calibration_status = "Poorly calibrated ✗"
    print(f"  Calibration: {calibration_status}")

    # Stratified metrics
    print("\nStratified Metrics by Hypoxia Tier:")
    print(f"  {'Tier':<20} {'N':<8} {'MAE':<12} {'RMSE':<12} {'PICP':<12}")
    print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

    for tier_name, tier_metrics in results['stratified'].items():
        n = tier_metrics['n_samples']
        mae = tier_metrics.get('mae', np.nan)
        rmse = tier_metrics.get('rmse', np.nan)
        picp = tier_metrics.get('picp', np.nan)

        tier_display = tier_name.replace('_', ' ').title()

        if n > 0:
            print(f"  {tier_display:<20} {n:<8} {mae:<12.2f} {rmse:<12.2f} {picp:<12.2%}")
        else:
            print(f"  {tier_display:<20} {n:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    print("\n" + "="*80)


def calculate_prediction_skill(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: Optional[np.ndarray] = None
) -> float:
    """Calculate prediction skill score relative to baseline.

    Skill score = 1 - (MSE_model / MSE_baseline)
    Positive values indicate improvement over baseline.

    Args:
        y_true: True values
        y_pred: Model predictions
        baseline_pred: Baseline predictions (if None, use climatology/mean)

    Returns:
        Skill score (higher is better, 0 = same as baseline, 1 = perfect)
    """
    # If no baseline provided, use climatology (mean)
    if baseline_pred is None:
        baseline_pred = np.full_like(y_true, y_true.mean())

    # Calculate MSE for both
    mse_model = mean_squared_error(y_true, y_pred)
    mse_baseline = mean_squared_error(y_true, baseline_pred)

    # Calculate skill
    if mse_baseline == 0:
        return np.nan

    skill = 1 - (mse_model / mse_baseline)

    return skill
