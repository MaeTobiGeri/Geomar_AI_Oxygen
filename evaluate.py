"""Evaluation suite for weighted hypoxia prediction model.

Implements SPEC.md §9 evaluation protocol:
1. Weighted aggregate metrics (correlation, RMSE) using sample weights
2. Threshold-sweep classification metrics (ROC-AUC, F1, precision/recall)
3. Event-study plots for held-out hypoxic episodes
4. Persistence baseline comparison on tail/event metrics

Usage:
    python evaluate.py --checkpoint models/hypoxia_tft/best_model.ckpt
    python evaluate.py --checkpoint models/hypoxia_tft/best_model.ckpt --output-dir evaluation_results/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, f1_score,
    precision_score, recall_score, accuracy_score
)
from scipy.stats import pearsonr
from pytorch_forecasting import TemporalFusionTransformer
import lightning.pytorch as pl

from src import data_ingestion, pipeline, labeling, features, dataset

# Suppress warnings
warnings.filterwarnings('ignore')


def weighted_correlation(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted Pearson correlation coefficient.

    Args:
        y_true: True values
        y_pred: Predicted values
        weights: Sample weights

    Returns:
        Weighted correlation coefficient
    """
    # Weighted means
    w_sum = weights.sum()
    mean_true = (y_true * weights).sum() / w_sum
    mean_pred = (y_pred * weights).sum() / w_sum

    # Weighted covariance and standard deviations
    cov = ((y_true - mean_true) * (y_pred - mean_pred) * weights).sum() / w_sum
    std_true = np.sqrt(((y_true - mean_true)**2 * weights).sum() / w_sum)
    std_pred = np.sqrt(((y_pred - mean_pred)**2 * weights).sum() / w_sum)

    if std_true == 0 or std_pred == 0:
        return 0.0

    return cov / (std_true * std_pred)


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted RMSE.

    Args:
        y_true: True values
        y_pred: Predicted values
        weights: Sample weights

    Returns:
        Weighted RMSE
    """
    mse = ((y_true - y_pred)**2 * weights).sum() / weights.sum()
    return np.sqrt(mse)


def compute_aggregate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    split_name: str
) -> Dict[str, float]:
    """Compute weighted aggregate metrics for a split.

    Args:
        y_true: True oxygen values
        y_pred: Predicted oxygen values (median/P50)
        weights: Sample weights
        split_name: Name of split (train/val/test)

    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*80}")
    print(f"Weighted Aggregate Metrics - {split_name.upper()}")
    print(f"{'='*80}")

    w_corr = weighted_correlation(y_true, y_pred, weights)
    w_rmse = weighted_rmse(y_true, y_pred, weights)

    # Also compute unweighted for comparison
    unw_corr = pearsonr(y_true, y_pred)[0]
    unw_rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    metrics = {
        f"{split_name}_weighted_corr": w_corr,
        f"{split_name}_weighted_rmse": w_rmse,
        f"{split_name}_unweighted_corr": unw_corr,
        f"{split_name}_unweighted_rmse": unw_rmse,
    }

    print(f"Weighted Correlation: {w_corr:.4f}")
    print(f"Weighted RMSE: {w_rmse:.2f} µmol/L")
    print(f"Unweighted Correlation: {unw_corr:.4f} (for comparison)")
    print(f"Unweighted RMSE: {unw_rmse:.2f} µmol/L (for comparison)")

    return metrics


def threshold_sweep_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 60.0,
    output_dir: Path = None
) -> Dict[str, float]:
    """Evaluate regression output as classifier across threshold sweep.

    Per SPEC.md §9: treat predictions as probabilistic classifier for hypoxia.

    Args:
        y_true: True oxygen values
        y_pred: Predicted oxygen values
        threshold: Hypoxia threshold (default 60 µmol/L)
        output_dir: Directory to save plots

    Returns:
        Dictionary of classification metrics
    """
    print(f"\n{'='*80}")
    print(f"Threshold-Sweep Classification Metrics")
    print(f"{'='*80}")

    # Binary labels: 1 if hypoxic (below threshold), 0 otherwise
    y_true_binary = (y_true < threshold).astype(int)

    # Use prediction as "probability score" for ROC
    # Lower oxygen = higher probability of hypoxia
    # So use (threshold - y_pred) as the score
    y_score = threshold - y_pred

    # ROC curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)

    # Precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, y_score)

    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else pr_thresholds[-1]

    # Compute metrics at optimal threshold
    y_pred_binary = (y_score >= optimal_threshold).astype(int)

    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    metrics = {
        "roc_auc": roc_auc,
        "optimal_f1": f1,
        "optimal_precision": prec,
        "optimal_recall": rec,
        "optimal_accuracy": acc,
        "hypoxia_prevalence": y_true_binary.mean(),
    }

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Optimal F1: {f1:.4f}")
    print(f"Optimal Precision: {prec:.4f}")
    print(f"Optimal Recall: {rec:.4f}")
    print(f"Optimal Accuracy: {acc:.4f}")
    print(f"Hypoxia Prevalence: {y_true_binary.mean():.2%}")
    print(f"\nCalibration (SPEC.md §9): F1 ∈ [0.2, 0.4] and AUC ∈ [0.7, 0.85] expected")

    # Plot ROC curve
    if output_dir:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ROC curve
        ax1.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve - Hypoxia Classification')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Precision-Recall curve
        ax2.plot(recall, precision, linewidth=2)
        ax2.axhline(y=y_true_binary.mean(), color='k', linestyle='--',
                    label=f'Baseline (prevalence={y_true_binary.mean():.2%})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "roc_pr_curves.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nSaved ROC/PR curves to {output_dir / 'roc_pr_curves.png'}")

    return metrics


def generate_predictions(
    model: TemporalFusionTransformer,
    dataloader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate predictions from trained model.

    Args:
        model: Trained TFT model
        dataloader: Data loader
        device: Device to run on

    Returns:
        Tuple of (y_true, y_pred_p50, y_pred_p10, y_pred_p90, weights)
    """
    model.eval()
    model.to(device)

    all_y_true = []
    all_y_pred_p50 = []
    all_y_pred_p10 = []
    all_y_pred_p90 = []
    all_weights = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            x, y = batch
            for key in x:
                if isinstance(x[key], torch.Tensor):
                    x[key] = x[key].to(device)
            y = (y[0].to(device), y[1].to(device))

            # Get predictions (returns quantiles: [P10, P50, P90])
            y_pred = model(x)

            # Extract predictions for first step ahead (index 0 in time dimension)
            # Shape: [batch, time_steps, quantiles]
            y_pred_p10 = y_pred[:, 0, 0].cpu().numpy()  # P10
            y_pred_p50 = y_pred[:, 0, 1].cpu().numpy()  # P50 (median)
            y_pred_p90 = y_pred[:, 0, 2].cpu().numpy()  # P90

            # True values and weights
            y_true = y[0][:, 0, 0].cpu().numpy()  # First target, first step

            # Extract weights if available
            if hasattr(batch, 'weight'):
                weights = batch.weight.cpu().numpy()
            else:
                # Extract from x if stored there
                if 'weight' in x:
                    weights = x['weight'][:, 0].cpu().numpy()
                else:
                    weights = np.ones_like(y_true)

            all_y_true.append(y_true)
            all_y_pred_p50.append(y_pred_p50)
            all_y_pred_p10.append(y_pred_p10)
            all_y_pred_p90.append(y_pred_p90)
            all_weights.append(weights)

    return (
        np.concatenate(all_y_true),
        np.concatenate(all_y_pred_p50),
        np.concatenate(all_y_pred_p10),
        np.concatenate(all_y_pred_p90),
        np.concatenate(all_weights),
    )


def persistence_baseline(y_true: np.ndarray) -> np.ndarray:
    """Simple persistence forecast: predict = last known value.

    Per SPEC.md §9: persistence can look good in aggregate due to autocorrelation
    but fails at anticipating onsets.

    Args:
        y_true: True oxygen time series

    Returns:
        Persistence predictions (shifted by 1 timestep)
    """
    # Shift forward by 1: predict[t] = actual[t-1]
    y_persist = np.roll(y_true, 1)
    y_persist[0] = y_true[0]  # No prediction for first timestep
    return y_persist


def plot_event_studies(
    df_featured: pd.DataFrame,
    episodes: pd.DataFrame,
    model_preds: Dict[str, np.ndarray],
    output_dir: Path,
    max_episodes: int = 6
):
    """Plot event studies for held-out hypoxic episodes.

    Per SPEC.md §9: show observation vs weighted model for historical episodes
    to demonstrate tail prediction improvement.

    Args:
        df_featured: Full featured dataframe with dates
        episodes: DataFrame of identified hypoxic episodes
        model_preds: Dictionary with 'dates', 'y_true', 'y_pred', 'y_persist'
        output_dir: Output directory for plots
        max_episodes: Maximum number of episodes to plot
    """
    print(f"\n{'='*80}")
    print(f"Event Study Plots")
    print(f"{'='*80}")

    # Select longest episodes for visualization
    episodes_sorted = episodes.sort_values('duration_weeks', ascending=False)
    episodes_to_plot = episodes_sorted.head(max_episodes)

    print(f"Plotting {len(episodes_to_plot)} longest hypoxic episodes")

    for idx, episode in episodes_to_plot.iterrows():
        start = episode['start_date']
        end = episode['end_date']
        duration = episode['duration_weeks']
        min_o2 = episode['min_o2']

        # Add context: 8 weeks before and 4 weeks after
        context_start = start - pd.Timedelta(weeks=8)
        context_end = end + pd.Timedelta(weeks=4)

        # Find indices in prediction arrays
        dates = model_preds['dates']
        mask = (dates >= context_start) & (dates <= context_end)

        if mask.sum() < 5:  # Skip if too few points
            continue

        dates_ep = dates[mask]
        y_true_ep = model_preds['y_true'][mask]
        y_pred_ep = model_preds['y_pred'][mask]
        y_persist_ep = model_preds['y_persist'][mask]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(dates_ep, y_true_ep, 'ko-', label='Observed', linewidth=2, markersize=5)
        ax.plot(dates_ep, y_pred_ep, 'b^-', label='TFT Model', linewidth=1.5, markersize=4, alpha=0.8)
        ax.plot(dates_ep, y_persist_ep, 'g--', label='Persistence Baseline', linewidth=1.5, alpha=0.6)

        # Threshold lines
        ax.axhline(y=60, color='orange', linestyle='--', linewidth=2, label='Hypoxic (60 µmol/L)', alpha=0.7)
        ax.axhline(y=30, color='red', linestyle='--', linewidth=2, label='Severe (30 µmol/L)', alpha=0.7)

        # Shade hypoxic zone
        ax.fill_between(dates_ep, 0, 60, color='orange', alpha=0.1)

        # Shade actual episode period
        ax.axvspan(start, end, color='red', alpha=0.15, label='Episode Period')

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Oxygen (µmol/L)', fontsize=11)
        ax.set_title(
            f'Episode {idx+1}: {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}\n'
            f'Duration: {duration} weeks, Min O₂: {min_o2:.1f} µmol/L',
            fontsize=12
        )
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"event_study_{idx+1}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(episodes_to_plot)} event study plots to {output_dir}/")


def compare_persistence_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persist: np.ndarray,
    weights: np.ndarray,
    threshold: float = 60.0
) -> Dict[str, float]:
    """Compare model vs persistence baseline on tail metrics.

    Per SPEC.md §9: persistence can look good in aggregate but fails on tail.

    Args:
        y_true: True values
        y_pred: Model predictions
        y_persist: Persistence predictions
        weights: Sample weights
        threshold: Hypoxia threshold

    Returns:
        Dictionary comparing model vs persistence
    """
    print(f"\n{'='*80}")
    print(f"Persistence Baseline Comparison")
    print(f"{'='*80}")

    # Aggregate metrics
    model_rmse = weighted_rmse(y_true, y_pred, weights)
    persist_rmse = weighted_rmse(y_true, y_persist, weights)

    # Tail metrics: focus on hypoxic samples
    hypoxic_mask = y_true < threshold
    if hypoxic_mask.sum() > 0:
        hypoxic_rmse_model = weighted_rmse(
            y_true[hypoxic_mask],
            y_pred[hypoxic_mask],
            weights[hypoxic_mask]
        )
        hypoxic_rmse_persist = weighted_rmse(
            y_true[hypoxic_mask],
            y_persist[hypoxic_mask],
            weights[hypoxic_mask]
        )
    else:
        hypoxic_rmse_model = np.nan
        hypoxic_rmse_persist = np.nan

    metrics = {
        "model_rmse_all": model_rmse,
        "persistence_rmse_all": persist_rmse,
        "model_rmse_hypoxic": hypoxic_rmse_model,
        "persistence_rmse_hypoxic": hypoxic_rmse_persist,
        "hypoxic_improvement": (hypoxic_rmse_persist - hypoxic_rmse_model) / hypoxic_rmse_persist * 100
        if not np.isnan(hypoxic_rmse_persist) and hypoxic_rmse_persist > 0 else 0.0
    }

    print(f"All Samples:")
    print(f"  Model RMSE: {model_rmse:.2f} µmol/L")
    print(f"  Persistence RMSE: {persist_rmse:.2f} µmol/L")
    print(f"\nHypoxic Samples Only (< {threshold} µmol/L):")
    print(f"  Model RMSE: {hypoxic_rmse_model:.2f} µmol/L")
    print(f"  Persistence RMSE: {hypoxic_rmse_persist:.2f} µmol/L")
    print(f"  Improvement: {metrics['hypoxic_improvement']:.1f}%")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate hypoxia prediction model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation outputs"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Hypoxia threshold in µmol/L (default: 60.0)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("HYPOXIA PREDICTION MODEL EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Hypoxia threshold: {args.threshold} µmol/L")

    # Run full pipeline to get data (same as train.py Phases 2-6)
    print("\n" + "-"*80)
    print("Phase 2: Data Ingestion")
    print("-"*80)
    df_combined = data_ingestion.load_and_clean_boknis_data()
    print(f"Ocean + weather data loaded: {len(df_combined)} rows")

    print("\n" + "-"*80)
    print("Phase 3: Weekly Resampling & Imputation")
    print("-"*80)
    df_weekly = pipeline.prepare_weekly_series(df_combined)
    print(f"Weekly series: {len(df_weekly)} rows")

    print("\n" + "-"*80)
    print("Phase 4: Hypoxia Labeling")
    print("-"*80)
    df_25m = labeling.select_target_series(df_weekly)
    print(f"25m target series: {len(df_25m)} rows")

    print("\n" + "-"*80)
    print("Phase 5: Feature Engineering")
    print("-"*80)
    df_features = features.engineer_features(df_25m, df_weekly)
    print(f"Engineered features: {len(df_features.columns)} columns")

    # Label hypoxia risk
    df_labeled = labeling.label_hypoxia_risk(df_features)

    # Drop rows with NaN values (TimeSeriesDataSet requirement)
    rows_before = len(df_labeled)
    df_labeled = df_labeled.dropna().reset_index(drop=True)
    rows_after = len(df_labeled)
    print(f"Dropped {rows_before - rows_after} rows with NaN values")
    print(f"Complete rows: {rows_after}")

    # Identify hypoxic episodes for event studies
    episodes = labeling.identify_hypoxic_episodes(df_labeled)
    print(f"Identified {len(episodes)} hypoxic episodes")

    print("\n" + "-"*80)
    print("Phase 6: Dataset Construction")
    print("-"*80)

    # Split train/val chronologically (same as train.py)
    train_df, val_df = dataset.split_train_validation(df_labeled, train_ratio=0.8)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Create dataloaders
    train_dl, val_dl, training_dataset = dataset.create_dataloaders(
        train_df,
        val_df,
        batch_size=64,
        encoder_length=8,
        decoder_length=4,
        num_workers=0,
    )

    # Load model
    print("\n" + "-"*80)
    print("Loading Model")
    print("-"*80)

    # Load with weights_only=False since we trust our checkpoint
    model = TemporalFusionTransformer.load_from_checkpoint(
        args.checkpoint,
        strict=False  # Allow missing/extra keys for compatibility
    )

    print(f"Model loaded from {args.checkpoint}")

    # Generate predictions
    print("\n" + "-"*80)
    print("Generating Predictions")
    print("-"*80)

    train_y_true, train_y_pred, train_p10, train_p90, train_weights = generate_predictions(model, train_dl)
    val_y_true, val_y_pred, val_p10, val_p90, val_weights = generate_predictions(model, val_dl)

    print(f"Train samples: {len(train_y_true)}")
    print(f"Val samples: {len(val_y_true)}")

    # Persistence baselines
    train_y_persist = persistence_baseline(train_y_true)
    val_y_persist = persistence_baseline(val_y_true)

    # Evaluate
    all_metrics = {}

    # 1. Weighted aggregate metrics
    train_metrics = compute_aggregate_metrics(train_y_true, train_y_pred, train_weights, "train")
    val_metrics = compute_aggregate_metrics(val_y_true, val_y_pred, val_weights, "val")
    all_metrics.update(train_metrics)
    all_metrics.update(val_metrics)

    # 2. Threshold-sweep classification metrics (on validation set)
    clf_metrics = threshold_sweep_analysis(val_y_true, val_y_pred, threshold=args.threshold, output_dir=output_dir)
    all_metrics.update(clf_metrics)

    # 3. Persistence baseline comparison
    persist_metrics = compare_persistence_baseline(
        val_y_true, val_y_pred, val_y_persist, val_weights, threshold=args.threshold
    )
    all_metrics.update(persist_metrics)

    # 4. Event study plots
    # Combine train + val for full timeline
    full_dates = pd.concat([train_df['Date'].reset_index(drop=True), val_df['Date'].reset_index(drop=True)])
    model_preds = {
        'dates': full_dates.values,
        'y_true': np.concatenate([train_y_true, val_y_true]),
        'y_pred': np.concatenate([train_y_pred, val_y_pred]),
        'y_persist': np.concatenate([train_y_persist, val_y_persist]),
    }

    plot_event_studies(df_labeled, episodes, model_preds, output_dir, max_episodes=6)

    # Save metrics
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
