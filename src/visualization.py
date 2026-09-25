"""Visualization tools for hypoxia prediction model.

Implements comprehensive visualization including:
- Full dataset prediction vs actual overlay (like reference paper)
- Uncertainty bands and confidence intervals
- Hypoxia threshold markers
- Attention weight heatmaps
- Feature importance plots

Based on reference model visualization and dashboard requirements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# Hypoxia thresholds
THRESHOLDS = {
    "severe": 30.0,
    "hypoxic": 60.0,
    "watch": 120.0,
}


def plot_full_dataset_predictions(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    title: str = "Model Predictions vs Actual Values",
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create publication-quality plot of predictions vs actual values.

    Similar to Figure 4 in reference paper (AE-DeepAR):
    - Actual values as line/scatter
    - Predicted values overlaid
    - Uncertainty band (if provided)
    - Hypoxia threshold lines

    Args:
        dates: Date series for x-axis
        y_true: Actual oxygen values
        y_pred: Predicted values (P50)
        y_lower: Lower prediction bound (P10), optional
        y_upper: Upper prediction bound (P90), optional
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save figure (if None, don't save)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual values
    ax.plot(dates, y_true, 'o-', label='Actual', color='#2C3E50',
            linewidth=2, markersize=4, alpha=0.7, zorder=3)

    # Plot predictions
    ax.plot(dates, y_pred, 's-', label='Predicted', color='#E74C3C',
            linewidth=2, markersize=4, alpha=0.7, zorder=2)

    # Plot uncertainty band if provided
    if y_lower is not None and y_upper is not None:
        ax.fill_between(dates, y_lower, y_upper,
                        alpha=0.2, color='#3498DB',
                        label='Prediction Interval (P10-P90)',
                        zorder=1)

    # Add hypoxia threshold lines
    ax.axhline(y=THRESHOLDS['severe'], color='red', linestyle='--',
               linewidth=2, label=f"Severe Hypoxia ({THRESHOLDS['severe']} µmol/L)",
               alpha=0.7)

    ax.axhline(y=THRESHOLDS['hypoxic'], color='orange', linestyle='--',
               linewidth=2, label=f"Hypoxic ({THRESHOLDS['hypoxic']} µmol/L)",
               alpha=0.7)

    ax.axhline(y=THRESHOLDS['watch'], color='gold', linestyle='--',
               linewidth=1.5, label=f"Watch ({THRESHOLDS['watch']} µmol/L)",
               alpha=0.5)

    # Shade hypoxic zone
    ax.axhspan(0, THRESHOLDS['hypoxic'], alpha=0.1, color='red', zorder=0)

    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Oxygen (µmol/L)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_full_dataset_predictions_interactive(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    title: str = "Model Predictions vs Actual Values"
) -> go.Figure:
    """Create interactive Plotly version of full dataset predictions.

    Args:
        dates: Date series for x-axis
        y_true: Actual oxygen values
        y_pred: Predicted values (P50)
        y_lower: Lower prediction bound (P10), optional
        y_upper: Upper prediction bound (P90), optional
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Plot actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode='lines+markers',
        name='Actual',
        line=dict(color='#2C3E50', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>O₂: %{y:.1f} µmol/L<extra></extra>'
    ))

    # Plot predictions
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='#E74C3C', width=2, dash='dot'),
        marker=dict(size=4, symbol='square'),
        hovertemplate='<b>Predicted</b><br>Date: %{x}<br>O₂: %{y:.1f} µmol/L<extra></extra>'
    ))

    # Plot uncertainty band if provided
    if y_lower is not None and y_upper is not None:
        # Upper bound
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_upper,
            mode='lines',
            name='P90 (Upper bound)',
            line=dict(color='lightblue', width=1),
            showlegend=True,
            hovertemplate='<b>P90</b><br>Date: %{x}<br>O₂: %{y:.1f} µmol/L<extra></extra>'
        ))

        # Lower bound
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_lower,
            mode='lines',
            name='P10 (Lower bound)',
            line=dict(color='lightblue', width=1),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.3)',
            showlegend=True,
            hovertemplate='<b>P10</b><br>Date: %{x}<br>O₂: %{y:.1f} µmol/L<extra></extra>'
        ))

    # Add threshold lines
    x_range = [dates.min(), dates.max()]

    fig.add_trace(go.Scatter(
        x=x_range,
        y=[THRESHOLDS['severe'], THRESHOLDS['severe']],
        mode='lines',
        name=f"Severe ({THRESHOLDS['severe']} µmol/L)",
        line=dict(color='red', width=2, dash='dash'),
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=x_range,
        y=[THRESHOLDS['hypoxic'], THRESHOLDS['hypoxic']],
        mode='lines',
        name=f"Hypoxic ({THRESHOLDS['hypoxic']} µmol/L)",
        line=dict(color='orange', width=2, dash='dash'),
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=x_range,
        y=[THRESHOLDS['watch'], THRESHOLDS['watch']],
        mode='lines',
        name=f"Watch ({THRESHOLDS['watch']} µmol/L)",
        line=dict(color='gold', width=1.5, dash='dash'),
        hoverinfo='skip'
    ))

    # Shade hypoxic zone
    fig.add_hrect(
        y0=0, y1=THRESHOLDS['hypoxic'],
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Hypoxic Zone",
        annotation_position="top left"
    )

    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Oxygen (µmol/L)",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        height=500,
        template="plotly_white"
    )

    return fig


def plot_residuals(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (16, 4)
) -> plt.Figure:
    """Plot prediction residuals over time.

    Args:
        dates: Date series for x-axis
        y_true: Actual values
        y_pred: Predicted values
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=figsize)

    # Plot residuals
    ax.plot(dates, residuals, 'o-', color='#34495E', markersize=3, linewidth=1, alpha=0.6)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Residuals Over Time', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance Scores",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot feature importance scores from autoencoder.

    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_scores = importance_scores[sorted_indices]

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(sorted_features))
    colors = plt.cm.viridis(sorted_scores / sorted_scores.max())

    ax.barh(y_pos, sorted_scores, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=0.5)

    plt.tight_layout()

    return fig


def plot_stratified_performance(
    stratified_metrics: Dict[str, Dict],
    metric_name: str = 'mae',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot performance metrics stratified by hypoxia tier.

    Args:
        stratified_metrics: Dictionary from metrics.stratified_metrics_by_tier()
        metric_name: Metric to plot ('mae', 'rmse', 'picp')
        title: Plot title (if None, auto-generate)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    tiers = list(stratified_metrics.keys())
    values = [stratified_metrics[tier].get(metric_name, np.nan) for tier in tiers]
    n_samples = [stratified_metrics[tier].get('n_samples', 0) for tier in tiers]

    # Clean tier names
    tier_labels = [tier.replace('_', ' ').title() for tier in tiers]

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot
    colors = ['#E74C3C', '#E67E22', '#F39C12', '#27AE60']  # Red to green gradient
    bars = ax.bar(tier_labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add sample counts on bars
    for bar, n in zip(bars, n_samples):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={n}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Formatting
    metric_labels = {
        'mae': 'Mean Absolute Error (µmol/L)',
        'rmse': 'Root Mean Squared Error (µmol/L)',
        'picp': 'Prediction Interval Coverage Probability',
        'mape': 'Mean Absolute Percentage Error (%)'
    }

    ax.set_ylabel(metric_labels.get(metric_name, metric_name.upper()), fontsize=12, fontweight='bold')
    ax.set_xlabel('Hypoxia Tier', fontsize=12, fontweight='bold')

    if title is None:
        title = f"{metric_labels.get(metric_name, metric_name.upper())} by Hypoxia Tier"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    return fig


def create_model_diagnostic_report(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    stratified_metrics: Optional[Dict] = None,
    feature_importance: Optional[Tuple[List[str], np.ndarray]] = None,
    save_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """Create comprehensive diagnostic report with multiple visualizations.

    Args:
        dates: Date series
        y_true: Actual values
        y_pred: Predicted values
        y_lower: Lower prediction bound
        y_upper: Upper prediction bound
        stratified_metrics: Stratified performance metrics
        feature_importance: Tuple of (feature_names, importance_scores)
        save_dir: Directory to save figures (if None, don't save)

    Returns:
        Dictionary mapping figure names to matplotlib figures
    """
    figures = {}

    # 1. Full dataset predictions
    fig1 = plot_full_dataset_predictions(
        dates, y_true, y_pred, y_lower, y_upper,
        title="Model Predictions vs Actual Values (Full Dataset)",
        figsize=(16, 6)
    )
    figures['predictions'] = fig1

    if save_dir:
        fig1.savefig(f"{save_dir}/full_dataset_predictions.png", dpi=300, bbox_inches='tight')

    # 2. Residuals
    fig2 = plot_residuals(dates, y_true, y_pred, figsize=(16, 4))
    figures['residuals'] = fig2

    if save_dir:
        fig2.savefig(f"{save_dir}/residuals.png", dpi=300, bbox_inches='tight')

    # 3. Feature importance (if provided)
    if feature_importance is not None:
        feature_names, importance_scores = feature_importance
        fig3 = plot_feature_importance(
            feature_names, importance_scores,
            top_n=20,
            title="Top 20 Features by Importance (Autoencoder Scores)"
        )
        figures['feature_importance'] = fig3

        if save_dir:
            fig3.savefig(f"{save_dir}/feature_importance.png", dpi=300, bbox_inches='tight')

    # 4. Stratified performance (if provided)
    if stratified_metrics is not None:
        fig4 = plot_stratified_performance(
            stratified_metrics,
            metric_name='mae',
            title="MAE by Hypoxia Tier"
        )
        figures['stratified_mae'] = fig4

        if save_dir:
            fig4.savefig(f"{save_dir}/stratified_mae.png", dpi=300, bbox_inches='tight')

        if 'picp' in list(stratified_metrics.values())[0]:
            fig5 = plot_stratified_performance(
                stratified_metrics,
                metric_name='picp',
                title="PICP by Hypoxia Tier"
            )
            figures['stratified_picp'] = fig5

            if save_dir:
                fig5.savefig(f"{save_dir}/stratified_picp.png", dpi=300, bbox_inches='tight')

    print(f"\nGenerated {len(figures)} diagnostic figures")
    if save_dir:
        print(f"Figures saved to: {save_dir}/")

    return figures
