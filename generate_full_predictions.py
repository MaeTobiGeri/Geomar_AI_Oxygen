"""Generate full dataset predictions for visualization.

Creates a comprehensive prediction vs actual overlay plot similar to Figure 4
in the reference paper. This is a standalone script that saves results for
later visualization in the dashboard.

Gaps in the data (rows dropped because of missing values) are NOT drawn as
empty stretches. The x-axis is "compressed": every row gets the next free
position, so the segments sit directly next to each other. A thin dotted line
marks every place where time was skipped. Training and validation rows are
shaded in different colours.

Usage:
    python generate_full_predictions.py [--checkpoint-path PATH] [--output-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src import data_ingestion, pipeline, labeling, features, dataset, model, visualization, metrics
from pytorch_forecasting import TemporalFusionTransformer


try:
    from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

    safe_classes = [
        GroupNormalizer, NaNLabelEncoder, QuantileLoss, TimeSeriesDataSet,
        pd.DataFrame, pd.Series, pd.Index, pd.RangeIndex, pd.DatetimeIndex
    ]
    torch.serialization.add_safe_globals(safe_classes)
except (ImportError, AttributeError):
    pass


SPLIT_COLORS = {"training": "tab:blue", "validation": "tab:orange"}
PLOTLY_SPLIT_COLORS = {"training": "#1f77b4", "validation": "#ff7f0e"}


def find_gap_positions(dates: pd.Series, gap_threshold_days: int) -> list:
    """Row positions that are the FIRST row after a time gap larger than the threshold."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    diffs = d.diff().dt.days
    return diffs.index[diffs > gap_threshold_days].tolist()


def split_spans(split: pd.Series) -> list:
    """Contiguous runs of the same split label -> list of (start_pos, end_pos, label)."""
    labels = split.to_numpy()
    spans, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            spans.append((start, i - 1, labels[start]))
            start = i
    return spans


def date_ticks(dates: pd.Series, n_ticks: int = 14):
    """Evenly spaced row positions with their real dates as labels."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    pos = np.unique(np.linspace(0, len(d) - 1, min(n_ticks, len(d))).astype(int))
    return pos, [d.iloc[p].strftime("%Y-%m") for p in pos]


def decorate_axis(ax, df: pd.DataFrame, gap_positions: list):
    """Add split shading, gap markers and date tick labels to a matplotlib axis."""
    used = set()
    for start, end, label in split_spans(df["Split"]):
        ax.axvspan(
            start - 0.5, end + 0.5,
            color=SPLIT_COLORS.get(label, "gray"), alpha=0.10,
            label=None if label in used else f"{label} data",
        )
        used.add(label)

    for k, pos in enumerate(gap_positions):
        ax.axvline(
            pos - 0.5, color="gray", linestyle=":", linewidth=0.9,
            label="time gap (skipped)" if k == 0 else None,
        )

    ticks, labels = date_ticks(df["Date"])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_xlabel("Date (compressed axis: gaps removed, dotted lines mark skipped time)")


def plot_compressed_predictions(df, gap_positions, title, figsize=(20, 6)):
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=figsize)
    decorate_axis(ax, df, gap_positions)

    ax.fill_between(x, df["P10"], df["P90"], color="tab:red", alpha=0.25, label="P10-P90 interval")
    ax.plot(x, df["P50"], color="tab:red", linewidth=1.2, label="Prediction (P50)")
    ax.plot(x, df["Actual"], color="black", linewidth=1.2, label="Actual")

    ax.set_ylabel("Oxygen [µmol/l]")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_compressed_residuals(df, gap_positions, figsize=(20, 4)):
    x = np.arange(len(df))
    residuals = df["Actual"] - df["P50"]
    fig, ax = plt.subplots(figsize=figsize)
    decorate_axis(ax, df, gap_positions)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(x, residuals, color="tab:purple", linewidth=1.0, label="Residual (actual - P50)")
    ax.set_ylabel("Residual [µmol/l]")
    ax.set_title("Prediction residuals")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_compressed_predictions_interactive(df, gap_positions, title):
    import plotly.graph_objects as go

    x = np.arange(len(df))
    dates = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d").to_numpy()
    split = df["Split"].to_numpy()
    custom = np.stack([dates, split], axis=-1)
    hover = "%{customdata[0]} (%{customdata[1]})<br>%{y:.1f} µmol/l<extra>%{fullData.name}</extra>"

    fig = go.Figure()

    for start, end, label in split_spans(df["Split"]):
        fig.add_vrect(
            x0=start - 0.5, x1=end + 0.5,
            fillcolor=PLOTLY_SPLIT_COLORS.get(label, "gray"), opacity=0.10,
            layer="below", line_width=0,
            annotation_text=f"{label} data", annotation_position="top left",
        )

    for pos in gap_positions:
        fig.add_vline(x=pos - 0.5, line_dash="dot", line_color="gray", line_width=1)

    fig.add_trace(go.Scatter(x=x, y=df["P90"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=df["P10"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(214,39,40,0.25)",
                             name="P10-P90 interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=df["P50"], mode="lines", line=dict(color="crimson", width=1.5),
                             name="Prediction (P50)", customdata=custom, hovertemplate=hover))
    fig.add_trace(go.Scatter(x=x, y=df["Actual"], mode="lines", line=dict(color="black", width=1.5),
                             name="Actual", customdata=custom, hovertemplate=hover))

    ticks, labels = date_ticks(df["Date"])
    fig.update_layout(
        title=title,
        xaxis=dict(tickmode="array", tickvals=ticks, ticktext=labels, tickangle=-45,
                   title="Date (compressed axis: dotted lines mark skipped time)"),
        yaxis_title="Oxygen [µmol/l]",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def generate_walk_forward_predictions(
    tft_model: TemporalFusionTransformer,
    df: pd.DataFrame,
    training_dataset,
    encoder_length: int = 8
) -> pd.DataFrame:
    """Generate walk-forward predictions for entire dataset.

    Args:
        tft_model: Trained model
        df: Full dataframe
        training_dataset: TimeSeriesDataSet for predictions
        encoder_length: Encoder length

    Returns:
        DataFrame with predictions
    """
    predictions_list = []
    start_idx = encoder_length

    print(f"\nGenerating walk-forward predictions...")
    print(f"  Dataset size: {len(df)}")
    print(f"  Predictions to generate: {len(df) - start_idx}")

    tft_model.eval()

    for i in range(start_idx, len(df)):
        df_context = df.iloc[:i+1].copy()

        try:
            pred_dataset = training_dataset.__class__.from_dataset(
                training_dataset,
                df_context,
                predict=True,
                stop_randomization=True
            )

            with torch.no_grad():
                raw_predictions = tft_model.predict(
                    pred_dataset,
                    mode="raw",
                    return_x=False
                )

            if hasattr(raw_predictions, 'prediction'):
                predictions = raw_predictions.prediction
            else:
                predictions = raw_predictions if torch.is_tensor(raw_predictions) else raw_predictions['prediction']

            predictions_np = predictions.cpu().numpy()

            if len(predictions_np) > 0:
                last_pred = predictions_np[-1, 0, :]

                if len(last_pred) >= 3:
                    p10, p50, p90 = last_pred[0], last_pred[1], last_pred[2]
                else:
                    p10 = p50 = p90 = last_pred[0] if len(last_pred) > 0 else np.nan

                predictions_list.append({
                    'Date': df.iloc[i]['Date'],
                    'Actual': df.iloc[i]['O2_umol_L'],
                    'P10': p10,
                    'P50': p50,
                    'P90': p90
                })

        except Exception as e:
            print(f"  Warning: Failed at index {i}: {e}")
            continue

        if (i - start_idx) % 100 == 0:
            progress = (i - start_idx) / (len(df) - start_idx) * 100
            print(f"  Progress: {progress:.1f}% ({i - start_idx}/{len(df) - start_idx})")

    results_df = pd.DataFrame(predictions_list)
    print(f"\nSuccessfully generated {len(results_df)} predictions")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Generate full dataset predictions")

    parser.add_argument("--checkpoint-path", type=str, default="models/hypoxia_tft/best_model.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/full_predictions",
                        help="Directory to save results")
    parser.add_argument("--encoder-length", type=int, default=8,
                        help="Encoder length (must match training)")
    parser.add_argument("--decoder-length", type=int, default=4,
                        help="Decoder length (must match training)")
    parser.add_argument("--train-split-ratio", type=float, default=0.8,
                        help="Train/val split ratio (must match training)")
    parser.add_argument("--gap-threshold-days", type=int, default=14,
                        help="Time jump (days) between consecutive rows that counts as a gap")

    args = parser.parse_args()

    print("="*80)
    print("FULL DATASET PREDICTION GENERATION")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint_path}")
    print(f"Output directory: {args.output_dir}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nLoading and preparing data...")
    df_combined = data_ingestion.load_and_clean_boknis_data()
    df_weekly = pipeline.prepare_weekly_series(df_combined)
    df_25m = labeling.select_target_series(df_weekly)
    df_features = features.engineer_features(df_25m, df_weekly)
    df_labeled = labeling.label_hypoxia_risk(df_features)
    df_labeled = df_labeled.dropna().reset_index(drop=True)

    print(f"  Data range: {df_labeled['Date'].min()} to {df_labeled['Date'].max()}")
    print(f"  Total samples: {len(df_labeled)}")

    train_df, val_df = dataset.split_train_validation(
        df_labeled, train_ratio=args.train_split_ratio
    )
    val_dates = set(pd.to_datetime(val_df["Date"]))
    print(f"  Training rows: {len(train_df)} ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"  Validation rows: {len(val_df)} ({val_df['Date'].min()} to {val_df['Date'].max()})")

    print("\nCreating dataset...")
    training_dataset = dataset.create_training_dataset(
        df_labeled,
        encoder_length=args.encoder_length,
        decoder_length=args.decoder_length
    )

    print(f"\nLoading model from {args.checkpoint_path}...")
    try:
        tft_model = TemporalFusionTransformer.load_from_checkpoint(
            args.checkpoint_path,
            strict=False
        )
        tft_model.eval()
        print("  Model loaded successfully")
    except Exception as e:
        print(f"  ERROR: Failed to load model: {e}")
        sys.exit(1)

    predictions_df = generate_walk_forward_predictions(
        tft_model,
        df_labeled,
        training_dataset,
        encoder_length=args.encoder_length
    )

    if predictions_df.empty:
        print("ERROR: no predictions were generated.")
        sys.exit(1)

    predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
    predictions_df = predictions_df.sort_values("Date").reset_index(drop=True)
    predictions_df["Split"] = np.where(
        predictions_df["Date"].isin(val_dates), "validation", "training"
    )

    gap_positions = find_gap_positions(predictions_df["Date"], args.gap_threshold_days)
    print(f"\nTime gaps larger than {args.gap_threshold_days} days: {len(gap_positions)}")
    for pos in gap_positions:
        before = predictions_df["Date"].iloc[pos - 1].date()
        after = predictions_df["Date"].iloc[pos].date()
        print(f"  {before} -> {after}  ({(after - before).days} days skipped)")

    predictions_path = output_path / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to: {predictions_path}")

    print("\nCalculating metrics...")
    y_true = predictions_df['Actual'].values
    y_pred_p10 = predictions_df['P10'].values
    y_pred_p50 = predictions_df['P50'].values
    y_pred_p90 = predictions_df['P90'].values

    evaluation_results = metrics.evaluate_predictions(
        y_true, y_pred_p10, y_pred_p50, y_pred_p90
    )

    metrics.print_evaluation_report(evaluation_results)

    val_mask = (predictions_df["Split"] == "validation").values
    validation_results = None
    if val_mask.sum() > 0:
        try:
            validation_results = metrics.evaluate_predictions(
                y_true[val_mask], y_pred_p10[val_mask], y_pred_p50[val_mask], y_pred_p90[val_mask]
            )
            print("\n" + "-"*80)
            print("VALIDATION-ONLY METRICS")
            print("-"*80)
            metrics.print_evaluation_report(validation_results)
        except Exception as e:
            print(f"  Warning: validation-only metrics failed: {e}")

    metrics_path = output_path / "metrics.json"

    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(metrics_path, 'w') as f:
        json.dump(convert_to_native(evaluation_results), f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    if validation_results is not None:
        val_metrics_path = output_path / "metrics_validation.json"
        with open(val_metrics_path, 'w') as f:
            json.dump(convert_to_native(validation_results), f, indent=2)
        print(f"Validation-only metrics saved to: {val_metrics_path}")

    print("\nGenerating visualizations...")

    fig1 = plot_compressed_predictions(
        predictions_df,
        gap_positions,
        title="Model Predictions vs Actual Oxygen Levels (Full Dataset, gaps removed)",
        figsize=(20, 6)
    )
    fig1_path = output_path / "full_dataset_predictions.png"
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig1_path}")
    plt.close(fig1)

    fig2 = plot_compressed_predictions_interactive(
        predictions_df,
        gap_positions,
        title="Model Predictions vs Actual Oxygen Levels (Interactive, gaps removed)"
    )
    fig2_path = output_path / "full_dataset_predictions_interactive.html"
    fig2.write_html(fig2_path)
    print(f"  Saved: {fig2_path}")

    fig3 = plot_compressed_residuals(predictions_df, gap_positions, figsize=(20, 4))
    fig3_path = output_path / "residuals.png"
    fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig3_path}")
    plt.close(fig3)

    if evaluation_results.get('stratified'):
        fig4 = visualization.plot_stratified_performance(
            evaluation_results['stratified'],
            metric_name='mae',
            title="MAE by Hypoxia Tier"
        )
        fig4_path = output_path / "stratified_mae.png"
        fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fig4_path}")
        plt.close(fig4)

        fig5 = visualization.plot_stratified_performance(
            evaluation_results['stratified'],
            metric_name='picp',
            title="PICP by Hypoxia Tier"
        )
        fig5_path = output_path / "stratified_picp.png"
        fig5.savefig(fig5_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fig5_path}")
        plt.close(fig5)

    summary = {
        'generated_at': datetime.now().isoformat(),
        'checkpoint_path': args.checkpoint_path,
        'num_predictions': len(predictions_df),
        'date_range': {
            'start': str(predictions_df['Date'].min()),
            'end': str(predictions_df['Date'].max())
        },
        'split': {
            'train_split_ratio': args.train_split_ratio,
            'training_rows': int((predictions_df['Split'] == 'training').sum()),
            'validation_rows': int((predictions_df['Split'] == 'validation').sum()),
        },
        'gaps': {
            'threshold_days': args.gap_threshold_days,
            'count': len(gap_positions),
        },
        'metrics_summary': {
            'mae': evaluation_results['standard_metrics']['mae'],
            'rmse': evaluation_results['standard_metrics']['rmse'],
            'mape': evaluation_results['standard_metrics']['mape'],
            'picp': evaluation_results['picp_p10_p90']
        }
    }

    summary_path = output_path / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_path}")
    print(f"\nKey files:")
    print(f"  - predictions.csv: Raw prediction data (with Split column)")
    print(f"  - full_dataset_predictions.png: Main visualization")
    print(f"  - full_dataset_predictions_interactive.html: Interactive plot")
    print(f"  - metrics.json / metrics_validation.json: Detailed metrics")
    print(f"  - summary.json: Summary statistics")


if __name__ == "__main__":
    main()