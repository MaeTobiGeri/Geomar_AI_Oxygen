"""Generate full dataset predictions for visualization.

Creates a comprehensive prediction vs actual overlay plot similar to Figure 4
in the reference paper. This is a standalone script that saves results for
later visualization in the dashboard.

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


# Register safe globals for checkpoint loading
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
        # Get data up to current point
        df_context = df.iloc[:i+1].copy()

        try:
            # Create prediction dataset
            pred_dataset = training_dataset.__class__.from_dataset(
                training_dataset,
                df_context,
                predict=True,
                stop_randomization=True
            )

            # Get prediction
            with torch.no_grad():
                raw_predictions = tft_model.predict(
                    pred_dataset,
                    mode="raw",
                    return_x=False
                )

            # Extract predictions
            if hasattr(raw_predictions, 'prediction'):
                predictions = raw_predictions.prediction
            else:
                predictions = raw_predictions if torch.is_tensor(raw_predictions) else raw_predictions['prediction']

            predictions_np = predictions.cpu().numpy()

            # Get the first prediction step
            if len(predictions_np) > 0:
                last_pred = predictions_np[-1, 0, :]  # Last sample, first time step, all quantiles

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
            # Skip if prediction fails
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

    args = parser.parse_args()

    print("="*80)
    print("FULL DATASET PREDICTION GENERATION")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint_path}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and prepare data (same as training)
    print("\nLoading and preparing data...")
    df_combined = data_ingestion.load_and_clean_boknis_data()
    df_weekly = pipeline.prepare_weekly_series(df_combined)
    df_25m = labeling.select_target_series(df_weekly)
    df_features = features.engineer_features(df_25m, df_weekly)
    df_labeled = labeling.label_hypoxia_risk(df_features)
    df_labeled = df_labeled.dropna().reset_index(drop=True)

    print(f"  Data range: {df_labeled['Date'].min()} to {df_labeled['Date'].max()}")
    print(f"  Total samples: {len(df_labeled)}")

    # Create dataset
    print("\nCreating dataset...")
    training_dataset = dataset.create_training_dataset(
        df_labeled,
        encoder_length=args.encoder_length,
        decoder_length=args.decoder_length
    )

    # Load model
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

    # Generate predictions
    predictions_df = generate_walk_forward_predictions(
        tft_model,
        df_labeled,
        training_dataset,
        encoder_length=args.encoder_length
    )

    # Save predictions
    predictions_path = output_path / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to: {predictions_path}")

    # Calculate metrics
    print("\nCalculating metrics...")
    y_true = predictions_df['Actual'].values
    y_pred_p10 = predictions_df['P10'].values
    y_pred_p50 = predictions_df['P50'].values
    y_pred_p90 = predictions_df['P90'].values

    evaluation_results = metrics.evaluate_predictions(
        y_true, y_pred_p10, y_pred_p50, y_pred_p90
    )

    # Print metrics
    metrics.print_evaluation_report(evaluation_results)

    # Save metrics (convert numpy types to native Python types)
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

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Full dataset prediction plot (matplotlib)
    fig1 = visualization.plot_full_dataset_predictions(
        predictions_df['Date'],
        y_true,
        y_pred_p50,
        y_pred_p10,
        y_pred_p90,
        title="Model Predictions vs Actual Oxygen Levels (Full Dataset)",
        figsize=(20, 6)
    )
    fig1_path = output_path / "full_dataset_predictions.png"
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig1_path}")
    plt.close(fig1)

    # 2. Interactive plot (Plotly) - save as HTML
    fig2 = visualization.plot_full_dataset_predictions_interactive(
        predictions_df['Date'],
        y_true,
        y_pred_p50,
        y_pred_p10,
        y_pred_p90,
        title="Model Predictions vs Actual Oxygen Levels (Interactive)"
    )
    fig2_path = output_path / "full_dataset_predictions_interactive.html"
    fig2.write_html(fig2_path)
    print(f"  Saved: {fig2_path}")

    # 3. Residuals plot
    fig3 = visualization.plot_residuals(
        predictions_df['Date'],
        y_true,
        y_pred_p50,
        figsize=(20, 4)
    )
    fig3_path = output_path / "residuals.png"
    fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig3_path}")
    plt.close(fig3)

    # 4. Stratified performance
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

    # Save summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'checkpoint_path': args.checkpoint_path,
        'num_predictions': len(predictions_df),
        'date_range': {
            'start': str(predictions_df['Date'].min()),
            'end': str(predictions_df['Date'].max())
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
    print(f"  - predictions.csv: Raw prediction data")
    print(f"  - full_dataset_predictions.png: Main visualization")
    print(f"  - full_dataset_predictions_interactive.html: Interactive plot")
    print(f"  - metrics.json: Detailed metrics")
    print(f"  - summary.json: Summary statistics")


if __name__ == "__main__":
    main()
