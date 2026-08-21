"""Verify that weighted loss mechanism is working correctly.

Per BUILD_PLAN.md Phase 7: "Confirm the weighted loss is actually being applied — e.g.
temporarily set all sample_weight values to a single large multiplier on hypoxic rows and
confirm training loss/gradients respond, before doing a full training run."

This test script:
1. Creates two datasets: one with extreme weights on hypoxic rows, one with uniform weights
2. Trains both for a few epochs
3. Compares training loss progression to confirm the weighted version prioritizes hypoxic rows

Usage:
    python verify_weighted_loss.py
"""

import pandas as pd
import numpy as np
import lightning.pytorch as pl
from pathlib import Path

from src import data_ingestion, pipeline, labeling, features, dataset, model


def create_extreme_weighted_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create a dataset with extreme weights on hypoxic rows for testing.

    Sets hypoxic rows (O2 < 60 µmol/L) to weight=100.0, normoxic to weight=1.0.
    This exaggerated difference should be clearly visible in training loss if weights work.

    Args:
        df: DataFrame with O2_umol_L and sample_weight columns

    Returns:
        DataFrame with modified sample_weight column
    """
    df = df.copy()
    hypoxic_mask = df["O2_umol_L"] < labeling.THRESHOLDS["hypoxic"]
    df.loc[hypoxic_mask, "sample_weight"] = 100.0
    df.loc[~hypoxic_mask, "sample_weight"] = 1.0
    return df


def create_uniform_weighted_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create a dataset with uniform weights (all 1.0) for baseline comparison.

    Args:
        df: DataFrame with sample_weight column

    Returns:
        DataFrame with all weights set to 1.0
    """
    df = df.copy()
    df["sample_weight"] = 1.0
    return df


def train_and_measure(df: pd.DataFrame, name: str, max_epochs: int = 5) -> dict:
    """Train a model on the given dataset and measure loss progression.

    Args:
        df: Labeled dataset with features and weights
        name: Name for this experiment (for logging)
        max_epochs: Number of epochs to train

    Returns:
        Dictionary with training metrics
    """
    print(f"\n{'='*80}")
    print(f"Training {name}")
    print(f"{'='*80}")

    # Train/val split
    train_df, val_df = dataset.split_train_validation(df, train_ratio=0.8)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # Check weight distribution
    hypoxic_mask = train_df["O2_umol_L"] < labeling.THRESHOLDS["hypoxic"]
    mean_weight_hypoxic = train_df.loc[hypoxic_mask, "sample_weight"].mean()
    mean_weight_normoxic = train_df.loc[~hypoxic_mask, "sample_weight"].mean()
    weight_ratio = mean_weight_hypoxic / mean_weight_normoxic if mean_weight_normoxic > 0 else 0

    print(f"Weight distribution:")
    print(f"  Hypoxic samples: mean weight = {mean_weight_hypoxic:.1f}")
    print(f"  Normoxic samples: mean weight = {mean_weight_normoxic:.1f}")
    print(f"  Ratio (hypoxic/normoxic): {weight_ratio:.1f}x")

    # Create dataloaders
    train_dl, val_dl, training_dataset = dataset.create_dataloaders(
        train_df,
        val_df,
        batch_size=64,
        encoder_length=8,
        decoder_length=4,
        num_workers=0,
    )

    # Create model
    tft = model.create_tft_model(training_dataset)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )

    # Train
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Get final validation loss
    val_results = trainer.validate(tft, dataloaders=val_dl, verbose=False)
    final_val_loss = val_results[0]["val_loss"]

    print(f"\nFinal validation loss: {final_val_loss:.4f}")

    return {
        "name": name,
        "weight_ratio": weight_ratio,
        "final_val_loss": final_val_loss,
        "hypoxic_samples": hypoxic_mask.sum(),
        "normoxic_samples": (~hypoxic_mask).sum(),
    }


def main():
    print("="*80)
    print("WEIGHTED LOSS VERIFICATION TEST")
    print("="*80)
    print("\nThis test verifies that the weight parameter actually affects training.")
    print("We'll train two models:")
    print("  1. EXTREME WEIGHTED: hypoxic rows weight=100.0, normoxic weight=1.0")
    print("  2. UNIFORM WEIGHTED: all rows weight=1.0 (baseline)")
    print("\nIf weights work correctly, the extreme weighted model should show")
    print("different loss behavior, prioritizing hypoxic sample accuracy.")

    # Load data (Phases 2-5)
    print("\nLoading data...")
    df_combined = data_ingestion.load_and_clean_boknis_data()

    # Weekly resampling
    df_weekly = pipeline.prepare_weekly_series(df_combined)

    # Select 25m target series
    df_25m = labeling.select_target_series(df_weekly)

    # Feature engineering
    df_features = features.engineer_features(df_25m, df_weekly)

    # Labeling
    df_labeled = labeling.label_hypoxia_risk(df_features)

    print(f"Total samples: {len(df_labeled)}")
    print(f"Hypoxic samples: {(df_labeled['O2_umol_L'] < labeling.THRESHOLDS['hypoxic']).sum()}")

    # Create two versions: extreme weighted and uniform weighted
    df_extreme = create_extreme_weighted_dataset(df_labeled)
    df_uniform = create_uniform_weighted_dataset(df_labeled)

    # Train both models
    results_extreme = train_and_measure(df_extreme, "EXTREME WEIGHTED (100x)", max_epochs=5)
    results_uniform = train_and_measure(df_uniform, "UNIFORM WEIGHTED (1x)", max_epochs=5)

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print(f"\nExtreme weighted model:")
    print(f"  Weight ratio: {results_extreme['weight_ratio']:.1f}x")
    print(f"  Final val loss: {results_extreme['final_val_loss']:.4f}")

    print(f"\nUniform weighted model:")
    print(f"  Weight ratio: {results_uniform['weight_ratio']:.1f}x")
    print(f"  Final val loss: {results_uniform['final_val_loss']:.4f}")

    # Sanity check: models should have different loss values if weights work
    loss_diff = abs(results_extreme['final_val_loss'] - results_uniform['final_val_loss'])
    loss_diff_pct = (loss_diff / results_uniform['final_val_loss']) * 100

    print(f"\nLoss difference: {loss_diff:.4f} ({loss_diff_pct:.1f}%)")

    if loss_diff_pct > 5:
        print("\n✓ VERIFICATION PASSED: Weighted loss is working!")
        print("  The extreme weighted model shows measurably different loss behavior,")
        print("  confirming that the weight parameter affects training.")
    else:
        print("\n✗ VERIFICATION FAILED: Weighted loss may not be working!")
        print("  The extreme weighted model shows very similar loss to uniform weights,")
        print("  suggesting the weight parameter is not being applied correctly.")
        print("\n  Check:")
        print("    1. TimeSeriesDataSet weight= parameter is set correctly")
        print("    2. QuantileLoss is using the weight in loss computation")
        print("    3. pytorch-forecasting version matches SPEC.md §6.5 (v1.7.0)")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    main()
