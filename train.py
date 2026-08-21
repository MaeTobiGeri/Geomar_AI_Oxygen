"""Training script for weighted hypoxia prediction model.

Implements BUILD_PLAN.md Phase 8: end-to-end training pipeline from data ingestion through
model training, with fixed checkpoint paths and reproducibility logging.

Runs Phases 2-7:
  - Phase 2: Data ingestion (ocean + weather)
  - Phase 3: Weekly resampling and imputation
  - Phase 4: Hypoxia labeling and sample weighting
  - Phase 5: Feature engineering
  - Phase 6: Dataset construction with chronological split
  - Phase 7: Model training with early stopping

Per SPEC.md §8 and §11, saves checkpoints to a FIXED path (not version_N) and logs
features/weights used alongside the checkpoint for reproducibility.

Usage:
    # Train with default hyperparameters
    python train.py

    # Train with custom hyperparameters
    python train.py --learning-rate 0.01 --hidden-size 32 --max-epochs 50

    # Load tuned hyperparameters from JSON
    python train.py --load-hyperparameters tuned_hyperparameters.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import lightning.pytorch as pl

from src import data_ingestion, pipeline, labeling, features, dataset, model


# Default configuration (can be overridden by CLI or JSON)
DEFAULT_CONFIG = {
    "encoder_length": 8,
    "decoder_length": 4,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 3,  # Early stopping patience (SPEC.md §8)
    "checkpoint_path": "models/hypoxia_tft",
    "train_split_ratio": 0.8,
}


def save_training_metadata(
    checkpoint_path: str,
    hyperparameters: dict,
    features_used: list,
    weight_config: dict,
    dataset_info: dict,
):
    """Save training configuration alongside checkpoint for reproducibility.

    Per BUILD_PLAN.md Phase 8: "Log which features were used (Phase 5 output) and the
    weight-tier configuration (Phase 4) alongside the checkpoint, so a saved model is
    reproducible without re-reading source code."

    Args:
        checkpoint_path: Directory where checkpoint is saved
        hyperparameters: Model hyperparameters used
        features_used: List of feature column names
        weight_config: Threshold and weight tier configuration
        dataset_info: Dataset statistics (train/val sizes, date ranges, etc.)
    """
    metadata = {
        "training_date": datetime.now().isoformat(),
        "hyperparameters": hyperparameters,
        "features": features_used,
        "weight_configuration": weight_config,
        "dataset_info": dataset_info,
    }

    metadata_path = Path(checkpoint_path) / "training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train weighted hypoxia prediction model")

    # Model hyperparameters (SPEC.md §7)
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="TFT hidden size (default: 16)")
    parser.add_argument("--attention-head-size", type=int, default=None,
                        help="TFT attention head size (default: 1)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate (default: 0.1)")
    parser.add_argument("--hidden-continuous-size", type=int, default=None,
                        help="Hidden continuous size (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate (default: 0.03)")
    parser.add_argument("--lstm-layers", type=int, default=None,
                        help="Number of LSTM layers (default: 1)")
    parser.add_argument("--gradient-clip-val", type=float, default=None,
                        help="Gradient clipping value (default: 0.1)")

    # Dataset parameters
    parser.add_argument("--encoder-length", type=int, default=DEFAULT_CONFIG["encoder_length"],
                        help="Encoder (lookback) length in weeks")
    parser.add_argument("--decoder-length", type=int, default=DEFAULT_CONFIG["decoder_length"],
                        help="Decoder (forecast) length in weeks")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Training batch size")

    # Training parameters
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_CONFIG["max_epochs"],
                        help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"],
                        help="Early stopping patience")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CONFIG["checkpoint_path"],
                        help="Fixed checkpoint directory path")

    # Load hyperparameters from JSON (from tune_hyperparameters.py)
    parser.add_argument("--load-hyperparameters", type=str, default=None,
                        help="Load hyperparameters from JSON file (overrides individual args)")

    # Data split
    parser.add_argument("--train-split-ratio", type=float, default=DEFAULT_CONFIG["train_split_ratio"],
                        help="Train/val split ratio")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")

    args = parser.parse_args()

    # Set random seed for reproducibility
    # workers=True ensures reproducibility across dataloader workers
    pl.seed_everything(args.seed, workers=True)

    # Allow non-deterministic operations for CUDA performance
    # Some operations (like upsample_linear1d) don't have deterministic GPU implementations
    import torch
    torch.use_deterministic_algorithms(False)

    # Load hyperparameters from JSON if provided
    if args.load_hyperparameters:
        print(f"Loading hyperparameters from: {args.load_hyperparameters}")
        with open(args.load_hyperparameters, "r") as f:
            tuned_config = json.load(f)
            tuned_hyperparameters = tuned_config["hyperparameters"]

        # Override with JSON values (but allow CLI to override JSON)
        for key, value in tuned_hyperparameters.items():
            arg_name = key.replace("_", "-")
            if not hasattr(args, key) or getattr(args, key.replace("-", "_")) is None:
                # Map JSON keys to argparse attribute names
                setattr(args, key, value)

    # Build hyperparameters dict (merge CLI args with defaults)
    hyperparameters = {}
    for key in ["hidden_size", "attention_head_size", "dropout", "hidden_continuous_size",
                "learning_rate", "lstm_layers", "gradient_clip_val"]:
        value = getattr(args, key, None)
        if value is not None:
            hyperparameters[key] = value

    # If no hyperparameters specified, use defaults from model.py
    if not hyperparameters:
        hyperparameters = model.DEFAULT_HYPERPARAMETERS.copy()

    print("="*80)
    print("WEIGHTED HYPOXIA PREDICTION MODEL TRAINING")
    print("="*80)
    print(f"\nHyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")

    print(f"\nDataset configuration:")
    print(f"  Encoder length: {args.encoder_length} weeks")
    print(f"  Decoder length: {args.decoder_length} weeks")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train/val split: {args.train_split_ratio:.0%}/{1-args.train_split_ratio:.0%}")

    print(f"\nTraining configuration:")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Checkpoint path: {args.checkpoint_path}")

    # Phase 2: Data ingestion
    print("\n" + "-"*80)
    print("Phase 2: Data Ingestion")
    print("-"*80)
    df_combined = data_ingestion.load_and_clean_boknis_data()
    print(f"Ocean + weather data loaded and merged: {len(df_combined)} rows")
    print(f"  Date range: {df_combined['Date'].min()} to {df_combined['Date'].max()}")

    # Phase 3: Weekly resampling and imputation
    print("\n" + "-"*80)
    print("Phase 3: Weekly Resampling & Imputation")
    print("-"*80)
    df_weekly = pipeline.prepare_weekly_series(df_combined)
    print(f"Weekly series: {len(df_weekly)} rows")

    # Phase 4: Hypoxia labeling and sample weighting
    print("\n" + "-"*80)
    print("Phase 4: Hypoxia Labeling & Sample Weighting")
    print("-"*80)
    df_25m = labeling.select_target_series(df_weekly)
    print(f"25m target series: {len(df_25m)} rows")

    # Phase 5: Feature engineering
    print("\n" + "-"*80)
    print("Phase 5: Feature Engineering")
    print("-"*80)
    df_features = features.engineer_features(df_25m, df_weekly)
    print(f"Engineered features: {len(df_features.columns)} columns")

    # Get list of features for metadata logging
    feature_cols = [col for col in df_features.columns
                    if col not in ["Date", "Depth_m", "Time_Idx", "O2_umol_L", "sample_weight",
                                   "oxygen_deficit", "month_sin", "month_cos"]]
    print(f"Feature columns: {', '.join(feature_cols)}")

    # Label hypoxia risk
    df_labeled = labeling.label_hypoxia_risk(df_features)

    # Drop rows with NaN values (from gaps wider than interpolation limit)
    # This is necessary because TimeSeriesDataSet doesn't accept NaN values
    rows_before = len(df_labeled)
    df_labeled = df_labeled.dropna().reset_index(drop=True)
    rows_after = len(df_labeled)
    rows_dropped = rows_before - rows_after
    print(f"\nDropped {rows_dropped} rows with NaN values ({rows_dropped/rows_before*100:.1f}%)")
    print(f"Complete rows remaining: {rows_after}")

    # Print weight statistics
    weight_config = {
        "thresholds": labeling.THRESHOLDS,
        "tier_weights": labeling.TIER_WEIGHTS,
    }
    print(f"\nWeight configuration:")
    print(f"  Thresholds: {labeling.THRESHOLDS}")
    print(f"  Tier weights: {labeling.TIER_WEIGHTS}")

    # Phase 6: Dataset construction
    print("\n" + "-"*80)
    print("Phase 6: Dataset Construction")
    print("-"*80)
    train_df, val_df = dataset.split_train_validation(
        df_labeled,
        train_ratio=args.train_split_ratio
    )
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    # Create dataloaders
    train_dl, val_dl, training_dataset = dataset.create_dataloaders(
        train_df,
        val_df,
        batch_size=args.batch_size,
        encoder_length=args.encoder_length,
        decoder_length=args.decoder_length,
        num_workers=args.num_workers,
    )

    # Sanity check batch weights
    print("\nSanity checking batch weights...")
    weight_check = dataset.sanity_check_batch_weights(train_dl)
    if weight_check["weight_present"]:
        print(f"  ✓ Weight tensor present in batches")
        print(f"  Mean weight (hypoxic): {weight_check['mean_weight_hypoxic']:.2f}")
        print(f"  Mean weight (normoxic): {weight_check['mean_weight_normoxic']:.2f}")
        print(f"  Ratio: {weight_check['weight_ratio']:.2f}x")
    else:
        print(f"  ✗ WARNING: Weight tensor not found in batches!")
        sys.exit(1)

    # Phase 7: Model training
    print("\n" + "-"*80)
    print("Phase 7: Model Training")
    print("-"*80)

    # Create model
    tft = model.create_tft_model(training_dataset, hyperparameters=hyperparameters)
    print(f"Model created: {tft.__class__.__name__}")

    # Create trainer
    trainer = model.create_trainer(
        max_epochs=args.max_epochs,
        patience=args.patience,
        gradient_clip_val=hyperparameters.get("gradient_clip_val", 0.1),
        checkpoint_path=args.checkpoint_path,
    )

    # Save training metadata before training
    dataset_info = {
        "total_samples": len(df_labeled),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "date_range": {
            "start": str(df_labeled["Date"].min()),
            "end": str(df_labeled["Date"].max()),
        },
        "train_date_range": {
            "start": str(train_df["Date"].min()),
            "end": str(train_df["Date"].max()),
        },
        "val_date_range": {
            "start": str(val_df["Date"].min()),
            "end": str(val_df["Date"].max()),
        },
        "encoder_length": args.encoder_length,
        "decoder_length": args.decoder_length,
        "batch_size": args.batch_size,
    }

    save_training_metadata(
        checkpoint_path=args.checkpoint_path,
        hyperparameters=hyperparameters,
        features_used=feature_cols,
        weight_config=weight_config,
        dataset_info=dataset_info,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Training complete
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best model checkpoint: {args.checkpoint_path}/best_model.ckpt")
    print(f"Training metadata: {args.checkpoint_path}/training_metadata.json")

    # Validate on best checkpoint
    print("\nValidating best checkpoint...")
    best_model_path = Path(args.checkpoint_path) / "best_model.ckpt"
    if best_model_path.exists():
        val_results = trainer.validate(
            ckpt_path=str(best_model_path),
            dataloaders=val_dl
        )
        print(f"Best validation loss: {val_results[0]['val_loss']:.4f}")
    else:
        print("Warning: Best checkpoint not found, using last checkpoint")
        val_results = trainer.validate(tft, dataloaders=val_dl)
        print(f"Final validation loss: {val_results[0]['val_loss']:.4f}")

    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()
