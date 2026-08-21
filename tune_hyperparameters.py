"""Hyperparameter tuning with Optuna for weighted TFT model.

Per SPEC.md §7 and BUILD_PLAN.md Phase 7: "Re-tune hyperparameters after switching to a
weighted loss. Reweighting the loss changes the optimization landscape — hyperparameters
tuned for a symmetric, unweighted objective aren't guaranteed to still be good."

This script uses Optuna's Tree-structured Parzen Estimator (TPE) to search hyperparameter
space, optimizing against validation loss on the weighted objective. The reference paper
(IRANNA) used this same approach.

Usage:
    python tune_hyperparameters.py --n-trials 50 --output tuned_hyperparameters.json
"""

import argparse
import json
import optuna
from optuna.samplers import TPESampler
import lightning.pytorch as pl
from pathlib import Path

from src import data_ingestion, pipeline, labeling, features, dataset, model


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function: train TFT with trial hyperparameters, return val_loss.

    Args:
        trial: Optuna trial object for suggesting hyperparameters

    Returns:
        Validation loss (lower is better)
    """
    # Suggest hyperparameters (ranges based on SPEC.md §7 starting values)
    hyperparameters = {
        "hidden_size": trial.suggest_int("hidden_size", 8, 64, step=8),
        "attention_head_size": trial.suggest_int("attention_head_size", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "hidden_continuous_size": trial.suggest_int("hidden_continuous_size", 4, 32, step=4),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.01, 1.0, log=True),
    }

    # Suggest dataset parameters
    encoder_length = trial.suggest_int("encoder_length", 4, 16, step=2)
    decoder_length = trial.suggest_int("decoder_length", 2, 8, step=2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Load and prepare data (Phases 2-5)
    print(f"\n[Trial {trial.number}] Loading data...")
    df_combined = data_ingestion.load_and_clean_boknis_data()

    # Weekly resampling (Phase 3)
    df_weekly = pipeline.prepare_weekly_series(df_combined)

    # Select 25m target series (Phase 4)
    df_25m = labeling.select_target_series(df_weekly)

    # Feature engineering (Phase 5)
    df_features = features.engineer_features(df_25m, df_weekly)

    # Labeling (Phase 4)
    df_labeled = labeling.label_hypoxia_risk(df_features)

    # Drop rows with NaN values (TimeSeriesDataSet requirement)
    df_labeled = df_labeled.dropna().reset_index(drop=True)

    # Train/val split (Phase 6)
    train_df, val_df = dataset.split_train_validation(df_labeled, train_ratio=0.8)

    # Create dataloaders (Phase 6)
    train_dl, val_dl, training_dataset = dataset.create_dataloaders(
        train_df,
        val_df,
        batch_size=batch_size,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        num_workers=0,  # Single process for Optuna
    )

    # Create model (Phase 7)
    tft = model.create_tft_model(training_dataset, hyperparameters=hyperparameters)

    # Create trainer with short training for fast tuning
    trainer = pl.Trainer(
        max_epochs=20,  # Short training for tuning (faster iterations)
        accelerator="auto",
        gradient_clip_val=hyperparameters["gradient_clip_val"],
        enable_progress_bar=False,  # Cleaner output for Optuna
        enable_model_summary=False,
        enable_checkpointing=False,  # Don't save checkpoints during tuning
        logger=False,  # Disable logging
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
            )
        ],
    )

    # Train
    print(f"[Trial {trial.number}] Training with hyperparameters: {hyperparameters}")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Get validation loss
    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        # If val_loss not in metrics, get it from validation
        val_results = trainer.validate(tft, dataloaders=val_dl, verbose=False)
        val_loss = val_results[0]["val_loss"]

    print(f"[Trial {trial.number}] Validation loss: {val_loss:.4f}")

    return float(val_loss)


def main():
    parser = argparse.ArgumentParser(description="Tune TFT hyperparameters with Optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--output", type=str, default="tuned_hyperparameters.json",
                        help="Output JSON file for best hyperparameters")
    parser.add_argument("--study-name", type=str, default="hypoxia_tft_tuning",
                        help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    args = parser.parse_args()

    # Create Optuna study with TPE sampler (same as reference paper)
    sampler = TPESampler(seed=42)  # Reproducibility
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )

    print(f"Starting hyperparameter tuning with {args.n_trials} trials...")
    print(f"Study name: {args.study_name}")
    if args.storage:
        print(f"Storage: {args.storage}")

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "="*80)
    print("Hyperparameter Tuning Complete!")
    print("="*80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best hyperparameters to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_config = {
        "hyperparameters": study.best_params,
        "val_loss": study.best_value,
        "trial_number": study.best_trial.number,
    }

    with open(output_path, "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest hyperparameters saved to: {output_path}")

    # Print summary statistics
    print(f"\nTuning summary:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")


if __name__ == "__main__":
    main()
