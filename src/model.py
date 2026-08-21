"""Model architecture and loss configuration. Follows Documentation/SPEC.md section 7.

Uses TemporalFusionTransformer with QuantileLoss for weighted training. Starting
hyperparameters from SPEC.md §7 (previous implementation's values), but these should be
re-tuned against the weighted objective once baseline training works (BUILD_PLAN.md Phase 7).

The TFT provides quantile predictions (P10/P50/P90) which are useful for threshold-crossing
probability alerts in the dashboard (SPEC.md §10). The weighted loss is applied via the
weight= parameter in TimeSeriesDataSet (Phase 6), which multiplies per-timestep loss before
aggregation in QuantileLoss (verified in SPEC.md §6.5).
"""

from typing import Dict, Any
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import TimeSeriesDataSet
import lightning.pytorch as pl


# Starting hyperparameters from SPEC.md §7 (previous implementation)
# These should be re-tuned against the weighted objective (BUILD_PLAN.md Phase 7)
DEFAULT_HYPERPARAMETERS = {
    "hidden_size": 16,
    "attention_head_size": 1,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "learning_rate": 0.03,
    "lstm_layers": 1,  # Default for TFT
    "gradient_clip_val": 0.1,  # Regularization (SPEC.md §8)
}

# Quantile predictions for uncertainty quantification (SPEC.md §7/§10)
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]  # P10, P50 (median), P90


def create_tft_model(
    training_dataset: TimeSeriesDataSet,
    hyperparameters: Dict[str, Any] = None,
    quantiles: list = None,
) -> TemporalFusionTransformer:
    """Build a TemporalFusionTransformer with QuantileLoss for weighted training.

    Args:
        training_dataset: TimeSeriesDataSet from Phase 6 (includes weight parameter)
        hyperparameters: Model hyperparameters (defaults to DEFAULT_HYPERPARAMETERS)
        quantiles: Quantile predictions to output (defaults to [0.1, 0.5, 0.9])

    Returns:
        TemporalFusionTransformer configured for weighted quantile regression

    Notes:
        - The weighted loss is enabled via training_dataset's weight parameter (SPEC.md §6.5)
        - QuantileLoss automatically applies the weight during loss computation
        - Starting hyperparameters from SPEC.md §7 should be re-tuned (BUILD_PLAN.md Phase 7)
        - Uses Adam optimizer per SPEC.md §7
    """
    if hyperparameters is None:
        hyperparameters = DEFAULT_HYPERPARAMETERS.copy()

    if quantiles is None:
        quantiles = DEFAULT_QUANTILES

    # Build TFT from dataset configuration
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        # Architecture hyperparameters (SPEC.md §7)
        hidden_size=hyperparameters.get("hidden_size", DEFAULT_HYPERPARAMETERS["hidden_size"]),
        attention_head_size=hyperparameters.get("attention_head_size", DEFAULT_HYPERPARAMETERS["attention_head_size"]),
        dropout=hyperparameters.get("dropout", DEFAULT_HYPERPARAMETERS["dropout"]),
        hidden_continuous_size=hyperparameters.get("hidden_continuous_size", DEFAULT_HYPERPARAMETERS["hidden_continuous_size"]),
        lstm_layers=hyperparameters.get("lstm_layers", DEFAULT_HYPERPARAMETERS["lstm_layers"]),
        # Loss configuration (SPEC.md §7/§8)
        loss=QuantileLoss(quantiles=quantiles),
        # Optimizer (SPEC.md §7)
        learning_rate=hyperparameters.get("learning_rate", DEFAULT_HYPERPARAMETERS["learning_rate"]),
        optimizer="adam",
        # Reduce learning rate on plateau (standard regularization)
        reduce_on_plateau_patience=4,
        reduce_on_plateau_reduction=2.0,
        # Logging
        log_interval=10,
        log_val_interval=1,
    )

    return tft


def create_trainer(
    max_epochs: int = 100,
    patience: int = 3,
    gradient_clip_val: float = None,
    checkpoint_path: str = "models/hypoxia_tft",
    enable_progress_bar: bool = True,
    enable_model_summary: bool = True,
) -> pl.Trainer:
    """Create a PyTorch Lightning Trainer with early stopping and checkpointing.

    Args:
        max_epochs: Maximum training epochs
        patience: Early stopping patience on val_loss (SPEC.md §8: previous impl used 3)
        gradient_clip_val: Gradient clipping value (regularization, SPEC.md §8)
        checkpoint_path: Fixed checkpoint directory (SPEC.md §11 pitfall: avoid version_N)
        enable_progress_bar: Show training progress bar
        enable_model_summary: Print model summary

    Returns:
        PyTorch Lightning Trainer configured for weighted training

    Notes:
        - Uses fixed checkpoint path to avoid SPEC.md §11's version_N pitfall
        - Early stopping on val_loss with patience from SPEC.md §8
        - Gradient clipping for regularization (SPEC.md §8)
    """
    if gradient_clip_val is None:
        gradient_clip_val = DEFAULT_HYPERPARAMETERS["gradient_clip_val"]

    # Early stopping callback (SPEC.md §8)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True,
    )

    # Checkpoint callback with FIXED PATH (SPEC.md §11 pitfall)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,  # Also save last checkpoint for resume
    )

    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # Use GPU if available
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
        # Deterministic for reproducibility
        deterministic=True,
    )

    return trainer


def get_model_config(model: TemporalFusionTransformer) -> Dict[str, Any]:
    """Extract model configuration for logging alongside checkpoint.

    Returns a dictionary with hyperparameters that can be saved to reproduce the model.
    This satisfies BUILD_PLAN.md Phase 8's requirement to log which hyperparameters were
    used alongside the checkpoint.

    Args:
        model: Trained TemporalFusionTransformer

    Returns:
        Dictionary with model configuration
    """
    config = {
        "hidden_size": model.hparams.hidden_size,
        "attention_head_size": model.hparams.attention_head_size,
        "dropout": model.hparams.dropout,
        "hidden_continuous_size": model.hparams.hidden_continuous_size,
        "lstm_layers": model.hparams.lstm_layers,
        "learning_rate": model.hparams.learning_rate,
        "loss": str(model.loss),
        "optimizer": model.hparams.optimizer,
    }

    return config
