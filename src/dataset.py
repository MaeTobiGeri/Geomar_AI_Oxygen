"""Dataset construction for weighted time-series training. Follows Documentation/SPEC.md
section 6.5 and BUILD_PLAN.md Phase 6.

Builds a pytorch-forecasting TimeSeriesDataSet with the sample_weight column from Phase 4
(src/labeling.py) to enable weighted loss training. Uses chronological train/validation
splits (not random-block) since TFT has sequential encoder/decoder context that needs
realistic evaluation (SPEC.md §8). Separate held-out event sets support Phase 9's event
studies.

The weight parameter mechanism was verified against pytorch-forecasting 1.7.0 in SPEC.md
§6.5: TimeSeriesDataSet.__init__ accepts weight="column_name", and MultiHorizonMetric.update
multiplies per-timestep loss by the weight before aggregation.
"""

import pandas as pd
from typing import Tuple, List, Optional
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import torch

# Importing from labeling module for threshold/event extraction
from src.labeling import identify_hypoxic_episodes, THRESHOLDS


# Default encoder/decoder lengths (SPEC.md §7 starting hyperparameters to tune)
DEFAULT_ENCODER_LENGTH = 8  # 8 weeks lookback
DEFAULT_DECODER_LENGTH = 4  # 4 weeks forecast horizon

# Chronological split ratio (80% train, 20% validation)
TRAIN_SPLIT_RATIO = 0.80


def create_training_dataset(
    df: pd.DataFrame,
    encoder_length: int = DEFAULT_ENCODER_LENGTH,
    decoder_length: int = DEFAULT_DECODER_LENGTH,
    target: str = "O2_umol_L",
    weight: str = "sample_weight",
    time_idx: str = "Time_Idx",
    group_ids: List[str] = None,
    time_varying_known_reals: List[str] = None,
    time_varying_unknown_reals: List[str] = None,
    static_categoricals: List[str] = None,
    max_prediction_length: Optional[int] = None,
    max_encoder_length: Optional[int] = None,
) -> TimeSeriesDataSet:
    """Build a TimeSeriesDataSet for weighted training.

    Args:
        df: DataFrame with engineered features (from Phase 5), target, and sample_weight
        encoder_length: Lookback window length (weeks)
        decoder_length: Forecast horizon (weeks)
        target: Target variable column name
        weight: Sample weight column name (from Phase 4 labeling)
        time_idx: Time index column name (added by Phase 3 pipeline)
        group_ids: List of columns defining series groups (default: ["Depth_m"])
        time_varying_known_reals: Known future covariates (default: month encodings)
        time_varying_unknown_reals: Unknown future covariates (default: all engineered features)
        static_categoricals: Static categorical features (default: None for single-depth series)
        max_prediction_length: Override decoder_length if provided
        max_encoder_length: Override encoder_length if provided

    Returns:
        TimeSeriesDataSet configured with weight parameter for weighted loss

    Notes:
        - The weight parameter enables per-sample loss reweighting (SPEC.md §6.5)
        - group_ids defaults to ["Depth_m"] to handle multi-depth data if present,
          but for the 25m target series this is effectively a single group
        - Target normalization uses GroupNormalizer for proper scaling
    """
    if max_prediction_length is None:
        max_prediction_length = decoder_length
    if max_encoder_length is None:
        max_encoder_length = encoder_length

    # Default group_ids: Depth_m (even though we're only using 25m, keep structure general)
    if group_ids is None:
        group_ids = ["Depth_m"]

    # Default time-varying features: month encodings are known, all others unknown
    if time_varying_known_reals is None:
        time_varying_known_reals = ["month_sin", "month_cos"]

    if time_varying_unknown_reals is None:
        # All engineered features from Phase 5 (these will be in the dataframe)
        time_varying_unknown_reals = [
            # Core physical measurements
            "Temp_C", "Salinity", "NO3", "NO2", "PO4", "Silicate", "Chl_a",
            # Weather variables
            "Air_Temp_C", "Wind_Speed_ms", "Wind_Dir_deg", "Wind_U", "Wind_V",
            # Surface readings (Phase 5 features)
            "Surface_Temp_C", "Surface_O2_umol_L",
            # Vertical gradients (Phase 5 features)
            "Vertical_Temp_Grad", "Vertical_O2_Grad",
            # Candidate features (Phase 5)
            "Wind_Mixing_Energy",
            "Chl_a_lag_1W", "Chl_a_lag_2W", "Chl_a_lag_3W", "Chl_a_lag_4W",
        ]

        # O2_Derivative_1W is optional (INCLUDE_O2_DERIVATIVE flag in features.py)
        if "O2_Derivative_1W" in df.columns:
            time_varying_unknown_reals.append("O2_Derivative_1W")

        # Filter to only columns actually present in df
        time_varying_unknown_reals = [
            col for col in time_varying_unknown_reals if col in df.columns
        ]

    if static_categoricals is None:
        static_categoricals = []

    # Build TimeSeriesDataSet with weight parameter (SPEC.md §6.5)
    training = TimeSeriesDataSet(
        df,
        time_idx=time_idx,
        target=target,
        group_ids=group_ids,
        min_encoder_length=max_encoder_length // 2,  # Allow shorter sequences
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=group_ids,
            transformation="softplus"  # Softplus for non-negative oxygen values
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        # CRITICAL: weight parameter enables per-sample loss reweighting (SPEC.md §6.5)
        weight=weight,
        # Allow gaps in weekly time series after NaN removal (dropna creates non-consecutive weeks)
        allow_missing_timesteps=True,
    )

    return training


def split_train_validation(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    time_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation split (SPEC.md §8, BUILD_PLAN.md Phase 6).

    Args:
        df: Full dataset sorted by time
        train_ratio: Fraction of data for training (default 0.8)
        time_col: Column name for time/date

    Returns:
        (train_df, val_df) chronologically split

    Notes:
        - NOT random-block split like the reference paper (SPEC.md §8)
        - TFT has sequential encoder/decoder context requiring chronological evaluation
        - Ensures no future data leaks into training
    """
    # Ensure sorted by time
    df = df.sort_values(time_col).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    return train_df, val_df


def get_held_out_events(
    df: pd.DataFrame,
    min_weeks: int = 2,
    val_start_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Extract held-out hypoxic episodes for event-study evaluation (SPEC.md §8/§9).

    Uses identify_hypoxic_episodes from Phase 4 to find known historical events.
    These are held out from training for dedicated out-of-sample event studies
    (BUILD_PLAN.md Phase 9).

    Args:
        df: Full 25m target series with O2_umol_L
        min_weeks: Minimum episode duration to include (default 2)
        val_start_date: If provided, only include events after this date (validation set)

    Returns:
        DataFrame with columns: episode_id, start_date, end_date, duration_weeks, min_o2

    Notes:
        - Cross-references THRESHOLDS["hypoxic"] from Phase 4 (60 µmol/L default)
        - Can be filtered to validation period only for fair evaluation
        - These episodes support Phase 9's event-study plots (obs vs weighted vs unweighted)
    """
    episodes = identify_hypoxic_episodes(df)

    # Filter by minimum duration
    episodes = episodes[episodes["duration_weeks"] >= min_weeks].reset_index(drop=True)

    # Optionally filter to validation period
    if val_start_date is not None:
        episodes = episodes[episodes["start_date"] >= val_start_date].reset_index(drop=True)

    return episodes


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 0,
    **dataset_kwargs,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, TimeSeriesDataSet]:
    """Create train and validation dataloaders with chronological split.

    Args:
        train_df: Training data (chronologically earlier)
        val_df: Validation data (chronologically later)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments for create_training_dataset

    Returns:
        (train_dataloader, val_dataloader, training_dataset)

    Notes:
        - training_dataset is returned for creating validation dataset (must use same scaling)
        - Chronological split ensures realistic evaluation (SPEC.md §8)
    """
    # Create training dataset
    training = create_training_dataset(train_df, **dataset_kwargs)

    # Create validation dataset using same parameters (critical for consistent normalization)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size * 10,  # Larger batch for validation (faster)
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader, training


def sanity_check_batch_weights(
    dataloader: torch.utils.data.DataLoader,
    weight_col: str = "sample_weight",
    hypoxic_threshold: float = None,
) -> dict:
    """Sanity check: verify weight configuration and distribution.

    In pytorch-forecasting v1.7.0, weights are not passed in batch dicts but are accessed
    from the underlying dataframe during loss computation. This function verifies:
    1. The TimeSeriesDataSet has weight attribute set
    2. The underlying data has the weight column
    3. Weight distribution is correct (higher on hypoxic samples)

    Args:
        dataloader: A dataloader to inspect
        weight_col: Name of weight column (default "sample_weight")
        hypoxic_threshold: O2 threshold for hypoxic classification (default: THRESHOLDS["hypoxic"])

    Returns:
        Dictionary with sanity check results:
            - weight_present: bool, whether weight configuration exists
            - mean_weight_hypoxic: mean weight on hypoxic samples
            - mean_weight_normoxic: mean weight on normoxic samples
            - weight_ratio: hypoxic/normoxic ratio (should be > 1.0)

    Notes:
        - This is a BUILD_PLAN.md Phase 6 requirement to verify weight= works
        - Should show hypoxic samples have higher weights (6.0-12.0) than normoxic (1.0-3.0)
    """
    if hypoxic_threshold is None:
        hypoxic_threshold = THRESHOLDS["hypoxic"]

    # Access the TimeSeriesDataSet from the dataloader
    # pytorch-forecasting dataloaders wrap the dataset in ._dataset or .dataset
    dataset = getattr(dataloader, 'dataset', None)
    if dataset is None:
        dataset = getattr(dataloader, '_dataset', None)

    if dataset is None:
        return {
            "weight_present": False,
            "mean_weight_hypoxic": None,
            "mean_weight_normoxic": None,
            "weight_ratio": None,
        }

    # Check if weight attribute is set on the TimeSeriesDataSet
    weight_attr = getattr(dataset, 'weight', None)
    weight_present = weight_attr is not None

    if not weight_present:
        return {
            "weight_present": False,
            "mean_weight_hypoxic": None,
            "mean_weight_normoxic": None,
            "weight_ratio": None,
        }

    # Access the underlying data dict (pytorch-forecasting stores data as dict of arrays)
    data = dataset.data

    # Check weight array exists in data dict
    if 'weight' not in data:
        return {
            "weight_present": False,
            "mean_weight_hypoxic": None,
            "mean_weight_normoxic": None,
            "weight_ratio": None,
        }

    # Extract weight and target arrays
    # data is a dict with keys: 'reals', 'categoricals', 'groups', 'target', 'weight', 'time'
    import numpy as np
    weights = np.array(data['weight']).flatten()  # Shape: (n_samples,)
    targets = np.array(data['target']).flatten()  # Shape: (n_samples,) - target may be (1, n) or (n,)

    # Compute weight statistics
    hypoxic_mask = targets < hypoxic_threshold
    normoxic_mask = ~hypoxic_mask

    mean_weight_hypoxic = weights[hypoxic_mask].mean() if hypoxic_mask.any() else None
    mean_weight_normoxic = weights[normoxic_mask].mean() if normoxic_mask.any() else None

    # Ratio (should be > 1.0)
    if mean_weight_hypoxic is not None and mean_weight_normoxic is not None:
        weight_ratio = mean_weight_hypoxic / mean_weight_normoxic
    else:
        weight_ratio = None

    return {
        "weight_present": True,
        "mean_weight_hypoxic": mean_weight_hypoxic,
        "mean_weight_normoxic": mean_weight_normoxic,
        "weight_ratio": weight_ratio,
    }
