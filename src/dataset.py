import pandas as pd
from typing import Tuple, List, Optional
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import torch

from src.labeling import identify_hypoxic_episodes, THRESHOLDS


DEFAULT_ENCODER_LENGTH = 8  
DEFAULT_DECODER_LENGTH = 4  

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
    
    if max_prediction_length is None:
        max_prediction_length = decoder_length
    if max_encoder_length is None:
        max_encoder_length = encoder_length

    if group_ids is None:
        group_ids = ["Depth_m"]

    if time_varying_known_reals is None:
        time_varying_known_reals = ["month_sin", "month_cos"]

    if time_varying_unknown_reals is None:
        time_varying_unknown_reals = [
            # Core physical measurements at 25m
            "Temp_C", "Salinity", "NO3", "NO2", "PO4", "Silicate",
            # Weather variables
            "Air_Temp_C", "Wind_Speed_ms", "Wind_Dir_deg", "Wind_U", "Wind_V",
            # Temporal features from data ingestion
            "Season_sin", "Season_cos", "Years_since_start", "Days_since_prev", "Segment_ID",
            # Surface readings (1m depth)
            "Surface_Temp_C", "Surface_O2_umol_L",
            # Vertical gradients
            "Vertical_Temp_Grad", "Vertical_O2_Grad",
            # Depth 1 lagged features (2-week lag)
            "Depth1_Chl_a_lag2W", "Depth1_Nitrat_lag2W", "Depth1_Phosphat_lag2W", "Depth1_Temp_lag2W",
        ]

        if "O2_Derivative_1W" in df.columns:
            time_varying_unknown_reals.append("O2_Derivative_1W")

        # Filter to only columns actually present in df
        time_varying_unknown_reals = [
            col for col in time_varying_unknown_reals if col in df.columns
        ]

    if static_categoricals is None:
        static_categoricals = []

    training = TimeSeriesDataSet(
        df,
        time_idx=time_idx,
        target=target,
        group_ids=group_ids,
        min_encoder_length=max_encoder_length // 2,  
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=group_ids,
            transformation="softplus"  
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        weight=weight,
        allow_missing_timesteps=True,
    )

    return training


def split_train_validation(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    time_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:

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

    episodes = identify_hypoxic_episodes(df)

    episodes = episodes[episodes["duration_weeks"] >= min_weeks].reset_index(drop=True)

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

    training = create_training_dataset(train_df, **dataset_kwargs)

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size * 10,  
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader, training


def sanity_check_batch_weights(
    dataloader: torch.utils.data.DataLoader,
    weight_col: str = "sample_weight",
    hypoxic_threshold: float = None,
) -> dict:

    if hypoxic_threshold is None:
        hypoxic_threshold = THRESHOLDS["hypoxic"]

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

    weight_attr = getattr(dataset, 'weight', None)
    weight_present = weight_attr is not None

    if not weight_present:
        return {
            "weight_present": False,
            "mean_weight_hypoxic": None,
            "mean_weight_normoxic": None,
            "weight_ratio": None,
        }

    data = dataset.data
    if 'weight' not in data:
        return {
            "weight_present": False,
            "mean_weight_hypoxic": None,
            "mean_weight_normoxic": None,
            "weight_ratio": None,
        }

    import numpy as np
    weights = np.array(data['weight']).flatten()  
    targets = np.array(data['target']).flatten()  

    hypoxic_mask = targets < hypoxic_threshold
    normoxic_mask = ~hypoxic_mask

    mean_weight_hypoxic = weights[hypoxic_mask].mean() if hypoxic_mask.any() else None
    mean_weight_normoxic = weights[normoxic_mask].mean() if normoxic_mask.any() else None

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
