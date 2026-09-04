"""Streamlit dashboard for hypoxia prediction at Boknis Eck.

Implements SPEC.md §10 and BUILD_PLAN.md Phase 10:
- Load fixed-path checkpoint (avoiding §11 pitfall of timestamp-based loading)
- Threshold reference lines and hypoxic zone shading
- Risk readout from P10/P50/P90 quantile predictions
- Forecast horizon control bounded to reliable range

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import torch

from src import data_ingestion, pipeline, labeling, features, dataset, model
from pytorch_forecasting import TemporalFusionTransformer

# Hypoxia thresholds from SPEC.md §6.1
THRESHOLDS = {
    "severe": 30.0,      # µmol/L
    "hypoxic": 60.0,     # µmol/L
    "watch": 120.0,      # µmol/L
}

# Forecast horizon bounds (weeks) - empirically established from Phase 9
# Conservative: use decoder_length as max reliable horizon
MIN_HORIZON = 1
MAX_HORIZON = 4  # Matches decoder_length from training
DEFAULT_HORIZON = 4

# Fixed checkpoint path (SPEC.md §10/§11: avoid version_N pitfall)
# For local use:
DEFAULT_CHECKPOINT = "models/hypoxia_tft/best_model.ckpt"

# For Google Drive (when running in Colab):
# DEFAULT_CHECKPOINT = "/content/drive/MyDrive/Geomar_Checkpoints/best_model.ckpt"


@st.cache_resource
def load_model_and_data(checkpoint_path: str):
    """Load trained model and prepared data.

    Cached to avoid reloading on every interaction.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Tuple of (model, df_features, training_dataset)
    """
    # Load and prepare data (Phases 2-5)
    df_combined = data_ingestion.load_and_clean_boknis_data()
    df_weekly = pipeline.prepare_weekly_series(df_combined)
    df_25m = labeling.select_target_series(df_weekly)
    df_features = features.engineer_features(df_25m, df_weekly)
    df_labeled = labeling.label_hypoxia_risk(df_features)

    # Drop NaN rows
    df_labeled = df_labeled.dropna().reset_index(drop=True)

    # Create dataset for prediction (Phase 6)
    # Use all data for context
    training_dataset = dataset.create_training_dataset(
        df_labeled,
        encoder_length=8,
        decoder_length=4,
    )

    # Load model
    tft_model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint_path,
        strict=False
    )
    tft_model.eval()

    return tft_model, df_labeled, training_dataset


def make_prediction(
    model: TemporalFusionTransformer,
    df: pd.DataFrame,
    training_dataset,
    forecast_date: pd.Timestamp,
    horizon_weeks: int = 4
):
    """Generate forecast from a specific date.

    Args:
        model: Trained TFT model
        df: Prepared dataframe
        training_dataset: TimeSeriesDataSet for prediction
        forecast_date: Date to forecast from
        horizon_weeks: Number of weeks to forecast

    Returns:
        Dictionary with forecast dates, P10/P50/P90 predictions
    """
    # Find the row for this date
    date_mask = df['Date'] <= forecast_date
    if date_mask.sum() < 8:  # Need at least encoder_length history
        return None

    # Get the most recent data up to forecast_date
    df_context = df[date_mask].copy()

    # Create prediction dataset using from_dataset() method
    # This ensures the new dataset uses the same parameters as training
    pred_dataset = training_dataset.__class__.from_dataset(
        training_dataset,
        df_context,
        predict=True,
        stop_randomization=True
    )

    # Get predictions
    with torch.no_grad():
        predictions = model.predict(
            pred_dataset,
            mode="raw",
            return_x=False
        )

    # Extract quantiles: shape [n_samples, decoder_length, n_quantiles]
    # Take last sample (most recent forecast)
    if len(predictions.shape) == 3:
        p10 = predictions[-1, :horizon_weeks, 0].numpy()
        p50 = predictions[-1, :horizon_weeks, 1].numpy()
        p90 = predictions[-1, :horizon_weeks, 2].numpy()
    else:
        # Fallback
        p50 = predictions[-1, :horizon_weeks].numpy()
        p10 = p50.copy()
        p90 = p50.copy()

    # Generate forecast dates (weekly)
    forecast_dates = pd.date_range(
        start=forecast_date + timedelta(weeks=1),
        periods=horizon_weeks,
        freq='W'
    )

    return {
        'dates': forecast_dates,
        'p10': p10,
        'p50': p50,
        'p90': p90,
        'forecast_from': forecast_date
    }


def compute_hypoxia_risk(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray, threshold: float = 60.0):
    """Compute hypoxia risk from quantile predictions.

    Per SPEC.md §10: derive alert from regression output using quantile spread.

    Args:
        p10: 10th percentile predictions
        p50: 50th percentile (median) predictions
        p90: 90th percentile predictions
        threshold: Hypoxia threshold (default 60 µmol/L)

    Returns:
        Dictionary with risk metrics
    """
    # Simple risk estimate: if P90 < threshold, high risk
    # If P50 < threshold, moderate risk
    # If P10 < threshold, low risk

    high_risk_weeks = (p90 < threshold).sum()
    moderate_risk_weeks = (p50 < threshold).sum()
    any_risk_weeks = (p10 < threshold).sum()

    total_weeks = len(p50)

    # Risk probability: percentage of forecast horizon at risk
    high_risk_pct = (high_risk_weeks / total_weeks) * 100
    moderate_risk_pct = (moderate_risk_weeks / total_weeks) * 100

    return {
        'high_risk_weeks': high_risk_weeks,
        'moderate_risk_weeks': moderate_risk_weeks,
        'any_risk_weeks': any_risk_weeks,
        'total_weeks': total_weeks,
        'high_risk_pct': high_risk_pct,
        'moderate_risk_pct': moderate_risk_pct,
    }


def plot_forecast(historical_data: pd.DataFrame, forecast: dict, show_history_weeks: int = 12):
    """Create interactive Plotly forecast visualization.

    Per SPEC.md §10: includes threshold lines and hypoxic zone shading.

    Args:
        historical_data: DataFrame with Date and O2_umol_L columns
        forecast: Dictionary from make_prediction()
        show_history_weeks: Number of weeks of history to show

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Historical data (last N weeks before forecast)
    forecast_from = forecast['forecast_from']
    history_start = forecast_from - timedelta(weeks=show_history_weeks)
    hist_data = historical_data[
        (historical_data['Date'] >= history_start) &
        (historical_data['Date'] <= forecast_from)
    ].copy()

    # Plot historical observations
    fig.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=hist_data['O2_umol_L'],
        mode='lines+markers',
        name='Historical O₂',
        line=dict(color='black', width=2),
        marker=dict(size=6)
    ))

    # Plot forecast P50 (median)
    fig.add_trace(go.Scatter(
        x=forecast['dates'],
        y=forecast['p50'],
        mode='lines+markers',
        name='Forecast (P50)',
        line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    # Plot uncertainty band (P10-P90)
    fig.add_trace(go.Scatter(
        x=forecast['dates'],
        y=forecast['p90'],
        mode='lines',
        name='P90 (High estimate)',
        line=dict(color='lightblue', width=1),
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=forecast['dates'],
        y=forecast['p10'],
        mode='lines',
        name='P10 (Low estimate)',
        line=dict(color='lightblue', width=1),
        fill='tonexty',
        fillcolor='rgba(173, 216, 230, 0.3)',
        showlegend=True
    ))

    # Add threshold lines (SPEC.md §10)
    # Combine historical and forecast dates for full x-range
    all_dates = pd.concat([hist_data['Date'], pd.Series(forecast['dates'])])
    x_range = [all_dates.min(), all_dates.max()]

    # Severe hypoxia threshold (30 µmol/L)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=[THRESHOLDS['severe'], THRESHOLDS['severe']],
        mode='lines',
        name=f"Severe ({THRESHOLDS['severe']} µmol/L)",
        line=dict(color='red', width=2, dash='dot'),
    ))

    # Hypoxic threshold (60 µmol/L)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=[THRESHOLDS['hypoxic'], THRESHOLDS['hypoxic']],
        mode='lines',
        name=f"Hypoxic ({THRESHOLDS['hypoxic']} µmol/L)",
        line=dict(color='orange', width=2, dash='dot'),
    ))

    # Watch threshold (120 µmol/L)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=[THRESHOLDS['watch'], THRESHOLDS['watch']],
        mode='lines',
        name=f"Watch ({THRESHOLDS['watch']} µmol/L)",
        line=dict(color='yellow', width=2, dash='dot'),
    ))

    # Shade hypoxic zone (SPEC.md §10)
    fig.add_hrect(
        y0=0, y1=THRESHOLDS['hypoxic'],
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Hypoxic Zone",
        annotation_position="top left"
    )

    # Layout
    fig.update_layout(
        title=f"Oxygen Forecast from {forecast_from.strftime('%Y-%m-%d')}",
        xaxis_title="Date",
        yaxis_title="Oxygen (µmol/L)",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=600
    )

    return fig


def main():
    st.set_page_config(
        page_title="Boknis Eck Hypoxia Forecast",
        layout="wide"
    )

    st.title("Boknis Eck Hypoxia Prediction Dashboard")
    st.markdown("**Weighted TFT model for oxygen forecasting at 25m depth**")

    # Sidebar: Configuration
    st.sidebar.header("Configuration")

    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint Path",
        value=DEFAULT_CHECKPOINT,
        help="Path to trained model checkpoint (fixed path, not version_N)"
    )

    # Load model and data
    try:
        with st.spinner("Loading model and data..."):
            tft_model, df_data, training_dataset = load_model_and_data(checkpoint_path)
        st.sidebar.success("Model loaded")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Display data info
    st.sidebar.metric("Data Range", f"{df_data['Date'].min().date()} to {df_data['Date'].max().date()}")
    st.sidebar.metric("Total Samples", len(df_data))

    # Main panel: Forecast controls
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Select Forecast Date")

        # Date selector
        min_date = df_data['Date'].min() + timedelta(weeks=8)  # Need encoder history
        max_date = df_data['Date'].max()

        forecast_date = st.date_input(
            "Forecast from date:",
            value=max_date.date(),
            min_value=min_date.date(),
            max_value=max_date.date(),
            help="Select a date with sufficient history (at least 8 weeks)"
        )

        forecast_date = pd.Timestamp(forecast_date)

    with col2:
        st.subheader("Forecast Horizon")

        # Horizon control (SPEC.md §10: bounded to reliable range)
        horizon_weeks = st.slider(
            "Weeks ahead:",
            min_value=MIN_HORIZON,
            max_value=MAX_HORIZON,
            value=DEFAULT_HORIZON,
            help=f"Forecast horizon bounded to empirically reliable range ({MIN_HORIZON}-{MAX_HORIZON} weeks)"
        )

    # Generate forecast
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            forecast = make_prediction(
                tft_model,
                df_data,
                training_dataset,
                forecast_date,
                horizon_weeks
            )

        if forecast is None:
            st.error("Insufficient historical data for this date. Select a later date.")
            st.stop()

        # Compute risk metrics (SPEC.md §10)
        risk = compute_hypoxia_risk(
            forecast['p10'],
            forecast['p50'],
            forecast['p90'],
            threshold=THRESHOLDS['hypoxic']
        )

        # Display risk readout (SPEC.md §10)
        st.subheader("Hypoxia Risk Assessment")

        risk_col1, risk_col2, risk_col3 = st.columns(3)

        with risk_col1:
            st.metric(
                "High Risk Weeks",
                f"{risk['high_risk_weeks']}/{risk['total_weeks']}",
                help="Weeks where P90 < 60 µmol/L (likely hypoxic)"
            )

        with risk_col2:
            st.metric(
                "Moderate Risk Weeks",
                f"{risk['moderate_risk_weeks']}/{risk['total_weeks']}",
                help="Weeks where P50 < 60 µmol/L (50% chance hypoxic)"
            )

        with risk_col3:
            st.metric(
                "Any Risk Weeks",
                f"{risk['any_risk_weeks']}/{risk['total_weeks']}",
                help="Weeks where P10 < 60 µmol/L (possible hypoxic)"
            )

        # Risk summary
        if risk['high_risk_pct'] > 50:
            st.error(f"**HIGH RISK**: {risk['high_risk_pct']:.0f}% probability of hypoxic conditions in forecast period")
        elif risk['moderate_risk_pct'] > 50:
            st.warning(f"**MODERATE RISK**: {risk['moderate_risk_pct']:.0f}% probability of hypoxic conditions in forecast period")
        elif risk['any_risk_pct'] > 0:
            st.info(f"**LOW RISK**: {risk['any_risk_pct']:.0f}% chance of hypoxic conditions in forecast period")
        else:
            st.success("**NO RISK**: No hypoxic conditions forecast in this period")

        # Plot forecast
        st.subheader("Oxygen Forecast")
        fig = plot_forecast(df_data, forecast, show_history_weeks=12)
        st.plotly_chart(fig, use_container_width=True)

        # Show forecast table
        with st.expander("Detailed Forecast Data"):
            forecast_df = pd.DataFrame({
                'Date': forecast['dates'],
                'P10 (Low)': forecast['p10'],
                'P50 (Median)': forecast['p50'],
                'P90 (High)': forecast['p90'],
            })
            forecast_df['Status'] = forecast_df['P50 (Median)'].apply(
                lambda x: 'Severe' if x < THRESHOLDS['severe']
                else 'Hypoxic' if x < THRESHOLDS['hypoxic']
                else 'Watch' if x < THRESHOLDS['watch']
                else 'Normal'
            )
            st.dataframe(forecast_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Model**: Weighted Temporal Fusion Transformer | "
        "**Data**: Boknis Eck Time Series Station (25m depth) | "
        "**Thresholds**: Severe < 30, Hypoxic < 60, Watch < 120 µmol/L"
    )


if __name__ == "__main__":
    main()
