# Boknis Eck Hypoxia Prediction Model

Weighted Temporal Fusion Transformer for oxygen forecasting at Boknis Eck Time Series Station (25m depth).

## Project Overview

This project implements a machine learning model to predict hypoxic events (low oxygen conditions) in the Boknis Eck fjord using:
- **Temporal Fusion Transformer (TFT)** for multi-horizon time series forecasting
- **Weighted loss function** to prioritize accurate hypoxia prediction
- **Quantile predictions** (P10/P50/P90) for uncertainty quantification
- **Interactive dashboard** for visualizing forecasts and risk assessment

## Key Features

- **Multi-step forecasting**: Predict oxygen levels 1-4 weeks ahead
- **Weighted training**: 12x higher weight on severe hypoxic samples
- **Risk assessment**: Probability-based hypoxia alerts from quantile spread
- **Historical event studies**: Analysis of past hypoxic episodes
- **Comprehensive evaluation**: ROC-AUC, F1, precision/recall metrics

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Geomar_AI_Oxygen.git
cd Geomar_AI_Oxygen

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Quick test (5 epochs, ~5 min)
python train.py --max-epochs 5 --batch-size 32

# Full training (100 epochs with early stopping, ~30-60 min on GPU)
python train.py --max-epochs 100 --batch-size 64 --patience 3

# Training with tuned hyperparameters
python train.py --load-hyperparameters tuned_hyperparameters.json
```

### Hyperparameter Tuning

```bash
# Run Optuna optimization (20 trials, ~2-4 hours on GPU)
python tune_hyperparameters.py --n-trials 20 --output tuned_hyperparameters.json
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --checkpoint models/hypoxia_tft/best_model.ckpt

# Outputs:
# - evaluation_results/evaluation_metrics.json (all metrics)
# - evaluation_results/roc_pr_curves.png (ROC and PR curves)
# - evaluation_results/event_study_*.png (6 historical hypoxic episodes)
```

### Dashboard

```bash
# Launch interactive Streamlit dashboard
streamlit run app.py

# Or specify custom checkpoint path
streamlit run app.py -- --checkpoint path/to/checkpoint.ckpt
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
Geomar_AI_Oxygen/
├── app.py                      # Streamlit dashboard (Phase 10)
├── train.py                    # End-to-end training pipeline (Phase 8)
├── evaluate.py                 # Comprehensive evaluation suite (Phase 9)
├── tune_hyperparameters.py     # Optuna hyperparameter optimization
├── verify_weighted_loss.py     # Weighted loss verification test
├── requirements.txt            # Python dependencies
├── src/                        # Core pipeline modules
│   ├── data_ingestion.py       # Load and merge ocean + weather data (Phase 2)
│   ├── pipeline.py             # Weekly resampling and imputation (Phase 3)
│   ├── labeling.py             # Hypoxia labeling and weighting (Phase 4)
│   ├── features.py             # Feature engineering (Phase 5)
│   ├── dataset.py              # TimeSeriesDataSet construction (Phase 6)
│   └── model.py                # TFT model and trainer setup (Phase 7)
├── tests/                      # Unit tests (Phase 11)
│   ├── test_data_ingestion.py
│   ├── test_pipeline.py
│   ├── test_labeling.py
│   ├── test_features.py
│   └── test_dataset.py
├── Documentation/               # Specifications and planning
│   ├── SPEC.md                 # Technical specification
│   ├── BUILD_PLAN.md           # Implementation phases
│   ├── CONCEPT.md              # Design decisions
│   ├── ENVIRONMENT.md          # Setup and dependencies
│   └── data/                   # Ocean and chlorophyll data
└── models/                     # Saved checkpoints
    └── hypoxia_tft/
        ├── best_model.ckpt
        └── training_metadata.json
```

## Model Performance

Based on evaluation with tuned hyperparameters:

### Classification Metrics (Hypoxia Detection)
- **ROC-AUC**: 0.950 (expected: 0.7-0.85, exceeded)
- **F1 Score**: 0.777 (expected: 0.2-0.4, exceeded)
- **Precision**: 0.821
- **Recall**: 0.738

### Regression Metrics
- **Train Correlation**: 0.987
- **Train RMSE**: 26.6 µmol/L (weighted)
- **Val Correlation**: 0.970
- **Val RMSE**: 6.5 µmol/L (weighted)

### vs. Persistence Baseline
- **88.2% improvement** on hypoxic samples (16.1 vs 135.8 µmol/L RMSE)

## Configuration

### Hypoxia Thresholds (SPEC.md §6.1)
- **Severe**: < 30 µmol/L (weight: 12.0)
- **Hypoxic**: < 60 µmol/L (weight: 6.0)
- **Watch**: < 120 µmol/L (weight: 3.0)
- **Normoxic**: ≥ 120 µmol/L (weight: 1.0)

### Model Architecture
- **Type**: Temporal Fusion Transformer
- **Encoder length**: 8 weeks (lookback window)
- **Decoder length**: 4 weeks (forecast horizon)
- **Loss function**: QuantileLoss with weighted samples
- **Quantiles**: [0.1, 0.5, 0.9] (P10/P50/P90)

### Default Hyperparameters
```python
{
    "hidden_size": 32,
    "attention_head_size": 4,
    "dropout": 0.1,
    "hidden_continuous_size": 20,
    "learning_rate": 0.016,
    "lstm_layers": 2,
    "gradient_clip_val": 0.83
}
```

## Dashboard Features

- **Date Selector**: Choose any date with sufficient history (≥8 weeks)
- **Forecast Horizon**: 1-4 weeks ahead (empirically reliable range)
- **Risk Assessment**:
  - High Risk: P90 < 60 µmol/L
  - Moderate Risk: P50 < 60 µmol/L
  - Low Risk: P10 < 60 µmol/L
- **Visualization**:
  - Historical observations (12 weeks before forecast)
  - P50 forecast (median prediction)
  - Uncertainty band (P10-P90)
  - Threshold reference lines (30, 60, 120 µmol/L)
  - Hypoxic zone shading

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_labeling.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Documentation

- **[SPEC.md](Documentation/SPEC.md)**: Complete technical specification
- **[BUILD_PLAN.md](Documentation/BUILD_PLAN.md)**: Implementation phases and status
- **[CONCEPT.md](Documentation/CONCEPT.md)**: Design decisions and architecture
- **[ENVIRONMENT.md](Documentation/ENVIRONMENT.md)**: Dependency management

## Contributing

1. Read `Documentation/SPEC.md` for technical requirements
2. Follow the phase-based implementation plan in `Documentation/BUILD_PLAN.md`
3. Run tests before submitting: `pytest tests/`
4. Ensure code follows existing patterns in `src/` modules

## License

[Add your license here]

## Acknowledgments

- **Data source**: Boknis Eck Time Series Station (GEOMAR)
- **Weather data**: DWD (German Weather Service) via wetterdienst
- **Model**: Temporal Fusion Transformer (pytorch-forecasting)

## Contact

[Add your contact information]

---

**Status**: Complete (Phases 1-10)
- Phase 1-7: Data pipeline and model training
- Phase 8: End-to-end training script
- Phase 9: Evaluation suite
- Phase 10: Interactive dashboard
