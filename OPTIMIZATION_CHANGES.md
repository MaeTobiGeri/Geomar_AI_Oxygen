# Data Pipeline Optimization - October 9, 2026

## Problem
Training was losing **85.4% of data** (1369 out of 1603 rows) due to NaN values, leaving only 234 samples.

## Root Causes Identified

1. **Chl_a at 25m depth**: 43.3% missing (worst offender)
2. **4-week lag features**: Created 4 additional NaN weeks at start
3. **16-week encoder length**: Required 20 consecutive complete weeks (16 encoder + 4 lag)
4. **Strict dropna()**: train.py drops ANY row with ANY NaN

## Data Coverage Analysis

### At 25m depth (target):
- ✅ Temp_C: 99.1% coverage
- ✅ Salinity: 98.9% coverage  
- ✅ O2: 64.5% coverage
- ✅ NO3: 64.7% coverage
- ✅ NO2: 62.4% coverage
- ✅ PO4: 64.5% coverage
- ✅ Silicate: 64.3% coverage
- ❌ **Chl_a: 43.3% coverage** (removed)

### At 1m depth (for lagged features):
- ✅ Chl_a: 63.1% coverage (kept as Depth1_Chl_a_lag2W)
- ✅ NO3: ~64% coverage
- ✅ PO4: ~64% coverage

## Changes Made

### 1. src/features.py
- ✅ Reduced lag from **4 weeks → 2 weeks** on all Depth1 features
  - `Depth1_Chl_a_lag4W` → `Depth1_Chl_a_lag2W`
  - `Depth1_Nitrat_lag4W` → `Depth1_Nitrat_lag2W`
  - `Depth1_Phosphat_lag4W` → `Depth1_Phosphat_lag2W`
  - `Depth1_Temp_lag4W` → `Depth1_Temp_lag2W`
- ✅ **Removed Chl_a at 25m** (poor coverage: 43.3%)
- ✅ **Kept Depth1_Chl_a_lag2W** from 1m depth (better coverage: 63.1%)

### 2. src/dataset.py
- ✅ Updated `time_varying_unknown_reals` to include ALL actual features:
  - Core: Temp_C, Salinity, NO3, NO2, PO4, Silicate
  - Weather: Air_Temp_C, Wind_Speed_ms, Wind_Dir_deg, Wind_U, Wind_V
  - Temporal: Season_sin, Season_cos, Years_since_start, Days_since_prev, Segment_ID
  - Surface: Surface_Temp_C, Surface_O2_umol_L
  - Gradients: Vertical_Temp_Grad, Vertical_O2_Grad
  - Depth1 lags: Depth1_Chl_a_lag2W, Depth1_Nitrat_lag2W, Depth1_Phosphat_lag2W, Depth1_Temp_lag2W
  - Optional: O2_Derivative_1W

### 3. train.py
- ✅ Reduced default **encoder_length: 16 → 8** weeks
- ✅ Reduced default **decoder_length: 8 → 4** weeks
- ✅ Improved feature column logging (sorted, with count)

### 4. tune_hyperparameters.py  
- ✅ Adjusted encoder search range: 12-20 → **6-12** weeks
- ✅ Adjusted decoder search range: 2-8 → **2-6** weeks

## Expected Impact

### Data Loss Reduction:
- **Before**: 85.4% data loss (234/1603 samples)
- **Expected After**: ~20-30% data loss (~1100-1300 samples)
- **Improvement**: ~4-5x more training data

### Why This Works:
1. Removing Chl_a at 25m eliminates worst offender (43% missing)
2. 2-week lag instead of 4-week halves initial NaN period
3. 8-week encoder instead of 16-week requires less consecutive data
4. Together: 10 consecutive complete weeks instead of 20

## Features Retained

All scientifically important features kept:
- ✅ Core oceanographic: Temp, Salinity, NO3, NO2, PO4, Silicate, O2
- ✅ Weather drivers: Wind speed/direction/components, Air temperature
- ✅ Stratification: Vertical gradients, Surface readings
- ✅ Biological proxy: Depth1_Chl_a_lag2W (from 1m, better coverage)
- ✅ Nutrients: Depth1 lagged NO3, PO4, Temp
- ✅ Temporal patterns: Seasonal cycles, long-term trends

## Next Steps

1. **Retrain model** with these optimized settings
2. **Compare metrics** to previous 234-sample training
3. **Verify** data loss is actually reduced to ~20-30%
4. **Tune hyperparameters** with new data size (more samples = may benefit from larger model)

## Commands to Run

```bash
# Quick test (verify data loss improved)
python train.py --max-epochs 5 --batch-size 32

# Full training with defaults
python train.py --max-epochs 100 --patience 3

# Hyperparameter tuning with new ranges
python tune_hyperparameters.py --n-trials 20 --output tuned_hyperparameters_v2.json
```

---

## UPDATE: Weather Features Removed

### Rationale
Wind data (Air_Temp_C, Wind_Speed_ms, Wind_Dir_deg, Wind_U, Wind_V) removed as they have limited direct impact at 25m depth. Wind primarily affects surface mixing, while hypoxia at 25m is driven by:
- Stratification (captured by Vertical_Temp_Grad, Surface_Temp_C)
- Biological processes (captured by Depth1_Chl_a_lag2W, nutrients)
- Deep water properties (captured by Temp_C, Salinity, O2)

### Benefits
1. **Simpler model**: Fewer features = less overfitting risk
2. **Faster training**: Reduced input dimensionality
3. **Better interpretability**: Focus on physically relevant features at depth

### Final Feature Set (19 features + optional O2_Derivative_1W)
- Core oceanographic (6): Temp_C, Salinity, NO3, NO2, PO4, Silicate
- Temporal (5): Season_sin, Season_cos, Years_since_start, Days_since_prev, Segment_ID
- Surface/Stratification (4): Surface_Temp_C, Surface_O2_umol_L, Vertical_Temp_Grad, Vertical_O2_Grad
- Depth1 lagged (4): Depth1_Chl_a_lag2W, Depth1_Nitrat_lag2W, Depth1_Phosphat_lag2W, Depth1_Temp_lag2W
