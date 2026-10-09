# Copernicus Marine Reanalysis Integration

## Overview

The pipeline now supports automatic gap-filling using **Copernicus Marine Service** Baltic Sea Biogeochemical Reanalysis data.

**Product**: BALTICSEA_MULTIYEAR_BGC_003_012
**Dataset**: cmems_mod_bal_bgc_my_P1M-m (monthly means)
**Coverage**: 1993-present
**Resolution**: ~1 nautical mile, 56 depth levels

## Variables Filled from Reanalysis

✅ **O2 (Dissolved Oxygen)** - fills 35% gaps at 25m
✅ **Chl_a (Chlorophyll-a)** - fills 43% gaps at 25m
✅ **NO3 (Nitrate)** - fills 30% gaps at 25m
✅ **PO4 (Phosphate)** - fills 30% gaps at 25m
✅ **Temp_C (Temperature)** - completes rare gaps
✅ **Salinity** - completes rare gaps

❌ **NO2, Silicate** - Not available in reanalysis (but we keep observed data with only 11 sample loss)

## Installation

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

This installs:
- `copernicusmarine>=1.0.0` - Copernicus data access
- `xarray>=2023.1.0` - NetCDF data handling

### 2. Set Up Copernicus Marine Credentials

**Create a free account**: https://data.marine.copernicus.eu/register

**Login via CLI**:
```bash
copernicusmarine login
```

Enter your username and password when prompted. Credentials are stored locally.

## Usage

### Basic Usage (Default - Reanalysis Enabled)

```python
from src import data_ingestion

# Load data with automatic gap-filling
df = data_ingestion.load_and_clean_boknis_data()
```

On first run, this will:
1. Download monthly reanalysis data for Boknis Eck (1993-2023)
2. Cache it locally in `.copernicus_cache/`
3. Interpolate monthly → weekly
4. Fill gaps in observed data
5. Track data source for weight adjustment

**Subsequent runs use the cached data** (fast!)

### Without Reanalysis (Observed Data Only)

```python
# Disable reanalysis gap-filling
df = data_ingestion.load_and_clean_boknis_data(use_reanalysis=False)
```

### Custom Date Range

```python
# Fetch reanalysis for specific period
df = data_ingestion.load_and_clean_boknis_data(
    use_reanalysis=True,
    reanalysis_start="2000-01-01",
    reanalysis_end="2020-12-31"
)
```

### Force Refresh (Re-download Data)

```python
from src import copernicus_reanalysis

# Clear cache and re-download
df_reanalysis = copernicus_reanalysis.load_or_fetch_reanalysis(
    force_refresh=True
)
```

## Data Source Tracking

The pipeline automatically tracks which values come from observations vs. reanalysis:

```python
# After gap-filling, these columns are added:
df['O2_umol_L_source']      # 0.0 = observed, 1.0 = reanalysis
df['Chl_a_source']          # 0.0 = observed, 1.0 = reanalysis
df['NO3_source']            # ...
df['PO4_source']
df['Temp_C_source']
df['Salinity_source']
```

## Weight Adjustment for Reanalysis Data

Reanalysis-filled samples receive **50% weight** by default (configurable).

### How It Works:

1. **Base weight** determined by oxygen severity tier:
   - Severe (<30 µmol/L): 12.0
   - Hypoxic (<60 µmol/L): 6.0
   - Watch (<120 µmol/L): 3.0
   - Normoxic (≥120 µmol/L): 1.0

2. **Reanalysis penalty** applied based on data source:
   - 100% observed: weight × 1.0 (no penalty)
   - 100% reanalysis: weight × 0.5 (50% penalty)
   - 50% mixed: weight × 0.75 (25% penalty)

### Example:

```python
# Severe hypoxic sample with reanalysis O2
base_weight = 12.0  # Severe tier
reanalysis_fraction = 1.0  # 100% reanalysis
penalty = 0.5
final_weight = 12.0 * 0.5 = 6.0

# Same sample with observed O2
reanalysis_fraction = 0.0  # 100% observed
final_weight = 12.0 * 1.0 = 12.0
```

### Adjust Penalty

To change the reanalysis penalty (in `src/labeling.py`):

```python
# More conservative (30% weight for reanalysis)
df = labeling.add_sample_weight(df, reanalysis_penalty=0.3)

# Treat equal to observations (100% weight)
df = labeling.add_sample_weight(df, reanalysis_penalty=1.0)
```

## Expected Impact

### Before Reanalysis:
- **85.4% data loss** (1369/1603 rows dropped)
- **234 samples** remaining after dropna()
- Major gaps in Chl_a (43% missing), O2 (35%), nutrients (30%)

### After Reanalysis:
- **~5-10% data loss** (NO2/Silicate only)
- **~1450-1550 samples** expected (6x more data!)
- Complete coverage for O2, Chl_a, NO3, PO4, Temp, Salinity

## Cache Management

### Cache Location:
```
.copernicus_cache/
├── boknis_eck_reanalysis_1993-01-01_2023-12-31.csv
└── temp_reanalysis.nc (temporary, deleted after processing)
```

### Clear Cache:
```bash
rm -rf .copernicus_cache/
```

### Check Cache:
```python
from pathlib import Path

cache_dir = Path(".copernicus_cache")
if cache_dir.exists():
    files = list(cache_dir.glob("*.csv"))
    print(f"Cached files: {[f.name for f in files]}")
```

## Troubleshooting

### Error: "copernicusmarine package not installed"
```bash
pip install copernicusmarine xarray
```

### Error: "Authentication failed"
```bash
# Re-login with correct credentials
copernicusmarine login

# Or set environment variables
export COPERNICUS_MARINE_SERVICE_USERNAME="your_username"
export COPERNICUS_MARINE_SERVICE_PASSWORD="your_password"
```

### Error: "Dataset not found"
Check that product ID is correct:
- Product: BALTICSEA_MULTIYEAR_BGC_003_012
- Dataset: cmems_mod_bal_bgc_my_P1M-m

Browse available products: https://data.marine.copernicus.eu/products

### Slow First Run
Initial download may take 5-10 minutes depending on connection. Data is cached for subsequent runs.

### Missing Variables
If certain variables fail to download:
1. Check Copernicus service status
2. Verify date range (1993-present)
3. Check credentials and quota

## Data Quality Notes

### Reanalysis Limitations:
- **Monthly resolution** (interpolated to weekly) - less precise than observations
- **Model output** not direct measurements - may have biases
- **Spatial resolution** ~1.8km - point measurements more accurate at station

### When to Use:
✅ Filling sparse historical data (pre-2000)
✅ Creating continuous time series for training
✅ Supplementing gaps in nutrients/chlorophyll

### When to Avoid:
❌ High-precision validation sets (use observed only)
❌ Calibrating sensors
❌ When observed data already complete

## Advanced: Manual Reanalysis Fetching

```python
from src import copernicus_reanalysis

# Fetch reanalysis data
df_reanalysis = copernicus_reanalysis.fetch_copernicus_data(
    start_date="2010-01-01",
    end_date="2020-12-31",
    force_refresh=False
)

# Interpolate to weekly
df_weekly = copernicus_reanalysis.interpolate_monthly_to_weekly(df_reanalysis)

# Merge with observations
df_merged = copernicus_reanalysis.merge_with_observations(
    df_observed=your_observed_data,
    df_reanalysis=df_weekly,
    variables_to_fill=["O2_umol_L", "Chl_a", "NO3", "PO4"]
)
```

## References

- **Copernicus Marine Service**: https://marine.copernicus.eu/
- **Product Documentation**: https://data.marine.copernicus.eu/product/BALTICSEA_MULTIYEAR_BGC_003_012
- **QUID Document**: https://documentation.marine.copernicus.eu/QUID/CMEMS-BAL-QUID-003-012.pdf
- **Python Package**: https://pypi.org/project/copernicusmarine/

## Citation

If you use this reanalysis data in publications:

> Baltic Sea Biogeochemical Reanalysis (BALTICSEA_MULTIYEAR_BGC_003_012),
> Copernicus Marine Service, https://doi.org/10.48670/moi-00012
