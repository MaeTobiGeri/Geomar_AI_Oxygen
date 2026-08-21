# Google Colab Training Instructions

This guide explains how to train the hypoxia prediction model on Google Colab for free GPU access.

## Quick Start

1. **Upload the notebook to Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File > Upload notebook`
   - Upload `colab_training.ipynb` from this repository
   - OR click `File > Open notebook > GitHub` and paste your repository URL

2. **Enable GPU** (IMPORTANT for faster training):
   - Click `Runtime > Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Click `Save`

3. **Update Repository URL**:
   - In the first code cell under "Setup Environment", update this line:
     ```python
     REPO_URL = "https://github.com/YOUR_USERNAME/Geomar_AI_Oxygen.git"
     ```
   - Replace `YOUR_USERNAME` with your actual GitHub username

4. **Run the notebook**:
   - Click `Runtime > Run all` to execute all cells
   - OR run cells individually by clicking the play button on each cell

## Notebook Structure

### 1. Setup Environment
- Checks GPU availability
- Clones repository from GitHub
- Installs dependencies from `requirements.txt`
- Optionally mounts Google Drive for checkpoint persistence

### 2. Verify Data and Pipeline
- Checks data files are present
- Runs unit tests to verify pipeline integrity

### 3. Training Options (Choose One)

#### Option A: Quick Test Training (5 epochs)
- Fast test to verify everything works
- ~5-10 minutes on T4 GPU
- Use this first to catch any issues

#### Option B: Full Training (Default Hyperparameters)
- Trains with SPEC.md §7 hyperparameters
- ~1-2 hours (early stopping usually triggers around epoch 20-40)
- Good for baseline model

#### Option C: Hyperparameter Tuning
- Uses Optuna to search hyperparameter space
- WARNING: Takes 4-20 hours depending on `N_TRIALS`
- Automatically trains with best hyperparameters after tuning

#### Option D: Weighted Loss Verification
- Tests that the weighting mechanism works
- Trains two models: extreme weighted vs uniform
- Quick verification (~10-15 minutes)

### 4. View Results and Download Checkpoints
- Views training metadata
- Lists checkpoint files
- Downloads checkpoints if not using Google Drive

### 5. Cleanup (Optional)
- Removes temporary files to free disk space

## Google Drive Integration

### Why Mount Google Drive?
- Checkpoints persist after Colab session ends
- Can resume training from saved checkpoints
- No need to re-download checkpoints

### How to Enable
In the "Mount Google Drive" cell, ensure:
```python
MOUNT_DRIVE = True
```

Colab will prompt you to:
1. Click the authorization link
2. Select your Google account
3. Grant access permissions

Checkpoints will be saved to:
```
/content/drive/MyDrive/Geomar_Checkpoints/
```

### How to Disable
If you don't want to use Drive:
```python
MOUNT_DRIVE = False
```

⚠️ **WARNING**: Checkpoints will be lost when session ends! Download them first.

## Training Time Estimates

On Google Colab's free T4 GPU:

| Training Mode | Time Estimate |
|---------------|---------------|
| Quick test (5 epochs) | 5-10 minutes |
| Full training (100 epochs, early stopping) | 1-2 hours |
| Hyperparameter tuning (20 trials) | 4-8 hours |
| Hyperparameter tuning (50 trials) | 10-20 hours |
| Weighted loss verification | 10-15 minutes |

## Tips and Best Practices

### 1. Always Use GPU Runtime
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```
Should print `True`. If `False`, change runtime type to GPU.

### 2. Monitor GPU Usage
Add a cell with:
```python
!nvidia-smi
```
This shows GPU utilization, memory usage, and temperature.

### 3. Start with Quick Test
Before running long training:
1. Run Option A (Quick Test)
2. Verify it completes without errors
3. Then proceed to full training

### 4. Handle Session Timeouts
Google Colab free tier has limits:
- **12 hour session limit**: Session ends after 12 hours
- **Idle timeout**: Session ends after ~90 minutes of inactivity

**Solutions**:
- Mount Google Drive to save checkpoints
- For long tuning runs, use Colab Pro ($10/month, 24hr sessions)
- Split tuning into multiple runs (e.g., 2x 25 trials instead of 1x 50)

### 5. Resume Training
If session ends mid-training:
1. Rerun setup cells
2. Checkpoints in Drive are preserved
3. Modify training command to resume (if implementing resume functionality)

### 6. Download Checkpoints
If not using Drive, download before session ends:
```python
from google.colab import files
files.download('models/hypoxia_tft/best_model.ckpt')
files.download('models/hypoxia_tft/training_metadata.json')
```

## Troubleshooting

### Problem: "No module named 'src'"
**Solution**: Make sure you ran the "Clone repository" cell. Current directory should be inside the repository.

### Problem: Version conflicts / dependency errors during install
**Expected behavior**: You may see warnings like:
```
google-colab 1.0.0 requires pandas==2.2.3, but you have pandas 3.0.3
```

**Solution**: These warnings are SAFE TO IGNORE. The notebook uses Colab-compatible versions:
- Uses Colab's pre-installed torch, numpy, pandas
- Only installs project-specific packages (pytorch-forecasting, lightning, etc.)
- The `requirements.txt` file is for LOCAL installation only
- Colab uses `requirements-colab.txt` strategy (minimal installs)

### Problem: "CUDA out of memory"
**Solutions**:
- Reduce `--batch-size` (try 32 or 16 instead of 64)
- Reduce `--hidden-size` (try 8 instead of 16)
- Restart runtime: `Runtime > Restart runtime`

### Problem: "Session crashed" during training
**Solutions**:
- Probably GPU memory issue - reduce batch size or model size
- Check GPU memory before training: `!nvidia-smi`

### Problem: Data files not found
**Solution**: Ensure data files are in the repository:
```
Documentation/data/BoknisEck_1957-2014.csv
Documentation/data/BoknisEck_2015-2023.csv
Documentation/data/BoknisEck_chl_2015-2021.tab
```

### Problem: Training is very slow
**Check**:
1. GPU is enabled: `torch.cuda.is_available()` should be `True`
2. GPU is being used: `!nvidia-smi` should show Python process

### Problem: "Rate limit exceeded" when cloning
**Solution**:
- Wait a few minutes and try again
- Or download repository as ZIP and upload to Colab manually

## Output Files

After training completes, you'll have:

### Checkpoint Files
- `best_model.ckpt` - Best model by validation loss
- `last.ckpt` - Last epoch checkpoint (for resuming)

### Metadata Files
- `training_metadata.json` - Full training configuration
  - Hyperparameters used
  - Features selected
  - Weight configuration
  - Dataset info (date ranges, sizes)
  - Training timestamp

### Tuning Files (if using Option C)
- `tuned_hyperparameters.json` - Best hyperparameters from Optuna
  - Best hyperparameters
  - Best validation loss
  - Trial number

## Using Trained Model Locally

After training on Colab, to use the model locally:

1. **Download checkpoint and metadata** from Google Drive or Colab

2. **Place in local repository**:
   ```bash
   mkdir -p models/hypoxia_tft
   mv best_model.ckpt models/hypoxia_tft/
   mv training_metadata.json models/hypoxia_tft/
   ```

3. **Load model** in Python:
   ```python
   from pytorch_forecasting import TemporalFusionTransformer

   model = TemporalFusionTransformer.load_from_checkpoint(
       "models/hypoxia_tft/best_model.ckpt"
   )
   ```

4. **Use for inference** (Phase 9/10 evaluation and dashboard)

## Cost Comparison

| Option | GPU Access | Session Limit | Cost |
|--------|------------|---------------|------|
| **Colab Free** | T4 (limited) | 12 hours | Free |
| **Colab Pro** | T4/P100 (priority) | 24 hours | $10/month |
| **Colab Pro+** | V100/A100 (priority) | No limit | $50/month |
| **Local RTX 3060** | 12GB VRAM | Unlimited | One-time hardware cost |

**Recommendation**: Start with Colab Free for development and testing. Upgrade to Pro if you need longer sessions for hyperparameter tuning.

## Next Steps After Training

1. **Evaluate model performance** (Phase 9):
   - Run evaluation suite on validation set
   - Check metrics on held-out hypoxic episodes
   - Compare weighted vs unweighted model performance

2. **Build dashboard** (Phase 10):
   - Create Streamlit app for interactive forecasting
   - Load trained checkpoint
   - Visualize predictions with uncertainty quantification

3. **Iterate if needed**:
   - If tail metrics are poor, run hyperparameter tuning
   - Adjust weight tiers if imbalance is too aggressive/gentle
   - Add/remove features based on importance analysis

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Documentation/SPEC.md and Documentation/BUILD_PLAN.md
3. Open an issue on GitHub
