"""Autoencoder-based feature selection for hypoxia prediction.

Implements feature selection approach from reference paper (AE-DeepAR):
- Trains shallow autoencoder on engineered features
- Calculates feature importance scores: S = diag(W₁ᵀW₁)
- Selects top N features based on reconstruction contribution
- Provides both automatic selection and correlation-based filtering

Based on BUILD_PLAN.md Phase 5 enhancements and reference model comparison.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json


class FeatureAutoencoder(nn.Module):
    """Shallow autoencoder for feature selection.

    Architecture:
        Input → Dense(hidden_dim) → Sigmoid → Dense(input_dim) → Output

    The encoder weights W₁ are used to calculate feature importance scores.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 8):
        """Initialize autoencoder.

        Args:
            input_dim: Number of input features
            hidden_dim: Size of latent representation (default: 8)
        """
        super().__init__()

        # Encoder: compress features
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()

        # Decoder: reconstruct features
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """Forward pass through autoencoder.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Reconstructed features [batch_size, input_dim]
        """
        # Encode
        latent = self.activation(self.encoder(x))

        # Decode
        reconstructed = self.decoder(latent)

        return reconstructed

    def encode(self, x):
        """Encode input to latent representation.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Latent representation [batch_size, hidden_dim]
        """
        return self.activation(self.encoder(x))

    def get_feature_scores(self) -> np.ndarray:
        """Calculate feature importance scores from encoder weights.

        Per reference paper: S = diag(W₁ᵀW₁)
        Higher scores indicate more important features for reconstruction.

        Returns:
            Array of feature scores [input_dim]
        """
        # Get encoder weights (W₁)
        W1 = self.encoder.weight.data.cpu().numpy()  # Shape: [hidden_dim, input_dim]

        # Calculate S = diag(W₁ᵀW₁)
        scores = np.diag(W1.T @ W1)

        return scores


def train_autoencoder(
    features_df: pd.DataFrame,
    feature_columns: List[str],
    hidden_dim: int = 8,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    device: str = "cpu"
) -> Tuple[FeatureAutoencoder, Dict]:
    """Train autoencoder on feature data.

    Args:
        features_df: DataFrame with engineered features
        feature_columns: List of column names to use as features
        hidden_dim: Latent dimension size (default: 8)
        epochs: Training epochs (default: 100)
        learning_rate: Learning rate (default: 0.001)
        batch_size: Batch size (default: 64)
        device: Device to train on ("cpu" or "cuda")

    Returns:
        Tuple of (trained_model, training_stats)
    """
    # Extract feature matrix
    X = features_df[feature_columns].values

    # Normalize features (important for autoencoder training)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8  # Avoid division by zero
    X_normalized = (X - mean) / std

    # Convert to tensor
    X_tensor = torch.FloatTensor(X_normalized).to(device)

    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Initialize model
    input_dim = len(feature_columns)
    model = FeatureAutoencoder(input_dim, hidden_dim).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    training_stats = {
        'losses': [],
        'mean': mean,
        'std': std
    }

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_X, in dataloader:
            # Forward pass
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_X)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        training_stats['losses'].append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    model.eval()

    return model, training_stats


def select_features_by_scores(
    feature_scores: np.ndarray,
    feature_columns: List[str],
    n_features: Optional[int] = None,
    threshold: Optional[float] = None
) -> List[str]:
    """Select top features based on autoencoder scores.

    Args:
        feature_scores: Array of feature importance scores
        feature_columns: List of feature names
        n_features: Number of top features to select (if None, use threshold)
        threshold: Score threshold for selection (if None, use n_features)

    Returns:
        List of selected feature names
    """
    # Create feature-score mapping
    feature_score_pairs = list(zip(feature_columns, feature_scores))

    # Sort by score (descending)
    feature_score_pairs.sort(key=lambda x: x[1], reverse=True)

    if n_features is not None:
        # Select top N features
        selected = [name for name, score in feature_score_pairs[:n_features]]
    elif threshold is not None:
        # Select features above threshold
        selected = [name for name, score in feature_score_pairs if score > threshold]
    else:
        raise ValueError("Must specify either n_features or threshold")

    return selected


def correlation_based_filtering(
    features_df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    correlation_threshold: float = 0.5
) -> List[str]:
    """Filter features based on correlation with target variable.

    Per reference paper: select features with Pearson correlation > threshold.

    Args:
        features_df: DataFrame with features and target
        target_column: Name of target column
        feature_columns: List of candidate feature names
        correlation_threshold: Minimum absolute correlation (default: 0.5)

    Returns:
        List of features passing correlation threshold
    """
    # Calculate correlations
    correlations = features_df[feature_columns].corrwith(
        features_df[target_column]
    ).abs()

    # Filter by threshold
    selected = correlations[correlations >= correlation_threshold].index.tolist()

    return selected


def hybrid_feature_selection(
    features_df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    correlation_threshold: float = 0.3,
    n_autoencoder_features: Optional[int] = None,
    autoencoder_threshold: Optional[float] = None,
    hidden_dim: int = 8,
    epochs: int = 100,
    device: str = "cpu"
) -> Dict:
    """Perform hybrid feature selection combining correlation and autoencoder.

    Process:
    1. Filter by correlation with target (initial screening)
    2. Train autoencoder on correlated features
    3. Select top features by autoencoder scores

    Args:
        features_df: DataFrame with features and target
        target_column: Name of target column
        feature_columns: List of candidate feature names
        correlation_threshold: Minimum correlation for initial filter (default: 0.3)
        n_autoencoder_features: Number of features to select via autoencoder
        autoencoder_threshold: Score threshold for autoencoder selection
        hidden_dim: Autoencoder latent dimension (default: 8)
        epochs: Autoencoder training epochs (default: 100)
        device: Device for training ("cpu" or "cuda")

    Returns:
        Dictionary with selection results and diagnostics
    """
    print("\n" + "="*80)
    print("HYBRID FEATURE SELECTION")
    print("="*80)

    # Step 1: Correlation-based filtering
    print(f"\nStep 1: Correlation filtering (threshold: {correlation_threshold})")
    print(f"  Candidate features: {len(feature_columns)}")

    correlated_features = correlation_based_filtering(
        features_df,
        target_column,
        feature_columns,
        correlation_threshold
    )

    print(f"  Features passing correlation threshold: {len(correlated_features)}")

    if len(correlated_features) == 0:
        print("  WARNING: No features passed correlation threshold!")
        return {
            'selected_features': [],
            'correlation_filtered': [],
            'autoencoder_scores': {},
            'model': None,
            'stats': None
        }

    # Step 2: Train autoencoder on correlated features
    print(f"\nStep 2: Training autoencoder (hidden_dim: {hidden_dim}, epochs: {epochs})")

    model, stats = train_autoencoder(
        features_df,
        correlated_features,
        hidden_dim=hidden_dim,
        epochs=epochs,
        device=device
    )

    # Step 3: Get feature scores
    print("\nStep 3: Calculating feature importance scores")
    feature_scores = model.get_feature_scores()

    # Create score mapping
    score_dict = dict(zip(correlated_features, feature_scores))

    # Print top 10 features by score
    sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 features by autoencoder score:")
    for i, (feat, score) in enumerate(sorted_features[:10], 1):
        print(f"    {i}. {feat}: {score:.4f}")

    # Step 4: Select final features
    print("\nStep 4: Final feature selection")

    selected_features = select_features_by_scores(
        feature_scores,
        correlated_features,
        n_features=n_autoencoder_features,
        threshold=autoencoder_threshold
    )

    print(f"  Selected features: {len(selected_features)}")
    print(f"  Feature names: {', '.join(selected_features)}")

    # Return results
    return {
        'selected_features': selected_features,
        'correlation_filtered': correlated_features,
        'autoencoder_scores': score_dict,
        'all_scores': dict(zip(correlated_features, feature_scores)),
        'model': model,
        'stats': stats
    }


def save_feature_selection_results(
    results: Dict,
    output_path: str
):
    """Save feature selection results to JSON.

    Args:
        results: Results dictionary from hybrid_feature_selection()
        output_path: Path to save JSON file
    """
    # Prepare serializable results (exclude model)
    serializable = {
        'selected_features': results['selected_features'],
        'correlation_filtered': results['correlation_filtered'],
        'autoencoder_scores': results['autoencoder_scores'],
        'num_selected': len(results['selected_features']),
        'num_correlated': len(results['correlation_filtered'])
    }

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nFeature selection results saved to: {output_path}")
