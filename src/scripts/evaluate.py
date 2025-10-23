"""
Evaluation script for LSTM Autoencoder
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from tqdm import tqdm

from config import config
from model import create_model
from dataset import load_processed_data, create_dataloaders


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate reconstruction metrics.
    
    Args:
        y_true: Ground truth array of shape (n_samples, seq_length, n_features)
        y_pred: Predicted array of same shape
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays for sklearn metrics
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    metrics = {
        'mse': mean_squared_error(y_true_flat, y_pred_flat),
        'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        'mae': mean_absolute_error(y_true_flat, y_pred_flat),
        'r2': r2_score(y_true_flat, y_pred_flat),
        'explained_variance': explained_variance_score(y_true_flat, y_pred_flat)
    }
    
    # Calculate per-feature metrics
    for i, feature_name in enumerate(config.FEATURE_COLUMNS):
        y_true_feature = y_true[:, :, i].reshape(-1)
        y_pred_feature = y_pred[:, :, i].reshape(-1)
        
        metrics[f'r2_{feature_name}'] = r2_score(y_true_feature, y_pred_feature)
        metrics[f'rmse_{feature_name}'] = np.sqrt(mean_squared_error(y_true_feature, y_pred_feature))
    
    return metrics


def evaluate_model(model: torch.nn.Module, 
                   dataloader,
                   device: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        Tuple of (originals, reconstructions, latents)
    """
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    all_originals = []
    all_reconstructions = []
    all_latents = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            
            reconstructed, latent = model(batch)
            
            all_originals.append(batch.cpu().numpy())
            all_reconstructions.append(reconstructed.cpu().numpy())
            all_latents.append(latent.cpu().numpy())
    
    originals = np.concatenate(all_originals, axis=0)
    reconstructions = np.concatenate(all_reconstructions, axis=0)
    latents = np.concatenate(all_latents, axis=0)
    
    return originals, reconstructions, latents


def plot_reconstruction_samples(originals: np.ndarray,
                                reconstructions: np.ndarray,
                                n_samples: int = 5,
                                save_path: Path = None):
    """
    Plot original vs reconstructed sequences.
    
    Args:
        originals: Original sequences (n_sequences, seq_length, n_features)
        reconstructions: Reconstructed sequences (same shape)
        n_samples: Number of samples to plot
        save_path: Path to save figure
    """
    n_samples = min(n_samples, len(originals))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Randomly select samples
    indices = np.random.choice(len(originals), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        orig = originals[idx]
        recon = reconstructions[idx]
        
        for j, feature_name in enumerate(config.FEATURE_COLUMNS):
            ax = axes[i, j]
            
            # Plot original and reconstruction
            ax.plot(orig[:, j], label='Original', alpha=0.7, linewidth=2)
            ax.plot(recon[:, j], label='Reconstructed', alpha=0.7, linewidth=2, linestyle='--')
            
            ax.set_title(f'Sample {i+1} - {feature_name}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved reconstruction plot to: {save_path}")
    
    plt.close()


def plot_latent_space(latents: np.ndarray, save_path: Path = None):
    """
    Visualize latent space distribution.
    
    Args:
        latents: Latent representations (n_sequences, latent_dim)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Distribution of latent dimensions
    ax1 = axes[0]
    for i in range(min(8, latents.shape[1])):  # Plot first 8 dimensions
        ax1.hist(latents[:, i], bins=50, alpha=0.5, label=f'Dim {i}')
    ax1.set_title('Latent Dimensions Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation heatmap (if latent_dim is small enough)
    ax2 = axes[1]
    if latents.shape[1] <= 32:
        import seaborn as sns
        corr = np.corrcoef(latents.T)
        sns.heatmap(corr, ax=ax2, cmap='coolwarm', center=0, square=True, 
                   cbar_kws={'label': 'Correlation'})
        ax2.set_title('Latent Dimensions Correlation')
    else:
        ax2.text(0.5, 0.5, f'Latent dim too large ({latents.shape[1]})\nto show correlation',
                ha='center', va='center', fontsize=12)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved latent space plot to: {save_path}")
    
    plt.close()


def print_evaluation_report(metrics: Dict[str, float]):
    """Print formatted evaluation report."""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   MSE:  {metrics['mse']:.8f}")
    print(f"   RMSE: {metrics['rmse']:.8f}")
    print(f"   MAE:  {metrics['mae']:.8f}")
    print(f"   R²:   {metrics['r2']:.6f}")
    print(f"   Explained Variance: {metrics['explained_variance']:.6f}")
    
    print(f"\n📈 Per-Feature R² Scores:")
    for feature in config.FEATURE_COLUMNS:
        r2 = metrics[f'r2_{feature}']
        print(f"   {feature:15s}: {r2:.6f}")
    
    print(f"\n📉 Per-Feature RMSE:")
    for feature in config.FEATURE_COLUMNS:
        rmse = metrics[f'rmse_{feature}']
        print(f"   {feature:15s}: {rmse:.8f}")
    
    # Quality assessment
    r2_score = metrics['r2']
    print(f"\n🎯 Quality Assessment:")
    if r2_score >= config.MIN_R2_EXCELLENT:
        print(f"   ⭐⭐⭐ EXCELLENT (R² >= {config.MIN_R2_EXCELLENT})")
    elif r2_score >= config.MIN_R2_GOOD:
        print(f"   ⭐⭐ GOOD (R² >= {config.MIN_R2_GOOD})")
    elif r2_score >= config.MIN_R2_ACCEPTABLE:
        print(f"   ⭐ ACCEPTABLE (R² >= {config.MIN_R2_ACCEPTABLE})")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT (R² < {config.MIN_R2_ACCEPTABLE})")
    
    print("="*60)


def evaluate_phase1(checkpoint_path: Path,
                    processed_file: Path,
                    output_dir: Path = None):
    """
    Complete evaluation for Phase 1.
    
    Args:
        checkpoint_path: Path to model checkpoint
        processed_file: Path to processed data
        output_dir: Directory to save results
    """
    if output_dir is None:
        output_dir = Path("/data/pool/c8x-98x/bridge_data/100_days/data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("PHASE 1: EVALUATION")
    print("="*80)
    
    # Load data
    train, val, test = load_processed_data(processed_file)
    train_loader, val_loader, test_loader = create_dataloaders(train, val, test)
    
    # Load model
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n Loaded model from: {checkpoint_path}")
    print(f"   Training epoch: {checkpoint['epoch']}")
    print(f"   Validation loss: {checkpoint['val_loss']:.6f}")
    
    # Evaluate on test set
    print(f"\n🔍 Evaluating on test set...")
    test_originals, test_reconstructions, test_latents = evaluate_model(
        model, test_loader, config.DEVICE
    )
    
    # Calculate metrics
    test_metrics = calculate_metrics(test_originals, test_reconstructions)
    
    # Print report
    print_evaluation_report(test_metrics)
    
    # Create visualizations
    print(f"\n Creating visualizations...")
    
    # Reconstruction samples
    plot_reconstruction_samples(
        test_originals,
        test_reconstructions,
        n_samples=5,
        save_path=output_dir / "reconstruction_samples.png"
    )
    
    # Latent space
    plot_latent_space(
        test_latents,
        save_path=output_dir / "latent_space.png"
    )
    
    # Save metrics to file
    import json
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n💾 Saved metrics to: {metrics_file}")
    
    # Calculate compression statistics
    original_size = test_originals.nbytes
    compressed_size = test_latents.nbytes
    compression_ratio = original_size / compressed_size
    
    print(f"\n💾 Storage Statistics:")
    print(f"   Original size: {original_size/1024/1024:.2f} MB")
    print(f"   Compressed size: {compressed_size/1024/1024:.2f} MB")
    print(f"   Compression ratio: {compression_ratio:.2f}x")
    print(f"   Space saved: {(1 - compressed_size/original_size)*100:.1f}%")
    
    return test_metrics


if __name__ == "__main__":
    # Evaluate Phase 1
    checkpoint_path = Path("/data/pool/c8x-98x/bridge_data/100_days/data/checkpoints/best.pt")
    processed_file = Path("/data/pool/c8x-98x/bridge_data/100_days/data/processed/20241127_processed.npz")
    output_dir = Path("/data/pool/c8x-98x/bridge_data/100_days/data/results")
    
    if checkpoint_path.exists() and processed_file.exists():
        print("Starting Phase 1 evaluation...")
        
        metrics = evaluate_phase1(
            checkpoint_path=checkpoint_path,
            processed_file=processed_file,
            output_dir=output_dir
        )
        
        print("\n🎉 Phase 1 evaluation complete!")
        
    else:
        if not checkpoint_path.exists():
            print(f" Checkpoint not found: {checkpoint_path}")
            print("Run train.py first!")
        if not processed_file.exists():
            print(f" Processed file not found: {processed_file}")
            print("Run preprocess.py first!")