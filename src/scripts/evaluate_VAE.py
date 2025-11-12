"""
Evaluation script for LSTM Variational Autoencoder (VAE)

This is parallel to evaluate.py but for VAE model
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

from config import config
from VAE import LSTMVariationalAutoencoder, vae_loss_function
from utils import set_seed


def load_vae_model(checkpoint_path: str, device: str = None) -> LSTMVariationalAutoencoder:

    if device is None:
        device = config.DEVICE
    
    print(f"\nLoading VAE model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with saved config
    model_config = checkpoint.get('config', {})
    model = LSTMVariationalAutoencoder(
        input_dim=model_config.get('input_dim', config.N_FEATURES),
        latent_dim=model_config.get('latent_dim', config.LATENT_DIM),
        output_dim=model_config.get('output_dim', config.N_FEATURES),
        seq_length=model_config.get('seq_length', config.SEQUENCE_LENGTH)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f" Model loaded from epoch {checkpoint['epoch']}")
    print(f"   Best val loss: {checkpoint['best_val_loss']:.6f}")
    
    return model


@torch.no_grad()
def evaluate_vae(model: LSTMVariationalAutoencoder,
                 X_test: torch.Tensor,
                 device: str = None,
                 kl_weight: float = 1.0,
                 use_abs: bool = True) -> Dict:

    if device is None:
        device = config.DEVICE
    
    model.eval()
    X_test = X_test.to(device)
    
    print("\n🔍 Evaluating VAE model...")
    print(f"Test data shape: {X_test.shape}")
    
    # Forward pass
    reconstructed, mu, log_var = model(X_test)
    
    # Calculate VAE loss
    total_loss, loss_dict = vae_loss_function(
        reconstructed, X_test, mu, log_var, kl_weight=kl_weight
    )
    
    # Convert to numpy for metric calculation
    y_true = X_test.cpu().numpy()
    y_pred = reconstructed.cpu().numpy()
    mu_np = mu.cpu().numpy()
    log_var_np = log_var.cpu().numpy()
    
    # Apply absolute value if requested (for consistency with preprocessing)
    if use_abs:
        y_true = np.abs(y_true)
        y_pred = np.abs(y_pred)
    
    # Flatten for metrics
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    # Calculate reconstruction metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    ev = explained_variance_score(y_true_flat, y_pred_flat)
    
    # Latent space statistics
    latent_stats = {
        'mu_mean': np.mean(mu_np),
        'mu_std': np.std(mu_np),
        'log_var_mean': np.mean(log_var_np),
        'log_var_std': np.std(log_var_np),
        'sigma_mean': np.mean(np.exp(0.5 * log_var_np)),  # std = exp(0.5 * log_var)
    }
    
    metrics = {
        # VAE losses
        'total_loss': loss_dict['total_loss'],
        'mse_loss': loss_dict['mse_loss'],
        'kl_divergence': loss_dict['kl_divergence'],
        'weighted_kl': loss_dict['weighted_kl'],
        
        # Reconstruction metrics
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': ev,
        
        # Latent space statistics
        'latent_stats': latent_stats
    }
    
    return metrics


def print_vae_metrics(metrics: Dict):
    """Print VAE evaluation metrics in a formatted way."""
    print("\n" + "="*60)
    print("VAE EVALUATION METRICS")
    print("="*60)
    
    print("\n Loss Components:")
    print(f"  Total Loss:       {metrics['total_loss']:.6f}")
    print(f"  MSE Loss:         {metrics['mse_loss']:.6f}")
    print(f"  KL Divergence:    {metrics['kl_divergence']:.6f}")
    print(f"  Weighted KL:      {metrics['weighted_kl']:.6f}")
    
    print("\n Reconstruction Quality:")
    print(f"  MSE:              {metrics['mse']:.6f}")
    print(f"  RMSE:             {metrics['rmse']:.6f}")
    print(f"  MAE:              {metrics['mae']:.6f}")
    print(f"  R² Score:         {metrics['r2']:.6f}")
    print(f"  Explained Var:    {metrics['explained_variance']:.6f}")
    
    # R² interpretation
    r2 = metrics['r2']
    if r2 >= 0.95:
        print(f"  Quality:          ⭐⭐⭐ EXCELLENT")
    elif r2 >= 0.92:
        print(f"  Quality:          ⭐⭐ GOOD")
    elif r2 >= 0.85:
        print(f"  Quality:          ⭐ ACCEPTABLE")
    elif r2 >= 0.0:
        print(f"  Quality:          ⚠️  POOR")
    else:
        print(f"  Quality:          ❌ BROKEN (Negative R²)")
    
    print("\n🎲 Latent Space Statistics:")
    ls = metrics['latent_stats']
    print(f"  μ (mean):         {ls['mu_mean']:.6f} ± {ls['mu_std']:.6f}")
    print(f"  log(σ²) (mean):   {ls['log_var_mean']:.6f} ± {ls['log_var_std']:.6f}")
    print(f"  σ (mean):         {ls['sigma_mean']:.6f}")
    
    print("\n💡 Interpretation:")
    print("  - MSE Loss: Reconstruction error (lower is better)")
    print("  - KL Divergence: How far latent distribution is from N(0,1)")
    print("  - Low KL: Model uses latent space effectively")
    print("  - High KL: Model may ignore latent space (posterior collapse)")
    
    print("="*60 + "\n")


def visualize_vae_latent_space(model: LSTMVariationalAutoencoder,
                                X_test: torch.Tensor,
                                save_path: Path = None,
                                device: str = None):
 
    if device is None:
        device = config.DEVICE
    
    model.eval()
    X_test = X_test.to(device)
    
    # Encode to get mu and log_var
    with torch.no_grad():
        mu, log_var = model.encoder(X_test)
        sigma = torch.exp(0.5 * log_var)
    
    mu_np = mu.cpu().numpy()
    sigma_np = sigma.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('VAE Latent Space Analysis', fontsize=16)
    
    # Plot 1: Distribution of mu (first 2 dimensions)
    ax = axes[0, 0]
    if mu_np.shape[1] >= 2:
        ax.scatter(mu_np[:, 0], mu_np[:, 1], alpha=0.5, s=10)
        ax.set_xlabel('Latent Dim 0 (μ)')
        ax.set_ylabel('Latent Dim 1 (μ)')
        ax.set_title('Latent Space (μ) - First 2 Dimensions')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Need at least 2 latent dims', ha='center', va='center')
    
    # Plot 2: Distribution of sigma (first 2 dimensions)
    ax = axes[0, 1]
    if sigma_np.shape[1] >= 2:
        scatter = ax.scatter(sigma_np[:, 0], sigma_np[:, 1], 
                           c=np.arange(len(sigma_np)), cmap='viridis', 
                           alpha=0.5, s=10)
        ax.set_xlabel('Latent Dim 0 (σ)')
        ax.set_ylabel('Latent Dim 1 (σ)')
        ax.set_title('Latent Space (σ) - First 2 Dimensions')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Sample Index')
    else:
        ax.text(0.5, 0.5, 'Need at least 2 latent dims', ha='center', va='center')
    
    # Plot 3: Histogram of mu values
    ax = axes[1, 0]
    ax.hist(mu_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='N(0,1) mean')
    ax.set_xlabel('μ values')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of μ (should be centered at 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of sigma values
    ax = axes[1, 1]
    ax.hist(sigma_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(1, color='red', linestyle='--', linewidth=2, label='N(0,1) std')
    ax.set_xlabel('σ values')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of σ (should be centered at 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Latent space visualization saved: {save_path}")
    
    plt.close()


def compare_ae_vs_vae(ae_metrics: Dict, vae_metrics: Dict):
  
    print("\n" + "="*60)
    print("AUTOENCODER vs VAE COMPARISON")
    print("="*60)
    
    print("\n📊 Reconstruction Quality:")
    print(f"{'Metric':<20} {'Autoencoder':<15} {'VAE':<15} {'Winner':<10}")
    print("-" * 60)
    
    metrics_to_compare = ['r2', 'mse', 'rmse', 'mae']
    
    for metric in metrics_to_compare:
        ae_val = ae_metrics.get(metric, 0)
        vae_val = vae_metrics.get(metric, 0)
        
        # For R², higher is better. For others, lower is better
        if metric == 'r2':
            winner = "AE" if ae_val > vae_val else "VAE"
            symbol = "🏆" if winner == "VAE" else ""
        else:
            winner = "AE" if ae_val < vae_val else "VAE"
            symbol = "🏆" if winner == "VAE" else ""
        
        print(f"{metric.upper():<20} {ae_val:<15.6f} {vae_val:<15.6f} {winner:<10} {symbol}")
    
    print("\n💡 Key Differences:")
    print("  ✅ Autoencoder: Deterministic, direct compression")
    print("  ✅ VAE: Probabilistic, learns distribution, regularized latent space")
    print("  ✅ VAE typically has slightly worse reconstruction (due to regularization)")
    print("  ✅ VAE has better latent space structure (good for generation/interpolation)")
    
    print("="*60 + "\n")


def evaluate_vae_phase1(checkpoint_path: str = None,
                        data_path: str = None,
                        kl_weight: float = 1.0,
                        save_visualizations: bool = True):

    set_seed(42)
    
    # Paths
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_DIR / "vae_best_model.pt"
    if data_path is None:
        data_path = config.PROCESSED_DATA_DIR / "processed_data.npz"
    
    # Load model
    model = load_vae_model(checkpoint_path)
    
    # Load test data
    print(f"\n Loading test data from: {data_path}")
    data = np.load(data_path)
    X_test = torch.FloatTensor(data['X_test'])
    print(f"Test shape: {X_test.shape}")
    
    # Evaluate
    metrics = evaluate_vae(model, X_test, kl_weight=kl_weight, use_abs=True)
    
    # Print metrics
    print_vae_metrics(metrics)
    
    # Save visualizations
    if save_visualizations:
        vis_dir = config.FIGURES_DIR / "vae"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Latent space visualization
        latent_path = vis_dir / "vae_latent_space.png"
        visualize_vae_latent_space(model, X_test, latent_path)
    
    # Save metrics
    metrics_path = config.METRICS_DIR / "vae_test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            metrics_json[key] = {k: float(v) for k, v in value.items()}
        else:
            metrics_json[key] = float(value)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f" Metrics saved: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    print("Evaluating LSTM Variational Autoencoder (VAE)...")
    
    # Evaluate VAE
    vae_metrics = evaluate_vae_phase1(
        kl_weight=1.0,
        save_visualizations=True
    )
    
    print("\n VAE evaluation complete!")