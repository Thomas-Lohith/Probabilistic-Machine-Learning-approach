"""
VAE Evaluation Module
Based on working evaluate.py, modified for Variational Autoencoder
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

from config import config


def evaluate_vae(model, test_data, kl_weight=1.0, use_abs=True):

    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Move data to device
        if isinstance(test_data, np.ndarray):
            test_data = torch.FloatTensor(test_data)
        test_data = test_data.to(device)
        
        # Forward pass
        reconstructed, mu, log_var = model(test_data)
        
        # Move to CPU for metric calculation
        original = test_data.cpu().numpy()
        recon = reconstructed.cpu().numpy()
        mu_np = mu.cpu().numpy()
        log_var_np = log_var.cpu().numpy()
        
        # Flatten for metrics
        original_flat = original.reshape(-1, original.shape[-1])
        recon_flat = recon.reshape(-1, recon.shape[-1])
        
        # Basic metrics (same as AE)
        mse = mean_squared_error(original_flat, recon_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(original_flat, recon_flat)
        
        # R² score
        if use_abs:
            r2 = r2_score(np.abs(original_flat), np.abs(recon_flat))
        else:
            r2 = r2_score(original_flat, recon_flat)
        
        # Per-feature metrics
        n_features = original.shape[-1]
        per_feature_metrics = {}
        
        for feat_idx in range(n_features):
            feat_original = original[:, :, feat_idx].flatten()
            feat_recon = recon[:, :, feat_idx].flatten()
            
            feat_mse = mean_squared_error(feat_original, feat_recon)
            feat_rmse = np.sqrt(feat_mse)
            feat_mae = mean_absolute_error(feat_original, feat_recon)
            
            if use_abs:
                feat_r2 = r2_score(np.abs(feat_original), np.abs(feat_recon))
            else:
                feat_r2 = r2_score(feat_original, feat_recon)
            
            per_feature_metrics[f'feature_{feat_idx}'] = {
                'mse': float(feat_mse),
                'rmse': float(feat_rmse),
                'mae': float(feat_mae),
                'r2': float(feat_r2)
            }
        
        # VAE-specific metrics
        # KL divergence
        kl_divergence = -0.5 * np.mean(1 + log_var_np - mu_np**2 - np.exp(log_var_np))
        kl_divergence = kl_divergence / mu_np.shape[0]  # Normalize by batch size
        
        # Reconstruction loss (same as used in training)
        recon_loss = float(F.mse_loss(reconstructed, test_data, reduction='mean').item())
        
        # Total VAE loss (ELBO)
        total_loss = recon_loss + kl_weight * kl_divergence
        
        # Latent space statistics
        latent_mean = np.mean(mu_np, axis=0)
        latent_std = np.std(mu_np, axis=0)
        latent_var_mean = np.mean(np.exp(log_var_np), axis=0)
        
        # Compile metrics
        metrics = {
            # Basic metrics
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            
            # VAE-specific metrics
            'kl_divergence': float(kl_divergence),
            'recon_loss': float(recon_loss),
            'total_loss': float(total_loss),
            
            # Latent space statistics
            'latent_mean': latent_mean.tolist(),
            'latent_std': latent_std.tolist(),
            'latent_var_mean': latent_var_mean.tolist(),
            
            # Per-feature metrics
            'per_feature': per_feature_metrics
        }
        
        return metrics


def print_vae_metrics(metrics):
    """
    Print VAE evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics from evaluate_vae()
    """
    print("\n" + "="*80)
    print("VAE EVALUATION METRICS")
    print("="*80)
    
    # Overall metrics
    print("\n📊 Overall Metrics:")
    print(f"   RMSE:           {metrics['rmse']:.6f}")
    print(f"   MAE:            {metrics['mae']:.6f}")
    print(f"   R² Score:       {metrics['r2']:.6f}")
    print(f"   MSE:            {metrics['mse']:.6f}")
    
    # VAE-specific metrics
    print("\n🔬 VAE Metrics:")
    print(f"   Reconstruction Loss:  {metrics['recon_loss']:.6f}")
    print(f"   KL Divergence:        {metrics['kl_divergence']:.6f}")
    print(f"   Total Loss (ELBO):    {metrics['total_loss']:.6f}")
    
    # KL divergence interpretation
    kl = metrics['kl_divergence']
    print(f"\n💡 KL Divergence Interpretation:")
    if kl < 1.0:
        print(f"   ⚠️  Very low ({kl:.3f}) - possible posterior collapse")
        print(f"      Model may not be using latent space effectively")
    elif kl < 5.0:
        print(f"   ✅ Good range ({kl:.3f}) - healthy latent space")
    elif kl < 10.0:
        print(f"   ⚠️  Slightly high ({kl:.3f}) - may need tuning")
    else:
        print(f"   ❌ Too high ({kl:.3f}) - model struggling")
    
    # Latent space statistics
    print(f"\n🎯 Latent Space Statistics:")
    latent_mean = np.array(metrics['latent_mean'])
    latent_std = np.array(metrics['latent_std'])
    print(f"   Mean (μ):     {latent_mean.mean():.4f} ± {latent_mean.std():.4f}")
    print(f"   Std (σ):      {latent_std.mean():.4f} ± {latent_std.std():.4f}")
    
    # Per-feature metrics
    if 'per_feature' in metrics:
        print(f"\n📈 Per-Feature Metrics:")
        for feat_name, feat_metrics in metrics['per_feature'].items():
            print(f"\n   {feat_name}:")
            print(f"      RMSE: {feat_metrics['rmse']:.6f}")
            print(f"      MAE:  {feat_metrics['mae']:.6f}")
            print(f"      R²:   {feat_metrics['r2']:.6f}")
    
    print("\n" + "="*80)


def load_vae_model(checkpoint_path):

    from model_vae import create_vae_model
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model
    model = create_vae_model()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to eval mode
    model.eval()
    
    print(f"✅ Loaded VAE model from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"   KL Weight: {checkpoint.get('kl_weight', 'unknown')}")
    
    return model


def evaluate_vae_phase1(checkpoint_path, processed_file, output_dir, kl_weight=1.0):
    """
    Evaluate VAE for Phase 1 (single day).
    
    Args:
        checkpoint_path: Path to model checkpoint
        processed_file: Path to processed data
        output_dir: Directory to save results
        kl_weight: KL weight used during training
        
    Returns:
        Dictionary of metrics
    """
    import json
    from dataset import load_processed_data
    from evaluate import plot_reconstruction_samples, plot_latent_space
    
    print("\n" + "="*80)
    print("PHASE 1 VAE EVALUATION")
    print("="*80)
    
    # Load model
    model = load_vae_model(checkpoint_path)
    
    # Load test data
    _, _, test_data = load_processed_data(processed_file)
    test_data = torch.FloatTensor(test_data)
    
    print(f"\nTest data shape: {test_data.shape}")
    
    # Evaluate
    metrics = evaluate_vae(model, test_data, kl_weight=kl_weight, use_abs=True)
    
    # Print metrics
    print_vae_metrics(metrics)
    
    # Get reconstructions and latent for visualization
    model.eval()
    with torch.no_grad():
        test_data_device = test_data.to(config.DEVICE)
        reconstructed, mu, log_var = model(test_data_device)
        
        test_originals = test_data.cpu().numpy()
        test_reconstructions = reconstructed.cpu().numpy()
        test_latents = mu.cpu().numpy()
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot reconstruction samples
    plot_reconstruction_samples(
        test_originals,
        test_reconstructions,
        n_samples=5,
        save_path=results_dir / "vae_reconstruction_samples.png",
        use_abs=True
    )
    print(f"   ✅ Reconstruction samples saved")
    
    # Plot latent space
    plot_latent_space(
        test_latents,
        save_path=results_dir / "vae_latent_space.png"
    )
    print(f"   ✅ Latent space plot saved")
    
    # Save metrics
    metrics_path = results_dir / "vae_test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved: {metrics_path}")
    
    print(f"\n✅ Results saved to: {results_dir}")
    
    return metrics


if __name__ == "__main__":
    print("VAE Evaluation Module")
    print("Import this module and use evaluate_vae() or evaluate_vae_phase1()")