"""
VAE Inference Script for Bridge Accelerometer Data Compression
==============================================================
Comprehensive inference tool supporting 5 modes:
1. compress   - Encode data to latent space (7-8x compression)
2. reconstruct - Decode from latent space
3. roundtrip  - Full quality test (encode → decode → compare)
4. anomaly    - Detect unusual patterns
5. generate   - Create synthetic data
Usage:
    python inference_vae.py compress data.csv --output results/
    python inference_vae.py roundtrip data.csv --output results/ --visualize
    python inference_vae.py anomaly data.csv --output results/ --threshold auto
    python inference_vae.py generate --output results/ --n-samples 100
Requirements:
    - Trained VAE checkpoint (vae_best_model.pt)
    - Scaler (auto-loaded from checkpoint directory)
    - Your existing modules (config, model_vae, load_data, etc.)
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json
import joblib
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Import from your existing modules
try:
    from config import config
    from model_vae import create_vae_model
    from load_data import load_single_csv, extract_features
except ImportError as e:
    print(f" ERROR: Could not import required modules: {e}")
    print("Make sure you're running from: /data/pool/c8x-98x/pml/src/scripts/")
    sys.exit(1)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def load_checkpoint(checkpoint_path, device=None):
    """Load VAE model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\n Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = create_vae_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
    
    return model, checkpoint

def load_scaler(checkpoint_path):
    """Auto-load scaler from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path).parent
    scaler_path = checkpoint_dir.parent / "processed" / "scaler.pkl"
    
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f" Loaded scaler: {scaler_path}")
        return scaler
    else:
        print(f" Scaler not found: {scaler_path}")
        return None

def standardize_data(data, scaler=None):
    """Standardize data using scaler."""
    original_shape = data.shape
    data_2d = data.reshape(-1, data.shape[-1])
    
    if scaler is None:
        scaler = StandardScaler()
        data_std = scaler.fit_transform(data_2d)
    else:
        data_std = scaler.transform(data_2d)
    
    return data_std.reshape(original_shape), scaler

def create_sequences(features, sequence_length):
    """Create non-overlapping sequences."""
    n_timesteps, n_features = features.shape
    n_sequences = n_timesteps // sequence_length
    
    truncated_length = n_sequences * sequence_length
    features_truncated = features[:truncated_length]
    
    sequences = features_truncated.reshape(n_sequences, sequence_length, n_features)
    return sequences

def load_csv_file(file_path, scaler=None):
    """Load and preprocess single CSV file."""
    print(f"\n Loading: {file_path}")
    
    df = load_single_csv(file_path)
    features = extract_features(df)
    sequences = create_sequences(features, config.SEQUENCE_LENGTH)
    
    print(f" Created {len(sequences)} sequences")
    
    sequences, scaler = standardize_data(sequences, scaler)
    return sequences, scaler

# ============================================================================
# MODE 1: COMPRESS
# ============================================================================
def compress_mode(args):
    """Compress data to latent space."""
    print("\n" + "#"*80)
    print("MODE: COMPRESS")
    print("#"*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and scaler
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    scaler = load_scaler(args.checkpoint) if not args.no_scaler else None
    
    # Load data
    input_path = Path(args.input)
    if input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        all_sequences = []
        for csv_file in csv_files:
            sequences, scaler = load_csv_file(csv_file, scaler)
            all_sequences.append(sequences)
        sequences = np.concatenate(all_sequences, axis=0)
    elif input_path.suffix == '.csv':
        sequences, scaler = load_csv_file(input_path, scaler)
    else:
        data = np.load(input_path)
        sequences = data['test'] if 'test' in data else data
        sequences, scaler = standardize_data(sequences, scaler)
    
    print(f"\n Encoding {len(sequences)} sequences...")
    
    # Encode
    model.eval()
    with torch.no_grad():
        sequences_tensor = torch.from_numpy(sequences).float().to(device)
        latent_vectors = model.encode(sequences_tensor).cpu().numpy()
    
    # Calculate compression
    original_size = sequences.nbytes
    compressed_size = latent_vectors.nbytes
    ratio = original_size / compressed_size
    
    print(f"\n Compression Results:")
    print(f"   Original: {original_size/1024:.2f} KB")
    print(f"   Compressed: {compressed_size/1024:.2f} KB")
    print(f"   Ratio: {ratio:.2f}x")
    print(f"   Saved: {(original_size-compressed_size)/1024:.2f} KB")
    
    # Save
    output_file = output_dir / "compressed.npz"
    np.savez_compressed(
        output_file,
        latent=latent_vectors,
        shape=sequences.shape,
        ratio=ratio
    )
    
    print(f"\n Saved: {output_file}")
    return latent_vectors

# ============================================================================
# MODE 2: RECONSTRUCT
# ============================================================================
def reconstruct_mode(args):
    """Reconstruct from latent vectors or CSV."""
    print("\n" + "="*80)
    print("MODE: RECONSTRUCT")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    scaler = load_scaler(args.checkpoint) if not args.no_scaler else None
    
    # Load input
    input_path = Path(args.input)
    if input_path.suffix in ['.npz', '.npy']:
        data = np.load(input_path)
        if 'latent' in data:
            latent = data['latent']
            seq_length = data['shape'][1] if 'shape' in data else config.SEQUENCE_LENGTH
        else:
            # It's sequences, encode first
            sequences = data['test'] if 'test' in data else data
            sequences, scaler = standardize_data(sequences, scaler)
            with torch.no_grad():
                seq_tensor = torch.from_numpy(sequences).float().to(device)
                latent = model.encode(seq_tensor).cpu().numpy()
            seq_length = sequences.shape[1]
    else:
        # CSV input
        sequences, scaler = load_csv_file(input_path, scaler)
        with torch.no_grad():
            seq_tensor = torch.from_numpy(sequences).float().to(device)
            latent = model.encode(seq_tensor).cpu().numpy()
        seq_length = sequences.shape[1]
    
    print(f"\n Reconstructing...")
    
    # Decode
    model.eval()
    with torch.no_grad():
        latent_tensor = torch.from_numpy(latent).float().to(device)
        reconstructed = model.decode(latent_tensor, seq_length).cpu().numpy()
    
    # Inverse standardize
    if scaler is not None:
        shape = reconstructed.shape
        reconstructed = scaler.inverse_transform(
            reconstructed.reshape(-1, shape[-1])
        ).reshape(shape)
    
    # Save
    output_file = output_dir / "reconstructed.npz"
    np.savez_compressed(output_file, reconstructed=reconstructed)
    
    print(f"\n Saved: {output_file}")
    return reconstructed

# ============================================================================
# MODE 3: ROUNDTRIP (Most Comprehensive)
# ============================================================================
def roundtrip_mode(args):
    """Complete quality test: encode → decode → compare."""
    print("\n" + "="*80)
    print("MODE: ROUNDTRIP (QUALITY TEST)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    scaler = load_scaler(args.checkpoint) if not args.no_scaler else None
    
    # Load data
    input_path = Path(args.input)
    if input_path.suffix == '.csv':
        sequences, scaler = load_csv_file(input_path, scaler)
    elif input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        all_seq = []
        for f in csv_files:
            seq, scaler = load_csv_file(f, scaler)
            all_seq.append(seq)
        sequences = np.concatenate(all_seq, axis=0)
    else:
        data = np.load(input_path)
        sequences = data['test'] if 'test' in data else data
        sequences, scaler = standardize_data(sequences, scaler)
    
    print(f"\n Testing {len(sequences)} sequences...")
    
    # Encode
    model.eval()
    with torch.no_grad():
        seq_tensor = torch.from_numpy(sequences).float().to(device)
        latent = model.encode(seq_tensor)
        latent_np = latent.cpu().numpy()
    
    # Compression stats
    orig_size = sequences.nbytes
    comp_size = latent_np.nbytes
    ratio = orig_size / comp_size
    
    print(f"\n Compression: {ratio:.2f}x")
    
    # Decode
    with torch.no_grad():
        reconstructed = model.decode(latent, sequences.shape[1]).cpu().numpy()
    
    # Compute metrics
    mse = np.mean((sequences - reconstructed) ** 2)
    mae = np.mean(np.abs(sequences - reconstructed))
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((sequences - reconstructed) ** 2)
    ss_tot = np.sum((sequences - sequences.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n Metrics (Standardized Scale):")
    print(f"   R²:   {r2:.6f}")
    print(f"   MSE:  {mse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    
    # Original scale metrics
    if scaler is not None:
        seq_orig = scaler.inverse_transform(
            sequences.reshape(-1, sequences.shape[-1])
        ).reshape(sequences.shape)
        rec_orig = scaler.inverse_transform(
            reconstructed.reshape(-1, reconstructed.shape[-1])
        ).reshape(reconstructed.shape)
        
        mse_orig = np.mean((seq_orig - rec_orig) ** 2)
        mae_orig = np.mean(np.abs(seq_orig - rec_orig))
        
        print(f"\n Metrics (Original Scale):")
        print(f"   MSE: {mse_orig:.6f}")
        print(f"   MAE: {mae_orig:.6f}")
    
    # Save results
    metrics = {
        'r2': float(r2),
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'compression_ratio': float(ratio)
    }
    
    if scaler is not None:
        metrics['original_scale'] = {
            'mse': float(mse_orig),
            'mae': float(mae_orig)
        }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    np.savez_compressed(
        output_dir / "roundtrip.npz",
        original=sequences,
        reconstructed=reconstructed,
        latent=latent_np
    )
    
    # Visualizations
    if args.visualize:
        print(f"\n Creating visualizations...")
        visualize_roundtrip(sequences, reconstructed, latent_np, output_dir, scaler)
    
    print(f"\n Results saved to: {output_dir}")
    return metrics

# ============================================================================
# MODE 4: ANOMALY DETECTION
# ============================================================================
def anomaly_mode(args):
    """Detect anomalous sequences based on reconstruction error."""
    print("\n" + "="*80)
    print("MODE: ANOMALY DETECTION")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    scaler = load_scaler(args.checkpoint) if not args.no_scaler else None
    
    # Load data
    input_path = Path(args.input)
    if input_path.suffix == '.csv':
        sequences, scaler = load_csv_file(input_path, scaler)
    elif input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        all_seq = []
        for f in csv_files:
            seq, scaler = load_csv_file(f, scaler)
            all_seq.append(seq)
        sequences = np.concatenate(all_seq, axis=0)
    else:
        data = np.load(input_path)
        sequences = data['test'] if 'test' in data else data
        sequences, scaler = standardize_data(sequences, scaler)
    
    print(f"\n Analyzing {len(sequences)} sequences...")
    
    # Encode-decode
    model.eval()
    with torch.no_grad():
        seq_tensor = torch.from_numpy(sequences).float().to(device)
        latent = model.encode(seq_tensor)
        reconstructed = model.decode(latent, sequences.shape[1]).cpu().numpy()
    
    # Compute per-sequence errors
    mse_per_seq = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
    mae_per_seq = np.mean(np.abs(sequences - reconstructed), axis=(1, 2))

    print('the mse per seq is:', mse_per_seq)
    
    # Determine threshold
    if args.threshold == 'auto':
        threshold = np.percentile(mae_per_seq, 90)
        print(f"\n Auto threshold (90th percentile): {threshold:.6f}")
    else:
        threshold = float(args.threshold)
        print(f"\n Manual threshold: {threshold}")

    
    # Find anomalies
    anomalies = mse_per_seq > threshold
    n_anomalies = np.sum(anomalies)
    pct = (n_anomalies / len(sequences)) * 100
    
    print(f"\n Results:")
    print(f"   Total sequences: {len(sequences)}")
    print(f"   Anomalies: {n_anomalies} ({pct:.2f}%)")
    print(f"   Normal: {len(sequences) - n_anomalies}")
    
    # Top anomalies
    print(f"\n Top 10 anomalies:")
    top_idx = np.argsort(mse_per_seq)[-10:][::-1]
    for i, idx in enumerate(top_idx, 1):
        print(f"   {i}. Seq {idx}: MSE={mse_per_seq[idx]:.6f}")
    
    # Save report
    report = {
        'total': int(len(sequences)),
        'anomalies': int(n_anomalies),
        'percentage': float(pct),
        'threshold': float(threshold),
        'anomaly_indices': np.where(anomalies)[0].tolist(),
        'scores': mse_per_seq.tolist()
    }
    
    with open(output_dir / "anomaly_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    np.savez_compressed(
        output_dir / "anomalies.npz",
        anomalous=sequences[anomalies],
        normal=sequences[~anomalies],
        indices=np.where(anomalies)[0]
    )
    
    # Visualizations
    if args.visualize:
        print(f"\n Creating visualizations...")
        visualize_anomalies(mse_per_seq, anomalies, threshold, output_dir)
    
    print(f"\n Results saved to: {output_dir}")
    return report

# ============================================================================
# MODE 5: GENERATE
# ============================================================================
def generate_mode(args):
    """Generate synthetic sequences from latent space."""
    print("\n" + "="*80)
    print("MODE: GENERATE SYNTHETIC DATA")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    scaler = load_scaler(args.checkpoint) if not args.no_scaler else None
    
    n_samples = args.n_samples if hasattr(args, 'n_samples') else 100
    seq_length = config.SEQUENCE_LENGTH
    
    print(f"\n Generating {n_samples} sequences...")
    
    # Generate
    generated = model.sample(n_samples, seq_length, device=device).cpu().numpy()
    
    # Inverse standardize
    if scaler is not None:
        shape = generated.shape
        generated = scaler.inverse_transform(
            generated.reshape(-1, shape[-1])
        ).reshape(shape)
    
    print(f"\n Generated data:")
    print(f"   Shape: {generated.shape}")
    print(f"   Min: {generated.min():.4f}")
    print(f"   Max: {generated.max():.4f}")
    print(f"   Mean: {generated.mean():.4f}")
    
    # Save
    output_file = output_dir / "generated.npz"
    np.savez_compressed(output_file, generated=generated)
    
    # Visualizations
    if args.visualize:
        print(f"\n Creating visualizations...")
        visualize_generated(generated, output_dir)
    
    print(f"\n Saved: {output_file}")
    return generated

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def visualize_roundtrip(original, reconstructed, latent, output_dir, scaler):
    """Create roundtrip visualizations."""
    output_dir = Path(output_dir)
    
    # Use original scale if available
    if scaler is not None:
        orig = scaler.inverse_transform(original.reshape(-1, original.shape[-1])).reshape(original.shape)
        recon = scaler.inverse_transform(reconstructed.reshape(-1, reconstructed.shape[-1])).reshape(reconstructed.shape)
    else:
        orig = original
        recon = reconstructed
    
    # 1. Reconstruction comparison
    n_samples = min(5, len(orig))
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        for j, feat in enumerate(config.FEATURE_COLUMNS):
            ax = axes[i, j]
            ax.plot(orig[i, :, j], label='Original', linewidth=2)
            ax.plot(recon[i, :, j], label='Reconstructed', linestyle='--', linewidth=2)
            ax.set_title(f'Seq {i} - {feat}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Latent space (PCA if > 2D)
    if latent.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent)
        title = f'Latent Space (PCA {pca.explained_variance_ratio_.sum():.1%})'
    else:
        latent_2d = latent
        title = 'Latent Space'
    
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5, s=10)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_space.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved visualizations")

def visualize_anomalies(scores, anomalies, threshold, output_dir):
    """Create anomaly visualizations."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Scatter
    axes[0].scatter(range(len(scores)), scores, c=anomalies, cmap='coolwarm', alpha=0.6, s=10)
    axes[0].axhline(threshold, color='red', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Sequence Index')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Anomaly Detection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(scores[~anomalies], bins=50, alpha=0.7, label='Normal')
    axes[1].hist(scores[anomalies], bins=50, alpha=0.7, label='Anomaly')
    axes[1].axvline(threshold, color='red', linestyle='--', label='Threshold')
    axes[1].set_xlabel('MSE')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'anomalies.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved visualizations")

def visualize_generated(generated, output_dir):
    """Create generated data visualizations."""
    output_dir = Path(output_dir)
    
    n_samples = min(3, len(generated))
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        for j, feat in enumerate(config.FEATURE_COLUMNS):
            ax = axes[i, j]
            ax.plot(generated[i, :, j], linewidth=2)
            ax.set_title(f'Generated {i} - {feat}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved visualizations")

# ============================================================================
# MAIN CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='VAE Inference for Bridge Accelerometer Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_vae.py compress data.csv --output results/
  python inference_vae.py roundtrip data.csv --output results/ --visualize
  python inference_vae.py anomaly data.csv --output results/ --threshold auto
  python inference_vae.py generate --output results/ --n-samples 100
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode')
    subparsers.required = True
    
    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--checkpoint', type=str,
                       default='phase2_VAE_results_sep_loss/checkpoints/vae_best_model.pt',
                       help='Checkpoint path')
    common.add_argument('--output', type=str, required=True, help='Output directory')
    common.add_argument('--no-scaler', action='store_true', help='Skip scaler')
    common.add_argument('--visualize', action='store_true', help='Create plots')
    
    # Compress
    compress_parser = subparsers.add_parser('compress', parents=[common])
    compress_parser.add_argument('input', type=str, help='Input data')
    
    # Reconstruct
    reconstruct_parser = subparsers.add_parser('reconstruct', parents=[common])
    reconstruct_parser.add_argument('input', type=str, help='Input data')
    
    # Roundtrip
    roundtrip_parser = subparsers.add_parser('roundtrip', parents=[common])
    roundtrip_parser.add_argument('input', type=str, help='Input data')
    
    # Anomaly
    anomaly_parser = subparsers.add_parser('anomaly', parents=[common])
    anomaly_parser.add_argument('input', type=str, help='Input data')
    anomaly_parser.add_argument('--threshold', type=str, default='auto',
                               help='Threshold (number or "auto")')
    
    # Generate
    generate_parser = subparsers.add_parser('generate', parents=[common])
    generate_parser.add_argument('--n-samples', type=int, default=100,
                                help='Number of samples')
    
    args = parser.parse_args()
    
    # Execute
    print("\n" + "="*80)
    print(f"VAE INFERENCE - {args.mode.upper()}")
    print("="*80)
    
    try:
        if args.mode == 'compress':
            compress_mode(args)
        elif args.mode == 'reconstruct':
            reconstruct_mode(args)
        elif args.mode == 'roundtrip':
            roundtrip_mode(args)
        elif args.mode == 'anomaly':
            anomaly_mode(args)
        elif args.mode == 'generate':
            generate_mode(args)
        
        print("\n" + "="*80)
        print(" COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
