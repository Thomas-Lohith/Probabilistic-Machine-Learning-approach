
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
import json
from tqdm import tqdm

from config import config
from load_data import load_single_csv, explore_data, extract_features
from VAE import create_vae_model  # Use VAE model
from train_VAE import VAETrainer  # Use VAE trainer
from dataset import load_processed_data, create_dataloaders
from evaluate_VAE import evaluate_vae, print_vae_metrics, load_vae_model  # Use VAE evaluation
from evaluate import plot_reconstruction_samples, plot_latent_space, plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Phase 2: Multi-Day Training (VAE)')
    
    parser.add_argument('--data-dir', type=str, 
                        default='/data/pool/c8x-98x/bridge_data/100_days',
                        help='Directory containing CSV files')
    
    parser.add_argument('--n-days', type=int, default=5,
                        help='Number of days to process (default: 5)')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYYMMDD format), e.g., 20241126')
    
    parser.add_argument('--file-list', type=str, nargs='+', default=None,
                        help='Specific files to process')
    
    parser.add_argument('--output-dir', type=str, default= None,
                        help='Output directory for results')
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (fewer epochs)')
    
    parser.add_argument('--skip-exploration', action='store_true',
                        help='Skip data exploration phase')
    
    parser.add_argument('--reuse-processed', action='store_true',
                        help='Reuse existing processed data if available')
    
    # VAE-specific arguments
    parser.add_argument('--kl-weight', type=float, default=1.0,
                        help='KL divergence weight (beta)')
    
    parser.add_argument('--no-kl-annealing', action='store_true',
                        help='Disable KL annealing')
    
    return parser.parse_args()


def find_csv_files(data_dir: Path, n_days: int, start_date: str = None) -> list:

    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Get all CSV files
    all_files = sorted(data_dir.glob("*.csv"))
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"\n📁 Found {len(all_files)} CSV files in {data_dir}")
    
    # Filter by start date if provided
    if start_date:
        all_files = [f for f in all_files if f.stem >= start_date]
        print(f"   Filtered to {len(all_files)} files >= {start_date}")
    
    # Select first n_days files
    selected_files = all_files[:n_days]
    
    print(f"\n Selected {len(selected_files)} files for Phase 2 (VAE):")
    for i, f in enumerate(selected_files, 1):
        print(f"   {i}. {f.name}")
    
    return selected_files


def preprocess_multiple_days(csv_files: list, output_dir: Path, reuse: bool = False):
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if combined file already exists
    combined_file = output_dir / "phase2_combined_processed.npz"
    
    if combined_file.exists() and reuse:
        print(f"\n Reusing existing processed data: {combined_file}")
        return combined_file
    
    print("\n" + "="*80)
    print("PREPROCESSING MULTIPLE DAYS")
    print("="*80)
    
    all_sequences = []
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\nProcessing file {i}/{len(csv_files)}: {csv_file.name}")
        
        # Load data
        df = load_single_csv(csv_file)

        df.drop(['day', 'hour_file', 'start_time', 'end_time', 'variance'], axis=1)
        
         # Quick stats
        print(f"   Records: {len(df):,}")
        print(f"   Duration: {len(df) / 86400:.2f} days")
        print(f"   Features: {list(df.columns)}")
        print(f"   Missing_data: {(df.isna().sum())}")

        if df.isna().any().any():
            missing_percent = df.isna().mean().mean() * 100
            print('****There are missing data in this file and the percentage is:', missing_percent)
            print(f'therfore skipping this file:', {csv_file})
            continue
        # Explore (optional, quick)
        #explore_data(df)
        
        # Extract features
        features_df = extract_features(df)
        
        # Create sequences
        from preprocess_data import DataPreprocessor
        Preprocessor = DataPreprocessor()
        
        sequences = Preprocessor.create_sequences(
            features_df, 
            sequence_length=config.SEQUENCE_LENGTH,
            #stride=config.STRIDE
        )
        
        all_sequences.append(sequences)
        print(f"   Created {len(sequences)} sequences")
    
    # Combine all sequences
    combined_sequences = np.concatenate(all_sequences, axis=0)
    print(f"\n Combined shape: {combined_sequences.shape}")
    
    # Split into train/val/test
    n_samples = len(combined_sequences)
    train_size = int(n_samples * config.TRAIN_RATIO)
    val_size = int(n_samples * config.VAL_RATIO)
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    X_train = combined_sequences[train_indices]
    X_val = combined_sequences[val_indices]
    X_test = combined_sequences[test_indices]
    
    print(f"\nSplit sizes:")
    print(f"   Train: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
    print(f"   Val:   {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
    print(f"   Test:  {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")


    # ============================================
    # NORMALIZATION - CRITICAL FOR VAE
    # ============================================
    print("\n" + "="*80)
    print("NORMALIZATION (StandardScaler)")
    print("="*80)
    
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    # Statistics before normalization
    print("\nBefore normalization:")
    print(f"   Train - Min: {X_train.min():.4f}, Max: {X_train.max():.4f}")
    print(f"   Train - Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
    
    # Reshape for fitting: (n_samples * seq_length, n_features)
    original_train_shape = X_train.shape
    original_val_shape = X_val.shape
    original_test_shape = X_test.shape
    
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit scaler on training data ONLY
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train_flat)
    
    # Transform validation and test using fitted scaler
    X_val_normalized = scaler.transform(X_val_flat)
    X_test_normalized = scaler.transform(X_test_flat)
    
    # Reshape back to sequences: (n_samples, seq_length, n_features)
    X_train = X_train_normalized.reshape(original_train_shape)
    X_val = X_val_normalized.reshape(original_val_shape)
    X_test = X_test_normalized.reshape(original_test_shape)
    
    # Statistics after normalization
    print("\nAfter normalization:")
    print(f"   Train - Min: {X_train.min():.4f}, Max: {X_train.max():.4f}")
    print(f"   Train - Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
    print(f"   Val   - Mean: {X_val.mean():.4f}, Std: {X_val.std():.4f}")
    print(f"   Test  - Mean: {X_test.mean():.4f}, Std: {X_test.std():.4f}")
    
    # Print per-feature statistics
    print("\nPer-feature statistics (training data):")
    for feat_idx in range(X_train.shape[-1]):
        feat_data = X_train[:, :, feat_idx]
        print(f"   Feature {feat_idx}: mean={feat_data.mean():.4f}, std={feat_data.std():.4f}")
    
    # Verify normalization
    if abs(X_train.mean()) < 0.1 and abs(X_train.std() - 1.0) < 0.2:
        print("\n Normalization successful! Data is properly scaled for VAE.")
    else:
        print("\n⚠️  WARNING: Normalization may not be working correctly!")
        print(f"   Expected: mean ≈ 0, std ≈ 1")
        print(f"   Got: mean = {X_train.mean():.4f}, std = {X_train.std():.4f}")
    
    # ============================================
    # Save scaler
    # ============================================

    with open(output_dir, 'wb') as f:
        joblib.dump(scaler, f)
        print(f"\n Scaler saved: {output_dir}")
        print(f"   Scaler type: {type(scaler).__name__}")
        print(f"   Scaler mean: {scaler.mean_}")
        print(f"   Scaler scale: {scaler.scale_}")
    
    # ============================================
    # Save processed data
    # ============================================
    # Save
    np.savez_compressed(
        combined_file,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        csv_files=[str(f) for f in csv_files]
    )
    
    print(f"\n Saved processed data: {combined_file}")
    print("\nSummary:")
    print(f"   • Normalized with StandardScaler ✅")
    print(f"   • Train mean ≈ 0, std ≈ 1 ✅")
    print(f"   • Scaler saved for denormalization ✅")
    print(f"   • Ready for VAE training! 🚀")    
    
    # Save
    np.savez_compressed(
        combined_file,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        csv_files=[str(f) for f in csv_files]
    )
    
    print(f"\n Saved processed data: {combined_file}")
    
    return combined_file


def train_phase2_vae(processed_file: Path, 
                     output_dir: Path, 
                     quick_test: bool = False,
                     kl_weight: float = 1.0,
                     kl_annealing: bool = True):
    print("\n" + "="*80)
    print("TRAINING VAE (PHASE 2)")
    print("="*80)
    
    # Load data
    data = np.load(processed_file)
    X_train = torch.FloatTensor(data['X_train'])
    X_val = torch.FloatTensor(data['X_val'])
    X_test = torch.FloatTensor(data['X_test'])
    
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(X_train, X_val, X_test)
    
    # Create model
    model = create_vae_model()
    
    # Create trainer
    epochs = 10 if quick_test else config.EPOCHS
    
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        kl_weight=kl_weight,
        kl_annealing=kl_annealing,
        kl_annealing_epochs=min(20, epochs // 5)
    )
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    history = trainer.train(epochs=epochs, checkpoint_dir=checkpoint_dir)
    
    return model, history


def evaluate_phase2_vae(checkpoint_path: Path,
                        processed_file: Path,
                        output_dir: Path,
                        kl_weight: float = 1.0):
  
    print("\n" + "="*80)
    print("EVALUATION (PHASE 2 VAE)")
    print("="*80)
    
    # Load model
    model = load_vae_model(checkpoint_path)
    
    # Load test data
    data = np.load(processed_file)
    X_test = torch.FloatTensor(data['X_test'])
    
    print(f"Test shape: {X_test.shape}")
    
    # Evaluate
    metrics = evaluate_vae(model, X_test, kl_weight=kl_weight, use_abs=True)
    
    # Print metrics
    print_vae_metrics(metrics)
    
    # Get reconstructions and latent representations for visualization
    print(f"\n📊 Generating data for visualizations...")
    model.eval()
    with torch.no_grad():
        # Get reconstructions, mu, and log_var
        reconstructed, mu, log_var = model(X_test.to(config.DEVICE))
        
        # Move to CPU for visualization
        test_originals = X_test.cpu().numpy()
        test_reconstructions = reconstructed.cpu().numpy()
        test_latents = mu.cpu().numpy()  # Use mean of latent distribution
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    results_dir = output_dir / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training history
    history_file = checkpoint_path.parent / "vae_training_history.json"
    if history_file.exists():
        print(f"   Creating training history plot...")
        plot_training_history(
            history_file,
            save_path=results_dir / "vae_training_history.png"
        )
    else:
        print(f"   ⚠️  Training history file not found: {history_file}")
    
    # 2. Reconstruction samples
    print(f"   Creating reconstruction samples plot...")
    plot_reconstruction_samples(
        test_originals,
        test_reconstructions,
        n_samples=5,
        save_path=results_dir / "vae_reconstruction_samples.png",
        use_abs=True
    )
    
    # 3. Latent space visualization
    print(f"   Creating latent space plot...")
    plot_latent_space(
        test_latents,
        save_path=results_dir / "vae_latent_space.png"
    )
    
    print(f"\n✅ Visualizations saved to: {results_dir}")
    
    # Save metrics
    metrics_path = output_dir / "vae_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            metrics_json[key] = {k: float(v) for k, v in value.items()}
        else:
            metrics_json[key] = float(value)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\n💾 Metrics saved: {metrics_path}")
    
    return metrics


def run_phase2_vae_pipeline(args):
    
    print("\n" + "="*80)
    print("PHASE 2: MULTI-DAY TRAINING (VAE)")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Number of days: {args.n_days}")
    print(f"KL Weight (β): {args.kl_weight}")
    print(f"KL Annealing: {not args.no_kl_annealing}")
    print("="*80)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # STEP 1: Find CSV files
    # ============================================
    if args.file_list:
        csv_files = [Path(f) for f in args.file_list]
    else:
        csv_files = find_csv_files(
            Path(args.data_dir),
            args.n_days,
            args.start_date
        )
    
    # ============================================
    # STEP 2: Preprocess
    # ============================================
    processed_dir = output_dir / "processed"
    processed_file = preprocess_multiple_days(
        csv_files,
        processed_dir,
        reuse=args.reuse_processed
    )
    
    # ============================================
    # STEP 3: Train VAE
    # ============================================
    model, history = train_phase2_vae(
        processed_file,
        output_dir,
        quick_test=args.quick,
        kl_weight=args.kl_weight,
        kl_annealing=not args.no_kl_annealing
    )
    
    # ============================================
    # STEP 4: Evaluate VAE
    # ============================================
    checkpoint_path = output_dir / "checkpoints" / "vae_final_model.pt"
    metrics = evaluate_phase2_vae(
        checkpoint_path,
        processed_file,
        output_dir,
        kl_weight=args.kl_weight
    )
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "="*80)
    print("PHASE 2 VAE PIPELINE COMPLETE! 🎉")
    print("="*80)
    
    print(f"\n Results Summary:")
    print(f"   Days processed: {args.n_days}")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   KL Divergence: {metrics['kl_divergence']:.6f}")
    
    print(f"\n💡 VAE Performance:")
    kl = metrics['kl_divergence']
    if kl < 1.0:
        print(f"     KL very low ({kl:.3f}) - possible posterior collapse")
    elif kl < 5.0:
        print(f"    KL in good range ({kl:.3f}) - healthy latent space")
    elif kl < 10.0:
        print(f"    KL slightly high ({kl:.3f}) - may need tuning")
    else:
        print(f"    KL too high ({kl:.3f}) - model struggling")
    
    print(f"\n Output directory: {output_dir}")
    print(f"   Processed data: {processed_dir}")
    print(f"   Checkpoints: {output_dir / 'checkpoints'}")
    print(f"   Metrics: {output_dir / 'vae_metrics.json'}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    args = parse_args()
    run_phase2_vae_pipeline(args)