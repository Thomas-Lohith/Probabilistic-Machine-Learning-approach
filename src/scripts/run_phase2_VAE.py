import argparse
from pathlib import Path
import numpy as np
import torch
import json
from tqdm import tqdm

from config import config
from load_data import load_single_csv, explore_data, extract_features
from model_vae import create_vae_model
from train_VAE import VAETrainer
from dataset import create_dataloaders, load_processed_data
from evaluate_VAE import evaluate_vae, print_vae_metrics, load_vae_model
from evaluate import plot_reconstruction_samples, plot_latent_space, plot_training_history

# Import from working run_phase2.py
from run_phase2 import find_csv_files, load_multiple_days, preprocess_multi_day


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
    
    parser.add_argument('--output-dir', type=str, default='/data/pool/c8x-98x/pml/src/phase2_VAE_results',
                        help='Output directory for results')
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (10 epochs)')
    
    parser.add_argument('--skip-exploration', action='store_true',
                        help='Skip data exploration phase')
    
    parser.add_argument('--reuse-processed', action='store_true',
                        help='Reuse existing processed data if available')
    
    # VAE-specific arguments
    parser.add_argument('--kl-weight', type=float, default=1.0,
                        help='KL divergence weight (beta in β-VAE, default: 1.0)')
    
    parser.add_argument('--no-kl-annealing', action='store_true',
                        help='Disable KL annealing (not recommended)')
    
    return parser.parse_args()


def train_phase2_vae(processed_file, output_dir, quick_test=False,
                     kl_weight=1.0, kl_annealing=False):
    """
    Train VAE model on Phase 2 multi-day data.
    
    Args:
        processed_file: Path to processed data
        output_dir: Output directory for checkpoints
        quick_test: Whether to run quick test
        kl_weight: KL divergence weight
        kl_annealing: Whether to use KL annealing
        
    Returns:
        Tuple of (model, history)
    """
    print("\n" + "="*80)
    print("TRAINING VAE (PHASE 2)")
    print("="*80)
    
    train, val, test = load_processed_data(processed_file)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(train, val, test)
    
    print(f"\nDataset sizes:")
    print(f"   Train: {len(train):,} sequences")
    print(f"   Val:   {len(val):,} sequences")
    print(f"   Test:  {len(test):,} sequences")
    print(f"   Total: {len(train) + len(val) + len(test):,} sequences")
    
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
    history = trainer.train(checkpoint_dir=checkpoint_dir)
    
    return model, history


def evaluate_phase2_vae(checkpoint_path, processed_file, output_dir, kl_weight=1.0):

    print("\n" + "="*80)
    print("EVALUATION (PHASE 2 VAE)")
    print("="*80)
    
    # Load model
    model = load_vae_model(checkpoint_path)
    
    # Load data
    from dataset import load_processed_data, create_dataloaders
    train, val, test = load_processed_data(processed_file)
    train_loader, val_loader, test_loader = create_dataloaders(train, val, test)
    
    print(f"Test shape: {test.shape}")
    
    # Load scaler for denormalization (if it exists)
    scaler_file = processed_file.parent / "scaler.pkl"
    scaler = None
    
    if scaler_file.exists():
        import joblib
        scaler = joblib.load(scaler_file)
        print(f" Loaded scaler: {scaler_file}")
        print(f"   Scaler type: {type(scaler).__name__}")
    else:
        print(f"  WARNING: Scaler not found: {scaler_file}")
        print("   Metrics will be on normalized scale")
    
    # Evaluate
    metrics = evaluate_vae(model, test, kl_weight=kl_weight, use_abs=True)
    
    # Print metrics
    print_vae_metrics(metrics)
    
    # Get reconstructions and latent representations for visualization
    print(f"\n📊 Generating data for visualizations...")
    model.eval()
    with torch.no_grad():
        # Get reconstructions, mu, and log_var
        test_tensor = torch.from_numpy(test).float().to(config.DEVICE)
        reconstructed, mu, log_var = model(test_tensor)
        
        # Move to CPU
        test_originals_norm = test
        test_reconstructions_norm = reconstructed.cpu().numpy()
        test_latents = mu.cpu().numpy()
        
        # Denormalize for visualization (if scaler available)
        if scaler is not None:
            print(f"🔄 Denormalizing data for visualization...")
            
            original_shape = test_originals_norm.shape
            test_originals = scaler.inverse_transform(
                test_originals_norm.reshape(-1, test_originals_norm.shape[-1])
            ).reshape(original_shape)
            
            test_reconstructions = scaler.inverse_transform(
                test_reconstructions_norm.reshape(-1, test_reconstructions_norm.shape[-1])
            ).reshape(original_shape)
            
            print(f"   Original scale - Min: {test_originals.min():.4f}, Max: {test_originals.max():.4f}")
            
            # Calculate metrics on original scale
            mse_original = np.mean((test_originals - test_reconstructions) ** 2)
            mae_original = np.mean(np.abs(test_originals - test_reconstructions))
            print(f"\n📊 Metrics on original scale:")
            print(f"   MSE: {mse_original:.6f}")
            print(f"   MAE: {mae_original:.6f}")
        else:
            test_originals = test_originals_norm
            test_reconstructions = test_reconstructions_norm
    
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
    
    print(f"\n Visualizations saved to: {results_dir}")
    
    # Save metrics
    metrics_path = output_dir / "vae_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Metrics saved: {metrics_path}")
    
    return metrics


def run_phase2_vae_pipeline(args):
    """
    Run complete Phase 2 VAE pipeline.
    
    Args:
        args: Command line arguments
    """
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
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # ERROR FIXED: Define processed_file path before checking if it exists
    processed_file = processed_dir / "phase2_combined_processed.npz"
    
    # ============================================
    # STEP 1: Find CSV files
    # ============================================
    if args.file_list:
        print(f"\n📁 Using provided file list ({len(args.file_list)} files)")
        file_list = [Path(f) for f in args.file_list]
    else:
        file_list = find_csv_files(
            Path(args.data_dir),
            args.n_days,
            args.start_date
        )
    
    # ============================================
    # STEP 2: Load and preprocess data
    # ============================================
    print("\n" + "="*80)
    print("STEP 2: DATA LOADING AND PREPROCESSING")
    print("="*80)
    
    # Check if we should reuse existing processed data
    if args.reuse_processed and processed_file.exists():
        print(f"\n♻️  Reusing existing processed data: {processed_file}")
    else:
        # Step 2.1: Load multi-day data
        data_dict = load_multiple_days(file_list, args.skip_exploration)
        
        # Step 2.2: Preprocess (use function from run_phase2.py)
        processed_file = preprocess_multi_day(data_dict, output_dir)
    
    print(f"✅ Processed data ready: {processed_file}")
    
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
    checkpoint_path = output_dir / "checkpoints" / "vae_best_model.pt"
    
    if not checkpoint_path.exists():
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
    
    print(f"\n📊 Results Summary:")
    print(f"   Days processed:     {args.n_days}")
    print(f"   R² Score:           {metrics['r2']:.4f}")
    print(f"   RMSE:               {metrics['rmse']:.6f}")
    print(f"   MAE:                {metrics['mae']:.6f}")
    print(f"   KL Divergence:      {metrics['kl_divergence']:.6f}")
    print(f"   Reconstruction Loss: {metrics['recon_loss']:.6f}")
    
    print(f"\n💡 VAE Performance:")
    kl = metrics['kl_divergence']
    if kl < 1.0:
        print(f"   ⚠️  KL very low ({kl:.3f}) - possible posterior collapse")
    elif kl < 5.0:
        print(f"   ✅ KL in good range ({kl:.3f}) - healthy latent space")
    elif kl < 10.0:
        print(f"   ⚠️  KL slightly high ({kl:.3f}) - may need tuning")
    else:
        print(f"   ❌ KL too high ({kl:.3f}) - model struggling")
    
    print(f"\n📁 Output Directory: {output_dir}")
    print(f"   Processed data: {processed_dir}")
    print(f"   Checkpoints:    {output_dir / 'checkpoints'}")
    print(f"   Visualizations: {output_dir / 'visualizations'}")
    print(f"   Metrics:        {output_dir / 'vae_metrics.json'}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    args = parse_args()
    run_phase2_vae_pipeline(args)
    #last experiment; 
    # python3 run_phase2_VAE.py --n-days 100 --kl-weight 0.1 
    #python3 run_phase2_VAE.py --n-days 100 --kl-weight 0.9 --no-kl-annealing
    #python3 run_phase2_VAE.py --n-days 100 --kl-weight 1