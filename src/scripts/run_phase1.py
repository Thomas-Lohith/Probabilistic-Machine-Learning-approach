"""
Phase 1: Complete Pipeline - Single Day Implementation
Run this script to execute the entire Phase 1 pipeline.
"""

from pathlib import Path
import sys

# Import our modules
from config import config
from load_data import get_data_summary
from preprocess_data import preprocess_single_day
from train import train_phase1
from evaluate import evaluate_phase1


def check_environment():
    """Check if environment is properly set up."""
    print("\n" + "="*80)
    print("ENVIRONMENT CHECK")
    print("="*80)
    
    import torch
    import pandas
    import numpy
    import sklearn
    
    print(f" Python version: {sys.version.split()[0]}")
    print(f" PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f" CUDA version: {torch.version.cuda}")
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f" Device: {config.DEVICE}")
    print("="*80)


def run_phase1_pipeline(data_file: Path = None,
                       skip_exploration: bool = False,
                       quick_test: bool = False):
    """
    Run complete Phase 1 pipeline.
    
    Args:
        data_file: Path to CSV file (default: config.PHASE1_TEST_FILE)
        skip_exploration: Skip data exploration step
        quick_test: Use fewer epochs for quick testing
    """
    
    # Check environment
    check_environment()
    
    # Set paths
    if data_file is None:
        data_file = Path("/data/pool/c8x-98x/bridge_data/100_days") / config.PHASE1_TEST_FILE
    
    processed_dir = Path("/data/pool/c8x-98x/bridge_data/100_days/data/processed")
    checkpoint_dir = Path("/data/pool/c8x-98x/bridge_data/100_days/data/checkpoints")
    results_dir = Path("/data/pool/c8x-98x/bridge_data/100_days/data/results")
    
    # Create directories
    processed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("PHASE 1: SINGLE DAY PIPELINE")
    print("="*80)
    print(f"Input file: {data_file}")
    print(f"Quick test mode: {quick_test}")
    print("="*80)
    
    # Check if data file exists
    if not data_file.exists():
        print(f"\n ERROR: Data file not found: {data_file}")
        print(f"\n Searching for CSV files in: {data_file.parent}")
        if data_file.parent.exists():
            csv_files = sorted(data_file.parent.glob("*.csv"))
            if csv_files:
                print(f"Found {len(csv_files)} CSV files:")
                for i, f in enumerate(csv_files[:10]):
                    print(f"  {i+1}. {f.name}")
                if len(csv_files) > 10:
                    print(f"  ... and {len(csv_files)-10} more")
            else:
                print("No CSV files found!")
        else:
            print(f"Directory does not exist: {data_file.parent}")
        return
    
    # ============================================
    # STEP 1: Data Exploration (Optional)
    # ============================================
    if not skip_exploration:
        print("\n" + "="*80)
        print("STEP 1: DATA EXPLORATION")
        print("="*80)
        
        try:
            summary = get_data_summary(data_file)
            print(" Data exploration complete!")
        except Exception as e:
            print(f"  Error in data exploration: {e}")
            print("Continuing anyway...")
    else:
        print("\n Skipping data exploration")
    
    # ============================================
    # STEP 2: Preprocessing
    # ============================================
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING")
    print("="*80)
    
    processed_file = processed_dir / f"{data_file.stem}_processed.npz"
    
    if processed_file.exists():
        print(f"  Processed file already exists: {processed_file}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Using existing processed file...")
        else:
            print("Reprocessing...")
            train, val, test, preprocessor = preprocess_single_day(
                data_file,
                output_dir=processed_dir,
                save_processed=True
            )
            print(" Preprocessing complete!")
    else:
        train, val, test, preprocessor = preprocess_single_day(
            data_file,
            output_dir=processed_dir,
            save_processed=True
        )
        print(" Preprocessing complete!")
    
    # ============================================
    # STEP 3: Training
    # ============================================
    print("\n" + "="*80)
    print("STEP 3: TRAINING")
    print("="*80)
    
    best_checkpoint = checkpoint_dir / "best.pt"
    
    if best_checkpoint.exists():
        print(f"⚠️  Checkpoint already exists: {best_checkpoint}")
        response = input("Retrain from scratch? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping training, using existing checkpoint...")
        else:
            print("Training from scratch...")
            trainer, history = train_phase1(
                processed_file=processed_file,
                save_dir=checkpoint_dir,
                quick_test=quick_test
            )
            print(" Training complete!")
    else:
        trainer, history = train_phase1(
            processed_file=processed_file,
            save_dir=checkpoint_dir,
            quick_test=quick_test
        )
        print(" Training complete!")
    
    # ============================================
    # STEP 4: Evaluation
    # ============================================
    print("\n" + "="*80)
    print("STEP 4: EVALUATION")
    print("="*80)
    
    if not best_checkpoint.exists():
        print(f" ERROR: Best checkpoint not found: {best_checkpoint}")
        print("Training may have failed!")
        return
    
    metrics = evaluate_phase1(
        checkpoint_path=best_checkpoint,
        processed_file=processed_file,
        output_dir=results_dir
    )
    
    print(" Evaluation complete!")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "="*80)
    print("PHASE 1 PIPELINE COMPLETE! 🎉")
    print("="*80)
    
    print(f"\n📁 Output Locations:")
    print(f"   Processed data: {processed_dir}")
    print(f"   Model checkpoint: {checkpoint_dir}")
    print(f"   Results: {results_dir}")
    
    print(f"\n📊 Key Results:")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   Compression Ratio: {config.compression_ratio:.2f}x")
    
    print(f"\n🎯 Next Steps:")
    if metrics['r2'] >= config.MIN_R2_EXCELLENT:
        print(f"    Excellent results! Ready for Phase 2 (multiple days)")
    elif metrics['r2'] >= config.MIN_R2_GOOD:
        print(f"   Good results! Can proceed to Phase 2 or tune hyperparameters")
    elif metrics['r2'] >= config.MIN_R2_ACCEPTABLE:
        print(f"    Acceptable results. Consider:")
        print(f"      - Increase latent_dim (current: {config.LATENT_DIM})")
        print(f"      - Add more LSTM layers")
        print(f"      - Train for more epochs")
    else:
        print(f"   ⚠️  Results need improvement. Try:")
        print(f"      - Increase latent_dim significantly")
        print(f"      - Check data preprocessing")
        print(f"      - Increase model capacity")
    
    print("\n" + "="*80)


def quick_test():
    """Run a quick test with reduced epochs."""
    print("\n QUICK TEST MODE")
    print("Using fewer epochs for fast iteration...\n")
    
    run_phase1_pipeline(quick_test=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1: Single Day Pipeline")
    parser.add_argument("--file", type=str, default=None,
                       help="Path to CSV file (default: config.PHASE1_TEST_FILE)")
    parser.add_argument("--skip-exploration", action="store_true",
                       help="Skip data exploration step")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode with fewer epochs")
    
    args = parser.parse_args()
    
    data_file = Path(args.file) if args.file else None
    
    run_phase1_pipeline(
        data_file=data_file,
        skip_exploration=args.skip_exploration,
        quick_test=args.quick
    )