
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
from preprocess_data import DataPreprocessor
from model import create_model
from train import Trainer
from evaluate import evaluate_model, calculate_metrics, print_evaluation_report
from evaluate import plot_reconstruction_samples, plot_latent_space, plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Phase 2: Multi-Day Training')
    
    parser.add_argument('--data-dir', type=str, default='/data/pool/c8x-98x/bridge_data/100_days', help='Directory containing CSV files')
    
    parser.add_argument('--n-days', type=int, default=5,
                        help='Number of days to process (default: 5)')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYYMMDD format), e.g., 20241126')
    
    parser.add_argument('--file-list', type=str, nargs='+', default=None,
                        help='Specific files to process')
    
    parser.add_argument('--output-dir', type=str, default='/data/pool/c8x-98x/pml/src/phase2_results',
                        help='Output directory for results')
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (fewer epochs)')
    
    parser.add_argument('--skip-exploration', action='store_false',
                        help='Skip data exploration phase')
    
    parser.add_argument('--reuse-processed', action='store_true',
                        help='Reuse existing processed data if available')
    
    return parser.parse_args()


def find_csv_files(data_dir: Path, n_days: int, start_date: str = None) -> list:
   
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Get all CSV files
    all_files = sorted(data_dir.glob("*.csv"))
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"\n Found {len(all_files)} CSV files in {data_dir}")
    
    # Filter by start date if provided
    if start_date:
        all_files = [f for f in all_files if f.stem >= start_date]
        print(f"   Filtered to {len(all_files)} files >= {start_date}")
    
    # Select first n_days files
    selected_files = all_files[:n_days]
    
    #to Show file names inorder to check the dates 
    # print(f"\n Selected {len(selected_files)} files for Phase 2:") 
    # for i, f in enumerate(selected_files, 1):
    #     print(f"   {i}. {f.name}")
    
    return selected_files


def load_multiple_days(file_list: list,output_dir, skip_exploration: bool = False ) -> dict:
   
    print("\n" + "="*80)
    print("PHASE 2: LOADING MULTIPLE DAYS")
    print("="*80)
    
    all_data = []
    day_info = []
    missing_info = []
    missing_numbers = []
    
    for i, csv_file in enumerate(file_list, 1):
        print(f"\n Loading Day {i}/{len(file_list)}: {csv_file.name}")
        
        try:
            # Load CSV
            df = load_single_csv(csv_file)
            # Quick stats
            
            print(f"   Records: {len(df):,}")
            print(f"   Duration: {len(df) / 86400:.2f} days")
            print(f"   Features: {list(df.columns)}")
            print(f"   Missing_data: {(df.isna().sum())}")
            print(f'   max values: {(df.max())}')
            print(f'   min values: {(df.min())}')
            
            if df.isna().any().any(): 
                missing_percent = df.isna().mean().mean() * 100
                missing = df.isna().any().any()
                # print('****There are missing data in this file and the percentage is:', missing_percent)
                # print(f'therfore skipping this file:', {csv_file})
                # continue
            # Store info
            day_info.append({
                'file': str(csv_file),
                'date': csv_file.stem,
                'n_records': len(df),
                'start_idx': sum(d['n_records'] for d in day_info),
                'end_idx': sum(d['n_records'] for d in day_info) + len(df)
            })
            
            missing_numbers.append(missing)
            missing_info.append(np.float16(missing_percent))
            all_data.append(df)
            
            # Detailed exploration for first file only
            if i == 1 and not skip_exploration:
                print(f"\n Detailed statistics for first file:")
                stats = explore_data(df,output_dir, show_plots= True)

            
                
        except Exception as e:
            print(f"    Error loading {csv_file.name}: {e}")
            raise
    
    # Combine all dataframes
    print(f"\n Combining {len(all_data)} days...")
    print("="*80)
    print('missing list:', list(map(float, missing_numbers))) 
    print('the missing percents of each day in the list:',)
    print(*missing_info, sep=', ')
    
    import pandas as pd
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print("="*80)
    print(f"\n Combined Dataset:")
    print(f'   the max values: {(combined_df.max())}')
    print(f'   the min values: {(combined_df.min())}')
    print(f"   Total records: {len(combined_df):,}")
    print(f"   Total duration: {len(combined_df) / 86400:.2f} days")
    print(f"   Size in memory: {combined_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    return {
        'data': combined_df,
        'day_info': day_info,
        'n_days': len(file_list)
    }


def preprocess_multi_day(data_dict: dict, output_dir: Path) -> Path:
    
    print("\n" + "="*80)
    print("PHASE 2: PREPROCESSING MULTI-DAY DATA")
    print("="*80)
    
    #from preprocess import DataPreprocessor
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Create filename based on date range
    first_date = data_dict['day_info'][0]['date']
    last_date = data_dict['day_info'][-1]['date']
    n_days = data_dict['n_days']

    output_dir = output_dir / 'processed'
    
    output_file = output_dir / f"phase2_{n_days}days_processed.npz"
    
    print(f"\n Processing {n_days} days of data...")
    print(f"   First day: {first_date}")
    print(f"   Last day: {last_date}")
    print(f"   Output: {output_file}")
    
    # Extract features from combined dataframe
    features = data_dict['data'][config.FEATURE_COLUMNS].values
    
    # Apply absolute values (matching load_data.py behavior)
    features = np.abs(features)
    print(f"\n Applied absolute values to features")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Handle missing values
    features = preprocessor.handle_missing_values(features, method='remove') #this interpolation technique can be addded into configuration
    
    # Normalize (fit on ALL multi-day data)
    features_normalized = preprocessor.normalize(features, fit=True)
    
    # Create sequences
    sequences = preprocessor.create_sequences(
        features_normalized,
        sequence_length=config.SEQUENCE_LENGTH,
        overlap=False  # Non-overlapping for efficiency
    )
    
    # Split data
    train, val, test = preprocessor.split_data(
        sequences,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO
    )
    
    # Save processed data
    np.savez_compressed(
        output_file,
        train=train,
        val=val,
        test=test,
        original_shape=features.shape
    )
    print(f"\n💾 Saved processed data to: {output_file}")
    
    # Save scaler
    scaler_file = output_dir/ 'scaler.pkl'
    preprocessor.save_scaler(scaler_file)
    
    # Save day boundaries for later analysis
    day_boundaries_file = output_dir / f"phase2_{n_days}days_boundaries.json"
    with open(day_boundaries_file, 'w') as f:
        json.dump(data_dict['day_info'], f, indent=2)
    
    print(f" Saved day boundaries to: {day_boundaries_file}")
    
    return output_file


def train_phase2_model(processed_file: Path, output_dir: Path, quick: bool = False):

    print("\n" + "="*80)
    print("PHASE 2: TRAINING ON MULTI-DAY DATA")
    print("="*80)
    
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    from dataset import load_processed_data, create_dataloaders
    train, val, test = load_processed_data(processed_file)
    train_loader, val_loader, test_loader = create_dataloaders(train, val, test)
    
    print(f"\n📊 Multi-day dataset sizes:")
    print(f"   Train: {len(train):,} sequences")
    print(f"   Val:   {len(val):,} sequences")
    print(f"   Test:  {len(test):,} sequences")
    print(f"   Total: {len(train) + len(val) + len(test):,} sequences")
    
    # Create model
    model = create_model()
    print(model)
    # Set epochs based on mode
    if quick:
        epochs = config.PHASE1_QUICK_EPOCHS
        print(f"\n⚡ Quick mode: {epochs} epochs")
    else:
        epochs = config.EPOCHS
        print(f"\n Full training: {epochs} epochs")
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )
    
    # Train
    history = trainer.train(save_dir=checkpoint_dir)
    
    # Return path to best checkpoint
    best_checkpoint = checkpoint_dir / "best.pt"
    return best_checkpoint


def evaluate_phase2(checkpoint_path: Path, 
                     processed_file: Path,
                     boundaries_file: Path,
                     output_dir: Path):
    print("\n" + "="*80)
    print("PHASE 2: EVALUATION WITH PER-DAY ANALYSIS")
    print("="*80)
    
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    from dataset import load_processed_data, create_dataloaders
    train, val, test = load_processed_data(processed_file)
    train_loader, val_loader, test_loader = create_dataloaders(train, val, test)
    
    # Load model
    model = create_model()
    print(model)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n Loaded model from: {checkpoint_path}")
    print(f"   Training epoch: {checkpoint['epoch']}")
    print(f"   Validation loss: {checkpoint['val_loss']:.6f}")
    
    # Overall evaluation
    print(f"\n Evaluating on test set...")
    test_originals, test_reconstructions, test_latents = evaluate_model(
        model, test_loader, config.DEVICE
    )
    
    print(f"\n Comparison Mode: Using absolute values")
    print(f"   Originals will be converted to abs() before comparison")
    print(f"   This matches the preprocessing in load_data.py")
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(test_originals, test_reconstructions, use_abs=True)
    
    print(f"\n" + "="*80)
    print("OVERALL METRICS (All Days Combined)")
    print("="*80)
    print_evaluation_report(overall_metrics)
    
    # Per-day evaluation
    if boundaries_file.exists():
        print(f"\n" + "="*80)
        print("PER-DAY ANALYSIS")
        print("="*80)
        
        with open(boundaries_file, 'r') as f:
            day_info = json.load(f)
        
        per_day_metrics = []
        
        for day in day_info:
            date = day['date']
            start_idx = day['start_idx']
            end_idx = day['end_idx']
            
            # Note: These indices are for the original data, not the test set
            # We'll evaluate on the whole test set but report as aggregate
            print(f"\n📅 {date}: {day['n_records']:,} records")
        
        # Just report that we trained on multiple days
        print(f"\n✅ Model trained on {len(day_info)} days of data")
        print(f"   The metrics above reflect performance across all {len(day_info)} days")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # Training history
    history_file = checkpoint_path.parent / "history.json"
    plot_training_history(
        history_file,
        save_path=results_dir / "training_history.png"
    )
    
    # Reconstruction samples
    plot_reconstruction_samples(
        test_originals,
        test_reconstructions,
        n_samples=8,
        save_path=results_dir / "reconstruction_samples.png",
        use_abs=True
    )
    
    # Latent space
    plot_latent_space(
        test_latents,
        save_path=results_dir / "latent_space.png"
    )
    
    # Save overall metrics
    metrics_file = results_dir / "overall_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    print(f"\n💾 Saved metrics to: {metrics_file}")
    
    print(f"\n✅ All results saved to: {results_dir}")


def main():
    """Main Phase 2 pipeline."""
    args = parse_args()
    
    print("="*80)
    print(" BRIDGE COMPRESSION - PHASE 2: MULTI-DAY TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data dir: {args.data_dir}")
    print(f"  N days: {args.n_days}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Output dir: {args.output_dir}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Find and load files
        if args.file_list:
            print(f"\n Using provided file list ({len(args.file_list)} files)")
            file_list = [Path(f) for f in args.file_list]
        else:
            file_list = find_csv_files(
                Path(args.data_dir),
                args.n_days,
                args.start_date
            )
        
        # Step 2: Load multi-day data
        data_dict = load_multiple_days(file_list, args.skip_exploration, output_dir)
        
        # Step 3: Preprocess
        processed_file = output_dir / "processed_data.npz"
        boundaries_file = output_dir / "day_boundaries.json"
        
        if args.reuse_processed and processed_file.exists():
            print(f"\n♻️  Reusing existing processed data: {processed_file}")
        else:
            processed_file = preprocess_multi_day(data_dict, output_dir)
            boundaries_file = output_dir / f"phase2_{data_dict['day_info'][0]['date']}_to_{data_dict['day_info'][-1]['date']}_{data_dict['n_days']}days_boundaries.json"
        
        # Step 4: Train
        best_checkpoint = train_phase2_model(
            processed_file,
            output_dir,
            quick=args.quick
        )
        
        # Step 5: Evaluate
        evaluate_phase2(
            best_checkpoint,
            processed_file,
            boundaries_file,
            output_dir
        )
        
        print("\n" + "="*80)
        print(" PHASE 2 COMPLETE!")
        print("="*80)
        print(f"\n All results saved to: {output_dir}/")
        print(f"\nNext steps:")
        print(f"  1. Check results in {output_dir}/results/")
        print(f"  2. Review metrics in overall_metrics.json")
        print(f"  3. If R² > 0.85, ready for Phase 3!")
        
    except Exception as e:
        print(f"\n Error in Phase 2 pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()