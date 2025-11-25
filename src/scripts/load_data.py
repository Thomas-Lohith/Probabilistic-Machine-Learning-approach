
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from config import config


def load_single_csv(file_path: Path) -> pd.DataFrame:
    print(f"Loading: {file_path}")
    
    df = pd.read_csv(file_path)
    df = df.drop(columns=['day', 'hour_file', 'start_time', 'end_time', 'variance'])

    #decreasing the computational time
    float64_cols = df.select_dtypes(include='float64').columns
    df[float64_cols] = df[float64_cols].astype('float32')

    

    # Verify columns
    expected_cols = config.COLUMNS
    if list(df.columns) != expected_cols:
        print(f"Warning: Column mismatch!")
        #print(f"Expected: {expected_cols}")
        #print(f"Got: {list(df.columns)}")
    
    print(f"Shape: {df.shape}")
    #print(f"Columns: {df.columns.tolist()}")

    
    
    return df


def explore_data(df: pd.DataFrame, show_plots: bool = True) -> dict:
    """
    Explore and analyze the loaded data.
    Args:
        df: DataFrame with accelerometer data
        show_plots: Whether to display plots 
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    # Basic info
    print(f"\n Dataset Shape: {df.shape}")
    print(f"   Rows: {df.shape[0]:,} (expected ~86,400 for 1 day)")
    print(f"   Columns: {df.shape[1]}")
    
    # Missing data analysis
    print(f"\n Missing Data:")
    missing_counts = df[config.FEATURE_COLUMNS].isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    for col in config.FEATURE_COLUMNS:
        count = missing_counts[col]
        pct = missing_pct[col]
        print(f"   {col:15s}: {count:6,} ({pct:5.2f}%)")
    
    stats['missing_counts'] = missing_counts.to_dict()
    stats['missing_pct'] = missing_pct.to_dict()
    
    # Feature statistics (excluding NaN)
    print(f"\nFeature Statistics (excluding NaN):")
    feature_stats = df[config.FEATURE_COLUMNS].describe()
    print(feature_stats)
    
    stats['feature_stats'] = feature_stats.to_dict()
    
    # Check for infinite values
    print(f"\n♾️  Infinite Values:")
    for col in config.FEATURE_COLUMNS:
        inf_count = np.isinf(df[col]).sum()
        print(f"   {col:15s}: {inf_count}")
    
    # Time range
    if 'start_time' in df.columns and 'end_time' in df.columns:
        print(f"\n Time Range:")
        print(f"   Start: {df['start_time'].iloc[0]}")
        print(f"   End:   {df['end_time'].iloc[-1]}")
    
    # Sample data
    print(f"\n First 5 rows:")
    print(df.head())
    
    print(f"\n Sample rows with missing data:")
    missing_rows = df[df[config.FEATURE_COLUMNS].isnull().any(axis=1)]
    if len(missing_rows) > 0:
        print(missing_rows.head(10))
    else:
        print("   No missing data found!")
    
    # Visualization
    if show_plots:
        plot_data_exploration(df)
    
    return stats


def plot_data_exploration(df: pd.DataFrame):
    """
    Create exploratory plots for the data.
    
    Args:
        df: DataFrame with accelerometer data
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Bridge Accelerometer Data Exploration', fontsize=16, y=1.00)
    
    # Get valid (non-NaN) data for each feature
    features = config.FEATURE_COLUMNS
    
    for idx, feature in enumerate(features):
        valid_data = df[feature].dropna()
        
        # Time series plot (left column)
        ax1 = axes[idx, 0]
        # Plot only first 10000 points for speed
        sample_size = min(10000, len(valid_data))
        sample_indices = np.linspace(0, len(valid_data)-1, sample_size, dtype=int)
        ax1.plot(sample_indices, valid_data.iloc[sample_indices], alpha=0.7, linewidth=0.5)
        ax1.set_title(f'{feature} - Time Series (sampled)')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel(feature)
        ax1.grid(True, alpha=0.3)
        
        # Distribution plot (right column)
        ax2 = axes[idx, 1]
        ax2.hist(valid_data, bins=100, alpha=0.7, edgecolor='black')
        ax2.set_title(f'{feature} - Distribution')
        ax2.set_xlabel(feature)
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = valid_data.mean()
        std_val = valid_data.std()
        ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2e}')
        ax2.legend()
    
    plt.tight_layout()
    
    #Save figure
    output_dir =Path("/data/pool/c8x-98x/pml/src/scripts/results/figures")
    #output_dir.mkdir(exist_ok= True)
    output_path = output_dir / "data_exploration.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Saved exploration plot to: {output_path}")
    
    plt.close()


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract feature array from DataFrame.
    
    Args:
        df: DataFrame with accelerometer data
        
    Returns:
        NumPy array of shape (n_samples, 3) with [mean, variance, log_variance]
    """
    features = df[config.FEATURE_COLUMNS].values
    features = np.abs(features)

    return features


def check_data_continuity(df: pd.DataFrame) -> dict:
    """
    Check if data is continuous (1 second intervals).
    
    Args:
        df: DataFrame with start_time and end_time columns
        
    Returns:
        Dictionary with continuity statistics
    """
    if 'start_time' not in df.columns:
        return {"error": "No start_time column"}
    
    # Parse timestamps
    df['start_dt'] = pd.to_datetime(df['start_time'], format='%Y/%m/%d %H:%M:%S:%f')
    
    # Calculate time differences
    time_diffs = df['start_dt'].diff()
    
    # Expected: 1 second
    expected_diff = pd.Timedelta(seconds=1)
    
    # Count gaps
    gaps = time_diffs[time_diffs != expected_diff].dropna()
    
    continuity_stats = {
        'total_rows': len(df),
        'gaps_found': len(gaps),
        'gap_percentage': (len(gaps) / len(df)) * 100,
        'expected_interval': '1 second',
        'unique_intervals': time_diffs.value_counts().head(10).to_dict()
    }
    
    print("\n⏱️  Data Continuity Check:")
    print(f"   Total rows: {continuity_stats['total_rows']:,}")
    print(f"   Gaps found: {continuity_stats['gaps_found']:,} ({continuity_stats['gap_percentage']:.2f}%)")
    
    return continuity_stats


def get_data_summary(file_path: Path) -> dict:
    """
    Get complete summary of a data file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Dictionary with complete data summary
    """
    df = load_single_csv(file_path)
    stats = explore_data(df, show_plots=True)
    continuity = check_data_continuity(df)
    
    summary = {
        'file': str(file_path),
        'shape': df.shape,
        'statistics': stats,
        'continuity': continuity,
    }
    
    return summary


if __name__ == "__main__":
    # Test with Phase 1 file
    test_file = Path("/data/pool/c8x-98x/bridge_data/100_days") / config.PHASE1_TEST_FILE
    
    if test_file.exists():
        print(f"Testing data loading with: {test_file}")
        summary = get_data_summary(test_file)
        #print(summary)
        print("\n" + "="*60)
        print(" DATA LOADING TEST COMPLETE")
        print("="*60)
    else:
        print(f"Test file not found: {test_file}")
        print(f"Available files in {test_file.parent}:")
        if test_file.parent.exists():
            csv_files = list(test_file.parent.glob("*.csv"))
            for f in csv_files[:5]:  # Show first 5
                print(f"   - {f.name}")
        else:
            print(f"   Directory does not exist!")