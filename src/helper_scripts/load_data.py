"""
Data loading utilities for bridge accelerometer data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_single_day(filepath: str) -> pd.DataFrame:
    """
    Load a single day's CSV file
    Args:
        filepath: Path to CSV file
    Returns:
        DataFrame with parsed timestamps
    """
    df = pd.read_csv(filepath)
    
    # Parse timestamps
    df['start_time'] = pd.to_datetime(df['start_time'], format='%Y/%m/%d %H:%M:%S:%f')
    df['end_time'] = pd.to_datetime(df['end_time'], format='%Y/%m/%d %H:%M:%S:%f')
    return df


def load_multiple_days(data_dir: str, num_days: Optional[int] = None) -> pd.DataFrame:
    """
    Load multiple days of data
    
    Args:
        data_dir: Directory containing CSV files
        num_days: Number of days to load (None = all)
        
    Returns:
        Combined DataFrame
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob('*.csv'))
    
    if num_days:
        csv_files = csv_files[:num_days]
    
    print(f"Loading {len(csv_files)} days of data...")
    
    all_data = []
    for csv_file in csv_files:
        try:
            df = load_single_day(csv_file)
            all_data.append(df)
            print(f"Loaded {csv_file.name}: {len(df)} records")
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records: {len(combined_df)}")
    
    return combined_df


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract feature array from DataFrame
    Args:
        df: Input DataFrame    
    Returns:
        Feature array [N, 3] containing mean, variance, log_variance
    """
    # Select feature columns
    feature_cols = ['mean', 'variance', 'log_variance']
    features = df[feature_cols].values
    
    # Handle NaN values
    nan_mask = np.isnan(features).any(axis=1)
    print(f"Rows with NaN: {nan_mask.sum()} ({nan_mask.mean()*100:.2f}%)")
    
    return features, nan_mask


def get_data_statistics(features: np.ndarray, nan_mask: np.ndarray) -> dict:
    """
    Compute basic statistics for features
    
    Args:
        features: Feature array
        nan_mask: Boolean mask for NaN values
        
    Returns:
        Dictionary of statistics
    """
    # Filter out NaN rows
    valid_features = features[~nan_mask]
    
    stats = {
        'total_samples': len(features),
        'valid_samples': len(valid_features),
        'nan_samples': nan_mask.sum(),
        'nan_percentage': nan_mask.mean() * 100,
        'mean': {
            'mean': np.mean(valid_features[:, 0]),
            'std': np.std(valid_features[:, 0]),
            'min': np.min(valid_features[:, 0]),
            'max': np.max(valid_features[:, 0]),
        },
        'variance': {
            'mean': np.mean(valid_features[:, 1]),
            'std': np.std(valid_features[:, 1]),
            'min': np.min(valid_features[:, 1]),
            'max': np.max(valid_features[:, 1]),
        },
        'log_variance': {
            'mean': np.mean(valid_features[:, 2]),
            'std': np.std(valid_features[:, 2]),
            'min': np.min(valid_features[:, 2]),
            'max': np.max(valid_features[:, 2]),
        }
    }
    
    return stats


# Example usage
if __name__ == "__main__":
    # Test with first day
    df = load_single_day('data/raw/20241126.csv')
    print(f"\nFirst day shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Extract features
    features, nan_mask = extract_features(df)
    print(f"\nFeature array shape: {features.shape}")
    
    # Get statistics
    stats = get_data_statistics(features, nan_mask)
    print(f"\nData Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Valid samples: {stats['valid_samples']}")
    print(f"NaN samples: {stats['nan_samples']} ({stats['nan_percentage']:.2f}%)")