
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import config
from load_data import load_single_csv, extract_features


class DataPreprocessor:
    """
    Handles all preprocessing steps for bridge accelerometer data.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def handle_missing_values(self, features: np.ndarray, method: str = 'interpolate') -> np.ndarray:

        print(f"\n Handling missing values (method: {method})...")
        
        # Count NaN before
        nan_count_before = np.isnan(features).sum()
        nan_pct_before = (nan_count_before / features.size) * 100
        print(f"   NaN values before: {nan_count_before:,} ({nan_pct_before:.2f}%)")
        
        if method == 'interpolate':
            # Linear interpolation for each feature
            for i in range(features.shape[1]):
                col = features[:, i]
                # Find NaN positions
                nan_mask = np.isnan(col)
                if nan_mask.any():
                    # Interpolate
                    valid_indices = np.where(~nan_mask)[0]
                    if len(valid_indices) > 0:
                        features[nan_mask, i] = np.interp(
                            np.where(nan_mask)[0],
                            valid_indices,
                            col[valid_indices]
                        )
        
        elif method == 'forward_fill':
            # Forward fill (propagate last valid value)
            df = pd.DataFrame(features)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')  # Handle leading NaN
            features = df.values
        
        elif method == 'remove':
            # Remove rows with any NaN (NOT RECOMMENDED - breaks temporal continuity)
            mask = ~np.isnan(features).any(axis=1)
            features = features[mask]
            print(f"   Removed {(~mask).sum()} rows with NaN")
        
        # Count NaN after
        nan_count_after = np.isnan(features).sum()
        nan_pct_after = (nan_count_after / features.size) * 100
        print(f"   NaN values after: {nan_count_after:,} ({nan_pct_after:.2f}%)")
        
        return features
    
    def normalize(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
    
        print(f"\n📏 Normalizing features (fit={fit})...")
        
        if fit:
            normalized = self.scaler.fit_transform(features)
            self.is_fitted = True
            print(f"   Fitted scaler on {features.shape[0]:,} samples")
            print(f"   Mean: {self.scaler.mean_}")
            print(f"   Std:  {self.scaler.scale_}")
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted! Call with fit=True first.")
            normalized = self.scaler.transform(features)
        
        return normalized
    
    def inverse_normalize(self, normalized_features: np.ndarray) -> np.ndarray:

        if not self.is_fitted:
            raise ValueError("Scaler not fitted!")
        
        return self.scaler.inverse_transform(normalized_features)
    
    def create_sequences(self, features: np.ndarray, 
                        sequence_length: int = None,
                        overlap: bool = False) -> np.ndarray:
        """
        Create sequences from continuous time series data.
        Args:
            features: Array of shape (n_samples, n_features)
            sequence_length: Length of each sequence (default: config.SEQUENCE_LENGTH)
            overlap: Whether sequences should overlap (default: False)        
        Returns:
            Array of shape (n_sequences, sequence_length, n_features)
        """
        if sequence_length is None:
            sequence_length = config.SEQUENCE_LENGTH
        
        print(f"\n Creating sequences...")
        print(f"   Input shape: {features.shape}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Overlap: {overlap}")
        
        n_samples = features.shape[0]
        
        if overlap:
            # Overlapping sequences (stride = 1)
            stride = 1
        else:
            # Non-overlapping sequences
            stride = sequence_length
        
        # Calculate number of sequences
        n_sequences = (n_samples - sequence_length) // stride + 1
        
        # Create sequences
        sequences = []
        for i in range(0, n_samples - sequence_length + 1, stride):
            seq = features[i:i + sequence_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        
        print(f"   Output shape: {sequences.shape}")
        print(f"   Number of sequences: {sequences.shape[0]:,}")
        
        return sequences
    
    def split_data(self, sequences: np.ndarray, 
                   train_ratio: float = None,
                   val_ratio: float = None,
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into train, validation, and test sets.
        
        Args:
            sequences: Array of shape (n_sequences, sequence_length, n_features)
            train_ratio: Fraction for training (default: config.TRAIN_RATIO)
            val_ratio: Fraction for validation (default: config.VAL_RATIO)
            random_state: Random seed
            
        Returns:
            Tuple of (train, val, test) arrays
        """
        if train_ratio is None:
            train_ratio = config.TRAIN_RATIO
        if val_ratio is None:
            val_ratio = config.VAL_RATIO
        
        test_ratio = 1.0 - train_ratio - val_ratio
        
        print(f"\n✂️  Splitting data...")
        print(f"   Total sequences: {sequences.shape[0]:,}")
        print(f"   Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            sequences,
            test_size=test_ratio,
            random_state=random_state,
            shuffle=True  # Shuffle for better generalization
        )
        
        # Second split: train vs val
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            shuffle=True
        )
        
        print(f"   Train: {train.shape[0]:,} sequences ({train.shape[0]/sequences.shape[0]:.1%})")
        print(f"   Val:   {val.shape[0]:,} sequences ({val.shape[0]/sequences.shape[0]:.1%})")
        print(f"   Test:  {test.shape[0]:,} sequences ({test.shape[0]/sequences.shape[0]:.1%})")
        
        return train, val, test
    
    def save_scaler(self, output_path: Path):
        """Save the fitted scaler to disk."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted!")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\n💾 Saved scaler to: {output_path}")
    
    def load_scaler(self, input_path: Path):
        """Load a fitted scaler from disk."""
        with open(input_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        print(f"\n📂 Loaded scaler from: {input_path}")


def preprocess_single_day(csv_path: Path, 
                          output_dir: Path = None,
                          save_processed: bool = True) -> Tuple:

    print("\n" + "="*60)
    print(f"PREPROCESSING: {csv_path.name}")
    print("="*60)
    
    # 1. Load data
    df = load_single_csv(csv_path)
    features = extract_features(df)
    
    print(f"\n📥 Loaded features: {features.shape}")
    
    # 2. Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # 3. Handle missing values
    features = preprocessor.handle_missing_values(features, method='interpolate')
    
    # 4. Normalize (fit on ALL data for single day - we'll fix this for multiple days)
    features_normalized = preprocessor.normalize(features, fit=True)
    
    # 5. Create sequences
    sequences = preprocessor.create_sequences(features_normalized)
    
    # 6. Split data
    train, val, test = preprocessor.split_data(sequences)
    
    # 7. Save processed data
    if save_processed:
        if output_dir is None:
            output_dir = Path("/data/pool/c8x-98x/pml/src/script/processed_data")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save arrays
        output_file = output_dir / f"{csv_path.stem}_processed.npz"
        np.savez_compressed(
            output_file,
            train=train,
            val=val,
            test=test,
            original_shape=features.shape
        )
        print(f"\n💾 Saved processed data to: {output_file}")
        
        # Save scaler
        scaler_file = output_dir / f"{csv_path.stem}_scaler.pkl"
        preprocessor.save_scaler(scaler_file)
    
    print("\n" + "="*60)
    print(" PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nData ready for training:")
    print(f"  Train: {train.shape}")
    print(f"  Val:   {val.shape}")
    print(f"  Test:  {test.shape}")
    
    return train, val, test, preprocessor


if __name__ == "__main__":
    # Test preprocessing on Phase 1 file
    test_file = Path("/data/pool/c8x-98x/bridge_data/100_days") / config.PHASE1_TEST_FILE
    
   