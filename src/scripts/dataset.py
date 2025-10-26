"""
PyTorch Dataset for bridge accelerometer sequences
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple

from config import config


class AccelerometerDataset(Dataset):
    """
    PyTorch Dataset for accelerometer sequences.
    """
    
    def __init__(self, sequences: np.ndarray):
        """
        Args:
            sequences: NumPy array of shape (n_sequences, seq_length, n_features)
        """
        self.sequences = torch.FloatTensor(sequences)
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sequence.
        
        Returns:
            Tensor of shape (seq_length, n_features)
        """
        return self.sequences[idx]


def create_dataloaders(train: np.ndarray,
                       val: np.ndarray,
                       test: np.ndarray,
                       batch_size: int = None,
                       num_workers: int = None,
                       pin_memory: bool = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        train: Training sequences
        val: Validation sequences
        test: Test sequences
        batch_size: Batch size (default: config.BATCH_SIZE)
        num_workers: Number of data loading workers (default: config.NUM_WORKERS)
        pin_memory: Whether to use pinned memory (default: config.PIN_MEMORY)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    if pin_memory is None:
        pin_memory = config.PIN_MEMORY
    
    # Create datasets
    train_dataset = AccelerometerDataset(train)
    val_dataset = AccelerometerDataset(val)
    test_dataset = AccelerometerDataset(test)
    
    print("\n" + "="*60)
    print("CREATING DATALOADERS")
    print("="*60)
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset):,} sequences")
    print(f"  Val:   {len(val_dataset):,} sequences")
    print(f"  Test:  {len(test_dataset):,} sequences")
    
    print(f"\nDataLoader settings:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"\nBatches per epoch:")
    print(f"  Train: {len(train_loader):,}")
    print(f"  Val:   {len(val_loader):,}")
    print(f"  Test:  {len(test_loader):,}")
    
    print("="*60)
    
    return train_loader, val_loader, test_loader


def load_processed_data(processed_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed data from .npz file.
    
    Args:
        processed_file: Path to .npz file
        
    Returns:
        Tuple of (train, val, test) arrays
    """
    print(f"\n📂 Loading processed data from: {processed_file}")
    
    data = np.load(processed_file)
    
    train = data['train']
    val = data['val']
    test = data['test']
    
    print(f"   Train: {train.shape}")
    print(f"   Val:   {val.shape}")
    print(f"   Test:  {test.shape}")
    
    return train, val, test


if __name__ == "__main__":
    # Test dataset creation
    print("Testing Dataset and DataLoader...")
    
    # Create dummy data
    n_sequences = 1000
    seq_length = config.SEQUENCE_LENGTH
    n_features = config.N_FEATURES
    
    dummy_train = np.random.randn(n_sequences, seq_length, n_features)
    dummy_val = np.random.randn(n_sequences//5, seq_length, n_features)
    dummy_test = np.random.randn(n_sequences//5, seq_length, n_features)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dummy_train,
        dummy_val,
        dummy_test,
        batch_size=64
    )
    
    # Test iteration
    print("\n🔄 Testing data iteration...")
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}: {batch.shape}")
        if batch_idx >= 2:  # Just test first 3 batches
            break
    
    print("\n Dataset test successful!")