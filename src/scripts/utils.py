
import torch
import numpy as np
import random
import os
from pathlib import Path
import json
from typing import Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f" Random seed set to: {seed}")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return {
        'trainable': trainable,
        'total': total,
        'frozen': total - trainable
    }


def save_json(data: Dict[str, Any], path: Path):

    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f" Saved JSON: {path}")


def load_json(path: Path) -> Dict[str, Any]:

    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f" Loaded JSON: {path}")
    return data


def get_device() -> str:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f" Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print(" Using CPU")
    
    return device


def format_time(seconds: float) -> str:
   
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_model_summary(model: torch.nn.Module, 
                       input_shape: tuple = None,
                       device: str = 'cpu'):
 
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen:    {params['frozen']:,}")
    print(f"  Total:     {params['total']:,}")
    
    # Model architecture
    print(f"\nArchitecture:")
    print(model)
    
    # Test forward pass if input shape provided
    if input_shape is not None:
        print(f"\nTesting forward pass with input shape: {input_shape}")
        try:
            dummy_input = torch.randn(1, *input_shape).to(device)
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_input)
                
            if isinstance(output, tuple):
                print(f"Output shapes:")
                for i, out in enumerate(output):
                    print(f"  Output {i}: {tuple(out.shape)}")
            else:
                print(f"Output shape: {tuple(output.shape)}")
                
            print("✅ Forward pass successful!")
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
    
    print("="*60 + "\n")


def ensure_dir(path: Path):
   
    path.mkdir(parents=True, exist_ok=True)


def get_git_hash() -> str:

    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return 'unknown'


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Any,
                   epoch: int,
                   loss: float,
                   path: Path,
                   **kwargs):
  
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'git_hash': get_git_hash(),
        **kwargs
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved: {path}")


def load_checkpoint(path: Path,
                   model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer = None,
                   scheduler: Any = None,
                   device: str = 'cpu') -> Dict:

    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f" Checkpoint loaded: {path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Loss: {checkpoint['loss']:.6f}")
    
    return checkpoint


class AverageMeter:

    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.6f}"


def print_training_header():
    """Print a nice header for training."""
    print("\n" + "="*80)
    print(" " * 30 + " TRAINING STARTED ")
    print("="*80 + "\n")


def print_training_footer(elapsed_time: float):
    """Print a nice footer after training."""
    print("\n" + "="*80)
    print(" " * 30 + " TRAINING COMPLETE ")
    print(f"Total time: {format_time(elapsed_time)}")
    print("="*80 + "\n")


def print_epoch_summary(epoch: int, 
                       total_epochs: int,
                       train_loss: float,
                       val_loss: float,
                       lr: float,
                       best: bool = False):
 
    star = "⭐" if best else "  "
    print(f"\n{star} Epoch {epoch+1}/{total_epochs}")
    print(f"   Train Loss: {train_loss:.6f}")
    print(f"   Val Loss:   {val_loss:.6f}")
    print(f"   LR:         {lr:.6f}")
    if best:
        print("⭐ New best model!")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    

