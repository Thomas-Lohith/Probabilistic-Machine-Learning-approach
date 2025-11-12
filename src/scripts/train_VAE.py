"""
Training script for LSTM Variational Autoencoder (VAE)

This is parallel to train.py but for VAE model with MSE + NLL loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, Tuple

from config import config
from VAE import create_vae_model, vae_loss_function
from utils import set_seed


class VAETrainer:

    
    def __init__(self, 
                 model, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = None,
                 learning_rate: float = None,
                 weight_decay: float = None,
                 kl_weight: float = 1.0,
                 kl_annealing: bool = False,
                 kl_annealing_epochs: int = 10):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else config.DEVICE
        
        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing
        self.kl_annealing_epochs = kl_annealing_epochs
        self.current_kl_weight = 0.0 if kl_annealing else kl_weight
        
        # Optimizer
        lr = learning_rate if learning_rate else config.LEARNING_RATE
        wd = weight_decay if weight_decay else config.WEIGHT_DECAY
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=config.BETAS,
            weight_decay=wd
        )
        
        # Scheduler (OneCycleLR)
        total_steps = len(train_loader) * config.EPOCHS
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.MAX_LR,
            total_steps=total_steps,
            pct_start=config.PCT_START,
            div_factor=config.DIV_FACTOR,
            final_div_factor=config.FINAL_DIV_FACTOR
        )
        
        # Mixed precision training
        self.use_amp = config.USE_MIXED_PRECISION
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_mse': [],
            'train_kl': [],
            'val_loss': [],
            'val_mse': [],
            'val_kl': [],
            'kl_weight': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def update_kl_weight(self, epoch: int):

        if self.kl_annealing and epoch < self.kl_annealing_epochs:
            # Linear annealing from 0 to kl_weight
            self.current_kl_weight = self.kl_weight * (epoch + 1) / self.kl_annealing_epochs
        else:
            self.current_kl_weight = self.kl_weight
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
        for batch_idx, data in enumerate(pbar):
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    reconstructed, mu, log_var = self.model(data)
                    loss, loss_dict = vae_loss_function(
                        reconstructed, data, mu, log_var, 
                        kl_weight=self.current_kl_weight
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    config.GRADIENT_CLIP_VALUE
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstructed, mu, log_var = self.model(data)
                loss, loss_dict = vae_loss_function(
                    reconstructed, data, mu, log_var,
                    kl_weight=self.current_kl_weight
                )
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    config.GRADIENT_CLIP_VALUE
                )
                
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss']
            total_mse += loss_dict['mse_loss']
            total_kl += loss_dict['kl_divergence']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'mse': f"{loss_dict['mse_loss']:.4f}",
                'kl': f"{loss_dict['kl_divergence']:.4f}",
                'kl_w': f"{self.current_kl_weight:.3f}"
            })
        
        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_kl = total_kl / num_batches
        
        return {
            'train_loss': avg_loss,
            'train_mse': avg_mse,
            'train_kl': avg_kl
        }
    
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]")
        
        for batch_idx, data in enumerate(pbar):
            data = data.to(self.device)
            
            if self.use_amp:
                with autocast():
                    reconstructed, mu, log_var = self.model(data)
                    loss, loss_dict = vae_loss_function(
                        reconstructed, data, mu, log_var,
                        kl_weight=self.current_kl_weight
                    )
            else:
                reconstructed, mu, log_var = self.model(data)
                loss, loss_dict = vae_loss_function(
                    reconstructed, data, mu, log_var,
                    kl_weight=self.current_kl_weight
                )
            
            total_loss += loss_dict['total_loss']
            total_mse += loss_dict['mse_loss']
            total_kl += loss_dict['kl_divergence']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'mse': f"{loss_dict['mse_loss']:.4f}",
                'kl': f"{loss_dict['kl_divergence']:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_kl = total_kl / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_mse': avg_mse,
            'val_kl': avg_kl
        }
    
    def train(self, epochs: int = None, checkpoint_dir: Path = None) -> Dict:

        epochs = epochs if epochs else config.EPOCHS
        checkpoint_dir = checkpoint_dir if checkpoint_dir else config.CHECKPOINT_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("STARTING VAE TRAINING")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {config.BATCH_SIZE}")
        print(f"Learning Rate: {config.LEARNING_RATE}")
        print(f"KL Weight: {self.kl_weight}")
        print(f"KL Annealing: {self.kl_annealing}")
        if self.kl_annealing:
            print(f"KL Annealing Epochs: {self.kl_annealing_epochs}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Update KL weight for annealing
            self.update_kl_weight(epoch)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_mse'].append(train_metrics['train_mse'])
            self.history['train_kl'].append(train_metrics['train_kl'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_mse'].append(val_metrics['val_mse'])
            self.history['val_kl'].append(val_metrics['val_kl'])
            self.history['kl_weight'].append(self.current_kl_weight)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Train - Loss: {train_metrics['train_loss']:.6f}, "
                  f"MSE: {train_metrics['train_mse']:.6f}, "
                  f"KL: {train_metrics['train_kl']:.6f}")
            print(f"  Val   - Loss: {val_metrics['val_loss']:.6f}, "
                  f"MSE: {val_metrics['val_mse']:.6f}, "
                  f"KL: {val_metrics['val_kl']:.6f}")
            print(f"  KL Weight: {self.current_kl_weight:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
                checkpoint_path = checkpoint_dir / f"vae_checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch)
                print(f"   Checkpoint saved: {checkpoint_path.name}")
            
            # Check for improvement
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                
                # Save best model
                best_path = checkpoint_dir / "vae_best_model.pt"
                self.save_checkpoint(best_path, epoch)
                print(f"   New best model! Val loss: {self.best_val_loss:.6f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{config.PATIENCE}")
            
            # Early stopping
            if self.patience_counter >= config.PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
            
            print()
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Best val loss: {self.best_val_loss:.6f}")
        print("="*60 + "\n")
        
        # Save final checkpoint and history
        final_path = checkpoint_dir / "vae_final_model.pt"
        self.save_checkpoint(final_path, epochs-1)
        
        history_path = checkpoint_dir / "vae_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': {
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim,
                'output_dim': self.model.output_dim,
                'seq_length': self.model.seq_length,
                'kl_weight': self.kl_weight,
                'kl_annealing': self.kl_annealing
            }
        }
        
        torch.save(checkpoint, path)


def train_vae_phase1(data_path: str = None,
                     checkpoint_dir: Path = None, 
                     epochs: int = None,
                     quick_test: bool = False,
                     kl_weight: float = 1.0,
                     kl_annealing: bool = True,
                     kl_annealing_epochs: int = 10):

    set_seed(42)
    
    # Handle quick_test
    if quick_test and epochs is None:
        epochs = 10
    
    # Load data
    if data_path is None:
        data_path = config.PROCESSED_DATA_DIR / "processed_data.npz"
    
    # Set checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = config.CHECKPOINT_DIR
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Loading data from: {data_path}")
        # Use the same data loading as train.py
    from dataset import load_processed_data, create_dataloaders
    
    # Load data using your existing function
    train, val, test = load_processed_data(data_path)
    
    print(f"Train shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Test shape: {test.shape}")
    
    # Create data loaders using your existing function
    train_loader, val_loader, test_loader = create_dataloaders(train, val, test)
    
    # Create model
    model = create_vae_model()
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        kl_weight=kl_weight,
        kl_annealing=kl_annealing,
        kl_annealing_epochs=kl_annealing_epochs
    )
    
    # Train
    history = trainer.train(epochs=epochs, checkpoint_dir=checkpoint_dir)
    
    return model, history


if __name__ == "__main__":
    print("Training LSTM Variational Autoencoder (VAE)...")
    print("Loss: MSE (reconstruction) + KL Divergence (NLL)")
    
    # Train VAE
    model, history = train_vae_phase1(
        epochs=50,
        kl_weight=1.0,
        kl_annealing=True,  # Use KL annealing to prevent posterior collapse
        kl_annealing_epochs=10
    )
    
    print("\n VAE training complete!")