
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np

from config import config
from model import create_model
from dataset import create_dataloaders, load_processed_data


class Trainer:
    """
    Trainer class for LSTM Autoencoder.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 device: str = None,
                 learning_rate: float = None,
                 epochs: int = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else config.DEVICE
        self.epochs = epochs if epochs else config.EPOCHS
        
        # Loss function
        self.criterion = nn.MSELoss()    #nn.MSELoss()
        
        # Optimizer
        if learning_rate is None:
            learning_rate = config.LEARNING_RATE
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS
        )
        
        # Learning rate scheduler (OneCycleLR)
        if config.USE_SCHEDULER:
            steps_per_epoch = len(train_loader)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.MAX_LR,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=config.PCT_START,
                div_factor=config.DIV_FACTOR,
                final_div_factor=config.FINAL_DIV_FACTOR
            )
        else:
            self.scheduler = None
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.USE_MIXED_PRECISION else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        print("\n" + "="*60)
        print("TRAINER INITIALIZED")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Optimizer: {config.OPTIMIZER}")
        print(f"Scheduler: {config.USE_SCHEDULER}")
        print(f"Mixed precision: {config.USE_MIXED_PRECISION}")
        print(f"Gradient clipping: {config.GRADIENT_CLIP_VALUE}")
        print("="*60)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    reconstructed, latent = self.model(batch)
                    loss = self.criterion(reconstructed, batch)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if config.GRADIENT_CLIP_VALUE:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP_VALUE)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward pass
                reconstructed, latent = self.model(batch)
                loss = self.criterion(reconstructed, batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if config.GRADIENT_CLIP_VALUE:
                    nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP_VALUE)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> float:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = batch.to(self.device)
                
                if self.scaler:
                    with autocast():
                        reconstructed, latent = self.model(batch)
                        loss = self.criterion(reconstructed, batch)
                else:
                    reconstructed, latent = self.model(batch)
                    loss = self.criterion(reconstructed, batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, save_dir: Path = None):
        """Complete training loop."""
        # if save_dir is None:
        #     save_dir = Path("/home/claude/checkpoints")
        # save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Track metrics
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.epochs} - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {current_lr:.2e}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(save_dir / "best.pt", epoch, val_loss, is_best=True)
                print(f"  ✅ Best model saved! (val_loss: {val_loss:.6f})")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")
            
            # Save latest checkpoint
            if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
                self.save_checkpoint(save_dir / "latest.pt", epoch, val_loss, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= config.PATIENCE:
                print(f"\n⏹️  Early stopping after {epoch+1} epochs (patience: {config.PATIENCE})")
                break
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:.6f}")
        print(f"Final train loss: {self.history['train_loss'][-1]:.6f}")
        
        # Save training history
        history_file = save_dir / "history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n💾 Saved training history to: {history_file}")
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'best_val_loss': self.best_val_loss,
            'is_best': is_best,
            'config': {
                'latent_dim': config.LATENT_DIM,
                'sequence_length': config.SEQUENCE_LENGTH,
                'n_features': config.N_FEATURES,
                'encoder_hidden_dims': config.ENCODER_HIDDEN_DIMS,
                'decoder_hidden_dims': config.DECODER_HIDDEN_DIMS,
            }
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ Loaded checkpoint from: {path}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Val loss: {checkpoint['val_loss']:.6f}")


def train_phase1(processed_file: Path, save_dir: Path = None, quick_test: bool = False):

    print("\n" + "="*80)
    print("PHASE 1: SINGLE DAY TRAINING")
    print("="*80)
    
    # Load processed data
    train, val, test = load_processed_data(processed_file)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(train, val, test)
    
    # Create model
    model = create_model()
    
    # Create trainer
    epochs = config.PHASE1_QUICK_EPOCHS if quick_test else config.EPOCHS
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )
    
    # Train
    history = trainer.train(save_dir=save_dir)
    
    return trainer, history


if __name__ == "__main__":
    # Phase 1 training
    processed_file = Path("/data/pool/c8x-98x/bridge_data/100_days/data/processed/20241127_processed.npz")
    save_dir = Path("/data/pool/c8x-98x/bridge_data/100_days/data/checkpoints")
    
    if processed_file.exists():
        print(f"Starting Phase 1 training with: {processed_file}")
        
        trainer, history = train_phase1(
            processed_file=processed_file,
            save_dir=save_dir,
            quick_test=False  # Set to True for quick testing
        )
        
        print("\n🎉 Phase 1 training complete!")
        
    else:
        print(f" Processed file not found: {processed_file}")
        print("Run preprocess.py first!")