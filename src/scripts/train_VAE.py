

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
from config import config


def vae_loss_function(reconstructed, original, mu, log_var, kl_weight=1.0):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, original, reduction='mean')
    
    # KL divergence
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Normalize by batch size
    kl_divergence = kl_divergence / mu.size(0)
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_divergence
    
    # Loss components for logging
    loss_dict = {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_divergence': kl_divergence.item(),
        'weighted_kl': (kl_weight * kl_divergence).item()
    }
    
    return total_loss, loss_dict


def vae_loss_function_corrected(reconstructed, original, mu, log_var, 
                     kl_weight=1.0): #optional weight_variance=2.5
    """
    Corrected VAE Loss Function addressing all identified issues
    
    Args:
        reconstructed: [batch_size, seq_len, 2] - (mean_pred, var_pred)
        original: [batch_size, seq_len, 2] - (mean_true, var_true)
        mu: [batch_size, latent_dim] - latent mean
        log_var: [batch_size, latent_dim] - latent log variance
        kl_weight: float - KL divergence weight (use with annealing!)
        weight_variance: float - variance reconstruction weight
    
    Returns:
        total_loss: scalar tensor
        loss_dict: dictionary with loss components for logging
    """
    
    # =========================================================================
    # FIX 1 & 2: WEIGHTED RECONSTRUCTION LOSS (addressing your variance issue)
    # =========================================================================
    # Split mean and variance components
    mean_true = original[..., 0]           # [batch_size, seq_len]
    var_true = original[..., 1]            # [batch_size, seq_len]
    
    mean_pred = reconstructed[..., 0]      # [batch_size, seq_len]
    var_pred = reconstructed[..., 1]       # [batch_size, seq_len]
    
    # Separate MSE for each component
    mse_mean = F.mse_loss(mean_pred, mean_true, reduction='mean')
    mse_var = F.mse_loss(var_pred, var_true, reduction='mean')
    
    # Weight variance higher to fix underestimation problem
    recon_loss = mse_mean +  mse_var #optional weight_variance *mse_var
    
    
    # =========================================================================
    # FIX 3 & 4: CORRECTED KL DIVERGENCE CALCULATION
    # =========================================================================
    # Clamp log_var for numerical stability (Fix 4)
    log_var_clamped = torch.clamp(log_var, min=-20, max=20)
    
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # CRITICAL: Use torch.mean() instead of torch.sum() (Fix 1)
    # This automatically averages over batch AND latent dimensions
    kl_divergence = -0.5 * torch.mean(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp())
    
    # Alternative (mathematically equivalent, but more explicit):
    # kl_divergence = -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1)
    # kl_divergence = torch.mean(kl_divergence)
    
    
    # =========================================================================
    # TOTAL LOSS
    # =========================================================================
    total_loss = recon_loss + kl_weight * kl_divergence
    
    
    # =========================================================================
    # FIX 5: PROPER LOSS COMPONENT LOGGING (make them additive!)
    # =========================================================================
    loss_dict = {
        # Individual components
        'recon_loss_mean': mse_mean.item(),
        'recon_loss_var': mse_var.item(),
        'recon_loss_total': recon_loss.item(),
        'kl_divergence': kl_divergence.item(),
        
        # Weighted components
        'kl_weight': kl_weight,
        'weighted_kl': (kl_weight * kl_divergence).item(),
        #'variance_weight': weight_variance,
        
        # Total (verify these sum: recon_loss + kl_weight * kl_div = total_loss)
        'total_loss': total_loss.item(),
        
        # Useful ratio for monitoring
        'loss_ratio': recon_loss.item() / (kl_weight * kl_divergence.item() + 1e-8)
    }
    
    return total_loss, loss_dict

class VAETrainer:
    """
    Trainer class for LSTM VAE.
    Similar to AE Trainer but with VAE-specific loss and KL annealing.
    """
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 learning_rate=config.LEARNING_RATE,
                 device=config.DEVICE,
                 kl_weight=1.0,
                 kl_annealing=True,
                 kl_annealing_epochs=20,
                 kl_min=0.0,
                 kl_max=1.0,
                 use_amp=False,
                 epochs: int = None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else config.DEVICE
        self.epochs =  epochs if epochs else config.EPOCHS
        
        # KL annealing parameters
        self.kl_weight_target = kl_weight
        self.kl_annealing = kl_annealing
        self.kl_annealing_epochs = kl_annealing_epochs
        self.kl_min = kl_min
        self.kl_max = kl_max if kl_weight <= kl_max else kl_weight
        self.current_kl_weight = kl_min if kl_annealing else kl_weight
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )
        
        # Mixed precision
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': [],
            'kl_weight': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def update_kl_weight(self, epoch):
        """
        Update KL weight using linear annealing.
        
        KL weight increases linearly from kl_min to kl_max over kl_annealing_epochs.
        
        Args:
            epoch: Current epoch number
        """
        if not self.kl_annealing:
            self.current_kl_weight = self.kl_weight_target
            return
        
        if epoch < self.kl_annealing_epochs:
            # Linear annealing
            self.current_kl_weight = self.kl_min + \
                (self.kl_max - self.kl_min) * (epoch / self.kl_annealing_epochs)
        else:
            self.current_kl_weight = self.kl_weight_target
    
    def train_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        # Update KL weight
        self.update_kl_weight(epoch)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        # ERROR FIXED: Changed from "for data, _ in enumerate(pbar)" to proper enumerate usage
        for batch_idx, data in enumerate(pbar):
            # Handle different dataloader return formats
            if isinstance(data, (tuple, list)):
                data = data[0]  # take only input tensor if tuple/list
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # Mixed precision training
                with autocast():
                    reconstructed, mu, log_var = self.model(data)
                    loss, loss_dict = vae_loss_function(
                        reconstructed, data, mu, log_var,
                        kl_weight=self.current_kl_weight
                    )
                
                self.scaler.scale(loss).backward()
                
                # ERROR FIXED: Changed GRADIENT_CLIP to GRADIENT_CLIP_VALUE (consistent naming)
                if config.GRADIENT_CLIP_VALUE:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        config.GRADIENT_CLIP_VALUE
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Normal training
                reconstructed, mu, log_var = self.model(data)
                loss, loss_dict = vae_loss_function(
                    reconstructed, data, mu, log_var,
                    kl_weight=self.current_kl_weight
                )
                
                loss.backward()
                
                # ERROR FIXED: Changed GRADIENT_CLIP to GRADIENT_CLIP_VALUE (consistent naming)
                if config.GRADIENT_CLIP_VALUE:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        config.GRADIENT_CLIP_VALUE
                    )
                # Optimizer step
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss']
            total_recon_loss += loss_dict['recon_loss']
            total_kl_loss += loss_dict['kl_divergence']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'kl': f"{loss_dict['kl_divergence']:.4f}",
                'β': f"{self.current_kl_weight:.3f}"
            })
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_recon = total_recon_loss / len(self.train_loader)
        avg_kl = total_kl_loss / len(self.train_loader)
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl
        }
    
    def validate(self):
        """
        Validate model.
        
        Returns:
            Dictionary with validation losses
        """
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        with torch.no_grad():
            # ERROR FIXED: Changed from "(data,)" to match train loop format
            for batch_idx, data in enumerate(self.val_loader):
                # Handle different dataloader return formats
                if isinstance(data, (tuple, list)):
                    data = data[0]
                data = data.to(self.device)
                
                reconstructed, mu, log_var = self.model(data)
                loss, loss_dict = vae_loss_function(
                    reconstructed, data, mu, log_var,
                    kl_weight=self.current_kl_weight
                )
                
                total_loss += loss_dict['total_loss']
                total_recon_loss += loss_dict['recon_loss']
                total_kl_loss += loss_dict['kl_divergence']
        
        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        avg_recon = total_recon_loss / len(self.val_loader)
        avg_kl = total_kl_loss / len(self.val_loader)
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl
        }
    
    def train(self, checkpoint_dir=None):
      
        print("\n" + "="*80)
        print("STARTING VAE TRAINING")
        print("="*80)
        print(f"Epochs: {self.epochs}")
        print(f"KL Weight (β): {self.kl_weight_target}")
        print(f"KL Annealing: {self.kl_annealing}")
        if self.kl_annealing:
            print(f"Annealing epochs: {self.kl_annealing_epochs}")
            print(f"KL range: [{self.kl_min}, {self.kl_max}]")
        print(f"Device: {self.device}")
        print("="*80)
        

        # Create checkpoint directory
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            

            # Update learning rate
            self.scheduler.step(val_metrics['total_loss'])
            
            epoch_time = time.time() - epoch_start
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['val_recon_loss'].append(val_metrics['recon_loss'])
            self.history['val_kl_loss'].append(val_metrics['kl_loss'])
            self.history['kl_weight'].append(self.current_kl_weight)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.epochs} - {epoch_time:.1f}s:")
            print(f"  Train Loss: {train_metrics['total_loss']:.6f} "
                  f"(Recon: {train_metrics['recon_loss']:.6f}, "
                  f"KL: {train_metrics['kl_loss']:.6f})")
            print(f"  Val Loss:   {val_metrics['total_loss']:.6f} "
                  f"(Recon: {val_metrics['recon_loss']:.6f}, "
                  f"KL: {val_metrics['kl_loss']:.6f})")
            print(f"  β: {self.current_kl_weight:.4f}, LR: {current_lr:.6f}")
            
            # Check for KL collapse
            if val_metrics['kl_loss'] < 0.1:
                print(f"  ⚠️  WARNING: KL divergence very low ({val_metrics['kl_loss']:.4f}) - possible posterior collapse!")
            
            # Save checkpoints
            if checkpoint_dir:
                # Save best model
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self.epochs_without_improvement = 0
                    best_path = checkpoint_dir / "vae_best_model.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['total_loss'],
                        'kl_weight': self.current_kl_weight
                    }, best_path)
                    print(f"  ✅ Best model saved: {best_path}")
                else:
                    self.epochs_without_improvement += 1
                    print(f"  No improvement for {self.epochs_without_improvement} epochs")
                
                # Save periodic checkpoint
                if (epoch + 1) % 20 == 0:
                    checkpoint_path = checkpoint_dir / f"vae_checkpoint_epoch_{epoch+1}.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['total_loss'],
                        'kl_weight': self.current_kl_weight
                    }, checkpoint_path)
                    print(f"  💾 Checkpoint saved: {checkpoint_path}")

                # Early stopping
                if self.epochs_without_improvement >= config.PATIENCE:
                    print(f"\n⏹️  Early stopping after {epoch+1} epochs (patience: {config.PATIENCE})")
                    break
            total_time = time.time() - start_time


        # Save final model
        if checkpoint_dir:
            final_path = checkpoint_dir / "vae_final_model.pt"
            torch.save({
                'epoch': self.epochs - 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'kl_weight': self.current_kl_weight
            }, final_path)
            print(f"\n✅ Final model saved: {final_path}")
            
            # Save training history
            history_path = checkpoint_dir / "vae_training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"✅ Training history saved: {history_path}")
        
        print("\n" + "="*80)
        print("VAE TRAINING COMPLETE!")
        print("="*80)
        
        return self.history


def train_vae_phase1(data_path, checkpoint_dir, quick_test=False, 
                     kl_weight=1.0, kl_annealing=True):
    """
    Train VAE for Phase 1 (single day).
    
    Args:
        data_path: Path to processed data
        checkpoint_dir: Directory to save checkpoints
        quick_test: Use fewer epochs for quick testing
        kl_weight: KL divergence weight (beta)
        kl_annealing: Whether to use KL annealing
        
    Returns:
        Trained model and history
    """
    from dataset import load_processed_data, create_dataloaders
    from model_vae import create_vae_model
    
    # Load data
    train_data, val_data, test_data = load_processed_data(data_path)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data
    )
    
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
    
    # Train
    history = trainer.train(checkpoint_dir=checkpoint_dir)
    
    return model, history


if __name__ == "__main__":
    print("VAE Training Module")
    print("Import this module and use VAETrainer class or train_vae_phase1 function")