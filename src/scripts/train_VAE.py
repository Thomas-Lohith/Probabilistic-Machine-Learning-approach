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

# ===== ELBO LOSS WITH TRAINABLE GLOBAL SIGMA² =====
class ELBOLossTrainableSigma(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))  # trainable global log sigma^2

    def vae_loss_function(self, reconstructed, original, mu, log_var, kl_weight=1.0):
        
        sigma2 = torch.exp(self.log_sigma2)  # scalar > 0
        
        # Gaussian NLL reconstruction: 0.5 * (MSE/σ² + log(σ²))
        mse = F.mse_loss(reconstructed, original, reduction='mean')
        recon_loss = 0.5 * (mse / sigma2 + torch.log(sigma2))
        
        # Standard VAE KL divergence (matches your original)
        kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kl_divergence = kl_divergence / mu.size(0)  # batch normalize
        
        total_loss = recon_loss + kl_weight * kl_divergence
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'weighted_kl': (kl_weight * kl_divergence).item(),
            'log_sigma2': self.log_sigma2.item(),
            'sigma2': torch.exp(self.log_sigma2).item()
        }
        return total_loss, loss_dict

# =====  VAE TRAINER WITH TRAINABLE SIGMA LOSS =====
class VAETrainer:
    """
    VAE Trainer with trainable sigma² ELBO loss and KL annealing.
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
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else config.DEVICE
        self.epochs = epochs if epochs else config.EPOCHS
        
        # === TRAINABLE LOSS MODULE ===
        self.criterion = ELBOLossTrainableSigma().to(self.device)
        
        # === OPTIMIZER INCLUDES MODEL + CRITERION PARAMETERS ===
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=learning_rate,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # KL annealing
        self.kl_weight_target = kl_weight
        self.kl_annealing = kl_annealing
        self.kl_annealing_epochs = kl_annealing_epochs
        self.kl_min = kl_min
        self.kl_max = kl_max if kl_weight <= kl_max else kl_weight
        self.current_kl_weight = kl_min if kl_annealing else kl_weight
        
        # Scheduler & AMP
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=100
        )
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
        # History tracking
        self.history = {
            'train_loss': [], 'train_recon_loss': [], 'train_kl_loss': [],
            'val_loss': [], 'val_recon_loss': [], 'val_kl_loss': [],
            'kl_weight': [], 'learning_rate': [], 'log_sigma2': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def update_kl_weight(self, epoch):
        if not self.kl_annealing:
            self.current_kl_weight = self.kl_weight_target
            return
        
        if epoch < self.kl_annealing_epochs:
            self.current_kl_weight = self.kl_min + \
                (self.kl_max - self.kl_min) * (epoch / self.kl_annealing_epochs)
        else:
            self.current_kl_weight = self.kl_weight_target
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = total_recon_loss = total_kl_loss = 0.0
        
        self.update_kl_weight(epoch)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, data in enumerate(pbar):
            if isinstance(data, (tuple, list)):
                data = data[0]
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    reconstructed, mu, log_var = self.model(data)
                    loss, loss_dict = self.criterion.vae_loss_function(
                        reconstructed, data, mu, log_var, 
                        kl_weight=self.current_kl_weight
                    )
                self.scaler.scale(loss).backward()
                if config.GRADIENT_CLIP_VALUE:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters() + list(self.criterion.parameters()),
                        config.GRADIENT_CLIP_VALUE
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstructed, mu, log_var = self.model(data)
                loss, loss_dict = self.criterion.vae_loss_function(
                    reconstructed, data, mu, log_var, 
                    kl_weight=self.current_kl_weight
                )
                loss.backward()
                if config.GRADIENT_CLIP_VALUE:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.criterion.parameters()),
                        config.GRADIENT_CLIP_VALUE
                    )
                self.optimizer.step()
            
            total_loss += loss_dict['total_loss']
            total_recon_loss += loss_dict['recon_loss']
            total_kl_loss += loss_dict['kl_divergence']
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'kl': f"{loss_dict['kl_divergence']:.4f}",
                'σ²': f"{loss_dict['sigma2']:.3f}",
                'β': f"{self.current_kl_weight:.3f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_recon = total_recon_loss / len(self.train_loader)
        avg_kl = total_kl_loss / len(self.train_loader)
        
        return {'total_loss': avg_loss, 'recon_loss': avg_recon, 'kl_loss': avg_kl}
    
    def validate(self):
        self.model.eval()
        total_loss = total_recon_loss = total_kl_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                if isinstance(data, (tuple, list)):
                    data = data[0]
                data = data.to(self.device)
                
                reconstructed, mu, log_var = self.model(data)
                loss, loss_dict = self.criterion.vae_loss_function(
                    reconstructed, data, mu, log_var, 
                    kl_weight=self.current_kl_weight
                )
                
                total_loss += loss_dict['total_loss']
                total_recon_loss += loss_dict['recon_loss']
                total_kl_loss += loss_dict['kl_divergence']
        
        avg_loss = total_loss / len(self.val_loader)
        avg_recon = total_recon_loss / len(self.val_loader)
        avg_kl = total_kl_loss / len(self.val_loader)
        
        return {'total_loss': avg_loss, 'recon_loss': avg_recon, 'kl_loss': avg_kl}
    
    def train(self, checkpoint_dir=None):
        print("\n" + "="*80)
        print("STARTING VAE TRAINING WITH TRAINABLE σ² ELBO LOSS")
        print("="*80)
        print(f"Initial σ²: {torch.exp(self.criterion.log_sigma2).item():.4f}")
        print(f"Epochs: {self.epochs}")
        print(f"KL Weight (β): {self.kl_weight_target}")
        print(f"KL Annealing: {self.kl_annealing}")
        if self.kl_annealing:
            print(f"  Annealing: [{self.kl_min} → {self.kl_max}] over {self.kl_annealing_epochs} epochs")
        print(f"Device: {self.device}")
        print("="*80)
        
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
            
            # Update LR scheduler
            self.scheduler.step(val_metrics['total_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['val_recon_loss'].append(val_metrics['recon_loss'])
            self.history['val_kl_loss'].append(val_metrics['kl_loss'])
            self.history['kl_weight'].append(self.current_kl_weight)
            self.history['learning_rate'].append(current_lr)
            self.history['log_sigma2'].append(self.criterion.log_sigma2.item())
            
            epoch_time = time.time() - epoch_start
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{self.epochs} ({epoch_time:.1f}s):")
            print(f"  Train: {train_metrics['total_loss']:.6f} (R:{train_metrics['recon_loss']:.6f}, K:{train_metrics['kl_loss']:.6f})")
            print(f"  Val:   {val_metrics['total_loss']:.6f} (R:{val_metrics['recon_loss']:.6f}, K:{val_metrics['kl_loss']:.6f})")
            print(f"  β: {self.current_kl_weight:.4f}, LR: {current_lr:.2e}, σ²: {torch.exp(self.criterion.log_sigma2).item():.4f}")
            
            # if val_metrics['kl_loss'] < 0.1:
            #     print(f"  ⚠️  KL very low ({val_metrics['kl_loss']:.4f}) - posterior collapse risk!")
            
            # Checkpoints
            if checkpoint_dir:
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self.epochs_without_improvement = 0
                    best_path = checkpoint_dir / "vae_best_model.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'criterion_state_dict': self.criterion.state_dict(),  # SAVE LEARNED SIGMA
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['total_loss'],
                        'kl_weight': self.current_kl_weight,
                        'log_sigma2': self.criterion.log_sigma2.item()
                    }, best_path)
                    print(f" $$$$ NEW BEST: {best_path}")
                else:
                    self.epochs_without_improvement += 1
                    print(f"  No improvement for {self.epochs_without_improvement} epochs")
                
                if (epoch + 1) % 50 == 0:
                    chk_path = checkpoint_dir / f"vae_epoch_{epoch+1}.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'criterion_state_dict': self.criterion.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['total_loss'],
                        'kl_weight': self.current_kl_weight
                    }, chk_path)
                    print(f"   Checkpoint: {chk_path}")
                
                if self.epochs_without_improvement >= config.PATIENCE:
                    print(f"\n  Early stopping after {epoch+1} epochs(patience: {config.PATIENCE})")
                    break
        
        total_time = time.time() - start_time
        
        # Final save
        if checkpoint_dir:
            final_path = checkpoint_dir / "vae_final_model.pt"
            torch.save({
                'epoch': self.epochs - 1,
                'model_state_dict': self.model.state_dict(),
                'criterion_state_dict': self.criterion.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'kl_weight': self.current_kl_weight,
                'final_sigma2': torch.exp(self.criterion.log_sigma2).item()
            }, final_path)
            
            history_path = checkpoint_dir / "vae_training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"\n Saved: {final_path}, {history_path}")
        
        print(f"\n TRAINING COMPLETE! Total time: {total_time/3600:.1f}h")
        print(f"Final σ²: {torch.exp(self.criterion.log_sigma2).item():.4f}")
        print("="*80)
        
        return self.history

# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    print("VAE Training Module with Trainable σ² ELBO")
    print("Usage: trainer = VAETrainer(model, train_loader, val_loader)")
    print("       history = trainer.train(checkpoint_dir='checkpoints/')")
