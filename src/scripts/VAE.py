
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from config import config


class LSTMVAEEncoder(nn.Module):
    
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dims: list = [128, 64, 32],
                 latent_dim: int = 16,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        super(LSTMVAEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Build LSTM layers (same as standard autoencoder)
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            lstm = nn.LSTM(
                input_size=current_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0
            )
            self.lstm_layers.append(lstm)
            self.dropout_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim * self.num_directions
        
        # VAE-specific: Two output layers for mu and log_var
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_log_var = nn.Linear(current_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
        batch_size = x.size(0)
        
        # Pass through LSTM layers
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            x, (h_n, c_n) = lstm(x)
            x = dropout(x)
        
        # Take last timestep
        x = x[:, -1, :]  # (batch_size, hidden_dim * num_directions)
        
        # Output mu and log_var (NOT variance, but log(variance) for numerical stability)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var


class LSTMVAEDecoder(nn.Module):
    
    def __init__(self,
                 latent_dim: int = 16,
                 hidden_dims: list = [32, 64, 128],
                 output_dim: int = 2,
                 seq_length: int = 60,
                 dropout: float = 0.2):
        super(LSTMVAEDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.dropout = dropout
        
        # Expand latent to first hidden dimension
        self.fc_expand = nn.Linear(latent_dim, hidden_dims[0])
        
        # Build LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            if i == 0:
                input_dim = hidden_dims[0]
            else:
                input_dim = hidden_dims[i-1]
            
            lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dims[i],
                num_layers=1,
                batch_first=True,
                dropout=0.0
            )
            self.lstm_layers.append(lstm)
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Final output layer
        self.fc_output = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:

        batch_size = z.size(0)
        
        # Expand and repeat
        x = self.fc_expand(z)
        x = x.unsqueeze(1).repeat(1, self.seq_length, 1)
        
        # Pass through LSTM layers
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, (h_n, c_n) = lstm(x)
            x = dropout(x)
        
        # Output reconstruction
        output = self.fc_output(x)
        
        return output


class LSTMVariationalAutoencoder(nn.Module):

    def __init__(self,
                 input_dim: int = None,
                 encoder_hidden_dims: list = None,
                 latent_dim: int = None,
                 decoder_hidden_dims: list = None,
                 output_dim: int = None,
                 seq_length: int = None,
                 dropout: float = None,
                 bidirectional: bool = None):
        super(LSTMVariationalAutoencoder, self).__init__()
        
        # Use config defaults if not provided
        if input_dim is None:
            input_dim = config.N_FEATURES
        if encoder_hidden_dims is None:
            encoder_hidden_dims = config.ENCODER_HIDDEN_DIMS
        if latent_dim is None:
            latent_dim = config.LATENT_DIM
        if decoder_hidden_dims is None:
            decoder_hidden_dims = config.DECODER_HIDDEN_DIMS
        if output_dim is None:
            output_dim = config.N_FEATURES
        if seq_length is None:
            seq_length = config.SEQUENCE_LENGTH
        if dropout is None:
            dropout = config.DROPOUT
        if bidirectional is None:
            bidirectional = config.ENCODER_BIDIRECTIONAL
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        
        # Build encoder and decoder
        self.encoder = LSTMVAEEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.decoder = LSTMVAEDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=output_dim,
            seq_length=seq_length,
            dropout=dropout
        )
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
       
        # Compute standard deviation from log variance
        # sigma = exp(0.5 * log_var) = sqrt(exp(log_var)) = sqrt(var)
        std = torch.exp(0.5 * log_var)
        # Sample epsilon from standard normal
        eps = torch.randn_like(std)
        # Reparameterization: z = mu + sigma * epsilon
        z = mu + std * eps
        
        return z
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        mu, log_var = self.encoder(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, log_var
    
    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:

        mu, log_var = self.encoder(x)
        
        if sample:
            return self.reparameterize(mu, log_var)
        else:
            return mu  # Deterministic encoding (use mean)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)
    
    def get_compression_ratio(self) -> float:
        """Calculate theoretical compression ratio."""
        original_size = self.seq_length * self.input_dim
        compressed_size = self.latent_dim
        return original_size / compressed_size
    
    def count_parameters(self) -> dict:
        """Count trainable parameters."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total_params = encoder_params + decoder_params
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params
        }


def vae_loss_function(reconstructed: torch.Tensor, 
                      original: torch.Tensor, 
                      mu: torch.Tensor, 
                      log_var: torch.Tensor,
                      kl_weight: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    # 1. Reconstruction Loss (MSE)
    # Mean Squared Error between original and reconstructed
    mse_loss = F.mse_loss(reconstructed, original, reduction='mean')
    
    # Alternative: Mean Absolute Error (uncomment to use MAE instead)
    # mae_loss = F.l1_loss(reconstructed, original, reduction='mean')

    # Alternative: NLL loss (uncomment to use nll instead)
    #nll_loss = F.nll_loss(reconstructed, original, ignore_index =1,  reduction = "mean")

    
    # 2. KL Divergence (Negative Log Likelihood / NLL)
    # KL[N(μ, σ²) || N(0, 1)] = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
    # 
    # Derivation:
    # We want KL divergence between learned distribution N(μ, σ²) 
    # and prior distribution N(0, 1)
    #
    # For Gaussian distributions:
    # KL(q||p) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #
    # Since we have log_var = log(sigma^2), we can write:
    # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Normalize by batch size (mean over batch)
    kl_divergence = kl_divergence / mu.size(0)
    
    # 3. Total Loss
    total_loss = mse_loss + kl_weight * kl_divergence
    
    # Return loss components for logging
    loss_dict = {
        'total_loss': total_loss.item(),
        'mse_loss': mse_loss.item(),
        'kl_divergence': kl_divergence.item(),
        'weighted_kl': (kl_weight * kl_divergence).item()
    }
    
    return total_loss, loss_dict


def create_vae_model(device: str = None) -> LSTMVariationalAutoencoder:
    if device is None:
        device = config.DEVICE
    
    model = LSTMVariationalAutoencoder()
    model = model.to(device)
    
    # Print model summary
    print("\n" + "="*60)
    print("VAE MODEL ARCHITECTURE")
    print("="*60)
    print(f"\nInput:  [{config.BATCH_SIZE}, {config.SEQUENCE_LENGTH}, {config.N_FEATURES}]")
    print(f"Latent: [{config.BATCH_SIZE}, {config.LATENT_DIM}] (probabilistic)")
    print(f"Output: [{config.BATCH_SIZE}, {config.SEQUENCE_LENGTH}, {config.N_FEATURES}]")
    print(f"\nCompression Ratio: {model.get_compression_ratio():.2f}x")
    
    params = model.count_parameters()
    print(f"\nParameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Total:   {params['total']:,}")
    
    print(f"\nVAE Features:")
    print(f"  - Probabilistic latent space (mu, log_var)")
    print(f"  - Reparameterization trick for backprop")
    print(f"  - Loss: MSE + KL Divergence")
    
    print(f"\nDevice: {device}")
    print("="*60)
    
    return model


if __name__ == "__main__":

    
    # Test VAE model creation
    print("Testing LSTM Variational Autoencoder...")
    
    model = create_vae_model()
    