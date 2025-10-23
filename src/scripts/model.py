"""
LSTM Autoencoder for time series compression
"""

import torch
import torch.nn as nn
from typing import Tuple

from config import config


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder: Compresses sequences into latent representation.
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dims: list = [64, 32],
                 latent_dim: int = 16,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super(LSTMEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Build LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            lstm = nn.LSTM(
                input_size=current_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0  # We'll apply dropout manually
            )
            self.lstm_layers.append(lstm)
            self.dropout_layers.append(nn.Dropout(dropout))
            
            # Update current dimension
            current_dim = hidden_dim * self.num_directions
        
        # Final fully connected layer to latent space
        self.fc_latent = nn.Linear(current_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)    
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        batch_size = x.size(0)
        
        # Pass through LSTM layers
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, (h_n, c_n) = lstm(x)
            x = dropout(x)
        
        # Take the last time step output
        # x shape: (batch_size, seq_length, hidden_dim * num_directions)
        x = x[:, -1, :]  # (batch_size, hidden_dim * num_directions)
        
        # Map to latent space
        latent = self.fc_latent(x)  # (batch_size, latent_dim)
        
        return latent


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder: Reconstructs sequences from latent representation.
    """
    
    def __init__(self,
                 latent_dim: int = 16,
                 hidden_dims: list = [64, 32],
                 output_dim: int = 3,
                 seq_length: int = 60,
                 dropout: float = 0.2):
        super(LSTMDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.dropout = dropout
        
        # Fully connected layer to expand latent
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
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            latent: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed sequence of shape (batch_size, seq_length, output_dim)
        """
        batch_size = latent.size(0)
        
        # Expand latent
        x = self.fc_expand(latent)  # (batch_size, hidden_dims[0])
        
        # Repeat for sequence length
        x = x.unsqueeze(1).repeat(1, self.seq_length, 1)  # (batch_size, seq_length, hidden_dims[0])
        
        # Pass through LSTM layers
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, (h_n, c_n) = lstm(x)
            x = dropout(x)
        
        # Map to output dimension
        output = self.fc_output(x)  # (batch_size, seq_length, output_dim)
        
        return output


class LSTMAutoencoder(nn.Module):
    """
    Complete LSTM Autoencoder combining encoder and decoder.
    """
    
    def __init__(self,
                 input_dim: int = None,
                 encoder_hidden_dims: list = None,
                 latent_dim: int = None,
                 decoder_hidden_dims: list = None,
                 output_dim: int = None,
                 seq_length: int = None,
                 dropout: float = None,
                 bidirectional: bool = None):
        super(LSTMAutoencoder, self).__init__()
        
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
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=output_dim,
            seq_length=seq_length,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)         
        Returns:
            Tuple of (reconstructed, latent):
                - reconstructed: (batch_size, seq_length, output_dim)
                - latent: (batch_size, latent_dim)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(latent)
    
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


def create_model(device: str = None) -> LSTMAutoencoder:
    """
    Create and initialize LSTM Autoencoder model.
    
    Args:
        device: Device to place model on (default: config.DEVICE)
        
    Returns:
        Initialized model on specified device
    """
    if device is None:
        device = config.DEVICE
    
    model = LSTMAutoencoder()
    model = model.to(device)
    
    # Print model summary
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(f"\nInput:  [{config.BATCH_SIZE}, {config.SEQUENCE_LENGTH}, {config.N_FEATURES}]")
    print(f"Latent: [{config.BATCH_SIZE}, {config.LATENT_DIM}]")
    print(f"Output: [{config.BATCH_SIZE}, {config.SEQUENCE_LENGTH}, {config.N_FEATURES}]")
    print(f"\nCompression Ratio: {model.get_compression_ratio():.2f}x")
    
    params = model.count_parameters()
    print(f"\nParameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Total:   {params['total']:,}")
    
    print(f"\nDevice: {device}")
    print("="*60)
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing LSTM Autoencoder...")
    
    model = create_model()
    
    # Create dummy input
    batch_size = 4
    seq_length = config.SEQUENCE_LENGTH
    n_features = config.N_FEATURES
    
    dummy_input = torch.randn(batch_size, seq_length, n_features).to(config.DEVICE)
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        reconstructed, latent = model(dummy_input)
    
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    print("\n Model test successful!")