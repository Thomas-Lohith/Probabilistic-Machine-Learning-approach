"""
VAE Model Architecture - CORRECTED VERSION
Based on working model.py, modified for Variational Autoencoder

FIX: Decoder LSTM layer input size calculation
Error was: RuntimeError: input.size(-1) must be equal to input_size. Expected 64, got 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class LSTMEncoder(nn.Module):
    
    def __init__(self, 
                 input_dim=config.N_FEATURES,
                 hidden_dims=None,
                 latent_dim=config.LATENT_DIM,
                 dropout=config.DROPOUT,
                 bidirectional=config.ENCODER_BIDIRECTIONAL):
        super(LSTMEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = config.ENCODER_HIDDEN_DIMS
            
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Build stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_dims):
            # First layer takes input_dim (N_FEATURES)
            # Subsequent layers take previous layer's output (accounting for bidirectional)
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = hidden_dims[i-1] * self.num_directions
            
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0,  # Dropout handled separately
                    bidirectional=bidirectional
                )
            )
        
        # Dropout between layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Output layers - VAE outputs mu and log_var
        final_hidden_dim = hidden_dims[-1] * self.num_directions
        self.fc_mu = nn.Linear(final_hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(final_hidden_dim, latent_dim)
        
    def forward(self, x):
       
        # Pass through stacked LSTM layers
        lstm_out = x
        for i, lstm_layer in enumerate(self.lstm_layers):
            lstm_out, (hidden, cell) = lstm_layer(lstm_out)
            
            # Apply dropout between layers (not after last layer)
            if self.dropout is not None and i < len(self.lstm_layers) - 1:
                lstm_out = self.dropout(lstm_out)
        
        # Use last hidden state from final layer
        # hidden shape: (num_directions, batch_size, hidden_dim)
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        else:
            hidden = hidden[0]
        
        # Project to mu and log_var
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        
        return mu, log_var


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder for VAE - FIXED VERSION
    Takes latent vector and reconstructs sequence.
    Supports multi-layer LSTM with different hidden dimensions.
    
    FIX: Corrected layer_input_dim calculation to prevent size mismatch errors.
    """
    
    def __init__(self,
                 latent_dim=config.LATENT_DIM,
                 hidden_dims=None,
                 output_dim=config.N_FEATURES,
                 dropout=config.DROPOUT):
        super(LSTMDecoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = config.DECODER_HIDDEN_DIMS
            
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.output_dim = output_dim
        
        # Project latent to first hidden dimension
        self.fc_latent = nn.Linear(latent_dim, hidden_dims[0])
        
        # Build stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_dims):
            # CRITICAL FIX: Proper input size calculation
            # - First layer (i=0): takes fc_latent output (hidden_dims[0])
            # - Subsequent layers: take previous LSTM output (hidden_dims[i-1])
            if i == 0:
                layer_input_dim = hidden_dims[0]
            else:
                layer_input_dim = hidden_dims[i-1]
            
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0  # Dropout handled separately
                )
            )
        
        # Dropout between layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, z, seq_length):

        batch_size = z.size(0)
        
        # Project latent to hidden dimension
        h = self.fc_latent(z)  # Shape: (batch_size, hidden_dims[0])
        h = F.relu(h)
        
        # Repeat for sequence length
        h = h.unsqueeze(1).repeat(1, seq_length, 1)  # Shape: (batch_size, seq_length, hidden_dims[0])
        
        # Pass through stacked LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            h, _ = lstm(h)
            
            # Apply dropout between layers (not after last layer)
            if self.dropout is not None and i < len(self.lstm_layers) - 1:
                h = self.dropout(h)
        
        # Project to output dimension
        output = self.fc_out(h)
        
        return output


class LSTMVAE(nn.Module):
    def __init__(self,
                 input_dim=config.N_FEATURES,
                 encoder_hidden_dims=None,
                 decoder_hidden_dims=None,
                 latent_dim=config.LATENT_DIM,
                 dropout=config.DROPOUT,
                 bidirectional=config.ENCODER_BIDIRECTIONAL):
        super(LSTMVAE, self).__init__()
        
        if encoder_hidden_dims is None:
            encoder_hidden_dims = config.ENCODER_HIDDEN_DIMS
        if decoder_hidden_dims is None:
            decoder_hidden_dims = config.DECODER_HIDDEN_DIMS
        
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
            output_dim=input_dim,
            dropout=dropout
        )
        
    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
    
        seq_length = x.size(1)
        
        # Encode
        mu, log_var = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstructed = self.decoder(z, seq_length)
        
        return reconstructed, mu, log_var
    
    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z, seq_length):
        return self.decoder(z, seq_length)
    
    def sample(self, num_samples, seq_length, device='cpu'):
        # Sample from standard normal
        z = torch.randn(num_samples, config.LATENT_DIM).to(device)
        
        # Decode
        with torch.no_grad():
            samples = self.decoder(z, seq_length)
        
        return samples


def create_vae_model():
    model = LSTMVAE(
        input_dim=config.N_FEATURES,
        encoder_hidden_dims=config.ENCODER_HIDDEN_DIMS,
        decoder_hidden_dims=config.DECODER_HIDDEN_DIMS,
        latent_dim=config.LATENT_DIM,
        dropout=config.DROPOUT,
        bidirectional=config.ENCODER_BIDIRECTIONAL
    )
    
    model = model.to(config.DEVICE)
    print(model)
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # print(f"\n📊 VAE Model Summary:")
    # print(f"   Architecture: LSTM Variational Autoencoder")
    # print(f"   Input dim: {config.N_FEATURES}")
    # print(f"   Encoder hidden dims: {config.ENCODER_HIDDEN_DIMS} (bidirectional={config.ENCODER_BIDIRECTIONAL})")
    # print(f"   Latent dim: {config.LATENT_DIM}")
    # print(f"   Decoder hidden dims: {config.DECODER_HIDDEN_DIMS}")
    # print(f"   Dropout: {config.DROPOUT}")
    # print(f"   Total parameters: {total_params:,}")
    # print(f"   Trainable parameters: {trainable_params:,}")
    # print(f"   Device: {config.DEVICE}")
    # print(f"   Compression ratio: {config.compression_ratio:.2f}x")
    
    # # Print detailed layer info
    # print(f"\n🔍 Decoder Layer Details:")
    # for i, hidden_dim in enumerate(config.DECODER_HIDDEN_DIMS):
    #     if i == 0:
    #         input_size = config.DECODER_HIDDEN_DIMS[0]
    #     else:
    #         input_size = config.DECODER_HIDDEN_DIMS[i-1]
    #     print(f"   Layer {i}: LSTM(input_size={input_size}, hidden_size={hidden_dim})")
    
    return model


if __name__ == "__main__":
    # Test VAE model
    print("="*80)
    print("TESTING VAE MODEL")
    print("="*80)
    
    # Create model
    model = create_vae_model()
    
    