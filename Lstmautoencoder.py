import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Custom Dataset for sequence data
class SequenceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.float32)

# LSTM Autoencoder Model
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        # Encoder
        _, (h_n, _) = self.encoder(x)  # h_n shape: [num_layers, batch, hidden_dim]
        h_n = h_n[-1]                  # Take last layer hidden state: [batch, hidden_dim]
        latent = self.fc(h_n)          # Latent representation: [batch, latent_dim]

        # Decoder
        dec_input = self.decoder_input(latent).unsqueeze(1).repeat(1, x.size(1), 1) # Repeat for each timestep
        out, _ = self.decoder(dec_input)
        return out

# Parameters
input_dim = 1           # Single sensor channel (vertical vibration)
hidden_dim = 64
latent_dim = 16
seq_len = 100           # Example sequence length
batch_size = 256
num_epochs = 20
learning_rate = 0.001

# Example data loading (replace with actual preprocessed data)
# data = np.load('preprocessed_vibration_data.npy') # shape (N, )
# Normalize  (data - mean) / std
# Prepare dataset and dataloader
# dataset = SequenceDataset(data, seq_len)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# For demonstration, creating random data
data = np.sin(np.linspace(0, 100*np.pi, 1000000))  # dummy sine wave data
data = (data - np.mean(data)) / np.std(data)
dataset = SequenceDataset(data, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss & optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in dataloader:
        batch = batch.unsqueeze(-1).to(device)  # shape: [batch, seq_len, input_dim]
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# Save trained model
torch.save(model.state_dict(), 'lstm_autoencoder_model.pth')

