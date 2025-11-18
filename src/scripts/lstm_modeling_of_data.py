"""
LSTM Feature Extractor with Downstream ANN - PyTorch Implementation
This script demonstrates how to:
1. Build an LSTM to learn features from sequential data
2. Extract features from the LSTM
3. Use those features as inputs for ANNs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os
from utils import set_seed


set_seed(42)


class LSTMFeatureExtractorNet(nn.Module):
    """
    LSTM network for feature extraction from sequential data
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, 
                 feature_dim: int = 32, dropout: float = 0.2, output_dim: Optional[int] = None):

        super(LSTMFeatureExtractorNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, feature_dim),
            nn.ReLU()
        )
        
        # Optional output layer for end-to-end training
        self.output_layer = None
        if output_dim is not None:
            if output_dim == 1:
                self.output_layer = nn.Linear(feature_dim, output_dim)
            else:
                self.output_layer = nn.Sequential(
                    nn.Linear(feature_dim, output_dim),
                    nn.Softmax(dim=1)
                )
    
    def forward(self, x, return_features=False):
        # LSTM forward pass
        # lstm_out shape: (batch, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # Shape: (batch, hidden_size)
        
        # Extract features
        features = self.feature_layer(last_hidden)  # Shape: (batch, feature_dim)
        
        if return_features or self.output_layer is None:
            return features
        
        # Pass through output layer if it exists
        output = self.output_layer(features)
        return output
    
    def extract_features(self, x):
        """
        Extract features without gradient computation
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_features=True)


class DownstreamANNNet(nn.Module):
    """
    Feedforward neural network that uses LSTM features as input
    """
    def __init__(self, input_dim: int, hidden_layers: List[int], 
                 output_dim: int, dropout: float = 0.3, task: str = 'regression'):
        """
        Args:
            input_dim: Dimension of input features (from LSTM)
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension
            dropout: Dropout probability
            task: 'regression' or 'classification'
        """
        super(DownstreamANNNet, self).__init__()
        
        self.task = task
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Add activation for classification
        if task == 'classification':
            if output_dim == 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class LSTMFeatureExtractor:
    """
    Wrapper class for training and using LSTM feature extractor
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 feature_dim: int = 32, dropout: float = 0.2, device: str = None):
        """
        Args:
            input_size: Number of features per timestep
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            feature_dim: Dimension of extracted features
            dropout: Dropout probability
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.model = None
        
        print(f"Using device: {self.device}")
    
    def build_model(self, output_dim: Optional[int] = None):
        """
        Build the LSTM model
        Args:
            output_dim: Optional output dimension for supervised pre-training
        """
        self.model = LSTMFeatureExtractorNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            feature_dim=self.feature_dim,
            dropout=self.dropout,
            output_dim=output_dim
        ).to(self.device)
        
        return self.model
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader = None,
                   epochs: int = 50, lr: float = 0.001, task: str = 'regression',
                   patience: int = 10, verbose: bool = True):
        """
        Train the LSTM model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        if task == 'classification':
            if self.model.output_layer is None:
                raise ValueError("Model needs output_dim for classification")
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_metric': [],
            'val_loss': [], 'val_metric': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_metric = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Compute loss
                if task == 'classification' and batch_y.dim() == 1:
                    loss = criterion(outputs, batch_y.long())
                else:
                    loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Compute metric
                if task == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    if batch_y.dim() > 1:
                        _, batch_y = torch.max(batch_y, 1)
                    train_metric += (predicted == batch_y).sum().item() / len(batch_y)
                else:
                    train_metric += torch.mean(torch.abs(outputs - batch_y)).item()
            
            train_loss /= len(train_loader)
            train_metric /= len(train_loader)
            
            history['train_loss'].append(train_loss)
            history['train_metric'].append(train_metric)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_metric = self.evaluate(val_loader, criterion, task)
                history['val_loss'].append(val_loss)
                history['val_metric'].append(val_metric)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 5 == 0:
                    metric_name = 'Acc' if task == 'classification' else 'MAE'
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.4f} | "
                          f"Val Loss: {val_loss:.4f}, Val {metric_name}: {val_metric:.4f}")
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    metric_name = 'Acc' if task == 'classification' else 'MAE'
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def evaluate(self, data_loader: DataLoader, criterion, task: str = 'regression'):
        """
        Evaluate the model
        """
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                
                if task == 'classification' and batch_y.dim() == 1:
                    loss = criterion(outputs, batch_y.long())
                else:
                    loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                if task == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    if batch_y.dim() > 1:
                        _, batch_y = torch.max(batch_y, 1)
                    total_metric += (predicted == batch_y).sum().item() / len(batch_y)
                else:
                    total_metric += torch.mean(torch.abs(outputs - batch_y)).item()
        
        return total_loss / len(data_loader), total_metric / len(data_loader)
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from input sequences
        Args:
            X: Input array of shape (n_samples, seq_len, n_features)
        Returns:
            Features array of shape (n_samples, feature_dim)
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            features = self.model.extract_features(X_tensor)
        
        return features.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str, output_dim: Optional[int] = None):
        """Load model weights"""
        if self.model is None:
            self.build_model(output_dim)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


class DownstreamANN:
    """
    Wrapper class for training ANN on extracted features
    """
    def __init__(self, input_dim: int, hidden_layers: List[int] = [64, 32],
                 output_dim: int = 1, dropout: float = 0.3, 
                 task: str = 'regression', device: str = None):
        """
        Args:
            input_dim: Dimension of input features
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension
            dropout: Dropout probability
            task: 'regression' or 'classification'
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.task = task
        self.model = None
    
    def build_model(self):
        """Build the ANN model"""
        self.model = DownstreamANNNet(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            output_dim=self.output_dim,
            dropout=self.dropout,
            task=self.task
        ).to(self.device)
        
        return self.model
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader = None,
                   epochs: int = 100, lr: float = 0.001, patience: int = 15,
                   verbose: bool = True):
        """
        Train the ANN model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        if self.task == 'classification':
            if self.output_dim == 1:
                criterion = nn.BCELoss()
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, verbose=verbose
        )
        
        history = {
            'train_loss': [], 'train_metric': [],
            'val_loss': [], 'val_metric': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_metric = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                if self.task == 'classification' and batch_y.dim() == 1:
                    loss = criterion(outputs, batch_y.long())
                else:
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if self.task == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    if batch_y.dim() > 1:
                        _, batch_y = torch.max(batch_y, 1)
                    train_metric += (predicted == batch_y).sum().item() / len(batch_y)
                else:
                    train_metric += torch.mean(torch.abs(outputs - batch_y)).item()
            
            train_loss /= len(train_loader)
            train_metric /= len(train_loader)
            
            history['train_loss'].append(train_loss)
            history['train_metric'].append(train_metric)
            
            # Validation
            if val_loader is not None:
                val_loss, val_metric = self.evaluate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_metric'].append(val_metric)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    metric_name = 'Acc' if self.task == 'classification' else 'MAE'
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.4f} | "
                          f"Val Loss: {val_loss:.4f}, Val {metric_name}: {val_metric:.4f}")
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    metric_name = 'Acc' if self.task == 'classification' else 'MAE'
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.4f}")
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def evaluate(self, data_loader: DataLoader, criterion):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                
                if self.task == 'classification' and batch_y.dim() == 1:
                    loss = criterion(outputs, batch_y.long())
                else:
                    loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                if self.task == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    if batch_y.dim() > 1:
                        _, batch_y = torch.max(batch_y, 1)
                    total_metric += (predicted == batch_y).sum().item() / len(batch_y)
                else:
                    total_metric += torch.mean(torch.abs(outputs - batch_y)).item()
        
        return total_loss / len(data_loader), total_metric / len(data_loader)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


def create_synthetic_data(n_samples=1000, sequence_length=10, n_features=5):
    """Create synthetic sequential data for demonstration"""
    X = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
    # Synthetic target based on sequence statistics
    y = np.mean(X[:, :, 0], axis=1) + np.std(X[:, :, 1], axis=1)
    y = y.astype(np.float32).reshape(-1, 1)
    
    return X, y


def main():
    print("=" * 60)
    print("LSTM Feature Extractor with Downstream ANN - PyTorch")
    print("=" * 60)
    
    # Configuration
    SEQUENCE_LENGTH = 10
    N_FEATURES = 5
    N_SAMPLES = 1000
    BATCH_SIZE = 32
    
    # Generate synthetic data
    print("\n1. Generating synthetic sequential data...")
    X, y = create_synthetic_data(N_SAMPLES, SEQUENCE_LENGTH, N_FEATURES)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Step 1: Build and train LSTM feature extractor
    print("\n2. Building and training LSTM feature extractor...")
    lstm_extractor = LSTMFeatureExtractor(
        input_size=N_FEATURES,
        hidden_size=64,
        num_layers=2,
        feature_dim=32,
        dropout=0.2
    )
    
    lstm_extractor.build_model(output_dim=1)
    print(f"Model parameters: {sum(p.numel() for p in lstm_extractor.model.parameters()):,}")
    
    history_lstm = lstm_extractor.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        task='regression',
        patience=10,
        verbose=True
    )
    
    # Step 2: Extract features
    print("\n3. Extracting features from LSTM...")
    X_train_features = lstm_extractor.extract_features(X_train)
    X_val_features = lstm_extractor.extract_features(X_val)
    X_test_features = lstm_extractor.extract_features(X_test)
    
    print(f"Extracted feature shape: {X_train_features.shape}")
    
    # Create DataLoaders for features
    train_feat_dataset = TensorDataset(
        torch.FloatTensor(X_train_features), 
        torch.FloatTensor(y_train)
    )
    val_feat_dataset = TensorDataset(
        torch.FloatTensor(X_val_features), 
        torch.FloatTensor(y_val)
    )
    test_feat_dataset = TensorDataset(
        torch.FloatTensor(X_test_features), 
        torch.FloatTensor(y_test)
    )
    
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_feat_loader = DataLoader(val_feat_dataset, batch_size=BATCH_SIZE)
    test_feat_loader = DataLoader(test_feat_dataset, batch_size=BATCH_SIZE)
    





    
    # Step 3: Build and train downstream ANN
    print("\n4. Building and training downstream ANN...")
    ann = DownstreamANN(
        input_dim=X_train_features.shape[1],
        hidden_layers=[64, 32, 16],
        output_dim=1,
        dropout=0.3,
        task='regression'
    )
    
    ann.build_model()
    print(f"Model parameters: {sum(p.numel() for p in ann.model.parameters()):,}")
    
    history_ann = ann.train_model(
        train_loader=train_feat_loader,
        val_loader=val_feat_loader,
        epochs=100,
        lr=0.001,
        patience=15,
        verbose=True
    )
    
    # Step 4: Evaluate
    print("\n5. Evaluating models...")
    
    # LSTM end-to-end performance
    criterion = nn.MSELoss()
    lstm_loss, lstm_mae = lstm_extractor.evaluate(test_loader, criterion, task='regression')
    print(f"LSTM (end-to-end) - Loss: {lstm_loss:.4f}, MAE: {lstm_mae:.4f}")
    
    # ANN performance on extracted features
    ann_loss, ann_mae = ann.evaluate(test_feat_loader, criterion)
    print(f"ANN (on LSTM features) - Loss: {ann_loss:.4f}, MAE: {ann_mae:.4f}")
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # LSTM training
    axes[0, 0].plot(history_lstm['train_loss'], label='Train Loss')
    axes[0, 0].plot(history_lstm['val_loss'], label='Val Loss')
    axes[0, 0].set_title('LSTM Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history_lstm['train_metric'], label='Train MAE')
    axes[0, 1].plot(history_lstm['val_metric'], label='Val MAE')
    axes[0, 1].set_title('LSTM Training MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # ANN training
    axes[1, 0].plot(history_ann['train_loss'], label='Train Loss')
    axes[1, 0].plot(history_ann['val_loss'], label='Val Loss')
    axes[1, 0].set_title('ANN Training Loss (on LSTM features)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history_ann['train_metric'], label='Train MAE')
    axes[1, 1].plot(history_ann['val_metric'], label='Val MAE')
    axes[1, 1].set_title('ANN Training MAE (on LSTM features)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/pytorch_training_history.png', dpi=300, bbox_inches='tight')
    print("\n6. Training plots saved to 'pytorch_training_history.png'")
    
    # Save models
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)
    lstm_extractor.save_model('/mnt/user-data/outputs/pytorch_lstm_model.pt')
    ann.save_model('/mnt/user-data/outputs/pytorch_ann_model.pt')
    print("\n7. Models saved successfully!")
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()