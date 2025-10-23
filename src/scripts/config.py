"""
Configuration file for Bridge Accelerometer Data Compression
"""

import torch
from pathlib import Path

class Config:
    # ============================================
    # Project Paths
    # ============================================
    PROJECT_ROOT = Path("/data/pool/c8x-98x/pml/src")
    
    # Data paths (we'll create these as needed)
    DATA_DIR = Path("/data/pool/c8x-98x/bridge_data/100_days")  # Original data location
    RAW_DATA_DIR = DATA_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "data" / "processed"
    COMPRESSED_DATA_DIR = DATA_DIR / "data" / "compressed"
    RECONSTRUCTED_DATA_DIR = DATA_DIR / "data" / "reconstructed"
    
    # Model paths
    MODEL_DIR = PROJECT_ROOT / "models"
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    CONFIG_DIR = MODEL_DIR / "configs"
    
    # Results paths
    RESULTS_DIR = PROJECT_ROOT / "results"
    FIGURES_DIR = RESULTS_DIR / "figures"
    METRICS_DIR = RESULTS_DIR / "metrics"
    
    # Logs
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # ============================================
    # Data Specifications
    # ============================================
    # CSV column names
    COLUMNS = ['day', 'hour_file', 'start_time', 'end_time', 'mean', 'variance', 'log_variance']
    FEATURE_COLUMNS = ['mean', 'variance', 'log_variance']
    
    # Time specifications
    SECONDS_PER_DAY = 86400
    SEQUENCE_LENGTH = 60  # 60 seconds per sequence
    N_FEATURES = 3  # mean, variance, log_variance
    
    # Data split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ============================================
    # Model Architecture
    # ============================================
    # LSTM Encoder
    ENCODER_HIDDEN_DIMS = [64, 32]  # LSTM layers (will be doubled due to bidirectional)
    ENCODER_BIDIRECTIONAL = True
    
    # Latent space
    LATENT_DIM = 32  # Compression bottleneck
    
    # LSTM Decoder
    DECODER_HIDDEN_DIMS = [64, 32]
    
    # Regularization
    DROPOUT = 0.2
    
    # ============================================
    # Training Hyperparameters
    # ============================================
    # Basic training
    BATCH_SIZE = 64  # Can increase to 512 on A100
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # Optimizer
    OPTIMIZER = "AdamW"
    BETAS = (0.9, 0.999)
    
    # Scheduler (OneCycleLR)
    USE_SCHEDULER = True
    MAX_LR = 1e-3
    PCT_START = 0.3  # 30% warmup
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1e4
    
    # Gradient clipping
    GRADIENT_CLIP_VALUE = 1.0
    
    # Mixed precision training
    USE_MIXED_PRECISION = True
    
    # ============================================
    # Training Settings
    # ============================================
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Checkpointing
    SAVE_EVERY_N_EPOCHS = 5
    PATIENCE = 10  # Early stopping patience
    
    # Logging
    LOG_EVERY_N_STEPS = 50
    VALIDATE_EVERY_N_EPOCHS = 1
    
    # ============================================
    # Evaluation Metrics
    # ============================================
    METRICS = ['mse', 'rmse', 'mae', 'r2', 'explained_variance']
    
    # Success thresholds
    MIN_R2_ACCEPTABLE = 0.85
    MIN_R2_GOOD = 0.92
    MIN_R2_EXCELLENT = 0.95
    
    # ============================================
    # Phase 1: Single Day Settings
    # ============================================
    PHASE1_TEST_FILE = "20241127.csv"  # Single day for initial testing
    PHASE1_QUICK_EPOCHS = 30  # Faster iteration for testing
    
    # ============================================
    # Computed Properties
    # ============================================
    @property
    def compression_ratio(self):
        """Calculate theoretical compression ratio"""
        original_size = self.SEQUENCE_LENGTH * self.N_FEATURES
        compressed_size = self.LATENT_DIM
        return original_size / compressed_size
    
    @property
    def sequences_per_day(self):
        """Calculate number of sequences per day (non-overlapping)"""
        return self.SECONDS_PER_DAY // self.SEQUENCE_LENGTH
    
    def __repr__(self):
        return f"""
        Bridge Accelerometer Compression Config
        ========================================
        Sequence Length: {self.SEQUENCE_LENGTH}s
        Features: {self.N_FEATURES}
        Latent Dimension: {self.LATENT_DIM}
        Compression Ratio: {self.compression_ratio:.2f}x
        
        Model: LSTM Autoencoder
        - Encoder: {self.ENCODER_HIDDEN_DIMS} (bidirectional={self.ENCODER_BIDIRECTIONAL})
        - Decoder: {self.DECODER_HIDDEN_DIMS}
        - Dropout: {self.DROPOUT}
        
        Training:
        - Batch Size: {self.BATCH_SIZE}
        - Epochs: {self.EPOCHS}
        - Learning Rate: {self.LEARNING_RATE}
        - Device: {self.DEVICE}
        
        Data:
        - Sequences per day: {self.sequences_per_day}
        - Train/Val/Test: {self.TRAIN_RATIO}/{self.VAL_RATIO}/{self.TEST_RATIO}
        """

# Create a global config instance
config = Config()

if __name__ == "__main__":
    print(config)
    print(f"\nCompression Ratio: {config.compression_ratio:.2f}x")
    print(f"Sequences per day: {config.sequences_per_day}")