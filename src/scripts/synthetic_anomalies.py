"""
Synthetic/Proxy Anomaly Generator for Bridge Accelerometer Data
================================================================
Injects controlled perturbations into time series data to simulate:
1. Noise bursts (sudden high-frequency spikes)
2. Frequency band drops (simulate sensor/structural issues)
3. Modal frequency shifts (structural changes)
4. Sensor degradation (drift, bias, noise increase)

Usage:
    python synthetic_anomalies.py --input data.csv --output anomalies/ --all
    python synthetic_anomalies.py --input processed.npz --noise-only --severity 0.3
    python synthetic_anomalies.py --mode latent --checkpoint model.pt --input data.npz

Author: Bridge Monitoring Team
Date: 2025
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, fft
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import your modules
try:
    from config import config
    from model_vae import create_vae_model
    from load_data import load_single_csv, extract_features
    from inference import load_checkpoint, load_scaler, create_sequences, standardize_data
except ImportError as e:
    print(f"⚠️  WARNING: Could not import some modules: {e}")
    print("Some functionality may be limited.")


# ============================================================================
# ANOMALY INJECTION CLASSES
# ============================================================================

class AnomalyInjector:
    """Base class for anomaly injection."""
    
    def __init__(self, severity: float = 0.3, random_seed: int = 42):
        """
        Args:
            severity: Anomaly strength [0.0, 1.0]
            random_seed: For reproducibility
        """
        self.severity = severity
        self.rng = np.random.RandomState(random_seed)
    
    def inject(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Inject anomaly into data.
        
        Args:
            data: Shape (n_sequences, seq_length, n_features) or (seq_length, n_features)
            
        Returns:
            anomalous_data: Same shape as input
            metadata: Dictionary with anomaly details
        """
        raise NotImplementedError


class NoiseBurstInjector(AnomalyInjector):
    """
    Inject sudden noise bursts (high-frequency spikes).
    Simulates: impact events, electrical interference, sensor glitches.
    """
    
    def inject(self, data: np.ndarray, 
               n_bursts: int = None,
               burst_duration: int = None,
               noise_std_multiplier: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Args:
            n_bursts: Number of bursts per sequence (default: 1-3)
            burst_duration: Duration in timesteps (default: 3-10)
            noise_std_multiplier: How much stronger than normal (default: 5-15x)
        """
        anomalous = data.copy()
        
        # Determine shape
        if data.ndim == 2:
            anomalous = anomalous[np.newaxis, :, :]
            single_sequence = True
        else:
            single_sequence = False
        
        n_sequences, seq_length, n_features = anomalous.shape
        
        # Defaults
        if n_bursts is None:
            n_bursts = self.rng.randint(1, 4)
        if burst_duration is None:
            burst_duration = self.rng.randint(3, 11)
        if noise_std_multiplier is None:
            noise_std_multiplier = 5 + self.severity * 10
        
        metadata = {
            'type': 'noise_burst',
            'n_bursts': n_bursts,
            'burst_duration': burst_duration,
            'noise_multiplier': noise_std_multiplier,
            'severity': self.severity,
            'burst_locations': []
        }
        
        for seq_idx in range(n_sequences):
            for _ in range(n_bursts):
                # Random location
                start_idx = self.rng.randint(0, seq_length - burst_duration)
                
                # Inject burst
                for feat_idx in range(n_features):
                    noise_std = np.std(anomalous[seq_idx, :, feat_idx])
                    burst_noise = self.rng.normal(0, noise_std * noise_std_multiplier, burst_duration)
                    anomalous[seq_idx, start_idx:start_idx+burst_duration, feat_idx] += burst_noise
                
                metadata['burst_locations'].append({
                    'sequence': seq_idx,
                    'start': start_idx,
                    'end': start_idx + burst_duration
                })
        
        if single_sequence:
            anomalous = anomalous[0]
        
        return anomalous, metadata


class FrequencyBandDropInjector(AnomalyInjector):
    """
    Remove specific frequency bands using bandstop filtering.
    Simulates: structural changes, resonance damping, sensor issues.
    
    NOTE: This works best on raw time-series data. For mean/variance features,
    the effect is applied to the underlying signal reconstruction.
    """
    
    def inject(self, data: np.ndarray,
               drop_band: Tuple[float, float] = None,
               sampling_rate: float = 10.0,
               filter_order: int = 4) -> Tuple[np.ndarray, Dict]:
        """
        Args:
            drop_band: (low_freq, high_freq) in Hz to remove (default: (2, 4) Hz)
            sampling_rate: Data sampling rate in Hz
            filter_order: Butterworth filter order
        """
        anomalous = data.copy()
        
        if data.ndim == 2:
            anomalous = anomalous[np.newaxis, :, :]
            single_sequence = True
        else:
            single_sequence = False
        
        n_sequences, seq_length, n_features = anomalous.shape
        
        # Default band to drop (modal frequency range for bridges)
        if drop_band is None:
            # Typical bridge frequencies: 0.5-5 Hz
            # Drop a band in this range
            center_freq = 1 + self.severity * 3  # 1-4 Hz
            bandwidth = 0.5 + self.severity * 1.5  # 0.5-2 Hz
            drop_band = (max(0.1, center_freq - bandwidth/2), 
                        min(sampling_rate/2 - 0.1, center_freq + bandwidth/2))
        
        # Design bandstop filter
        nyquist = sampling_rate / 2
        low_norm = drop_band[0] / nyquist
        high_norm = drop_band[1] / nyquist
        
        # Ensure valid frequency range
        low_norm = max(0.01, min(0.99, low_norm))
        high_norm = max(low_norm + 0.01, min(0.99, high_norm))
        
        sos = signal.butter(filter_order, [low_norm, high_norm], 
                           btype='bandstop', output='sos')
        
        metadata = {
            'type': 'frequency_band_drop',
            'drop_band_hz': drop_band,
            'sampling_rate': sampling_rate,
            'filter_order': filter_order,
            'severity': self.severity
        }
        
        # Apply filter to each sequence and feature
        for seq_idx in range(n_sequences):
            for feat_idx in range(n_features):
                anomalous[seq_idx, :, feat_idx] = signal.sosfilt(
                    sos, anomalous[seq_idx, :, feat_idx]
                )
        
        if single_sequence:
            anomalous = anomalous[0]
        
        return anomalous, metadata


class ModalFrequencyShiftInjector(AnomalyInjector):
    """
    Shift dominant frequency components up or down.
    Simulates: structural stiffness changes, temperature effects, damage.
    """
    
    def inject(self, data: np.ndarray,
               shift_percent: float = None,
               sampling_rate: float = 10.0) -> Tuple[np.ndarray, Dict]:
        """
        Args:
            shift_percent: Frequency shift as percentage (default: ±1-3%)
            sampling_rate: Data sampling rate in Hz
        """
        anomalous = data.copy()
        
        if data.ndim == 2:
            anomalous = anomalous[np.newaxis, :, :]
            single_sequence = True
        else:
            single_sequence = False
        
        n_sequences, seq_length, n_features = anomalous.shape
        
        # Default shift
        if shift_percent is None:
            # Random shift direction
            direction = self.rng.choice([-1, 1])
            shift_percent = direction * (1 + self.severity * 2)  # ±1-3%
        
        metadata = {
            'type': 'modal_frequency_shift',
            'shift_percent': shift_percent,
            'sampling_rate': sampling_rate,
            'severity': self.severity
        }
        
        # Apply frequency shift using phase modification in frequency domain
        for seq_idx in range(n_sequences):
            for feat_idx in range(n_features):
                signal_data = anomalous[seq_idx, :, feat_idx]
                
                # FFT
                spectrum = fft.rfft(signal_data)
                freqs = fft.rfftfreq(len(signal_data), 1/sampling_rate)
                
                # Shift frequencies
                shift_factor = 1 + (shift_percent / 100)
                
                # Interpolate spectrum to shifted frequencies
                # This is a simplified approach; more sophisticated methods exist
                magnitudes = np.abs(spectrum)
                phases = np.angle(spectrum)
                
                # Create shifted spectrum (approximate)
                # Scale frequency components
                shifted_magnitudes = np.interp(
                    freqs, 
                    freqs / shift_factor,
                    magnitudes,
                    left=0, right=0
                )
                
                shifted_spectrum = shifted_magnitudes * np.exp(1j * phases)
                
                # IFFT
                anomalous[seq_idx, :, feat_idx] = fft.irfft(shifted_spectrum, n=len(signal_data))
        
        if single_sequence:
            anomalous = anomalous[0]
        
        return anomalous, metadata


class SensorDegradationInjector(AnomalyInjector):
    """
    Simulate gradual sensor degradation: drift, bias, increased noise.
    Simulates: aging sensors, calibration drift, environmental effects.
    """
    
    def inject(self, data: np.ndarray,
               drift_rate: float = None,
               bias: float = None,
               noise_increase: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Args:
            drift_rate: Linear drift per timestep (default: based on severity)
            bias: Constant offset (default: based on severity)
            noise_increase: Noise std multiplier (default: 1.5-3x)
        """
        anomalous = data.copy()
        
        if data.ndim == 2:
            anomalous = anomalous[np.newaxis, :, :]
            single_sequence = True
        else:
            single_sequence = False
        
        n_sequences, seq_length, n_features = anomalous.shape
        
        # Defaults based on severity
        if drift_rate is None:
            drift_rate = self.severity * 0.001  # Small linear drift
        if bias is None:
            bias = self.severity * np.mean(np.abs(anomalous))
        if noise_increase is None:
            noise_increase = 1 + self.severity * 2  # 1.5-3x
        
        metadata = {
            'type': 'sensor_degradation',
            'drift_rate': drift_rate,
            'bias': bias,
            'noise_increase': noise_increase,
            'severity': self.severity
        }
        
        for seq_idx in range(n_sequences):
            for feat_idx in range(n_features):
                signal_data = anomalous[seq_idx, :, feat_idx]
                
                # 1. Add linear drift
                drift = np.linspace(0, drift_rate * seq_length, seq_length)
                signal_data += drift
                
                # 2. Add bias
                signal_data += bias
                
                # 3. Increase noise
                noise_std = np.std(signal_data)
                additional_noise = self.rng.normal(0, noise_std * (noise_increase - 1), seq_length)
                signal_data += additional_noise
                
                anomalous[seq_idx, :, feat_idx] = signal_data
        
        if single_sequence:
            anomalous = anomalous[0]
        
        return anomalous, metadata


class CompoundAnomalyInjector(AnomalyInjector):
    """
    Inject multiple anomaly types simultaneously.
    More realistic for real-world scenarios.
    """
    
    def inject(self, data: np.ndarray,
               anomaly_types: List[str] = None,
               **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Args:
            anomaly_types: List of anomaly types to inject
                Options: ['noise', 'freq_drop', 'freq_shift', 'degradation']
        """
        if anomaly_types is None:
            anomaly_types = ['noise', 'degradation']  # Common combination
        
        anomalous = data.copy()
        metadata = {
            'type': 'compound',
            'anomaly_types': anomaly_types,
            'severity': self.severity,
            'components': {}
        }
        
        # Apply each anomaly type
        for anom_type in anomaly_types:
            if anom_type == 'noise':
                injector = NoiseBurstInjector(self.severity, self.rng.randint(0, 10000))
                anomalous, meta = injector.inject(anomalous, **kwargs)
                metadata['components']['noise'] = meta
            
            elif anom_type == 'freq_drop':
                injector = FrequencyBandDropInjector(self.severity, self.rng.randint(0, 10000))
                anomalous, meta = injector.inject(anomalous, **kwargs)
                metadata['components']['freq_drop'] = meta
            
            elif anom_type == 'freq_shift':
                injector = ModalFrequencyShiftInjector(self.severity, self.rng.randint(0, 10000))
                anomalous, meta = injector.inject(anomalous, **kwargs)
                metadata['components']['freq_shift'] = meta
            
            elif anom_type == 'degradation':
                injector = SensorDegradationInjector(self.severity, self.rng.randint(0, 10000))
                anomalous, meta = injector.inject(anomalous, **kwargs)
                metadata['components']['degradation'] = meta
        
        return anomalous, metadata


# ============================================================================
# LATENT SPACE ANOMALY INJECTION
# ============================================================================

class LatentSpaceAnomalyInjector:
    """
    Inject anomalies directly in VAE latent space.
    More abstract but can be very effective for testing robustness.
    """
    
    def __init__(self, model, device='cpu', severity: float = 0.3, random_seed: int = 42):
        self.model = model
        self.device = device
        self.severity = severity
        self.rng = np.random.RandomState(random_seed)
        self.model.eval()
    
    def inject_latent_noise(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Add noise to latent representations."""
        # Encode
        with torch.no_grad():
            data_tensor = torch.from_numpy(data).float().to(self.device)
            latent = self.model.encode(data_tensor).cpu().numpy()
        
        # Add noise
        noise_std = np.std(latent) * self.severity
        noisy_latent = latent + self.rng.normal(0, noise_std, latent.shape)
        
        # Decode
        with torch.no_grad():
            noisy_latent_tensor = torch.from_numpy(noisy_latent).float().to(self.device)
            reconstructed = self.model.decode(noisy_latent_tensor, data.shape[1]).cpu().numpy()
        
        metadata = {
            'type': 'latent_noise',
            'noise_std': float(noise_std),
            'severity': self.severity
        }
        
        return reconstructed, noisy_latent, metadata
    
    def inject_latent_outliers(self, data: np.ndarray, n_outliers: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Replace random latent dimensions with extreme values."""
        with torch.no_grad():
            data_tensor = torch.from_numpy(data).float().to(self.device)
            latent = self.model.encode(data_tensor).cpu().numpy()
        
        anomalous_latent = latent.copy()
        
        # For each sequence, corrupt random dimensions
        n_sequences = latent.shape[0]
        latent_dim = latent.shape[1]
        
        for seq_idx in range(n_sequences):
            outlier_dims = self.rng.choice(latent_dim, n_outliers, replace=False)
            for dim in outlier_dims:
                # Replace with extreme value (3-5 std from mean)
                std = np.std(latent[:, dim])
                mean = np.mean(latent[:, dim])
                extreme_value = mean + self.rng.choice([-1, 1]) * (3 + self.severity * 2) * std
                anomalous_latent[seq_idx, dim] = extreme_value
        
        # Decode
        with torch.no_grad():
            anomalous_latent_tensor = torch.from_numpy(anomalous_latent).float().to(self.device)
            reconstructed = self.model.decode(anomalous_latent_tensor, data.shape[1]).cpu().numpy()
        
        metadata = {
            'type': 'latent_outliers',
            'n_outliers': n_outliers,
            'severity': self.severity
        }
        
        return reconstructed, anomalous_latent, metadata


# ============================================================================
# CSV OUTPUT FUNCTIONS
# ============================================================================

def sequences_to_csv_dataframe(sequences: np.ndarray,
                               start_datetime: datetime = None,
                               feature_names: List[str] = None) -> pd.DataFrame:
    """
    Convert sequence array back to CSV DataFrame format.
    
    Args:
        sequences: Shape (n_sequences, seq_length, n_features)
        start_datetime: Starting timestamp (default: 2024-11-27 00:00:00)
        feature_names: Feature column names (default: from config)
        
    Returns:
        DataFrame in original CSV format with columns:
        [day, hour_file, start_time, end_time, feature1, feature2, ...]
    """
    if feature_names is None:
        try:
            feature_names = config.FEATURE_COLUMNS
        except:
            feature_names = [f'feature_{i}' for i in range(sequences.shape[-1])]
    
    if start_datetime is None:
        start_datetime = datetime(2024, 11, 27, 0, 0, 0)
    
    n_sequences, seq_length, n_features = sequences.shape
    
    # Flatten sequences to rows (one row per second)
    total_seconds = n_sequences * seq_length
    
    rows = []
    current_time = start_datetime
    
    for seq_idx in range(n_sequences):
        for time_idx in range(seq_length):
            # Calculate timestamps
            end_time = current_time + timedelta(seconds=1)
            
            # Extract features
            feature_values = sequences[seq_idx, time_idx, :]
            
            # Create row
            row = {
                'day': current_time.strftime('%Y%m%d'),
                'hour_file': current_time.hour,
                'start_time': current_time.strftime('%Y/%m/%d %H:%M:%S:000'),
                'end_time': end_time.strftime('%Y/%m/%d %H:%M:%S:000'),
            }
            
            # Add feature columns
            for feat_idx, feat_name in enumerate(feature_names):
                row[feat_name] = float(feature_values[feat_idx])
            
            # Add variance column if only mean and log_variance provided
            if len(feature_names) == 2 and 'log_variance' in feature_names:
                log_var_idx = feature_names.index('log_variance')
                row['variance'] = float(np.exp(feature_values[log_var_idx]))
            
            rows.append(row)
            current_time = end_time
    
    df = pd.DataFrame(rows)
    
    # Ensure correct column order (matching original CSV)
    base_cols = ['day', 'hour_file', 'start_time', 'end_time']
    feature_cols = list(feature_names)
    
    # Add variance if it exists
    if 'variance' in df.columns and 'variance' not in feature_cols:
        feature_cols.append('variance')
    
    ordered_cols = base_cols + feature_cols
    df = df[ordered_cols]
    
    return df


def save_sequences_to_csv(sequences: np.ndarray,
                          output_path: Path,
                          metadata: Dict = None,
                          start_datetime: datetime = None,
                          feature_names: List[str] = None):
    """
    Save sequences to CSV file in original format.
    
    Args:
        sequences: Shape (n_sequences, seq_length, n_features)
        output_path: Path to save CSV
        metadata: Optional metadata to save as JSON sidecar
        start_datetime: Starting timestamp
        feature_names: Feature column names
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = sequences_to_csv_dataframe(sequences, start_datetime, feature_names)
    
    # Save CSV
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"💾 Saved CSV: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Save metadata as sidecar JSON
    if metadata is not None:
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Saved metadata: {metadata_path}")


def save_anomaly_dataset_as_csvs(dataset_path: Path,
                                 output_dir: Path,
                                 start_datetime: datetime = None):
    """
    Convert entire anomaly dataset NPZ to individual CSV files.
    
    Args:
        dataset_path: Path to anomaly_dataset.npz
        output_dir: Directory to save CSV files
        start_datetime: Starting timestamp
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("CONVERTING ANOMALY DATASET TO CSV")
    print("="*80)
    
    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)
    normal = data['normal']
    anomalous = data['anomalous']
    labels = data['labels']
    
    print(f"\nLoaded:")
    print(f"  Normal: {normal.shape}")
    print(f"  Anomalous: {anomalous.shape}")
    
    # Load metadata if available
    metadata_path = dataset_path.parent / "anomaly_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            global_metadata = json.load(f)
        individual_metadata = global_metadata.get('individual_metadata', [])
    else:
        individual_metadata = [{}] * len(anomalous)
    
    # Save normal data as single CSV
    print("\n📝 Saving normal data...")
    normal_csv = output_dir / "normal_data.csv"
    save_sequences_to_csv(
        normal,
        normal_csv,
        metadata={'type': 'normal', 'n_sequences': len(normal)},
        start_datetime=start_datetime
    )
    
    # Save anomalous data - group by type
    print("\n📝 Saving anomalous data by type...")
    
    # Group by anomaly type
    type_groups = {}
    for idx, label in enumerate(labels):
        anom_type = label['type']
        severity = label.get('severity', 'unknown')
        
        key = f"{anom_type}_severity{severity}"
        if key not in type_groups:
            type_groups[key] = []
        type_groups[key].append(idx)
    
    # Save each group
    for group_name, indices in tqdm(type_groups.items(), desc="Saving CSV groups"):
        group_sequences = anomalous[indices]
        group_labels = [labels[i] for i in indices]
        group_metadata = [individual_metadata[i] for i in indices]
        
        csv_path = output_dir / f"anomalous_{group_name}.csv"
        
        # Aggregate metadata
        meta = {
            'type': group_labels[0]['type'],
            'severity': group_labels[0].get('severity', 'unknown'),
            'n_sequences': len(group_sequences),
            'indices': indices,
            'labels': group_labels,
            'anomaly_details': group_metadata
        }
        
        save_sequences_to_csv(
            group_sequences,
            csv_path,
            metadata=meta,
            start_datetime=start_datetime
        )
    
    # Create summary
    summary = {
        'conversion_date': datetime.now().isoformat(),
        'source_npz': str(dataset_path),
        'output_directory': str(output_dir),
        'normal_csv': 'normal_data.csv',
        'anomalous_csvs': list(type_groups.keys()),
        'total_normal_sequences': len(normal),
        'total_anomalous_sequences': len(anomalous),
        'groups': {name: len(indices) for name, indices in type_groups.items()}
    }
    
    summary_path = output_dir / "csv_conversion_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Normal CSV: {normal_csv}")
    print(f"   Anomalous CSVs: {len(type_groups)} files")
    print(f"   Summary: {summary_path}")
    print("="*80)


# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_anomaly_dataset(normal_data: np.ndarray,
                            anomaly_types: List[str],
                            severity_levels: List[float],
                            n_samples_per_type: int,
                            output_dir: Path,
                            sampling_rate: float = 10.0,
                            random_seed: int = 42,
                            save_csv: bool = False,
                            start_datetime: datetime = None):
    """
    Generate comprehensive anomaly dataset.
    
    Args:
        normal_data: Normal sequences (n_sequences, seq_length, n_features)
        anomaly_types: List of anomaly types to generate
        severity_levels: List of severity values [0.0, 1.0]
        n_samples_per_type: How many samples per (type, severity) combination
        output_dir: Where to save results
        sampling_rate: Data sampling rate
        random_seed: For reproducibility
        save_csv: If True, also save as CSV files
        start_datetime: Starting timestamp for CSV (default: 2024-11-27)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING ANOMALY DATASET")
    print("="*80)
    print(f"Normal data shape: {normal_data.shape}")
    print(f"Anomaly types: {anomaly_types}")
    print(f"Severity levels: {severity_levels}")
    print(f"Samples per (type, severity): {n_samples_per_type}")
    print(f"Save CSV: {save_csv}")
    print("="*80)
    
    all_anomalous = []
    all_labels = []
    all_metadata = []
    
    rng = np.random.RandomState(random_seed)
    
    # For each anomaly type and severity
    for anom_type in tqdm(anomaly_types, desc="Anomaly types"):
        for severity in tqdm(severity_levels, desc=f"  {anom_type} severity", leave=False):
            
            # Create injector
            if anom_type == 'noise':
                injector = NoiseBurstInjector(severity, rng.randint(0, 100000))
            elif anom_type == 'freq_drop':
                injector = FrequencyBandDropInjector(severity, rng.randint(0, 100000))
            elif anom_type == 'freq_shift':
                injector = ModalFrequencyShiftInjector(severity, rng.randint(0, 100000))
            elif anom_type == 'degradation':
                injector = SensorDegradationInjector(severity, rng.randint(0, 100000))
            elif anom_type == 'compound':
                injector = CompoundAnomalyInjector(severity, rng.randint(0, 100000))
            else:
                print(f"⚠️  Unknown anomaly type: {anom_type}, skipping")
                continue
            
            # Generate samples
            for _ in range(n_samples_per_type):
                # Select random normal sequence
                idx = rng.randint(0, len(normal_data))
                normal_seq = normal_data[idx:idx+1]
                
                # Inject anomaly
                anomalous_seq, metadata = injector.inject(
                    normal_seq
                )
                
                all_anomalous.append(anomalous_seq[0])
                all_labels.append({
                    'type': anom_type,
                    'severity': severity,
                    'original_idx': idx
                })
                all_metadata.append(metadata)
    
    # Convert to arrays
    anomalous_data = np.array(all_anomalous)
    
    print(f"\n✅ Generated {len(anomalous_data)} anomalous sequences")
    
    # Save dataset
    dataset_file = output_dir / "anomaly_dataset.npz"
    np.savez_compressed(
        dataset_file,
        anomalous=anomalous_data,
        normal=normal_data,
        labels=all_labels
    )
    
    # Save metadata
    metadata_file = output_dir / "anomaly_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'n_anomalous': len(anomalous_data),
            'n_normal': len(normal_data),
            'anomaly_types': anomaly_types,
            'severity_levels': severity_levels,
            'n_samples_per_type': n_samples_per_type,
            'sampling_rate': sampling_rate,
            'timestamp': datetime.now().isoformat(),
            'individual_metadata': all_metadata
        }, f, indent=2)
    
    print(f"\n💾 Saved:")
    print(f"   Dataset: {dataset_file}")
    print(f"   Metadata: {metadata_file}")
    
    # Save as CSV if requested
    if save_csv:
        print(f"\n📝 Converting to CSV format...")
        csv_dir = output_dir / "csv_files"
        save_anomaly_dataset_as_csvs(
            dataset_file,
            csv_dir,
            start_datetime=start_datetime
        )
    
    return anomalous_data, all_labels, all_metadata


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_anomalies(normal: np.ndarray,
                       anomalous: np.ndarray,
                       metadata: Dict,
                       output_path: Path,
                       feature_names: List[str] = None):
    """Create comparison visualization."""
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(normal.shape[-1])]
    
    n_features = normal.shape[-1]
    
    fig, axes = plt.subplots(n_features, 2, figsize=(14, 4*n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Anomaly: {metadata['type']} (Severity: {metadata.get('severity', 'N/A')})", 
                 fontsize=14, fontweight='bold')
    
    for feat_idx in range(n_features):
        # Time series
        ax_time = axes[feat_idx, 0]
        ax_time.plot(normal[feat_idx], label='Normal', linewidth=2, alpha=0.8)
        ax_time.plot(anomalous[feat_idx], label='Anomalous', linewidth=2, alpha=0.8)
        ax_time.set_title(f'{feature_names[feat_idx]} - Time Domain')
        ax_time.set_xlabel('Time Step')
        ax_time.set_ylabel('Value')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        
        # Frequency domain
        ax_freq = axes[feat_idx, 1]
        
        # FFT
        normal_fft = np.abs(fft.rfft(normal[feat_idx]))
        anom_fft = np.abs(fft.rfft(anomalous[feat_idx]))
        freqs = fft.rfftfreq(len(normal[feat_idx]), 1/10.0)  # Assuming 10 Hz
        
        ax_freq.semilogy(freqs, normal_fft, label='Normal', linewidth=2, alpha=0.8)
        ax_freq.semilogy(freqs, anom_fft, label='Anomalous', linewidth=2, alpha=0.8)
        ax_freq.set_title(f'{feature_names[feat_idx]} - Frequency Domain')
        ax_freq.set_xlabel('Frequency (Hz)')
        ax_freq.set_ylabel('Magnitude')
        ax_freq.legend()
        ax_freq.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_visualization(dataset_path: Path, output_dir: Path):
    """Create summary visualization of anomaly dataset."""
    data = np.load(dataset_path, allow_pickle=True)
    anomalous = data['anomalous']
    normal = data['normal']
    labels = data['labels']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample from each anomaly type
    unique_types = list(set([label['type'] for label in labels]))
    
    for anom_type in unique_types:
        # Find examples of this type
        indices = [i for i, label in enumerate(labels) if label['type'] == anom_type]
        if not indices:
            continue
        
        # Pick one
        idx = indices[0]
        
        # Get corresponding normal sequence
        normal_idx = labels[idx]['original_idx']
        
        # Visualize
        visualize_anomalies(
            normal[normal_idx].T,
            anomalous[idx].T,
            labels[idx],
            output_dir / f"anomaly_{anom_type}.png",
            config.FEATURE_COLUMNS
        )
    
    print(f"\n📊 Created {len(unique_types)} summary visualizations in {output_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic anomalies for bridge accelerometer data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all anomaly types with default settings
  python synthetic_anomalies.py --input data.csv --output anomalies/ --all
  
  # Generate with CSV output
  python synthetic_anomalies.py --input data.csv --output anomalies/ --all --save-csv
  
  # Generate only noise bursts at high severity
  python synthetic_anomalies.py --input data.npz --noise-only --severity 0.8
  
  # Generate compound anomalies with custom start date
  python synthetic_anomalies.py --input data.csv --compound --output anomalies/ \
      --save-csv --start-date "2024-11-27 00:00:00"
  
  # Convert existing NPZ to CSV (after generation)
  python synthetic_anomalies.py convert-to-csv \
      --dataset anomalies/anomaly_dataset.npz \
      --output anomalies/csv_files/
  
  # Inject anomalies in latent space (requires trained model)
  python synthetic_anomalies.py --mode latent --checkpoint model.pt --input data.npz
        """
    )
    
    # Create subparsers for main mode vs convert mode
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Default (generate anomalies)
    parser.set_defaults(command='generate')
    
    # Convert-to-CSV subcommand
    convert_parser = subparsers.add_parser(
        'convert-to-csv',
        help='Convert existing NPZ anomaly dataset to CSV files'
    )
    convert_parser.add_argument('--dataset', type=str, required=True,
                               help='Path to anomaly_dataset.npz')
    convert_parser.add_argument('--output', type=str, required=True,
                               help='Output directory for CSV files')
    convert_parser.add_argument('--start-date', type=str,
                               help='Start datetime (YYYY-MM-DD HH:MM:SS, default: 2024-11-27 00:00:00)')
    
    # Main generation arguments (only if not converting)
    # Input/output
    parser.add_argument('--input', type=str, required=True,
                       help='Input data (.csv, .npz)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    
    # Mode
    parser.add_argument('--mode', type=str, default='data', choices=['data', 'latent'],
                       help='Inject in data space or latent space')
    parser.add_argument('--checkpoint', type=str,
                       help='VAE checkpoint (required for latent mode)')
    
    # Anomaly types (mutually exclusive groups)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true',
                      help='Generate all anomaly types')
    group.add_argument('--noise-only', action='store_true',
                      help='Only noise bursts')
    group.add_argument('--freq-only', action='store_true',
                      help='Only frequency anomalies (drop + shift)')
    group.add_argument('--degradation-only', action='store_true',
                      help='Only sensor degradation')
    group.add_argument('--compound', action='store_true',
                      help='Compound anomalies (multiple types)')
    
    # Parameters
    parser.add_argument('--severity', type=float, nargs='+', default=[0.3, 0.6, 0.9],
                       help='Severity levels (0-1, default: 0.3 0.6 0.9)')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Samples per (type, severity) combination')
    parser.add_argument('--sampling-rate', type=float, default=10.0,
                       help='Data sampling rate (Hz)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Options
    parser.add_argument('--no-scaler', action='store_true',
                       help='Skip scaler (latent mode only)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save anomalies as CSV files (in addition to NPZ)')
    parser.add_argument('--start-date', type=str,
                       help='Start datetime for CSV (YYYY-MM-DD HH:MM:SS, default: 2024-11-27 00:00:00)')
    
    args = parser.parse_args()
    
    # Handle convert-to-csv command separately
    if args.command == 'convert-to-csv':
        print("\n" + "="*80)
        print("CONVERTING NPZ TO CSV")
        print("="*80)
        
        # Parse start date
        if args.start_date:
            try:
                start_datetime = datetime.strptime(args.start_date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                print(f"⚠️  Invalid date format: {args.start_date}")
                start_datetime = datetime(2024, 11, 27, 0, 0, 0)
        else:
            start_datetime = datetime(2024, 11, 27, 0, 0, 0)
        
        # Convert
        save_anomaly_dataset_as_csvs(
            Path(args.dataset),
            Path(args.output),
            start_datetime=start_datetime
        )
        
        print("\n✅ Conversion complete!")
        return
    
    # Parse start date if provided
    start_datetime = None
    if args.start_date:
        try:
            start_datetime = datetime.strptime(args.start_date, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"⚠️  Invalid date format: {args.start_date}")
            print("   Using default: 2024-11-27 00:00:00")
            start_datetime = datetime(2024, 11, 27, 0, 0, 0)
    else:
        start_datetime = datetime(2024, 11, 27, 0, 0, 0)
    
    # Determine anomaly types
    if args.all:
        anomaly_types = ['noise', 'freq_drop', 'freq_shift', 'degradation']
    elif args.noise_only:
        anomaly_types = ['noise']
    elif args.freq_only:
        anomaly_types = ['freq_drop', 'freq_shift']
    elif args.degradation_only:
        anomaly_types = ['degradation']
    elif args.compound:
        anomaly_types = ['compound']
    else:
        # Default
        anomaly_types = ['noise', 'degradation']
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SYNTHETIC ANOMALY GENERATION")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Anomaly types: {anomaly_types}")
    print(f"Severity levels: {args.severity}")
    print("="*80)
    
    # Load data
    input_path = Path(args.input)
    
    if args.mode == 'latent':
        # Latent space mode - requires model
        if not args.checkpoint:
            print("❌ ERROR: --checkpoint required for latent mode")
            sys.exit(1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _ = load_checkpoint(args.checkpoint, device)
        scaler = load_scaler(args.checkpoint) if not args.no_scaler else None
        
        # Load data
        if input_path.suffix == '.npz':
            data = np.load(input_path)
            sequences = data['test'] if 'test' in data else data
        elif input_path.suffix == '.csv':
            sequences, scaler = load_csv_file(input_path, scaler)
        else:
            print(f"❌ ERROR: Unsupported file type: {input_path.suffix}")
            sys.exit(1)
        
        if scaler:
            sequences, _ = standardize_data(sequences, scaler)
        
        # Generate latent anomalies
        latent_injector = LatentSpaceAnomalyInjector(model, device, args.severity[0], args.seed)
        
        print("\n🔧 Generating latent noise anomalies...")
        noise_recon, noise_latent, noise_meta = latent_injector.inject_latent_noise(sequences)
        
        print("🔧 Generating latent outlier anomalies...")
        outlier_recon, outlier_latent, outlier_meta = latent_injector.inject_latent_outliers(sequences)
        
        # Save
        np.savez_compressed(
            output_dir / "latent_anomalies.npz",
            normal=sequences,
            noise_anomalous=noise_recon,
            outlier_anomalous=outlier_recon,
            noise_latent=noise_latent,
            outlier_latent=outlier_latent
        )
        
        with open(output_dir / "latent_metadata.json", 'w') as f:
            json.dump({
                'noise': noise_meta,
                'outlier': outlier_meta
            }, f, indent=2)
        
        print(f"\n✅ Saved latent anomalies to {output_dir}")
    
    else:
        # Data space mode
        if input_path.suffix == '.npz':
            data = np.load(input_path)
            sequences = data['test'] if 'test' in data else data
        elif input_path.suffix == '.csv':
            df = load_single_csv(input_path)
            features = extract_features(df)
            sequences = create_sequences(features, config.SEQUENCE_LENGTH)
        else:
            print(f"❌ ERROR: Unsupported file type: {input_path.suffix}")
            sys.exit(1)
        
        print(f"\n📊 Loaded {len(sequences)} normal sequences")
        
        # Generate anomaly dataset
        anomalous_data, labels, metadata = generate_anomaly_dataset(
            normal_data=sequences,
            anomaly_types=anomaly_types,
            severity_levels=args.severity,
            n_samples_per_type=args.n_samples,
            output_dir=output_dir,
            sampling_rate=args.sampling_rate,
            random_seed=args.seed,
            save_csv=args.save_csv,
            start_datetime=start_datetime
        )
        
        # Visualizations
        if args.visualize:
            print("\n📊 Creating visualizations...")
            create_summary_visualization(
                output_dir / "anomaly_dataset.npz",
                output_dir / "visualizations"
            )
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()