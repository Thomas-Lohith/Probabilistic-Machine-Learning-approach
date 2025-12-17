
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_time_series(df: pd.DataFrame, feature: str = 'mean', save_path: str = None):
    """
    Plot time series of a feature
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    
    ax.plot(df['start_time'], df[feature], linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel(feature.capitalize())
    ax.set_title(f'{feature.capitalize()} Over Time - First Day')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, save_path: str = None):
    """
    Plot distributions of all features
    """
    features = ['mean', 'variance', 'log_variance']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, feature in enumerate(features):
        data = df[feature].dropna()
        axes[idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel(feature.capitalize())
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Distribution of {feature.capitalize()}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_missing_data_pattern(df: pd.DataFrame, save_path: str = None):
    """
    Visualize pattern of missing data
    """
    # Check for NaN in each feature
    nan_mask = df[['mean', 'variance', 'log_variance']].isna()
    
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Plot NaN locations
    for idx, col in enumerate(['mean', 'variance', 'log_variance']):
        nan_indices = np.where(nan_mask[col])[0]
        ax.scatter(nan_indices, [idx] * len(nan_indices), marker='|', s=100, alpha=0.6)
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Mean', 'Variance', 'Log Variance'])
    ax.set_xlabel('Sample Index')
    ax.set_title('Missing Data Pattern - First Day')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """
    Plot correlation between features
    """
    features = ['mean', 'variance', 'log_variance']
    corr_matrix = df[features].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    from load_data import load_single_day
    
    # Load data
    df = load_single_day('/data/pool/c8x-98x/bridge_data/100_days/20241127.csv')
    
    # Create results directory if it doesn't exist
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("Generating visualizations...")
    
    plot_time_series(df, 'mean', 'results/figures/01_mean_timeseries.png')
    plot_feature_distributions(df, 'results/figures/02_feature_distributions.png')
    plot_missing_data_pattern(df, 'results/figures/03_missing_data.png')
    plot_correlation_matrix(df, 'results/figures/04_correlation.png')
    
    print("Visualizations saved to results/figures/")