import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           f1_score, cohen_kappa_score)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self):
        self.metrics = {}
        self.bin_edges = [float('-inf'), 7000, 7400, 8000, 8500, float('inf')]
        self.bin_labels = ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 5']
        
    def analyze_capacity_distribution(self, capacity: np.ndarray) -> None:
        print("\nDetailed Capacity Analysis:")
        print("-" * 50)
        print(f"Min capacity: {np.min(capacity):.2f}")
        print(f"Max capacity: {np.max(capacity):.2f}")
        print(f"Mean capacity: {np.mean(capacity):.2f}")
        print(f"Median capacity: {np.median(capacity):.2f}")
        print("\nValue counts in each range:")
        print(f"≤ 7000: {np.sum(capacity <= 7000)}")
        print(f"7000-7400: {np.sum((capacity > 7000) & (capacity <= 7400))}")
        print(f"7400-8000: {np.sum((capacity > 7400) & (capacity <= 8000))}")
        print(f"8000-8500: {np.sum((capacity > 8000) & (capacity <= 8500))}")
        print(f"> 8500: {np.sum(capacity > 8500)}")
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        #Calculate various regression metrics such as MSE, RMSE, MAE, R2.
        self.metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        return self.metrics
    
    def classify_capacity(self, capacity: np.ndarray) -> np.ndarray:
        #Classify capacity values into predefined bins.
        bins = np.zeros_like(capacity, dtype=int)
        
        # Bin 1: ≤ 7000
        bins[capacity <= 7000] = 1
        
        # Bin 2: 7000-7400
        bins[(capacity > 7000) & (capacity <= 7400)] = 2
        
        # Bin 3: 7400-8000
        bins[(capacity > 7400) & (capacity <= 8000)] = 3
        
        # Bin 4: 8000-8500
        bins[(capacity > 8000) & (capacity <= 8500)] = 4
        
        # Bin 5: > 8500
        bins[capacity > 8500] = 5
        
        return bins
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        #Evaluate f1 score and accuracy for binned capacities.
        true_bins = self.classify_capacity(y_true)
        pred_bins = self.classify_capacity(y_pred)
        
        # Calculate overall metrics
        overall_metrics = {
            'accuracy': np.mean(true_bins == pred_bins),
            'f1_macro': f1_score(true_bins, pred_bins, average='macro'),
            'f1_weighted': f1_score(true_bins, pred_bins, average='weighted'),
            'kappa': cohen_kappa_score(true_bins, pred_bins)
        }
        
        # Calculate per-bin metrics
        per_bin_metrics = {}
        for bin_idx in range(1, len(self.bin_edges)):
            bin_mask = (true_bins == bin_idx)
            if np.sum(bin_mask) > 0:
                bin_metrics = {
                    'accuracy': np.mean(true_bins[bin_mask] == pred_bins[bin_mask]),
                    'f1': f1_score(true_bins == bin_idx, pred_bins == bin_idx, zero_division=0)
                }
                per_bin_metrics[f'Bin {bin_idx}'] = bin_metrics
        
        return overall_metrics, per_bin_metrics
    
    def plot_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    save_path: str = None) -> None:
        # Create a figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot of predicted vs actual values
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2)
        ax1.set_xlabel('Actual Capacity')
        ax1.set_ylabel('Predicted Capacity')
        ax1.set_title('Predicted vs Actual Capacity')
        
        # Residual plot
        residuals = y_pred - y_true
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Capacity')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def get_bin_distribution(self, capacity: np.ndarray) -> Dict[str, int]:
        """
        Get the distribution of batteries across bins.
        
        Args:
            capacity (np.ndarray): Array of capacity values
            
        Returns:
            Dict[str, int]: Number of batteries in each bin
        """
        bins = self.classify_capacity(capacity)
        distribution = {}
        for i, label in enumerate(self.bin_labels, 1):
            count = np.sum(bins == i)
            distribution[label] = count
        return distribution

    def print_report(self) -> None:
        """Print a comprehensive evaluation report."""
        print("\nModel Evaluation Report")
        print("-" * 50)
        print("\nRegression Metrics:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
