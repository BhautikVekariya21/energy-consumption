"""
================================================================================
EVALUATION METRICS
================================================================================
Comprehensive metrics for evaluating electricity load forecasts.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats

from utils import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastMetrics:
    """Container for forecast metrics."""
    
    # Core metrics
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float
    
    # Additional metrics
    mse: float = 0.0
    max_error: float = 0.0
    median_ae: float = 0.0
    
    # Skill scores
    skill_vs_persistence: float = 0.0
    skill_vs_climatology: float = 0.0
    
    # Horizon-specific (optional)
    horizon_mae: List[float] = field(default_factory=list)
    horizon_r2: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'mae': float(self.mae),
            'rmse': float(self.rmse),
            'mape': float(self.mape),
            'smape': float(self.smape),
            'r2': float(self.r2),
            'mse': float(self.mse),
            'max_error': float(self.max_error),
            'median_ae': float(self.median_ae),
            'skill_vs_persistence': float(self.skill_vs_persistence),
            'skill_vs_climatology': float(self.skill_vs_climatology),
            'horizon_mae': [float(x) for x in self.horizon_mae],
            'horizon_r2': [float(x) for x in self.horizon_r2]
        }
    
    def summary(self) -> str:
        """Get formatted summary string."""
        lines = [
            "Forecast Metrics",
            "=" * 40,
            f"MAE:   {self.mae:,.2f}",
            f"RMSE:  {self.rmse:,.2f}",
            f"MAPE:  {self.mape:.2f}%",
            f"SMAPE: {self.smape:.2f}%",
            f"R²:    {self.r2:.4f}",
            f"",
            f"Max Error:  {self.max_error:,.2f}",
            f"Median AE:  {self.median_ae:,.2f}",
        ]
        
        if self.skill_vs_persistence != 0:
            lines.append(f"")
            lines.append(f"Skill vs Persistence: {self.skill_vs_persistence:.2f}%")
            lines.append(f"Skill vs Climatology: {self.skill_vs_climatology:.2f}%")
        
        return "\n".join(lines)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8
) -> Dict[str, float]:
    """
    Calculate comprehensive forecast metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Remove NaN/Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'error': 'No valid data points'}
    
    # Errors
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    # Core metrics
    mae = np.mean(abs_errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    
    # Percentage errors
    mape = np.mean(abs_errors / (np.abs(y_true) + epsilon)) * 100
    smape = np.mean(2 * abs_errors / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100
    
    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    
    # Additional metrics
    max_error = np.max(abs_errors)
    median_ae = np.median(abs_errors)
    
    # Correlation
    correlation, p_value = stats.pearsonr(y_true, y_pred)
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'smape': float(smape),
        'r2': float(r2),
        'max_error': float(max_error),
        'median_ae': float(median_ae),
        'correlation': float(correlation),
        'n_samples': len(y_true)
    }


def calculate_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, List[float]]:
    """
    Calculate metrics for each forecast horizon step.
    
    Args:
        y_true: True values of shape (n_samples, horizon)
        y_pred: Predicted values of shape (n_samples, horizon)
    
    Returns:
        Dictionary with lists of metrics per horizon
    """
    if len(y_true.shape) != 2:
        raise ValueError("Input must be 2D arrays of shape (n_samples, horizon)")
    
    horizon = y_true.shape[1]
    
    metrics = {
        'horizon_mae': [],
        'horizon_rmse': [],
        'horizon_mape': [],
        'horizon_r2': []
    }
    
    for h in range(horizon):
        h_metrics = calculate_metrics(y_true[:, h], y_pred[:, h])
        
        metrics['horizon_mae'].append(h_metrics['mae'])
        metrics['horizon_rmse'].append(h_metrics['rmse'])
        metrics['horizon_mape'].append(h_metrics['mape'])
        metrics['horizon_r2'].append(h_metrics['r2'])
    
    return metrics


def calculate_source_metrics(
    source_true: Dict[str, np.ndarray],
    source_pred: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for each energy source.
    
    Args:
        source_true: Dict of true values per source
        source_pred: Dict of predicted values per source
    
    Returns:
        Nested dict of metrics per source
    """
    results = {}
    
    for source_name in source_true.keys():
        if source_name in source_pred:
            true_vals = source_true[source_name]
            pred_vals = source_pred[source_name]
            
            results[source_name] = calculate_metrics(true_vals, pred_vals)
    
    return results


def calculate_skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline: str = 'persistence',
    lag: int = 24
) -> float:
    """
    Calculate skill score relative to baseline.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        baseline: Type of baseline ('persistence', 'climatology')
        lag: Lag for persistence baseline
    
    Returns:
        Skill score as percentage improvement
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if baseline == 'persistence':
        # Persistence forecast: use last known value
        baseline_pred = np.roll(y_true, lag)
        baseline_pred[:lag] = y_true[:lag]
    elif baseline == 'climatology':
        # Climatology: use mean
        baseline_pred = np.full_like(y_true, np.mean(y_true))
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    # Calculate errors
    model_mse = np.mean((y_true - y_pred) ** 2)
    baseline_mse = np.mean((y_true - baseline_pred) ** 2)
    
    # Skill score
    if baseline_mse > 0:
        skill = (1 - model_mse / baseline_mse) * 100
    else:
        skill = 0.0
    
    return float(skill)


class MetricsCalculator:
    """
    Class for comprehensive metrics calculation with additional features.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.results_history = []
    
    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        name: str = 'forecast'
    ) -> ForecastMetrics:
        """
        Calculate all metrics and return ForecastMetrics object.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            name: Name for this evaluation
        
        Returns:
            ForecastMetrics object
        """
        # Core metrics
        basic = calculate_metrics(y_true, y_pred)
        
        # Skill scores
        skill_persistence = calculate_skill_score(y_true, y_pred, 'persistence')
        skill_climatology = calculate_skill_score(y_true, y_pred, 'climatology')
        
        # Horizon metrics if 2D
        if len(y_true.shape) == 2:
            horizon_metrics = calculate_horizon_metrics(y_true, y_pred)
            horizon_mae = horizon_metrics['horizon_mae']
            horizon_r2 = horizon_metrics['horizon_r2']
        else:
            horizon_mae = []
            horizon_r2 = []
        
        metrics = ForecastMetrics(
            mae=basic['mae'],
            rmse=basic['rmse'],
            mape=basic['mape'],
            smape=basic['smape'],
            r2=basic['r2'],
            mse=basic['mse'],
            max_error=basic['max_error'],
            median_ae=basic['median_ae'],
            skill_vs_persistence=skill_persistence,
            skill_vs_climatology=skill_climatology,
            horizon_mae=horizon_mae,
            horizon_r2=horizon_r2
        )
        
        # Store in history
        self.results_history.append({
            'name': name,
            'metrics': metrics.to_dict()
        })
        
        return metrics
    
    def compare(
        self,
        results: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, ForecastMetrics]:
        """
        Compare multiple forecasts.
        
        Args:
            results: Dict of {name: (y_true, y_pred)}
        
        Returns:
            Dict of metrics for each forecast
        """
        comparisons = {}
        
        for name, (y_true, y_pred) in results.items():
            comparisons[name] = self.calculate(y_true, y_pred, name)
        
        return comparisons
    
    def print_comparison_table(
        self,
        results: Dict[str, ForecastMetrics]
    ):
        """Print formatted comparison table."""
        
        print("\n" + "=" * 80)
        print("FORECAST COMPARISON")
        print("=" * 80)
        print(f"{'Model':<20} | {'MAE':>12} | {'RMSE':>12} | {'MAPE':>10} | {'R²':>10}")
        print("-" * 80)
        
        for name, metrics in results.items():
            print(f"{name:<20} | {metrics.mae:>12,.2f} | {metrics.rmse:>12,.2f} | "
                  f"{metrics.mape:>9.2f}% | {metrics.r2:>10.4f}")
        
        print("=" * 80)