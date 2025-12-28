"""
================================================================================
VISUALIZATION TOOLS
================================================================================
Comprehensive visualization for electricity forecasting results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import pandas as pd

from config import get_settings
from utils import get_logger, format_number

logger = get_logger(__name__)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2196F3',
    'secondary': '#FF9800',
    'success': '#4CAF50',
    'danger': '#F44336',
    'warning': '#FFC107',
    'info': '#00BCD4',
    'actual': '#1976D2',
    'predicted': '#D32F2F',
    'uncertainty': '#E3F2FD'
}

SOURCE_COLORS = {
    'nuclear': '#9C27B0',
    'coal': '#795548',
    'natural_gas': '#FF9800',
    'hydro': '#2196F3',
    'wind': '#4CAF50',
    'solar': '#FFEB3B',
    'other': '#9E9E9E'
}


class Visualizer:
    """
    Main visualization class for electricity forecasting.
    """
    
    def __init__(
        self,
        save_dir: Optional[Path] = None,
        figure_size: Tuple[int, int] = (12, 6),
        dpi: int = 150,
        style: str = 'seaborn-v0_8-whitegrid'
    ):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save figures
            figure_size: Default figure size
            dpi: Figure DPI
            style: Matplotlib style
        """
        settings = get_settings()
        self.save_dir = Path(save_dir) if save_dir else settings.paths.plot_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.figure_size = figure_size
        self.dpi = dpi
        
        try:
            plt.style.use(style)
        except:
            pass
        
        logger.info(f"Visualizer initialized, saving to {self.save_dir}")
    
    def plot_predictions(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        title: str = "Load Forecast vs Actual",
        uncertainty: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        show: bool = True,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot predictions vs actual values.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            timestamps: Optional datetime index
            title: Plot title
            uncertainty: Optional (lower, upper) bounds
            show: Whether to display plot
            save_name: Filename to save (without extension)
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                                 gridspec_kw={'height_ratios': [3, 1]})
        
        # Flatten if needed
        actual = np.asarray(actual).flatten()
        predicted = np.asarray(predicted).flatten()
        
        if timestamps is None:
            x = np.arange(len(actual))
            xlabel = "Timestep"
        else:
            x = timestamps[:len(actual)]
            xlabel = "Date"
        
        # Main plot
        ax1 = axes[0]
        ax1.plot(x, actual, label='Actual', color=COLORS['actual'], 
                linewidth=1.5, alpha=0.8)
        ax1.plot(x, predicted, label='Predicted', color=COLORS['predicted'],
                linewidth=1.5, alpha=0.8, linestyle='--')
        
        # Uncertainty bands
        if uncertainty is not None:
            lower, upper = uncertainty
            lower = np.asarray(lower).flatten()
            upper = np.asarray(upper).flatten()
            ax1.fill_between(x, lower, upper, alpha=0.2, 
                           color=COLORS['predicted'], label='95% CI')
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel("Load (MW)", fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Error plot
        ax2 = axes[1]
        errors = actual - predicted
        ax2.bar(x, errors, alpha=0.6, color=COLORS['secondary'], width=1.0)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_xlabel(xlabel, fontsize=12)
        ax2.set_ylabel("Error (MW)", fontsize=12)
        ax2.set_title("Prediction Error", fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Format dates if using timestamps
        if timestamps is not None:
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_dir / f'{save_name}.png', dpi=self.dpi, 
                       bbox_inches='tight')
            logger.info(f"Saved: {self.save_dir / f'{save_name}.png'}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_source_breakdown(
        self,
        total_load: np.ndarray,
        sources: Dict[str, np.ndarray],
        timestamps: Optional[pd.DatetimeIndex] = None,
        title: str = "Generation by Source",
        show: bool = True,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot stacked area chart of generation by source.
        
        Args:
            total_load: Total load values
            sources: Dict of source name to generation values
            timestamps: Optional datetime index
            title: Plot title
            show: Whether to display
            save_name: Filename to save
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                gridspec_kw={'height_ratios': [2, 1]})
        
        total_load = np.asarray(total_load).flatten()
        n_points = len(total_load)
        
        if timestamps is None:
            x = np.arange(n_points)
        else:
            x = timestamps[:n_points]
        
        # Prepare source data
        source_arrays = []
        source_names = []
        colors = []
        
        for name, values in sources.items():
            source_arrays.append(np.asarray(values).flatten()[:n_points])
            source_names.append(name.replace('_', ' ').title())
            colors.append(SOURCE_COLORS.get(name, '#666666'))
        
        # Stacked area chart
        ax1 = axes[0]
        ax1.stackplot(x, source_arrays, labels=source_names, colors=colors, alpha=0.8)
        ax1.plot(x, total_load, color='black', linewidth=2, linestyle='--',
                label='Total Load', alpha=0.7)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel("Power (MW)", fontsize=12)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax1.grid(True, alpha=0.3)
        
        # Pie chart for average contribution
        ax2 = axes[1]
        avg_sources = [arr.mean() for arr in source_arrays]
        
        # Only show sources with > 1% contribution
        threshold = sum(avg_sources) * 0.01
        filtered_names = []
        filtered_values = []
        filtered_colors = []
        other_total = 0
        
        for name, value, color in zip(source_names, avg_sources, colors):
            if value >= threshold:
                filtered_names.append(name)
                filtered_values.append(value)
                filtered_colors.append(color)
            else:
                other_total += value
        
        if other_total > 0:
            filtered_names.append('Other')
            filtered_values.append(other_total)
            filtered_colors.append('#9E9E9E')
        
        wedges, texts, autotexts = ax2.pie(
            filtered_values,
            labels=filtered_names,
            colors=filtered_colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75
        )
        ax2.set_title("Average Generation Mix", fontsize=12)
        
        # Add total in center
        centre_circle = plt.Circle((0, 0), 0.4, fc='white')
        ax2.add_patch(centre_circle)
        ax2.text(0, 0, f'{sum(filtered_values)/1000:.1f}\nGW',
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_dir / f'{save_name}.png', dpi=self.dpi,
                       bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_horizon_performance(
        self,
        horizon_mae: List[float],
        horizon_r2: List[float],
        horizon_labels: Optional[List[str]] = None,
        title: str = "Forecast Performance by Horizon",
        show: bool = True,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot metrics across forecast horizon.
        
        Args:
            horizon_mae: MAE for each horizon step
            horizon_r2: R² for each horizon step
            horizon_labels: Optional labels for horizons
            title: Plot title
            show: Whether to display
            save_name: Filename to save
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        n_steps = len(horizon_mae)
        x = np.arange(1, n_steps + 1)
        
        if horizon_labels is None:
            horizon_labels = [f"H{i}" for i in x]
        
        # MAE plot
        ax1 = axes[0]
        bars1 = ax1.bar(x, horizon_mae, color=COLORS['primary'], alpha=0.7)
        ax1.set_xlabel("Forecast Horizon", fontsize=12)
        ax1.set_ylabel("MAE (MW)", fontsize=12)
        ax1.set_title("MAE by Horizon", fontsize=13, fontweight='bold')
        ax1.set_xticks(x[::max(1, n_steps//10)])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add trend line
        z = np.polyfit(x, horizon_mae, 1)
        p = np.poly1d(z)
        ax1.plot(x, p(x), '--', color=COLORS['danger'], linewidth=2,
                label=f'Trend (+{z[0]:.1f}/step)')
        ax1.legend()
        
        # R² plot
        ax2 = axes[1]
        bars2 = ax2.bar(x, horizon_r2, color=COLORS['success'], alpha=0.7)
        ax2.axhline(0.5, color=COLORS['warning'], linestyle='--', 
                   label='Fair (0.5)', alpha=0.7)
        ax2.axhline(0.7, color=COLORS['success'], linestyle='--',
                   label='Good (0.7)', alpha=0.7)
        ax2.set_xlabel("Forecast Horizon", fontsize=12)
        ax2.set_ylabel("R² Score", fontsize=12)
        ax2.set_title("R² by Horizon", fontsize=13, fontweight='bold')
        ax2.set_xticks(x[::max(1, n_steps//10)])
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_dir / f'{save_name}.png', dpi=self.dpi,
                       bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        show: bool = True,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training history curves.
        
        Args:
            history: Training history dict
            title: Plot title
            show: Whether to display
            save_name: Filename to save
        
        Returns:
            Matplotlib figure
        """
        # Determine number of subplots
        metrics = ['loss']
        if 'mae' in history:
            metrics.append('mae')
        if 'lr' in history or 'learning_rate' in history:
            metrics.append('lr')
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        
        epochs = range(1, len(history.get('loss', [])) + 1)
        
        for ax, metric in zip(axes, metrics):
            if metric == 'lr':
                key = 'lr' if 'lr' in history else 'learning_rate'
                if key in history:
                    ax.plot(epochs, history[key], color=COLORS['info'])
                    ax.set_ylabel("Learning Rate")
                    ax.set_yscale('log')
            else:
                if metric in history:
                    ax.plot(epochs, history[metric], label='Train', 
                           color=COLORS['primary'], linewidth=2)
                if f'val_{metric}' in history:
                    ax.plot(epochs, history[f'val_{metric}'], label='Val',
                           color=COLORS['secondary'], linewidth=2)
                    
                    # Mark best
                    best_idx = np.argmin(history[f'val_{metric}'])
                    best_val = history[f'val_{metric}'][best_idx]
                    ax.axvline(best_idx + 1, color=COLORS['success'], 
                              linestyle='--', alpha=0.5)
                    ax.scatter([best_idx + 1], [best_val], color=COLORS['success'],
                              s=100, zorder=5, label=f'Best: {best_val:.4f}')
                
                ax.set_ylabel(metric.upper())
                ax.legend()
            
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_dir / f'{save_name}.png', dpi=self.dpi,
                       bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance",
        show: bool = True,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance: Dict of feature name to importance score
            top_n: Number of top features to show
            title: Plot title
            show: Whether to display
            save_name: Filename to save
        
        Returns:
            Matplotlib figure
        """
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        names = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values, color=COLORS['primary'], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_dir / f'{save_name}.png', dpi=self.dpi,
                       bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig


# Convenience functions

def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    **kwargs
) -> plt.Figure:
    """Convenience function for plotting predictions."""
    viz = Visualizer()
    return viz.plot_predictions(actual, predicted, **kwargs)


def plot_training_history(
    history: Dict[str, List[float]],
    **kwargs
) -> plt.Figure:
    """Convenience function for plotting training history."""
    viz = Visualizer()
    return viz.plot_training_history(history, **kwargs)


def plot_source_breakdown(
    total_load: np.ndarray,
    sources: Dict[str, np.ndarray],
    **kwargs
) -> plt.Figure:
    """Convenience function for plotting source breakdown."""
    viz = Visualizer()
    return viz.plot_source_breakdown(total_load, sources, **kwargs)


def plot_horizon_performance(
    horizon_mae: List[float],
    horizon_r2: List[float],
    **kwargs
) -> plt.Figure:
    """Convenience function for plotting horizon performance."""
    viz = Visualizer()
    return viz.plot_horizon_performance(horizon_mae, horizon_r2, **kwargs)


def plot_feature_importance(
    importance: Dict[str, float],
    **kwargs
) -> plt.Figure:
    """Convenience function for plotting feature importance."""
    viz = Visualizer()
    return viz.plot_feature_importance(importance, **kwargs)


def create_forecast_report(
    actual: np.ndarray,
    predicted: np.ndarray,
    sources: Optional[Dict[str, np.ndarray]] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    history: Optional[Dict[str, List[float]]] = None,
    save_dir: Optional[Path] = None,
    report_name: str = "forecast_report"
) -> Path:
    """
    Create comprehensive forecast report with multiple plots.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        sources: Optional source breakdown
        feature_importance: Optional feature importance
        history: Optional training history
        save_dir: Directory to save report
        report_name: Name of the report
    
    Returns:
        Path to saved report directory
    """
    settings = get_settings()
    save_dir = Path(save_dir) if save_dir else settings.paths.plot_dir / report_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    viz = Visualizer(save_dir=save_dir)
    
    # Create all plots
    viz.plot_predictions(actual, predicted, show=False, 
                        save_name='predictions')
    
    if sources:
        viz.plot_source_breakdown(predicted, sources, show=False,
                                 save_name='source_breakdown')
    
    if feature_importance:
        viz.plot_feature_importance(feature_importance, show=False,
                                   save_name='feature_importance')
    
    if history:
        viz.plot_training_history(history, show=False,
                                 save_name='training_history')
    
    # Calculate horizon metrics if 2D
    if len(predicted.shape) == 2:
        from .metrics import calculate_horizon_metrics
        h_metrics = calculate_horizon_metrics(actual, predicted)
        viz.plot_horizon_performance(
            h_metrics['horizon_mae'],
            h_metrics['horizon_r2'],
            show=False,
            save_name='horizon_performance'
        )
    
    logger.info(f"Forecast report saved to {save_dir}")
    
    return save_dir