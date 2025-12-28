"""
================================================================================
EVALUATION MODULE
================================================================================
Metrics calculation and visualization for electricity forecasting.
"""

from .metrics import (
    calculate_metrics,
    calculate_horizon_metrics,
    calculate_source_metrics,
    MetricsCalculator,
    ForecastMetrics
)
from .visualizer import (
    Visualizer,
    plot_predictions,
    plot_training_history,
    plot_source_breakdown,
    plot_horizon_performance,
    plot_feature_importance,
    create_forecast_report
)

__all__ = [
    # Metrics
    'calculate_metrics', 'calculate_horizon_metrics',
    'calculate_source_metrics', 'MetricsCalculator', 'ForecastMetrics',
    
    # Visualization
    'Visualizer', 'plot_predictions', 'plot_training_history',
    'plot_source_breakdown', 'plot_horizon_performance',
    'plot_feature_importance', 'create_forecast_report'
]