"""
Data module for electricity forecasting system.
Handles data downloading, preprocessing, feature engineering, and dataset creation.
"""

from .downloader import DataDownloader, download_all_data
from .preprocessor import DataPreprocessor, preprocess_data
from .feature_engineer import FeatureEngineer, create_features
from .dataset import (
    TimeSeriesDataset,
    create_datasets,
    create_tf_dataset,
    DataSpec,
    ScalerManager,
    DataPipeline  # Added this
)

__all__ = [
    'DataDownloader', 'download_all_data',
    'DataPreprocessor', 'preprocess_data',
    'FeatureEngineer', 'create_features',
    'TimeSeriesDataset', 'create_datasets', 'create_tf_dataset', 
    'DataSpec', 'ScalerManager', 'DataPipeline'
]