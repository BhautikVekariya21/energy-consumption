"""
================================================================================
TENSORFLOW DATASET CREATION
================================================================================
Creates optimized TensorFlow datasets for training and inference.
Supports windowed sequences, multi-horizon prediction, and source-wise outputs.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pickle
from pathlib import Path

from config import get_settings, get_model_config
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class DataSpec:
    """Specification for dataset structure."""
    
    # Dimensions
    n_features: int
    n_targets: int
    lookback: int
    horizon: int
    
    # Feature information
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    source_names: List[str] = field(default_factory=list)
    
    # Scaling info
    feature_scaler_type: str = 'robust'
    target_scaler_type: str = 'standard'
    
    # Data shapes
    input_shape: Tuple[int, ...] = None
    output_shape: Tuple[int, ...] = None
    
    def __post_init__(self):
        self.input_shape = (self.lookback, self.n_features)
        self.output_shape = (self.horizon * self.n_targets,)
    
    def save(self, filepath: Path):
        """Save specification to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Path) -> 'DataSpec':
        """Load specification from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ScalerManager:
    """Manages scalers for features and targets."""
    
    def __init__(
        self,
        feature_scaler_type: str = 'robust',
        target_scaler_type: str = 'standard'
    ):
        """
        Initialize scaler manager.
        
        Args:
            feature_scaler_type: Type of scaler for features
            target_scaler_type: Type of scaler for targets
        """
        self.feature_scaler = self._create_scaler(feature_scaler_type)
        self.target_scaler = self._create_scaler(target_scaler_type)
        
        self.feature_scaler_type = feature_scaler_type
        self.target_scaler_type = target_scaler_type
        
        self.is_fitted = False
    
    def _create_scaler(self, scaler_type: str):
        """Create a scaler based on type."""
        scalers = {
            'standard': StandardScaler,
            'robust': RobustScaler,
            'minmax': MinMaxScaler
        }
        return scalers.get(scaler_type, RobustScaler)()
    
    def fit(self, features: np.ndarray, targets: np.ndarray):
        """Fit scalers on data."""
        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets.reshape(-1, 1))
        self.is_fitted = True
        
        logger.debug(f"Scalers fitted on {len(features)} samples")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features."""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit() first.")
        return self.feature_scaler.transform(features)
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets."""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit() first.")
        shape = targets.shape
        transformed = self.target_scaler.transform(targets.reshape(-1, 1))
        return transformed.reshape(shape)
    
    def inverse_transform_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        """Inverse transform targets."""
        shape = scaled_targets.shape
        original = self.target_scaler.inverse_transform(scaled_targets.reshape(-1, 1))
        return original.reshape(shape)
    
    def save(self, filepath: Path):
        """Save scalers to file."""
        data = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'feature_scaler_type': self.feature_scaler_type,
            'target_scaler_type': self.target_scaler_type,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: Path) -> 'ScalerManager':
        """Load scalers from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        manager = cls(
            data['feature_scaler_type'],
            data['target_scaler_type']
        )
        manager.feature_scaler = data['feature_scaler']
        manager.target_scaler = data['target_scaler']
        manager.is_fitted = data['is_fitted']
        
        return manager


class TimeSeriesDataset:
    """
    Time series dataset for electricity forecasting.
    
    Features:
    - Windowed sequence creation
    - Multiple target support (load + sources)
    - Efficient memory management
    - TensorFlow dataset creation
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = 'Load_MW',
        feature_cols: Optional[List[str]] = None,
        source_cols: Optional[List[str]] = None,
        lookback: int = 168,
        horizon: int = 24,
        scaler_manager: Optional[ScalerManager] = None,
        fit_scalers: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature column names
            source_cols: List of source estimation columns
            lookback: Number of historical timesteps
            horizon: Number of future timesteps to predict
            scaler_manager: Pre-fitted scaler manager
            fit_scalers: Whether to fit new scalers
        """
        self.df = df.copy()
        self.target_col = target_col
        self.lookback = lookback
        self.horizon = horizon
        
        # Determine feature columns
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]
        self.feature_cols = feature_cols
        
        # Determine source columns
        if source_cols is None:
            source_cols = [c for c in df.columns if c.startswith('est_') and c.endswith('_mw')]
        self.source_cols = source_cols
        
        # Extract data
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        
        # Extract source data if available
        if source_cols:
            self.sources = df[source_cols].values.astype(np.float32)
        else:
            self.sources = None
        
        # Handle scaling
        if fit_scalers:
            self.scaler = ScalerManager()
            self.scaler.fit(self.features, self.targets)
        elif scaler_manager:
            self.scaler = scaler_manager
        else:
            self.scaler = ScalerManager()
            self.scaler.fit(self.features, self.targets)
        
        # Transform data
        self.features_scaled = self.scaler.transform_features(self.features)
        self.targets_scaled = self.scaler.transform_targets(self.targets)
        
        # Create sequences
        self._create_sequences()
        
        # Create data specification
        self.spec = DataSpec(
            n_features=len(feature_cols),
            n_targets=1 + len(source_cols),
            lookback=lookback,
            horizon=horizon,
            feature_names=feature_cols,
            target_names=[target_col] + source_cols,
            source_names=source_cols
        )
        
        logger.info(f"TimeSeriesDataset created: {len(self)} samples, "
                   f"{self.spec.n_features} features, horizon={horizon}")
    
    def _create_sequences(self):
        """Create windowed sequences from data."""
        
        n_samples = len(self.df) - self.lookback - self.horizon + 1
        
        if n_samples <= 0:
            raise ValueError(
                f"Not enough data for lookback={self.lookback}, horizon={self.horizon}. "
                f"Have {len(self.df)} samples, need at least {self.lookback + self.horizon}"
            )
        
        # Pre-allocate arrays
        self.X = np.zeros(
            (n_samples, self.lookback, len(self.feature_cols)),
            dtype=np.float32
        )
        self.Y = np.zeros(
            (n_samples, self.horizon),
            dtype=np.float32
        )
        
        # Fill sequences
        for i in range(n_samples):
            self.X[i] = self.features_scaled[i:i + self.lookback]
            self.Y[i] = self.targets_scaled[i + self.lookback:i + self.lookback + self.horizon]
        
        # Create source targets if available
        if self.sources is not None:
            n_sources = len(self.source_cols)
            self.Y_sources = np.zeros(
                (n_samples, self.horizon, n_sources),
                dtype=np.float32
            )
            
            for i in range(n_samples):
                self.Y_sources[i] = self.sources[i + self.lookback:i + self.lookback + self.horizon]
        else:
            self.Y_sources = None
        
        # Make contiguous
        self.X = np.ascontiguousarray(self.X)
        self.Y = np.ascontiguousarray(self.Y)
        
        logger.debug(f"Created {n_samples} sequences")
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.Y[idx]
    
    def get_batch(
        self,
        indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of samples."""
        return self.X[indices], self.Y[indices]
    
    def inverse_transform(self, scaled_predictions: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original scale."""
        return self.scaler.inverse_transform_targets(scaled_predictions)
    
    def to_tf_dataset(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
        buffer_size: int = 10000,
        prefetch: bool = True,
        include_sources: bool = False
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            buffer_size: Shuffle buffer size
            prefetch: Whether to prefetch batches
            include_sources: Whether to include source predictions in output
        
        Returns:
            tf.data.Dataset
        """
        if include_sources and self.Y_sources is not None:
            # Multi-output: total load + source-wise
            dataset = tf.data.Dataset.from_tensor_slices((
                self.X,
                {
                    'total_load': self.Y,
                    'sources': self.Y_sources
                }
            ))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(buffer_size, len(self)))
        
        dataset = dataset.batch(batch_size)
        
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def create_datasets(
    df: pd.DataFrame,
    target_col: str = 'Load_MW',
    lookback: int = 168,
    horizon: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 256
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, DataSpec, ScalerManager]:
    """
    Create train, validation, and test datasets.
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        lookback: Historical window size
        horizon: Forecast horizon
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        batch_size: Batch size
    
    Returns:
        Tuple of (train_ds, val_ds, test_ds, spec, scaler)
    """
    logger.info("Creating datasets...")
    
    # Split data temporally
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"  Split: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    
    # Determine feature columns
    feature_cols = [c for c in df.columns if c != target_col]
    source_cols = [c for c in df.columns if c.startswith('est_') and c.endswith('_mw')]
    
    # Create training dataset (fit scalers)
    train_dataset = TimeSeriesDataset(
        train_df,
        target_col=target_col,
        feature_cols=feature_cols,
        source_cols=source_cols,
        lookback=lookback,
        horizon=horizon,
        fit_scalers=True
    )
    
    # Create validation dataset (use training scalers)
    val_dataset = TimeSeriesDataset(
        val_df,
        target_col=target_col,
        feature_cols=feature_cols,
        source_cols=source_cols,
        lookback=lookback,
        horizon=horizon,
        scaler_manager=train_dataset.scaler
    )
    
    # Create test dataset (use training scalers)
    test_dataset = TimeSeriesDataset(
        test_df,
        target_col=target_col,
        feature_cols=feature_cols,
        source_cols=source_cols,
        lookback=lookback,
        horizon=horizon,
        scaler_manager=train_dataset.scaler
    )
    
    # Convert to TensorFlow datasets
    include_sources = len(source_cols) > 0
    
    train_ds = train_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle=True,
        include_sources=include_sources
    )
    
    val_ds = val_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle=False,
        include_sources=include_sources
    )
    
    test_ds = test_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle=False,
        include_sources=include_sources
    )
    
    logger.info(f"  Created TF datasets with batch_size={batch_size}")
    
    return train_ds, val_ds, test_ds, train_dataset.spec, train_dataset.scaler


def create_tf_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Simple function to create TF dataset from numpy arrays.
    
    Args:
        X: Input features (samples, lookback, features)
        Y: Target values (samples, horizon)
        batch_size: Batch size
        shuffle: Whether to shuffle
    
    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, len(X)))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


class DataPipeline:
    """
    End-to-end data pipeline for electricity forecasting.
    
    Combines downloading, preprocessing, feature engineering,
    and dataset creation into a single interface.
    """
    
    def __init__(
        self,
        lookback: int = 168,
        horizon: int = 24,
        is_daily: bool = False,
        batch_size: int = 256
    ):
        """
        Initialize data pipeline.
        
        Args:
            lookback: Historical window size
            horizon: Forecast horizon
            is_daily: Whether to use daily data
            batch_size: Batch size for datasets
        """
        self.lookback = lookback
        self.horizon = horizon
        self.is_daily = is_daily
        self.batch_size = batch_size
        
        self.df_raw = None
        self.df_processed = None
        self.df_featured = None
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.spec = None
        self.scaler = None
        
        logger.info(f"DataPipeline initialized (lookback={lookback}, horizon={horizon})")
    
    def load_data(self, force_download: bool = False) -> pd.DataFrame:
        """Load raw data."""
        from data.downloader import download_all_data
        
        self.df_raw = download_all_data(force_download=force_download)
        return self.df_raw
    
    def preprocess(self, resample_freq: Optional[str] = None) -> pd.DataFrame:
        """Preprocess data."""
        from data.preprocessor import preprocess_data
        
        if self.df_raw is None:
            self.load_data()
        
        if resample_freq is None:
            resample_freq = 'D' if self.is_daily else None
        
        self.df_processed, stats = preprocess_data(
            self.df_raw,
            resample_freq=resample_freq
        )
        
        logger.info(f"Preprocessing: {stats}")
        
        return self.df_processed
    
    def engineer_features(self) -> pd.DataFrame:
        """Create features."""
        from data.feature_engineer import create_features
        
        if self.df_processed is None:
            self.preprocess()
        
        self.df_featured, engineer = create_features(
            self.df_processed,
            is_daily=self.is_daily
        )
        
        return self.df_featured
    
    def create_datasets(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create TensorFlow datasets."""
        
        if self.df_featured is None:
            self.engineer_features()
        
        train_ds, val_ds, test_ds, spec, scaler = create_datasets(
            self.df_featured,
            lookback=self.lookback,
            horizon=self.horizon,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            batch_size=self.batch_size
        )
        
        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds
        self.spec = spec
        self.scaler = scaler
        
        return train_ds, val_ds, test_ds
    
    def run(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, DataSpec]:
        """Run complete pipeline."""
        
        logger.info("Running complete data pipeline...")
        
        self.load_data()
        self.preprocess()
        self.engineer_features()
        train_ds, val_ds, test_ds = self.create_datasets()
        
        logger.info("Data pipeline complete")
        
        return train_ds, val_ds, test_ds, self.spec
    
    def save(self, directory: Path):
        """Save pipeline artifacts."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        if self.spec:
            self.spec.save(directory / 'data_spec.pkl')
        
        if self.scaler:
            self.scaler.save(directory / 'scalers.pkl')
        
        if self.df_featured is not None:
            self.df_featured.to_pickle(directory / 'featured_data.pkl')
        
        logger.info(f"Pipeline saved to {directory}")
    
    @classmethod
    def load(cls, directory: Path) -> 'DataPipeline':
        """Load pipeline from artifacts."""
        directory = Path(directory)
        
        pipeline = cls()
        
        if (directory / 'data_spec.pkl').exists():
            pipeline.spec = DataSpec.load(directory / 'data_spec.pkl')
            pipeline.lookback = pipeline.spec.lookback
            pipeline.horizon = pipeline.spec.horizon
        
        if (directory / 'scalers.pkl').exists():
            pipeline.scaler = ScalerManager.load(directory / 'scalers.pkl')
        
        if (directory / 'featured_data.pkl').exists():
            pipeline.df_featured = pd.read_pickle(directory / 'featured_data.pkl')
        
        logger.info(f"Pipeline loaded from {directory}")
        
        return pipeline