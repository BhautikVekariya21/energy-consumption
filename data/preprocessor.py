"""
================================================================================
DATA PREPROCESSOR
================================================================================
Handles data cleaning, outlier detection, missing value imputation,
and data quality checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from scipy import stats

from utils import get_logger

logger = get_logger(__name__)


class OutlierMethod(Enum):
    """Methods for outlier detection."""
    IQR = "iqr"
    ZSCORE = "zscore"
    MAD = "mad"  # Median Absolute Deviation
    ISOLATION_FOREST = "isolation_forest"


class ImputationMethod(Enum):
    """Methods for missing value imputation."""
    LINEAR = "linear"
    TIME = "time"
    SEASONAL = "seasonal"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    MEAN = "mean"
    MEDIAN = "median"


@dataclass
class PreprocessingStats:
    """Statistics from preprocessing."""
    original_rows: int
    final_rows: int
    missing_before: int
    missing_after: int
    outliers_detected: int
    outliers_replaced: int
    duplicates_removed: int
    
    def to_dict(self) -> Dict:
        return {
            'original_rows': self.original_rows,
            'final_rows': self.final_rows,
            'missing_before': self.missing_before,
            'missing_after': self.missing_after,
            'outliers_detected': self.outliers_detected,
            'outliers_replaced': self.outliers_replaced,
            'duplicates_removed': self.duplicates_removed,
            'data_retention_pct': round(self.final_rows / self.original_rows * 100, 2)
        }


class DataPreprocessor:
    """
    Comprehensive data preprocessing for time series.
    
    Features:
    - Missing value detection and imputation
    - Outlier detection and handling
    - Data quality checks
    - Temporal consistency validation
    """
    
    def __init__(
        self,
        outlier_method: OutlierMethod = OutlierMethod.IQR,
        outlier_threshold: float = 3.0,
        imputation_method: ImputationMethod = ImputationMethod.TIME,
        min_valid_load: float = 0.0,
        max_valid_load: Optional[float] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            outlier_method: Method for detecting outliers
            outlier_threshold: Threshold for outlier detection
            imputation_method: Method for imputing missing values
            min_valid_load: Minimum valid load value
            max_valid_load: Maximum valid load value (None for auto)
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.imputation_method = imputation_method
        self.min_valid_load = min_valid_load
        self.max_valid_load = max_valid_load
        
        self.stats: Optional[PreprocessingStats] = None
        
        logger.info(f"DataPreprocessor initialized "
                   f"(outlier: {outlier_method.value}, imputation: {imputation_method.value})")
    
    def preprocess(
        self,
        df: pd.DataFrame,
        target_col: str = 'Load_MW',
        resample_freq: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.
        
        Args:
            df: Input DataFrame with datetime index
            target_col: Name of target column
            resample_freq: Optional frequency to resample to ('H', 'D', etc.)
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")
        
        original_rows = len(df)
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Datetime' in df.columns:
                df = df.set_index('Datetime')
            else:
                raise ValueError("DataFrame must have datetime index or 'Datetime' column")
        
        # Sort by time
        df = df.sort_index()
        
        # Remove duplicates
        duplicates = df.index.duplicated()
        n_duplicates = duplicates.sum()
        df = df[~duplicates]
        
        if n_duplicates > 0:
            logger.info(f"  Removed {n_duplicates} duplicate timestamps")
        
        # Count initial missing
        missing_before = df[target_col].isna().sum()
        
        # Handle invalid values (negative, zero if inappropriate)
        invalid_mask = df[target_col] < self.min_valid_load
        if self.max_valid_load:
            invalid_mask |= df[target_col] > self.max_valid_load
        
        df.loc[invalid_mask, target_col] = np.nan
        
        # Detect outliers
        outlier_mask = self._detect_outliers(df[target_col])
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            logger.info(f"  Detected {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")
            df.loc[outlier_mask, target_col] = np.nan
        
        # Fill missing timestamps
        df = self._fill_missing_timestamps(df)
        
        # Impute missing values
        df = self._impute_missing(df, target_col)
        
        # Resample if requested
        if resample_freq:
            df = self._resample(df, target_col, resample_freq)
        
        # Final quality check
        missing_after = df[target_col].isna().sum()
        
        # Store stats
        self.stats = PreprocessingStats(
            original_rows=original_rows,
            final_rows=len(df),
            missing_before=missing_before + invalid_mask.sum(),
            missing_after=missing_after,
            outliers_detected=n_outliers,
            outliers_replaced=n_outliers,
            duplicates_removed=n_duplicates
        )
        
        logger.info(f"  Preprocessing complete: {original_rows:,} → {len(df):,} rows")
        logger.info(f"  Missing values: {missing_before} → {missing_after}")
        
        return df
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """
        Detect outliers in the series.
        
        Args:
            series: Data series to analyze
        
        Returns:
            Boolean mask of outliers
        """
        valid_data = series.dropna()
        
        if self.outlier_method == OutlierMethod.IQR:
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - self.outlier_threshold * IQR
            upper = Q3 + self.outlier_threshold * IQR
            
            outliers = (series < lower) | (series > upper)
            
        elif self.outlier_method == OutlierMethod.ZSCORE:
            mean = valid_data.mean()
            std = valid_data.std()
            
            z_scores = np.abs((series - mean) / std)
            outliers = z_scores > self.outlier_threshold
            
        elif self.outlier_method == OutlierMethod.MAD:
            median = valid_data.median()
            mad = np.median(np.abs(valid_data - median))
            
            # Modified z-score
            modified_z = 0.6745 * (series - median) / (mad + 1e-10)
            outliers = np.abs(modified_z) > self.outlier_threshold
            
        else:
            # Default: no outliers
            outliers = pd.Series(False, index=series.index)
        
        # Don't mark NaN as outliers
        outliers = outliers & series.notna()
        
        return outliers
    
    def _fill_missing_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill in missing timestamps to create continuous series.
        
        Args:
            df: DataFrame with datetime index
        
        Returns:
            DataFrame with all timestamps filled
        """
        # Determine frequency
        freq = pd.infer_freq(df.index)
        if freq is None:
            # Try to detect from most common difference
            diffs = df.index.to_series().diff().dropna()
            most_common = diffs.mode()[0]
            
            if most_common <= pd.Timedelta(hours=1):
                freq = 'H'
            elif most_common <= pd.Timedelta(days=1):
                freq = 'D'
            else:
                freq = 'H'  # Default to hourly
        
        # Create complete index
        full_index = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            freq=freq
        )
        
        # Reindex
        df = df.reindex(full_index)
        
        n_added = len(full_index) - len(df.dropna())
        if n_added > 0:
            logger.debug(f"  Added {n_added} missing timestamps")
        
        return df
    
    def _impute_missing(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """
        Impute missing values.
        
        Args:
            df: DataFrame with missing values
            target_col: Column to impute
        
        Returns:
            DataFrame with imputed values
        """
        if self.imputation_method == ImputationMethod.LINEAR:
            df[target_col] = df[target_col].interpolate(method='linear')
            
        elif self.imputation_method == ImputationMethod.TIME:
            df[target_col] = df[target_col].interpolate(method='time')
            
        elif self.imputation_method == ImputationMethod.SEASONAL:
            df = self._seasonal_impute(df, target_col)
            
        elif self.imputation_method == ImputationMethod.FORWARD_FILL:
            df[target_col] = df[target_col].fillna(method='ffill')
            
        elif self.imputation_method == ImputationMethod.BACKWARD_FILL:
            df[target_col] = df[target_col].fillna(method='bfill')
            
        elif self.imputation_method == ImputationMethod.MEAN:
            df[target_col] = df[target_col].fillna(df[target_col].mean())
            
        elif self.imputation_method == ImputationMethod.MEDIAN:
            df[target_col] = df[target_col].fillna(df[target_col].median())
        
        # Handle any remaining NaN at edges
        df[target_col] = df[target_col].fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def _seasonal_impute(
        self,
        df: pd.DataFrame,
        target_col: str,
        period: int = 168  # Weekly for hourly data
    ) -> pd.DataFrame:
        """
        Impute using seasonal patterns.
        
        Args:
            df: DataFrame with missing values
            target_col: Column to impute
            period: Seasonal period
        
        Returns:
            DataFrame with imputed values
        """
        values = df[target_col].values.copy()
        missing_mask = np.isnan(values)
        
        for i in np.where(missing_mask)[0]:
            # Try to find value from same position in previous periods
            candidates = []
            
            for offset in range(1, 5):  # Look back up to 4 periods
                prev_idx = i - offset * period
                next_idx = i + offset * period
                
                if prev_idx >= 0 and not np.isnan(values[prev_idx]):
                    candidates.append(values[prev_idx])
                if next_idx < len(values) and not np.isnan(values[next_idx]):
                    candidates.append(values[next_idx])
            
            if candidates:
                values[i] = np.mean(candidates)
        
        df[target_col] = values
        return df
    
    def _resample(
        self,
        df: pd.DataFrame,
        target_col: str,
        freq: str
    ) -> pd.DataFrame:
        """
        Resample data to different frequency.
        
        Args:
            df: DataFrame to resample
            target_col: Target column
            freq: Target frequency ('H', 'D', 'W', 'M')
        
        Returns:
            Resampled DataFrame
        """
        # Determine aggregation method based on frequency
        if freq in ['D', 'W', 'M', 'Q', 'Y']:
            # For daily and longer, compute multiple aggregations
            resampled = df.resample(freq).agg({
                target_col: ['mean', 'min', 'max', 'std', 'sum']
            })
            resampled.columns = [
                target_col, 
                f'{target_col}_min', 
                f'{target_col}_max',
                f'{target_col}_std',
                f'{target_col}_sum'
            ]
        else:
            # For hourly or sub-hourly, just take mean
            resampled = df.resample(freq).mean()
        
        logger.info(f"  Resampled to {freq}: {len(df):,} → {len(resampled):,} rows")
        
        return resampled
    
    def get_stats(self) -> Optional[Dict]:
        """Get preprocessing statistics."""
        if self.stats:
            return self.stats.to_dict()
        return None
    
    def validate_data(self, df: pd.DataFrame, target_col: str = 'Load_MW') -> Dict:
        """
        Validate preprocessed data quality.
        
        Args:
            df: DataFrame to validate
            target_col: Target column
        
        Returns:
            Validation report
        """
        report = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for missing values
        missing = df[target_col].isna().sum()
        if missing > 0:
            report['is_valid'] = False
            report['issues'].append(f"Contains {missing} missing values")
        
        # Check for negative values
        negatives = (df[target_col] < 0).sum()
        if negatives > 0:
            report['is_valid'] = False
            report['issues'].append(f"Contains {negatives} negative values")
        
        # Check for zeros (warning only)
        zeros = (df[target_col] == 0).sum()
        if zeros > 0:
            report['warnings'].append(f"Contains {zeros} zero values")
        
        # Check continuity
        if isinstance(df.index, pd.DatetimeIndex):
            diffs = df.index.to_series().diff().dropna()
            expected_diff = diffs.mode()[0]
            gaps = (diffs > expected_diff * 1.5).sum()
            
            if gaps > 0:
                report['warnings'].append(f"Contains {gaps} time gaps")
        
        # Check data range
        mean = df[target_col].mean()
        std = df[target_col].std()
        cv = std / mean if mean > 0 else 0
        
        if cv > 1.0:
            report['warnings'].append(f"High coefficient of variation: {cv:.2f}")
        
        report['stats'] = {
            'mean': float(mean),
            'std': float(std),
            'cv': float(cv),
            'min': float(df[target_col].min()),
            'max': float(df[target_col].max())
        }
        
        return report


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = 'Load_MW',
    resample_freq: Optional[str] = None,
    outlier_method: str = 'iqr',
    imputation_method: str = 'time'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function for preprocessing.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        resample_freq: Optional resample frequency
        outlier_method: Outlier detection method
        imputation_method: Missing value imputation method
    
    Returns:
        Tuple of (preprocessed DataFrame, stats dictionary)
    """
    preprocessor = DataPreprocessor(
        outlier_method=OutlierMethod(outlier_method),
        imputation_method=ImputationMethod(imputation_method)
    )
    
    processed_df = preprocessor.preprocess(
        df,
        target_col=target_col,
        resample_freq=resample_freq
    )
    
    stats = preprocessor.get_stats()
    
    return processed_df, stats