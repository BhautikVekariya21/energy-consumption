"""
================================================================================
FEATURE ENGINEERING
================================================================================
Comprehensive feature engineering for electricity load forecasting.
Includes temporal, lag, rolling, seasonal, and source-specific features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import holidays
from scipy import stats as scipy_stats

from config import get_settings
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Lag features
    lag_hours: List[int] = field(default_factory=lambda: [
        1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720
    ])
    lag_days: List[int] = field(default_factory=lambda: [
        1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 365
    ])
    
    # Rolling windows
    rolling_windows_hours: List[int] = field(default_factory=lambda: [
        6, 12, 24, 48, 168
    ])
    rolling_windows_days: List[int] = field(default_factory=lambda: [
        7, 14, 30, 60, 90, 180, 365
    ])
    
    # Cyclical features
    use_cyclical: bool = True
    
    # Holiday features
    use_holidays: bool = True
    holiday_country: str = 'US'
    
    # Fourier features
    use_fourier: bool = True
    fourier_periods: List[int] = field(default_factory=lambda: [24, 168, 365])
    fourier_orders: List[int] = field(default_factory=lambda: [3, 2, 2])
    
    # Source decomposition features
    use_source_features: bool = True


class FeatureEngineer:
    """
    Comprehensive feature engineering for time series forecasting.
    
    Creates features for:
    - Temporal patterns (hour, day, month, etc.)
    - Lag values
    - Rolling statistics
    - Seasonal patterns
    - Holiday effects
    - Source-wise generation estimates
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        self.settings = get_settings()
        
        # Holiday calendar
        if self.config.use_holidays:
            try:
                self.holiday_cal = holidays.country_holidays(
                    self.config.holiday_country
                )
            except:
                self.holiday_cal = holidays.US()
                logger.warning(f"Could not load holidays for {self.config.holiday_country}, using US")
        
        self.feature_names: List[str] = []
        self.source_features: List[str] = []
        
        logger.info("FeatureEngineer initialized")
    
    def create_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'Load_MW',
        is_daily: bool = False
    ) -> pd.DataFrame:
        """
        Create all features.
        
        Args:
            df: Input DataFrame with datetime index
            target_col: Target column name
            is_daily: Whether data is daily (vs hourly)
        
        Returns:
            DataFrame with all features
        """
        logger.info("Creating features...")
        
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index")
        
        # Create temporal features
        df = self._create_temporal_features(df, is_daily)
        
        # Create cyclical features
        if self.config.use_cyclical:
            df = self._create_cyclical_features(df, is_daily)
        
        # Create holiday features
        if self.config.use_holidays:
            df = self._create_holiday_features(df)
        
        # Create lag features
        df = self._create_lag_features(df, target_col, is_daily)
        
        # Create rolling features
        df = self._create_rolling_features(df, target_col, is_daily)
        
        # Create seasonal baseline features
        df = self._create_seasonal_features(df, target_col, is_daily)
        
        # Create Fourier features
        if self.config.use_fourier:
            df = self._create_fourier_features(df, is_daily)
        
        # Create source-wise features
        if self.config.use_source_features:
            df = self._create_source_features(df, target_col)
        
        # Create trend features
        df = self._create_trend_features(df, target_col)
        
        # Create interaction features
        df = self._create_interaction_features(df, target_col)
        
        # Handle NaN and infinities
        df = self._clean_features(df)
        
        # Store feature names (excluding target)
        self.feature_names = [c for c in df.columns if c != target_col]
        
        logger.info(f"  Created {len(self.feature_names)} features")
        
        return df
    
    def _create_temporal_features(
        self,
        df: pd.DataFrame,
        is_daily: bool
    ) -> pd.DataFrame:
        """Create basic temporal features."""
        
        idx = df.index
        
        # Time components
        if not is_daily:
            df['hour'] = idx.hour
            df['minute'] = idx.minute
        
        df['day'] = idx.day
        df['day_of_week'] = idx.dayofweek
        df['day_of_year'] = idx.dayofyear
        df['week_of_year'] = idx.isocalendar().week.astype(int)
        df['month'] = idx.month
        df['quarter'] = idx.quarter
        df['year'] = idx.year
        
        # Binary features
        df['is_weekend'] = (idx.dayofweek >= 5).astype(np.float32)
        df['is_monday'] = (idx.dayofweek == 0).astype(np.float32)
        df['is_friday'] = (idx.dayofweek == 4).astype(np.float32)
        
        # Season
        df['is_winter'] = idx.month.isin([12, 1, 2]).astype(np.float32)
        df['is_spring'] = idx.month.isin([3, 4, 5]).astype(np.float32)
        df['is_summer'] = idx.month.isin([6, 7, 8]).astype(np.float32)
        df['is_fall'] = idx.month.isin([9, 10, 11]).astype(np.float32)
        
        # Time of day categories (for hourly data)
        if not is_daily:
            df['is_night'] = ((idx.hour >= 22) | (idx.hour <= 5)).astype(np.float32)
            df['is_morning'] = ((idx.hour >= 6) & (idx.hour <= 11)).astype(np.float32)
            df['is_afternoon'] = ((idx.hour >= 12) & (idx.hour <= 17)).astype(np.float32)
            df['is_evening'] = ((idx.hour >= 18) & (idx.hour <= 21)).astype(np.float32)
            df['is_business_hour'] = (
                (idx.hour >= 9) & (idx.hour <= 17) & (idx.dayofweek < 5)
            ).astype(np.float32)
            df['is_peak_hour'] = (
                ((idx.hour >= 7) & (idx.hour <= 9)) | 
                ((idx.hour >= 17) & (idx.hour <= 20))
            ).astype(np.float32)
        
        # Month position
        df['is_month_start'] = (idx.day <= 3).astype(np.float32)
        df['is_month_end'] = (idx.day >= 28).astype(np.float32)
        df['is_quarter_start'] = idx.is_quarter_start.astype(np.float32)
        df['is_quarter_end'] = idx.is_quarter_end.astype(np.float32)
        df['is_year_start'] = idx.is_year_start.astype(np.float32)
        df['is_year_end'] = idx.is_year_end.astype(np.float32)
        
        return df
    
    def _create_cyclical_features(
        self,
        df: pd.DataFrame,
        is_daily: bool
    ) -> pd.DataFrame:
        """Create cyclical (sin/cos) encoding of temporal features."""
        
        if not is_daily:
            # Hour of day
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of month
        df['dom_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # Day of year
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Week of year
        df['woy_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['woy_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        # Month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Quarter
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        return df
    
    def _create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday-related features."""
        
        # Check if date is a holiday
        df['is_holiday'] = df.index.map(
            lambda x: 1.0 if x.date() in self.holiday_cal else 0.0
        ).astype(np.float32)
        
        # Days until next holiday
        def days_to_holiday(date, max_days=30):
            for i in range(max_days):
                check_date = date + pd.Timedelta(days=i)
                if check_date.date() in self.holiday_cal:
                    return i
            return max_days
        
        # This is slow, so we cache it
        dates = df.index.date
        unique_dates = pd.Series(dates).unique()
        
        holiday_map = {}
        for d in unique_dates:
            d_pd = pd.Timestamp(d)
            holiday_map[d] = days_to_holiday(d_pd)
        
        df['days_to_holiday'] = df.index.map(
            lambda x: holiday_map.get(x.date(), 30)
        ).astype(np.float32)
        
        # Days since last holiday
        def days_since_holiday(date, max_days=30):
            for i in range(max_days):
                check_date = date - pd.Timedelta(days=i)
                if check_date.date() in self.holiday_cal:
                    return i
            return max_days
        
        holiday_since_map = {}
        for d in unique_dates:
            d_pd = pd.Timestamp(d)
            holiday_since_map[d] = days_since_holiday(d_pd)
        
        df['days_since_holiday'] = df.index.map(
            lambda x: holiday_since_map.get(x.date(), 30)
        ).astype(np.float32)
        
        # Bridge day (day between holiday and weekend)
        df['is_bridge_day'] = (
            (df['is_holiday'].shift(1) == 1) | 
            (df['is_holiday'].shift(-1) == 1)
        ).astype(np.float32) * (1 - df['is_holiday']) * (1 - df['is_weekend'])
        
        return df
    
    def _create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_daily: bool
    ) -> pd.DataFrame:
        """Create lag features."""
        
        lags = self.config.lag_days if is_daily else self.config.lag_hours
        suffix = 'd' if is_daily else 'h'
        
        for lag in lags:
            df[f'lag_{lag}{suffix}'] = df[target_col].shift(lag)
        
        # Year-over-year lags
        if is_daily:
            df['lag_365d'] = df[target_col].shift(365)
            df['lag_364d'] = df[target_col].shift(364)  # Same day of week
            df['lag_371d'] = df[target_col].shift(371)  # Same day of week +1 week
            df['lag_730d'] = df[target_col].shift(730)  # 2 years ago
        else:
            df['lag_8760h'] = df[target_col].shift(8760)  # 1 year (365*24)
            df['lag_8736h'] = df[target_col].shift(8736)  # Same hour, same DoW, ~1 year
        
        # Same time last period
        if is_daily:
            df['same_dow_last_week'] = df[target_col].shift(7)
            df['same_dom_last_month'] = df[target_col].shift(30)
        else:
            df['same_hour_yesterday'] = df[target_col].shift(24)
            df['same_hour_last_week'] = df[target_col].shift(168)
        
        return df
    
    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_daily: bool
    ) -> pd.DataFrame:
        """Create rolling window features."""
        
        windows = self.config.rolling_windows_days if is_daily else self.config.rolling_windows_hours
        suffix = 'd' if is_daily else 'h'
        
        # Use shifted data to prevent leakage
        shifted = df[target_col].shift(1)
        
        for window in windows:
            # Basic statistics
            df[f'roll_{window}{suffix}_mean'] = shifted.rolling(window, min_periods=1).mean()
            df[f'roll_{window}{suffix}_std'] = shifted.rolling(window, min_periods=1).std()
            df[f'roll_{window}{suffix}_min'] = shifted.rolling(window, min_periods=1).min()
            df[f'roll_{window}{suffix}_max'] = shifted.rolling(window, min_periods=1).max()
            
            # Range and coefficient of variation
            df[f'roll_{window}{suffix}_range'] = (
                df[f'roll_{window}{suffix}_max'] - df[f'roll_{window}{suffix}_min']
            )
            df[f'roll_{window}{suffix}_cv'] = (
                df[f'roll_{window}{suffix}_std'] / (df[f'roll_{window}{suffix}_mean'] + 1e-10)
            )
            
            # Percentiles
            df[f'roll_{window}{suffix}_p25'] = shifted.rolling(window, min_periods=1).quantile(0.25)
            df[f'roll_{window}{suffix}_p75'] = shifted.rolling(window, min_periods=1).quantile(0.75)
            
            # Skewness (for larger windows)
            if window >= 24 or (is_daily and window >= 7):
                df[f'roll_{window}{suffix}_skew'] = shifted.rolling(window, min_periods=3).skew()
        
        # Exponential moving averages
        for span in [7, 14, 30] if is_daily else [24, 168, 720]:
            df[f'ewm_{span}_mean'] = shifted.ewm(span=span, adjust=False).mean()
        
        return df
    
    def _create_seasonal_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_daily: bool
    ) -> pd.DataFrame:
        """Create seasonal baseline features."""
        
        shifted = df[target_col].shift(1)
        
        if is_daily:
            # Day of year average
            df['seasonal_doy_mean'] = df.groupby('day_of_year')[target_col].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            
            # Day of week + month average
            df['seasonal_dow_month_mean'] = df.groupby(['day_of_week', 'month'])[target_col].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            
            # Monthly average
            df['seasonal_month_mean'] = df.groupby('month')[target_col].transform(
                lambda x: x.shift(1).expanding().mean()
            )
        else:
            # Hour of day average
            df['seasonal_hour_mean'] = df.groupby('hour')[target_col].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            
            # Hour + day of week average
            df['seasonal_hour_dow_mean'] = df.groupby(['hour', 'day_of_week'])[target_col].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            
            # Hour + month average
            df['seasonal_hour_month_mean'] = df.groupby(['hour', 'month'])[target_col].transform(
                lambda x: x.shift(1).expanding().mean()
            )
        
        # Deviation from seasonal baseline
        if 'seasonal_doy_mean' in df.columns:
            baseline = df['seasonal_doy_mean']
        elif 'seasonal_hour_dow_mean' in df.columns:
            baseline = df['seasonal_hour_dow_mean']
        else:
            baseline = df[target_col].rolling(168, min_periods=1).mean()
        
        df['deviation_from_seasonal'] = shifted - baseline
        df['ratio_to_seasonal'] = shifted / (baseline + 1e-10)
        
        return df
    
    def _create_fourier_features(
        self,
        df: pd.DataFrame,
        is_daily: bool
    ) -> pd.DataFrame:
        """Create Fourier series features for seasonality."""
        
        periods = self.config.fourier_periods
        orders = self.config.fourier_orders
        
        t = np.arange(len(df))
        
        for period, order in zip(periods, orders):
            for k in range(1, order + 1):
                df[f'fourier_sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
                df[f'fourier_cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
        
        return df
    
    def _create_source_features(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """
        Create features for source-wise generation estimation.
        
        This estimates the contribution of different energy sources
        based on typical patterns and the total load.
        """
        
        settings = get_settings()
        sources = settings.data.energy_sources
        mix = settings.data.default_generation_mix
        
        # Base load estimate (minimum load as baseload)
        df['baseload_estimate'] = df[target_col].rolling(168, min_periods=1).min()
        
        # Variable load
        df['variable_load'] = df[target_col] - df['baseload_estimate']
        
        # Source-wise estimates based on typical patterns
        for source in sources:
            fraction = mix.get(source, 0.1)
            
            if source == 'nuclear':
                # Nuclear: relatively constant baseload
                df[f'est_{source}_mw'] = df['baseload_estimate'] * fraction * 2.5
                
            elif source == 'coal':
                # Coal: baseload with some variation
                df[f'est_{source}_mw'] = (
                    df['baseload_estimate'] * fraction * 1.5 +
                    df['variable_load'] * fraction * 0.3
                )
                
            elif source == 'natural_gas':
                # Natural gas: flexible, follows demand
                df[f'est_{source}_mw'] = (
                    df['baseload_estimate'] * fraction * 0.5 +
                    df['variable_load'] * fraction * 1.5
                )
                
            elif source == 'hydro':
                # Hydro: seasonal variation
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * df['day_of_year'] / 365 - np.pi/2)
                df[f'est_{source}_mw'] = df[target_col] * fraction * seasonal_factor
                
            elif source == 'wind':
                # Wind: variable, often higher at night
                night_factor = 1 + 0.2 * df.get('is_night', 0)
                seasonal_factor = 1 + 0.2 * np.cos(2 * np.pi * df['day_of_year'] / 365)
                df[f'est_{source}_mw'] = df[target_col] * fraction * night_factor * seasonal_factor
                
            elif source == 'solar':
                # Solar: follows sun pattern
                if 'hour' in df.columns:
                    # Peak at noon
                    hour = df['hour']
                    solar_factor = np.maximum(0, np.cos((hour - 12) / 12 * np.pi))
                    # Seasonal variation
                    seasonal_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365)
                    df[f'est_{source}_mw'] = df[target_col] * fraction * solar_factor * seasonal_factor * 3
                else:
                    seasonal_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365)
                    df[f'est_{source}_mw'] = df[target_col] * fraction * seasonal_factor
                
            else:
                # Other sources
                df[f'est_{source}_mw'] = df[target_col] * fraction
            
            # Clip to reasonable values
            df[f'est_{source}_mw'] = df[f'est_{source}_mw'].clip(lower=0)
            
            # Store source feature names
            self.source_features.append(f'est_{source}_mw')
        
        # Normalize to sum to total (approximately)
        source_cols = [f'est_{s}_mw' for s in sources]
        total_sources = df[source_cols].sum(axis=1)
        scale_factor = df[target_col] / (total_sources + 1e-10)
        
        for col in source_cols:
            df[f'{col}_scaled'] = df[col] * scale_factor
            df[f'{col}_pct'] = df[col] / (df[target_col] + 1e-10) * 100
        
        return df
    
    def _create_trend_features(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """Create trend-related features."""
        
        shifted = df[target_col].shift(1)
        
        # Linear trend
        df['year_trend'] = (df['year'] - df['year'].min()) / max(1, df['year'].max() - df['year'].min())
        
        # Differences
        df['diff_1'] = shifted.diff(1)
        df['diff_7'] = shifted.diff(7) if len(df) > 7 else shifted.diff(1)
        df['diff_30'] = shifted.diff(30) if len(df) > 30 else shifted.diff(1)
        
        # Percent changes
        df['pct_change_1'] = shifted.pct_change(1)
        df['pct_change_7'] = shifted.pct_change(7) if len(df) > 7 else shifted.pct_change(1)
        
        # Short vs long term trend
        short_ma = shifted.rolling(7, min_periods=1).mean()
        long_ma = shifted.rolling(30, min_periods=1).mean()
        
        df['trend_short_vs_long'] = short_ma - long_ma
        df['trend_ratio'] = short_ma / (long_ma + 1e-10)
        
        # Momentum indicators
        df['momentum_7'] = shifted - shifted.shift(7)
        df['momentum_30'] = shifted - shifted.shift(30)
        
        # Rate of change
        df['roc_7'] = (shifted - shifted.shift(7)) / (shifted.shift(7) + 1e-10)
        
        return df
    
    def _create_interaction_features(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """Create interaction features."""
        
        # Weekend × season
        df['weekend_summer'] = df['is_weekend'] * df['is_summer']
        df['weekend_winter'] = df['is_weekend'] * df['is_winter']
        
        # Holiday × day of week
        if 'is_holiday' in df.columns:
            df['holiday_monday'] = df['is_holiday'] * df['is_monday']
            df['holiday_friday'] = df['is_holiday'] * df['is_friday']
        
        # Time interactions (for hourly data)
        if 'hour' in df.columns:
            df['hour_weekend'] = df['hour'] * df['is_weekend']
            df['hour_summer'] = df['hour'] * df['is_summer']
            df['hour_winter'] = df['hour'] * df['is_winter']
        
        # Load level indicators
        if target_col in df.columns:
            load_mean = df[target_col].mean()
            load_std = df[target_col].std()
            
            df['is_high_load'] = (df[target_col] > load_mean + load_std).astype(np.float32)
            df['is_low_load'] = (df[target_col] < load_mean - load_std).astype(np.float32)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features (handle NaN, inf, etc.)."""
        
        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with forward/backward fill, then 0
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Clip extreme values
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['year', 'month', 'day', 'hour', 'minute']:
                q01 = df[col].quantile(0.001)
                q99 = df[col].quantile(0.999)
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        # Convert to float32 for efficiency
        float_cols = df.select_dtypes(include=[np.float64]).columns
        df[float_cols] = df[float_cols].astype(np.float32)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def get_source_features(self) -> List[str]:
        """Get list of source-related feature names."""
        return self.source_features
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by type."""
        
        groups = {
            'temporal': [],
            'cyclical': [],
            'holiday': [],
            'lag': [],
            'rolling': [],
            'seasonal': [],
            'fourier': [],
            'source': [],
            'trend': [],
            'interaction': [],
            'other': []
        }
        
        for feat in self.feature_names:
            if feat.startswith(('hour', 'day', 'week', 'month', 'quarter', 'year', 'is_')):
                groups['temporal'].append(feat)
            elif feat.endswith(('_sin', '_cos')):
                groups['cyclical'].append(feat)
            elif 'holiday' in feat:
                groups['holiday'].append(feat)
            elif feat.startswith('lag_'):
                groups['lag'].append(feat)
            elif feat.startswith(('roll_', 'ewm_')):
                groups['rolling'].append(feat)
            elif feat.startswith('seasonal_'):
                groups['seasonal'].append(feat)
            elif feat.startswith('fourier_'):
                groups['fourier'].append(feat)
            elif feat.startswith('est_') or feat in self.source_features:
                groups['source'].append(feat)
            elif feat.startswith(('trend_', 'diff_', 'pct_', 'momentum_', 'roc_')):
                groups['trend'].append(feat)
            elif '_' in feat and any(x in feat for x in ['weekend', 'summer', 'winter']):
                groups['interaction'].append(feat)
            else:
                groups['other'].append(feat)
        
        return groups


def create_features(
    df: pd.DataFrame,
    target_col: str = 'Load_MW',
    is_daily: bool = False,
    config: Optional[FeatureConfig] = None
) -> Tuple[pd.DataFrame, FeatureEngineer]:
    """
    Convenience function for feature engineering.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        is_daily: Whether data is daily
        config: Feature configuration
    
    Returns:
        Tuple of (featured DataFrame, FeatureEngineer instance)
    """
    engineer = FeatureEngineer(config)
    featured_df = engineer.create_features(df, target_col, is_daily)
    
    return featured_df, engineer