"""
================================================================================
DATA DOWNLOADER
================================================================================
Handles downloading, caching, and loading of electricity consumption data.
Supports multiple data sources and automatic cache management.
"""

import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import io

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from config import get_settings
from utils import get_logger, Timer

logger = get_logger(__name__)


@dataclass
class DataSource:
    """Configuration for a single data source."""
    name: str
    url: str
    datetime_col: str = "Datetime"
    value_col: str = None  # Will use second column if None
    timezone: str = "America/New_York"
    
    def __post_init__(self):
        if self.value_col is None:
            self.value_col = f"{self.name}_MW"


# PJM Interconnection data sources
PJM_SOURCES = [
    DataSource(
        name="PJME",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJME_hourly.csv"
    ),
    DataSource(
        name="AEP",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/AEP_hourly.csv"
    ),
    DataSource(
        name="COMED",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/COMED_hourly.csv"
    ),
    DataSource(
        name="DAYTON",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/DAYTON_hourly.csv"
    ),
    DataSource(
        name="DEOK",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/DEOK_hourly.csv"
    ),
    DataSource(
        name="DOM",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/DOM_hourly.csv"
    ),
    DataSource(
        name="DUQ",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/DUQ_hourly.csv"
    ),
    DataSource(
        name="EKPC",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/EKPC_hourly.csv"
    ),
    DataSource(
        name="FE",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/FE_hourly.csv"
    ),
    DataSource(
        name="NI",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/NI_hourly.csv"
    ),
    DataSource(
        name="PJMW",
        url="https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJM_Load_hourly.csv"
    ),
]


class DataDownloader:
    """
    Handles downloading and caching of electricity data.
    
    Features:
    - Automatic caching with expiry
    - Progress bars for downloads
    - Multiple data source support
    - Robust error handling
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        cache_expiry_days: int = 7,
        timeout: int = 30
    ):
        """
        Initialize the downloader.
        
        Args:
            cache_dir: Directory for cached data
            use_cache: Whether to use caching
            cache_expiry_days: Days before cache expires
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        
        self.cache_dir = Path(cache_dir) if cache_dir else settings.paths.data_dir
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        self.timeout = timeout
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataDownloader initialized (cache: {self.cache_dir})")
    
    def _get_cache_path(self, source_name: str) -> Path:
        """Get cache file path for a data source."""
        return self.cache_dir / f"{source_name.lower()}_cache.pkl"
    
    def _get_combined_cache_path(self) -> Path:
        """Get cache file path for combined data."""
        return self.cache_dir / "combined_data_cache.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not cache_path.exists():
            return False
        
        # Check modification time
        mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = mod_time + timedelta(days=self.cache_expiry_days)
        
        return datetime.now() < expiry_time
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path):
        """Save data to cache."""
        cache_data = {
            'data': data,
            'timestamp': datetime.now(),
            'version': '1.0'
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.debug(f"Saved to cache: {cache_path}")
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            logger.debug(f"Loaded from cache: {cache_path}")
            return cache_data['data']
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def download_source(
        self,
        source: DataSource,
        force_download: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Download data from a single source.
        
        Args:
            source: DataSource configuration
            force_download: Skip cache and force download
        
        Returns:
            DataFrame with datetime index and load column
        """
        cache_path = self._get_cache_path(source.name)
        
        # Check cache
        if self.use_cache and not force_download and self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                logger.info(f"  ✓ {source.name}: Loaded from cache ({len(cached_data):,} records)")
                return cached_data
        
        # Download
        try:
            logger.info(f"  ↓ {source.name}: Downloading...")
            
            response = requests.get(source.url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(response.text))
            
            # Standardize columns
            if source.datetime_col in df.columns:
                df['Datetime'] = pd.to_datetime(df[source.datetime_col])
            else:
                df['Datetime'] = pd.to_datetime(df.iloc[:, 0])
            
            # Get load column
            if len(df.columns) >= 2:
                load_col = df.columns[1] if df.columns[1] != 'Datetime' else df.columns[0]
                df['Load_MW'] = pd.to_numeric(df[load_col], errors='coerce')
            
            # Clean up
            df = df[['Datetime', 'Load_MW']].copy()
            df = df.dropna()
            df = df.set_index('Datetime')
            df = df.sort_index()
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Save to cache
            if self.use_cache:
                self._save_to_cache(df, cache_path)
            
            logger.info(f"  ✓ {source.name}: Downloaded ({len(df):,} records)")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"  ✗ {source.name}: Network error - {e}")
            return None
        except Exception as e:
            logger.error(f"  ✗ {source.name}: Error - {e}")
            return None
    
    def download_all(
        self,
        sources: Optional[List[DataSource]] = None,
        combine: bool = True,
        force_download: bool = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Download data from all sources.
        
        Args:
            sources: List of DataSource configs (defaults to PJM_SOURCES)
            combine: Whether to combine all sources into one DataFrame
            force_download: Skip cache and force download
        
        Returns:
            Combined DataFrame or dict of DataFrames
        """
        if sources is None:
            sources = PJM_SOURCES[:4]  # Use first 4 by default
        
        # Check combined cache first
        if combine and self.use_cache and not force_download:
            combined_cache = self._get_combined_cache_path()
            if self._is_cache_valid(combined_cache):
                cached_data = self._load_from_cache(combined_cache)
                if cached_data is not None:
                    logger.info(f"Loaded combined data from cache ({len(cached_data):,} records)")
                    return cached_data
        
        # Download individual sources
        all_data = {}
        
        logger.info(f"Downloading {len(sources)} data sources...")
        
        for source in tqdm(sources, desc="Downloading", disable=True):
            df = self.download_source(source, force_download)
            if df is not None:
                all_data[source.name] = df
        
        if not all_data:
            raise ValueError("Failed to download any data sources")
        
        logger.info(f"Successfully downloaded {len(all_data)} sources")
        
        if combine:
            # Combine all sources by summing load
            combined = pd.concat(all_data.values(), axis=0)
            combined = combined.groupby(combined.index).agg({'Load_MW': 'sum'})
            combined = combined.sort_index()
            
            # Remove any remaining duplicates
            combined = combined[~combined.index.duplicated(keep='first')]
            
            # Save combined cache
            if self.use_cache:
                self._save_to_cache(combined, self._get_combined_cache_path())
            
            logger.info(f"Combined data: {len(combined):,} records "
                       f"({combined.index[0].date()} to {combined.index[-1].date()})")
            
            return combined
        
        return all_data
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Get information about the downloaded data.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary with data statistics
        """
        return {
            'n_records': len(df),
            'start_date': df.index[0].isoformat(),
            'end_date': df.index[-1].isoformat(),
            'date_range_days': (df.index[-1] - df.index[0]).days,
            'mean_load_mw': float(df['Load_MW'].mean()),
            'max_load_mw': float(df['Load_MW'].max()),
            'min_load_mw': float(df['Load_MW'].min()),
            'std_load_mw': float(df['Load_MW'].std()),
            'missing_hours': int(self._count_missing_hours(df)),
            'data_quality': self._assess_quality(df)
        }
    
    def _count_missing_hours(self, df: pd.DataFrame) -> int:
        """Count missing hours in the data."""
        expected_hours = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            freq='H'
        )
        return len(expected_hours) - len(df)
    
    def _assess_quality(self, df: pd.DataFrame) -> str:
        """Assess data quality."""
        missing_pct = self._count_missing_hours(df) / len(df) * 100
        
        if missing_pct < 1:
            return "excellent"
        elif missing_pct < 5:
            return "good"
        elif missing_pct < 10:
            return "fair"
        else:
            return "poor"
    
    def clear_cache(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*_cache.pkl"):
            cache_file.unlink()
            logger.info(f"Deleted: {cache_file}")
        
        logger.info("Cache cleared")


def download_all_data(
    force_download: bool = False,
    sources: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function to download all data.
    
    Args:
        force_download: Skip cache and force download
        sources: List of source names to download
    
    Returns:
        Combined DataFrame with all data
    """
    downloader = DataDownloader()
    
    if sources:
        source_configs = [s for s in PJM_SOURCES if s.name in sources]
    else:
        source_configs = None
    
    return downloader.download_all(sources=source_configs, force_download=force_download)