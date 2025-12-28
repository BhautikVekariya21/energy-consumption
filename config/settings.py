"""
================================================================================
GLOBAL SETTINGS AND CONFIGURATION
================================================================================
Industry-grade configuration management for electricity forecasting system.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import yaml


@dataclass
class DataSettings:
    """Data-related settings."""
    
    # Data sources
    data_sources: List[str] = field(default_factory=lambda: [
        'PJME', 'AEP', 'COMED', 'DAYTON'
    ])
    
    # Energy source types for generation mix
    energy_sources: List[str] = field(default_factory=lambda: [
        'nuclear', 'coal', 'natural_gas', 'hydro', 'wind', 'solar', 'other'
    ])
    
    # Typical generation mix (US average percentages)
    default_generation_mix: Dict[str, float] = field(default_factory=lambda: {
        'nuclear': 0.19,
        'coal': 0.20,
        'natural_gas': 0.40,
        'hydro': 0.06,
        'wind': 0.10,
        'solar': 0.04,
        'other': 0.01
    })
    
    # Data URLs
    base_url: str = "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data"
    
    # Cache settings
    cache_dir: str = "processed_data"
    use_cache: bool = True
    cache_expiry_days: int = 7


@dataclass
class TrainingSettings:
    """Training-related settings."""
    
    # Batch sizes
    batch_size: int = 256
    validation_batch_size: int = 512
    
    # Training parameters
    max_epochs: int = 200
    early_stopping_patience: int = 25
    reduce_lr_patience: int = 10
    min_lr: float = 1e-7
    
    # Optimizer settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Reproducibility
    seed: int = 42


@dataclass
class ForecastHorizon:
    """Single forecast horizon configuration."""
    name: str
    hours: int
    lookback_hours: int
    aggregation: str = 'hourly'  # 'hourly' or 'daily'
    
    @property
    def days(self) -> int:
        return self.hours // 24
    
    @property
    def lookback_days(self) -> int:
        return self.lookback_hours // 24


@dataclass
class ForecastSettings:
    """Forecast-related settings."""
    
    horizons: List[ForecastHorizon] = field(default_factory=lambda: [
        ForecastHorizon("1_day", 24, 168, "hourly"),
        ForecastHorizon("2_days", 48, 336, "hourly"),
        ForecastHorizon("1_week", 168, 336, "hourly"),
        ForecastHorizon("1_month", 720, 2160, "daily"),
        ForecastHorizon("1_quarter", 2160, 2880, "daily"),
        ForecastHorizon("6_months", 4380, 4380, "daily"),
        ForecastHorizon("1_year", 8760, 8760, "daily"),
    ])
    
    # Train/Val/Test split
    test_ratio: float = 0.15
    val_ratio: float = 0.10
    
    # Minimum training samples
    min_train_samples: int = 1000


@dataclass
class PathSettings:
    """Path-related settings."""
    
    # Base directories
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "processed_data"
    
    @property
    def model_dir(self) -> Path:
        return self.base_dir / "saved_models"
    
    @property
    def plot_dir(self) -> Path:
        return self.base_dir / "plots"
    
    @property
    def log_dir(self) -> Path:
        return self.base_dir / "logs"
    
    def create_directories(self):
        """Create all necessary directories."""
        for dir_path in [self.data_dir, self.model_dir, self.plot_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Master settings container."""
    
    data: DataSettings = field(default_factory=DataSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    forecast: ForecastSettings = field(default_factory=ForecastSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    
    # Project metadata
    project_name: str = "Electricity Forecast System"
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Initialize directories after creation."""
        self.paths.create_directories()
    
    def save(self, filepath: str):
        """Save settings to YAML file."""
        config_dict = {
            'project_name': self.project_name,
            'version': self.version,
            'data': {
                'sources': self.data.data_sources,
                'energy_sources': self.data.energy_sources,
                'generation_mix': self.data.default_generation_mix,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'max_epochs': self.training.max_epochs,
                'learning_rate': self.training.learning_rate,
                'early_stopping_patience': self.training.early_stopping_patience,
            },
            'forecast': {
                'horizons': [
                    {'name': h.name, 'hours': h.hours, 'lookback': h.lookback_hours}
                    for h in self.forecast.horizons
                ]
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'Settings':
        """Load settings from YAML file."""
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        settings = cls()
        settings.project_name = config.get('project_name', settings.project_name)
        settings.version = config.get('version', settings.version)
        
        return settings


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset global settings (useful for testing)."""
    global _settings
    _settings = None


# Convenience function
def configure(**kwargs) -> Settings:
    """Configure settings with custom values."""
    settings = get_settings()
    
    for key, value in kwargs.items():
        if hasattr(settings.training, key):
            setattr(settings.training, key, value)
        elif hasattr(settings.data, key):
            setattr(settings.data, key, value)
        elif hasattr(settings.forecast, key):
            setattr(settings.forecast, key, value)
    
    return settings