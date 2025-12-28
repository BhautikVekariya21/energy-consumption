"""
================================================================================
HELPER UTILITIES
================================================================================
Common utility functions used throughout the project.
"""

import os
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from contextlib import contextmanager

import numpy as np


def set_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # Additional TF settings for reproducibility
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except ImportError:
        pass


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cpu_count': os.cpu_count(),
        'gpu_available': False,
        'gpu_count': 0,
        'gpu_devices': [],
        'mixed_precision_available': False
    }
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        info['gpu_available'] = len(gpus) > 0
        info['gpu_count'] = len(gpus)
        
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                gpu_info = {
                    'name': gpu.name,
                    'device_type': gpu.device_type,
                    'details': details
                }
            except:
                gpu_info = {
                    'name': gpu.name,
                    'device_type': gpu.device_type
                }
            info['gpu_devices'].append(gpu_info)
        
        # Check mixed precision support
        try:
            from tensorflow.keras import mixed_precision
            info['mixed_precision_available'] = True
        except:
            pass
            
    except ImportError:
        pass
    
    return info


def format_time(seconds: float) -> str:
    """
    Format time duration to human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(value: float, precision: int = 2) -> str:
    """
    Format large numbers with appropriate units.
    
    Args:
        value: Number to format
        precision: Decimal precision
    
    Returns:
        Formatted string with units
    """
    if abs(value) >= 1e12:
        return f"{value/1e12:.{precision}f} TW"
    elif abs(value) >= 1e9:
        return f"{value/1e9:.{precision}f} GW"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f} MW"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f} kW"
    else:
        return f"{value:.{precision}f} W"


def format_energy(value_mwh: float, precision: int = 2) -> str:
    """
    Format energy values (MWh) with appropriate units.
    
    Args:
        value_mwh: Energy in MWh
        precision: Decimal precision
    
    Returns:
        Formatted string with units
    """
    if abs(value_mwh) >= 1e6:
        return f"{value_mwh/1e6:.{precision}f} TWh"
    elif abs(value_mwh) >= 1e3:
        return f"{value_mwh/1e3:.{precision}f} GWh"
    else:
        return f"{value_mwh:.{precision}f} MWh"


def save_json(data: Dict, filepath: Union[str, Path]):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        return obj
    
    # Recursively convert
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        else:
            return convert(obj)
    
    with open(filepath, 'w') as f:
        json.dump(deep_convert(data), f, indent=2)


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", logger=None):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed
            logger: Optional logger to use
        """
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        msg = f"{self.name}: {format_time(self.elapsed)}"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0


@contextmanager
def suppress_tf_warnings():
    """Context manager to suppress TensorFlow warnings."""
    import os
    import logging
    
    # Store original settings
    orig_tf_log = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
    
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        yield
    finally:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = orig_tf_log


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory info in GB
    """
    import psutil
    
    process = psutil.Process()
    mem_info = process.memory_info()
    
    result = {
        'rss_gb': mem_info.rss / 1e9,
        'vms_gb': mem_info.vms / 1e9,
    }
    
    # GPU memory if available
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        for i, gpu in enumerate(gpus):
            try:
                mem = tf.config.experimental.get_memory_info(gpu.name)
                result[f'gpu_{i}_used_gb'] = mem['current'] / 1e9
                result[f'gpu_{i}_peak_gb'] = mem['peak'] / 1e9
            except:
                pass
    except:
        pass
    
    return result