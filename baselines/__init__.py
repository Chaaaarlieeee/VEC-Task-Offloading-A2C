"""
Baseline Strategies Package
"""

from .local import run_local_baseline
from .offload import run_offload_baseline
from .random_strategy import run_random_baseline

__all__ = [
    'run_local_baseline',
    'run_offload_baseline',
    'run_random_baseline'
]

