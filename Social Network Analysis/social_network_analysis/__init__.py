"""
Social Network Analysis System

A comprehensive platform for analyzing social network data including:
- Network structure analysis
- Community detection
- Centrality calculations
- Influence propagation modeling
- Link prediction using graph neural networks
"""

from . import data
from . import graph
from . import analysis
from . import visualization
from . import ml

__version__ = "1.0.0"
__author__ = "Social Network Analysis Team"

__all__ = ['data', 'graph', 'analysis', 'visualization', 'ml']