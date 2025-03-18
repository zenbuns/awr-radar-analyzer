"""
Radar Point Cloud Analyzer package.

This package provides functionality for processing, visualizing, 
and analyzing radar point cloud data from AWR1843 mmWave radar.
"""

# Import the main class for easy access
from radar_analyzer.core import RadarPointCloudAnalyzer

# Import key submodules
from radar_analyzer import processing
from radar_analyzer import visualization
from radar_analyzer import utils

# Define what gets exported with "from radar_analyzer import *"
__all__ = ['RadarPointCloudAnalyzer']