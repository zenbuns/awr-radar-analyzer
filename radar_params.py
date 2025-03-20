#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data classes for radar experiment parameters and collected data.

This module provides the data structure definitions for radar experiments,
including configuration parameters and containers for collected data.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SamplingCircle:
    """
    Parameters for a sampling circle in the radar visualization.
    
    Attributes:
        enabled: Whether this circle is enabled/visible.
        distance: Distance to the center of the sampling circle in meters.
        radius: Radius of the sampling circle in meters.
        angle: Angle in degrees for this circle's position (0 = center, -60 = left, +60 = right).
        color: Color for this circle's visualization.
        label: Label for identifying this circle.
    """
    enabled: bool = True
    distance: float = 5.0
    radius: float = 0.5
    angle: float = 0.0  # 0 degrees = center, -60 = left, +60 = right
    color: str = "lime"
    label: str = "Primary"


@dataclass
class RadarExperimentParams:
    """
    Parameters for the radar experiment configuration.
    
    Attributes:
        max_range: Maximum radar detection range in meters.
        circle_interval: Distance interval between range circles in meters.
        collection_duration: Duration of data collection in seconds.
        current_config: Name of the current radar configuration.
        target_distance: Target distance for the experiment in meters.
        heatmap_resolution: Resolution of the heatmap grid in meters per cell.
        circle_distance: Distance to the center of the primary sampling circle in meters.
        circle_radius: Radius of the primary sampling circle in meters.
        circles: List of sampling circles for data collection.
        num_trials: Number of repeated trials for this experiment configuration.
    """

    max_range: float = 35.0
    circle_interval: float = 10.0
    collection_duration: int = 60
    current_config: str = "default_config"
    target_distance: float = 5.0
    heatmap_resolution: float = 0.5
    # Add these missing attributes that are used in the codebase
    circle_distance: float = 5.0  # Distance to primary sampling circle center
    circle_radius: float = 0.5    # Radius of primary sampling circle
    num_trials: int = 1  # Optionally used for multiple repeated trials
    
    # Multi-frame processing parameters
    enable_multi_frame: bool = True  # Whether to enable multi-frame processing
    multi_frame_count: int = 10  # Number of frames to combine
    multi_frame_method: str = "average"  # Method to combine frames ("average", "max", "sum")
    
    # Default sampling circles
    circles: List[SamplingCircle] = field(default_factory=lambda: [
        SamplingCircle(enabled=True, distance=5.0, radius=0.5, angle=0.0, color="lime", label="Primary"),
        SamplingCircle(enabled=False, distance=15.0, radius=0.5, angle=-60.0, color="cyan", label="Left"),
        SamplingCircle(enabled=False, distance=25.0, radius=0.5, angle=60.0, color="yellow", label="Right"),
    ])

    def __post_init__(self):
        """
        Ensure consistent values between individual attributes and circles list.
        """
        # Initialize primary circle with circle_distance and circle_radius
        if self.circles and len(self.circles) > 0:
            self.circles[0].distance = self.circle_distance
            self.circles[0].radius = self.circle_radius
    
    def update_circle_distance(self, index: int, distance: float) -> None:
        """
        Update the distance of a sampling circle.
        
        Args:
            index: Index of the circle to update (0-2)
            distance: New distance in meters
        """
        if 0 <= index < len(self.circles):
            self.circles[index].distance = distance
            # Update the primary circle distance attribute for backward compatibility
            if index == 0:
                self.circle_distance = distance
    
    def update_circle_radius(self, index: int, radius: float) -> None:
        """
        Update the radius of a sampling circle.
        
        Args:
            index: Index of the circle to update (0-2)
            radius: New radius in meters
        """
        if 0 <= index < len(self.circles):
            self.circles[index].radius = radius
            # Update the primary circle radius attribute for backward compatibility
            if index == 0:
                self.circle_radius = radius
    
    def toggle_circle(self, index: int, enabled: bool) -> None:
        """
        Toggle a sampling circle on/off.
        
        Args:
            index: Index of the circle to toggle (0-2)
            enabled: Whether the circle should be visible
        """
        if 0 <= index < len(self.circles):
            self.circles[index].enabled = enabled


@dataclass
class ExperimentData:
    """
    Container for experiment data, including time-series of circle statistics.
    
    This class stores all collected radar point data and derived statistics 
    for a single experiment run.
    
    Attributes:
        x_points: List of x-coordinates for detected radar points.
        y_points: List of y-coordinates for detected radar points.
        z_points: List of z-coordinates for detected radar points.
        intensities: List of intensity values for detected radar points.
        timestamps: List of timestamps for each collected data point.
        target_distances: List of target distances for each data point.
        time_series_timestamps: List of timestamps for circle statistics.
        circle_point_counts: List of point counts within sampling circle over time.
        circle_avg_intensities: List of average intensities within sampling circle over time.
        multi_frame_metrics: Dictionary containing metrics from multi-frame analysis.
        metadata: Dictionary to store additional metadata like distance bands.
    """

    x_points: List[float] = field(default_factory=list)
    y_points: List[float] = field(default_factory=list)
    z_points: List[float] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    target_distances: List[float] = field(default_factory=list)
    time_series_timestamps: List[float] = field(default_factory=list) 
    circle_point_counts: List[int] = field(default_factory=list)
    circle_avg_intensities: List[float] = field(default_factory=list)
    multi_frame_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Store additional metadata like distance bands

    def clear(self) -> None:
        """
        Clear all stored data.
        
        This method resets all data lists to empty, effectively clearing
        all collected experiment data.
        """
        self.x_points.clear()
        self.y_points.clear()
        self.z_points.clear()
        self.intensities.clear()
        self.timestamps.clear()
        self.target_distances.clear()
        self.time_series_timestamps.clear()
        self.circle_point_counts.clear()
        self.circle_avg_intensities.clear()
        self.multi_frame_metrics.clear()
        self.metadata.clear()

    def extend(self, other: "ExperimentData") -> None:
        """
        Extend data with another ExperimentData instance.
        
        This method adds all the data from another ExperimentData object to this one,
        effectively combining the two datasets.
        
        Args:
            other: Another instance of ExperimentData to extend from.
        """
        self.x_points.extend(other.x_points)
        self.y_points.extend(other.y_points)
        self.z_points.extend(other.z_points)
        self.intensities.extend(other.intensities)
        self.timestamps.extend(other.timestamps)
        self.target_distances.extend(other.target_distances)
        self.time_series_timestamps.extend(other.time_series_timestamps)
        self.circle_point_counts.extend(other.circle_point_counts)
        self.circle_avg_intensities.extend(other.circle_avg_intensities)
        
        # Merge multi-frame metrics (add any new keys from other)
        for key, value in other.multi_frame_metrics.items():
            if key not in self.multi_frame_metrics:
                self.multi_frame_metrics[key] = value

        # Merge metadata (add any new keys from other)
        for key, value in other.metadata.items():
            if key not in self.metadata:
                self.metadata[key] = value