#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization layer for radar scatter visualization.

This module provides an optimizer class to improve scatter plot
performance with large radar point cloud datasets.
"""

import time
import numpy as np

class ScatterOptimizer:
    """
    Optimization layer for scatter plot rendering decisions.
    
    This class implements intelligent downsampling and update frequency
    control to improve performance with large point clouds.
    
    Attributes:
        last_update_time: Last time the scatter plot was updated.
        update_interval: Minimum time between updates.
        max_points: Maximum number of points to display.
        adaptive_sampling: Whether to use adaptive sampling.
        frame_counter: Counter for frames processed.
        last_point_count: Number of points in the last dataset.
        startup_phase: Whether the optimizer is in the startup phase.
        startup_frames: Number of frames in the startup phase.
        last_draw_time: Last time the scatter plot was drawn.
        max_fps: Maximum frames per second for redrawing.
    """
    
    def __init__(self):
        """Initialize the ScatterOptimizer with default settings."""
        self.last_update_time = 0
        self.update_interval = 0.1  # seconds
        self.max_points = 5000
        self.adaptive_sampling = True
        self.frame_counter = 0
        self.last_point_count = 0
        self.startup_phase = True
        self.startup_frames = 10
        self.last_draw_time = 0
        self.max_fps = 30  # maximum frames per second
        
    def should_update(self, point_count):
        """
        Determine if scatter plot needs updating based on timing.
        
        Args:
            point_count: Number of points in the current dataset.
            
        Returns:
            bool: True if scatter should be updated, False otherwise.
        """
        current_time = time.time()
        self.frame_counter += 1
        
        # Calculate time since last draw
        elapsed = current_time - self.last_draw_time
        
        # Calculate minimum time between frames based on max_fps
        min_frame_time = 1.0 / self.max_fps
        
        # Always update during startup phase
        if self.startup_phase and self.frame_counter <= self.startup_frames:
            if self.frame_counter == self.startup_frames:
                self.startup_phase = False
            self.last_update_time = current_time
            self.last_draw_time = current_time
            self.last_point_count = point_count
            return True
        
        # Enforce frame rate limit
        if elapsed < min_frame_time:
            return False
            
        # Enforce minimum update interval
        if current_time - self.last_update_time < self.update_interval:
            return False
            
        # Always update if point count changed significantly
        if abs(point_count - self.last_point_count) > max(10, self.last_point_count * 0.05):
            self.last_update_time = current_time
            self.last_draw_time = current_time
            self.last_point_count = point_count
            return True
            
        # Update state and return decision
        self.last_update_time = current_time
        self.last_draw_time = current_time
        self.last_point_count = point_count
        return True
        
    def downsample(self, x, y, intensities, max_points=None):
        """
        Downsample point cloud data if needed.
        
        Args:
            x: X-coordinates array.
            y: Y-coordinates array.
            intensities: Intensity values array.
            max_points: Optional override for maximum points.
            
        Returns:
            Tuple of (x, y, intensities) after downsampling.
        """
        if max_points is None:
            max_points = self.max_points
            
        if len(x) <= max_points:
            return x, y, intensities
            
        # Calculate downsampling factor - keep higher percentage for larger datasets
        if len(x) > 50000:
            keep_percent = 0.05  # Keep 5% for very large datasets
        elif len(x) > 20000:
            keep_percent = 0.1   # Keep 10% for large datasets
        else:
            keep_percent = 0.2   # Keep 20% for medium datasets
            
        # Always keep at least max_points
        n_keep = max(max_points, int(len(x) * keep_percent))
        
        # Select random indices for downsampling
        indices = np.random.choice(len(x), n_keep, replace=False)
        
        return x[indices], y[indices], intensities[indices]
        
    def set_update_interval(self, interval):
        """
        Set the minimum time between updates.
        
        Args:
            interval: Time in seconds.
        """
        self.update_interval = max(0.05, float(interval))
        
    def set_max_points(self, max_points):
        """
        Set the maximum number of points to display.
        
        Args:
            max_points: Maximum point count.
        """
        self.max_points = max(1000, min(20000, int(max_points)))
        
    def set_max_fps(self, fps):
        """
        Set the maximum frames per second for redrawing.
        
        Args:
            fps: Maximum frames per second.
        """
        self.max_fps = max(1, min(60, int(fps)))
        
    def configure(self, update_interval=None, max_points=None, adaptive_sampling=None, max_fps=None):
        """
        Configure multiple parameters at once.
        
        Args:
            update_interval: Minimum time between updates in seconds.
            max_points: Maximum number of points to display.
            adaptive_sampling: Whether to use adaptive sampling.
            max_fps: Maximum frames per second for rendering.
        """
        if update_interval is not None:
            self.set_update_interval(update_interval)
            
        if max_points is not None:
            self.set_max_points(max_points)
            
        if adaptive_sampling is not None:
            self.adaptive_sampling = bool(adaptive_sampling)
            
        if max_fps is not None:
            self.set_max_fps(max_fps) 