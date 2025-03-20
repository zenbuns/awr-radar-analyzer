#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization layer for radar heatmap visualization.

This module provides an optimizer class to reduce unnecessary
recalculations and redraws in the visualization pipeline.
"""

import time
import numpy as np

class HeatmapOptimizer:
    """
    Optimization layer for heatmap rendering decisions.
    
    This class decides when to update the heatmap and contours
    based on data changes and timing intervals to improve performance.
    
    Attributes:
        last_update_time: Last time the heatmap was updated.
        last_contour_time: Last time contours were updated.
        last_data_hash: Hash value of the last processed data.
        update_interval: Minimum time between heatmap updates.
        contour_interval: Minimum time between contour updates.
    """
    
    def __init__(self):
        """Initialize the HeatmapOptimizer with default settings."""
        self.last_update_time = 0
        self.last_contour_time = 0
        self.last_data_hash = None
        self.update_interval = 0.1  # seconds
        self.contour_interval = 0.5  # seconds
        self.frame_counter = 0
        self.last_draw_time = 0
        self.max_fps = 30  # maximum frames per second
        self.startup_phase = True  # Special flag for startup phase
        self.startup_frames = 10   # Number of frames to always render during startup
        
    def should_update_heatmap(self, data):
        """
        Determine if heatmap needs updating based on data changes and timing.
        
        Args:
            data: Numpy array of heatmap data.
            
        Returns:
            bool: True if heatmap should be updated, False otherwise.
        """
        current_time = time.time()
        self.frame_counter += 1
        
        # CRITICAL FIX: Continue regular updates after startup phase
        # Ensure minimum update frequency regardless of data changes
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        
        # Always update during startup phase for consistent initial rendering
        if self.frame_counter <= 100 or self.startup_phase:
            if self.frame_counter == 100:
                self.startup_phase = False
            self.last_update_time = current_time
            return True
        
        # Skip extremely rapid updates (less than 10ms apart) to prevent CPU overload
        if current_time - self.last_update_time < 0.01:
            return False
            
        # Check for meaningful data changes with a more reliable hash method
        if data is not None and data.size > 0:
            # Use max value and nonzero count for more reliable change detection
            data_max = np.max(data) if data.size > 0 else 0
            nonzero_count = np.count_nonzero(data)
            data_fingerprint = int((data_max * 1000) + nonzero_count)
        else:
            data_fingerprint = 0
            
        # Force update on significant data change
        if data_fingerprint != self.last_data_hash:
            self.last_data_hash = data_fingerprint
            self.last_update_time = current_time
            return True
            
        # Skip update if data hasn't changed and interval hasn't elapsed
        return False
        
    def should_update_contours(self):
        """
        Determine if contours need updating based on timing.
        
        Contour calculation is expensive, so we update it less frequently.
        
        Returns:
            bool: True if contours should be updated, False otherwise.
        """
        # Always update during startup phase
        if self.startup_phase and self.frame_counter <= self.startup_frames:
            return True
            
        current_time = time.time()
        if current_time - self.last_contour_time < self.contour_interval:
            return False
            
        self.last_contour_time = current_time
        return True
    
    def should_redraw(self):
        """
        Determine if the canvas needs to be redrawn.
        
        This limits the maximum frame rate to avoid excessive CPU use.
        
        Returns:
            bool: True if canvas should be redrawn, False otherwise.
        """
        # Calculate time since last draw
        current_time = time.time()
        elapsed = current_time - self.last_draw_time
        
        # Calculate minimum time between frames based on max_fps
        min_frame_time = 1.0 / self.max_fps
        
        # Always draw during startup phase for responsiveness
        if self.startup_phase or self.frame_counter <= 100:
            self.last_draw_time = current_time
            return True
            
        # CRITICAL FIX: Ensure at least one redraw per second regardless of other conditions
        # This prevents the display from appearing frozen even when data isn't changing
        if elapsed > 1.0:
            self.last_draw_time = current_time
            return True
            
        # Enforce frame rate limit
        if elapsed < min_frame_time:
            return False
            
        # Update last draw time and allow redraw
        self.last_draw_time = current_time
        return True
    
    def set_update_interval(self, interval):
        """
        Set the minimum time between heatmap updates.
        
        Args:
            interval: Time in seconds.
        """
        # CRITICAL FIX: Ensure update_interval is never too large
        # Max value of 0.25 seconds ensures updates at least 4 times per second
        self.update_interval = max(0.05, min(0.25, float(interval)))
    
    def set_contour_interval(self, interval):
        """
        Set the minimum time between contour updates.
        
        Args:
            interval: Time in seconds.
        """
        self.contour_interval = max(0.2, float(interval))
    
    def set_max_fps(self, fps):
        """
        Set the maximum frames per second for redrawing.
        
        Args:
            fps: Maximum frames per second.
        """
        self.max_fps = max(1, min(60, int(fps)))
    
    def configure(self, min_time_interval=None, max_time_interval=None, change_threshold_percent=None):
        """
        Configure multiple parameters at once.
        
        Args:
            min_time_interval: Minimum time between heatmap updates in seconds.
            max_time_interval: Minimum time between contour updates in seconds.
            change_threshold_percent: Data change threshold as a percentage.
        """
        if min_time_interval is not None:
            self.set_update_interval(min_time_interval)
            
        if max_time_interval is not None:
            self.set_contour_interval(max_time_interval)
            
        if change_threshold_percent is not None:
            # Convert percentage to equivalent FPS value (higher percentage = lower FPS)
            # This is an approximation - higher threshold means we can accept lower FPS
            fps = 60 - (change_threshold_percent / 2)  # Scale: 0% -> 60fps, 100% -> 10fps
            fps = max(10, min(60, fps))  # Clamp between 10-60 fps
            self.set_max_fps(int(fps)) 