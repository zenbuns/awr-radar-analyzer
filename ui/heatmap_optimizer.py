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
        # Add cached data for optimization
        self.data_checksum_cache = {}
        self.last_data_shape = None
        self.change_threshold = 0.05  # 5% change threshold
        
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
        
        # Always update the first 10 frames to ensure data appears
        if self.frame_counter <= 10:
            self.last_update_time = current_time
            return True
        
        # During startup phase, always update for the first few frames
        if self.startup_phase and self.frame_counter <= self.startup_frames:
            # After startup_frames frames, exit startup phase
            if self.frame_counter == self.startup_frames:
                self.startup_phase = False
            self.last_update_time = current_time
            return True
        
        # Frame rate limiting - enforce minimum time between updates
        elapsed = current_time - self.last_update_time
        if elapsed < self.update_interval:
            return False
            
        # Use downsampled data fingerprinting for faster comparison
        # This is much more efficient than processing the full array
        if data is not None and data.size > 0:
            # Check if data shape changed
            if self.last_data_shape != data.shape:
                self.last_data_shape = data.shape
                self.last_update_time = current_time
                return True
                
            # Fast downsampled checksum for large arrays
            if data.size > 10000:
                # Take downsampled data points using strided slicing (much faster)
                stride = max(1, data.shape[0] // 20, data.shape[1] // 20)
                sample = data[::stride, ::stride]
                data_sum = np.sum(sample)
                nonzeros = np.count_nonzero(sample)
            else:
                # For smaller arrays, use the whole array
                data_sum = np.sum(data)
                nonzeros = np.count_nonzero(data)
                
            # Create fingerprint using sum and active cell count
            data_fingerprint = (int(data_sum * 1000), nonzeros)
            
            # Skip update if data hasn't changed significantly
            if self.last_data_hash is not None:
                last_sum, last_nonzeros = self.last_data_hash
                
                # Calculate relative change
                sum_change = abs(data_sum - last_sum/1000) / (last_sum/1000 + 1e-10)
                count_change = abs(nonzeros - last_nonzeros) / (last_nonzeros + 1e-10)
                
                # Only update if the change exceeds threshold or on periodic heartbeat
                if sum_change < self.change_threshold and count_change < self.change_threshold:
                    # Heartbeat update (lower frequency for unchanged data)
                    if self.frame_counter % 10 != 0:
                        return False
        else:
            data_fingerprint = (0, 0)
        
        # Update tracking state
        self.last_data_hash = data_fingerprint
        self.last_update_time = current_time
        return True
        
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
        current_time = time.time()
        min_interval = 1.0 / self.max_fps
        
        # Always redraw during startup
        if self.frame_counter <= 10:
            self.last_draw_time = current_time
            return True
            
        # Limit frame rate to max_fps
        if current_time - self.last_draw_time < min_interval:
            return False
            
        self.last_draw_time = current_time
        return True
    
    def set_update_interval(self, interval):
        """
        Set the minimum time between heatmap updates.
        
        Args:
            interval: Time in seconds.
        """
        self.update_interval = max(0.05, float(interval))
    
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
            # Set change threshold as a fraction (from percentage)
            self.change_threshold = change_threshold_percent / 100.0
            
            # Convert percentage to equivalent FPS value (higher percentage = lower FPS)
            # This is an approximation - higher threshold means we can accept lower FPS
            fps = 60 - (change_threshold_percent / 2)  # Scale: 0% -> 60fps, 100% -> 10fps
            fps = max(10, min(60, fps))  # Clamp between 10-60 fps
            self.set_max_fps(int(fps)) 