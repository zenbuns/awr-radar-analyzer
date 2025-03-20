#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing utilities for radar point clouds.

This module contains functions for processing radar point cloud data,
including filtering points, updating heatmaps, and calculating metrics.
"""

import numpy as np
import math
from typing import Tuple, Dict, Any


def calculate_heatmap_size(params) -> Tuple[int, int]:
    """
    Calculate the size of the heatmap grid based on max range and resolution.
    
    Adjusts the grid size to be even for memory efficiency and better visualization.

    Args:
        params: RadarExperimentParams object containing max_range and heatmap_resolution.

    Returns:
        A tuple containing the width and height of the heatmap grid in pixels.
    """
    grid_size = int(2 * params.max_range / params.heatmap_resolution)
    # Ensure grid size is even for better memory alignment
    if grid_size % 2 == 1:
        grid_size += 1
    return grid_size, grid_size


def filter_points_in_circle(
        analyzer,
        x: np.ndarray,
        y: np.ndarray,
        intensities: np.ndarray
) -> None:
    """
    Filter current points to find those within all enabled sampling circles.
    
    This method efficiently identifies points within each enabled sampling circle
    using vectorized operations and categorizes them by distance bands.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        x: X-coordinates of points.
        y: Y-coordinates of points.
        intensities: Intensity values of points.
    """
    try:
        if len(x) == 0:
            # Fast empty initialization
            for i in range(len(analyzer.params.circles)):
                circle_key = f'circle{i+1}' if i > 0 else 'circle'
                analyzer.current_data[f'{circle_key}_x'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_y'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_intensities'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_indices'] = np.array([], dtype=np.int32)
                analyzer.current_data[f'{circle_key}_distances'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_distance_bands'] = {}
            return
            
        # During collection from bag, only process primary circle unless specifically needed
        if analyzer.collecting_data and hasattr(analyzer, 'is_playing') and analyzer.is_playing:
            # Only process primary circle for collection
            circle = analyzer.params.circles[0]
            if not circle.enabled:
                return
                
            # Calculate circle center based on distance and angle
            angle_rad = math.radians(circle.angle)
            circle_center_x = circle.distance * math.sin(angle_rad)
            circle_center_y = circle.distance * math.cos(angle_rad)
            
            # Fast vectorized distance calculation
            dx = x - circle_center_x
            dy = y - circle_center_y
            dist_sq = dx * dx + dy * dy
            radius_sq = circle.radius ** 2
            
            # Find points within circle
            indices = np.where(dist_sq <= radius_sq)[0]
            
            # Store results directly without extra processing
            analyzer.current_data['circle_x'] = x[indices]
            analyzer.current_data['circle_y'] = y[indices]
            analyzer.current_data['circle_intensities'] = intensities[indices]
            analyzer.current_data['circle_indices'] = indices
            
            # Calculate basic distances for essential functionality
            if len(indices) > 0:
                # Calculate distance from origin (sensor)
                distances = np.sqrt(x[indices]**2 + y[indices]**2)
                analyzer.current_data['circle_distances'] = distances
                
                # Only create minimal distance bands needed for core analysis
                dist_bands = {}
                bands = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]
                for i, (min_dist, max_dist) in enumerate(bands):
                    band_indices = np.where((distances >= min_dist) & (distances < max_dist))[0]
                    dist_bands[f'{min_dist}-{max_dist}'] = {
                        'indices': band_indices,
                        'count': len(band_indices)
                    }
                analyzer.current_data['circle_distance_bands'] = dist_bands
            else:
                analyzer.current_data['circle_distances'] = np.array([], dtype=np.float32)
                analyzer.current_data['circle_distance_bands'] = {}
                
            return
        
        # Process each enabled circle with full processing for visualization
        analyzer.get_logger().debug(f"Processing points for {len(analyzer.params.circles)} circles, total input points: {len(x)}")
        
        for i, circle in enumerate(analyzer.params.circles):
            if not circle.enabled:
                # Initialize empty arrays for disabled circles
                circle_key = f'circle{i+1}' if i > 0 else 'circle'
                analyzer.current_data[f'{circle_key}_x'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_y'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_intensities'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_indices'] = np.array([], dtype=np.int32)
                analyzer.current_data[f'{circle_key}_distances'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_distance_bands'] = {}
                continue
            
            # Calculate circle center based on distance and angle
            angle_rad = math.radians(circle.angle)
            circle_center_x = circle.distance * math.sin(angle_rad)
            circle_center_y = circle.distance * math.cos(angle_rad)
            
            # Efficient vectorized distance calculation
            dx = x - circle_center_x
            dy = y - circle_center_y
            dist_sq = dx * dx + dy * dy
            radius_sq = circle.radius ** 2
            
            # Find points within circle
            indices = np.where(dist_sq <= radius_sq)[0]
            
            # Get key for this circle (use 'circle' for first one for backward compatibility)
            circle_key = f'circle{i+1}' if i > 0 else 'circle'
            
            # Update data
            circle_x = x[indices]
            circle_y = y[indices]
            circle_intensities = intensities[indices]
            
            # Store in current data
            analyzer.current_data[f'{circle_key}_x'] = circle_x
            analyzer.current_data[f'{circle_key}_y'] = circle_y
            analyzer.current_data[f'{circle_key}_intensities'] = circle_intensities
            analyzer.current_data[f'{circle_key}_indices'] = indices
            
            # Calculate distance from origin (sensor)
            if len(indices) > 0:
                distances = np.sqrt(circle_x**2 + circle_y**2)
                analyzer.current_data[f'{circle_key}_distances'] = distances
                
                # Create distance bands
                dist_bands = {}
                max_range = analyzer.params.max_range
                band_size = max_range / 10  # Divide range into 10 bands
                for i in range(10):
                    min_dist = i * band_size
                    max_dist = (i + 1) * band_size
                    band_indices = np.where((distances >= min_dist) & (distances < max_dist))[0]
                    dist_bands[f'{min_dist:.1f}-{max_dist:.1f}'] = {
                        'indices': band_indices,
                        'count': len(band_indices)
                    }
                    
                # Also add standard bands for easier comparison
                std_bands = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]
                for min_dist, max_dist in std_bands:
                    if f'{min_dist:.1f}-{max_dist:.1f}' not in dist_bands:
                        band_indices = np.where((distances >= min_dist) & (distances < max_dist))[0]
                        dist_bands[f'{min_dist}-{max_dist}'] = {
                            'indices': band_indices,
                            'count': len(band_indices)
                        }
                        
                analyzer.current_data[f'{circle_key}_distance_bands'] = dist_bands
            else:
                analyzer.current_data[f'{circle_key}_distances'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_distance_bands'] = {}
    
    except Exception as e:
        analyzer.get_logger().error(f"Error filtering points: {str(e)}")


def update_heatmap_vectorized(
        analyzer,
        x: np.ndarray,
        y: np.ndarray,
        intensity: np.ndarray
) -> None:
    """
    Update persistent (summation) heatmap with vectorized operations.
    
    This method adds new point data to the persistent heatmap using
    efficient numpy operations.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        x: X-coordinates of points.
        y: Y-coordinates of points.
        intensity: Intensity values of points.
    """
    if len(x) == 0:
        return

    try:
        max_range = analyzer.params.max_range
        res = analyzer.params.heatmap_resolution
        grid_size_x, grid_size_y = calculate_heatmap_size(analyzer.params)

        # Convert coordinates to grid indices
        grid_x = np.floor((x + max_range) / res).astype(np.int32)
        grid_y = np.floor(y / res).astype(np.int32)

        # Filter out points outside grid bounds
        valid_idx = (0 <= grid_x) & (grid_x < grid_size_x) & (0 <= grid_y) & (grid_y < grid_size_y)
        if not np.any(valid_idx):
            return

        grid_x = grid_x[valid_idx]
        grid_y = grid_y[valid_idx]
        intensity_valid = intensity[valid_idx]

        # Use np.add.at for efficient accumulation
        np.add.at(analyzer.heatmap_data, (grid_y, grid_x), intensity_valid)
    except Exception as e:
        analyzer.get_logger().error(f"Error updating heatmap: {str(e)}")


def update_live_heatmap_vectorized(
        analyzer,
        x: np.ndarray,
        y: np.ndarray,
        intensity: np.ndarray
) -> None:
    """
    Update the live (decaying) heatmap with vectorized operations.
    
    This method adds new point data to the live heatmap using
    efficient numpy operations. The live heatmap decays over time to
    emphasize recent data.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        x: X-coordinates of points.
        y: Y-coordinates of points.
        intensity: Intensity values of points.
    """
    if len(x) == 0:
        return

    try:
        max_range = analyzer.params.max_range
        res = analyzer.params.heatmap_resolution
        grid_size_x, grid_size_y = calculate_heatmap_size(analyzer.params)

        # CRITICAL FIX: Ensure heatmap array is initialized
        if analyzer.live_heatmap_data is None or analyzer.live_heatmap_data.shape[0] != grid_size_y:
            analyzer.live_heatmap_data = np.zeros((grid_size_y, grid_size_x), dtype=np.float32)
            analyzer.get_logger().info(f"Re-initialized heatmap with shape {grid_size_y}x{grid_size_x}")

        # Convert coordinates to grid indices
        grid_x = np.floor((x + max_range) / res).astype(np.int32)
        grid_y = np.floor(y / res).astype(np.int32)

        # Filter out points outside grid bounds
        valid_idx = (0 <= grid_x) & (grid_x < grid_size_x) & (0 <= grid_y) & (grid_y < grid_size_y)
        if not np.any(valid_idx):
            return

        grid_x = grid_x[valid_idx]
        grid_y = grid_y[valid_idx]
        intensity_valid = intensity[valid_idx]
        
        # CRITICAL FIX: Ensure intensity values are normalized
        if np.max(intensity_valid) > 100:
            # Intensity seems to be in raw format, normalize to 0-1 range
            intensity_valid = intensity_valid / np.max(intensity_valid)
            
        elif np.max(intensity_valid) < 0.001 and len(intensity_valid) > 0:
            # Intensity is too small, scale it up to be visible
            intensity_valid = np.ones_like(intensity_valid) * 0.5
        
        # CRITICAL FIX: Ensure we detect grid indexing issues
        if np.max(grid_y) >= grid_size_y or np.max(grid_x) >= grid_size_x:
            # Log warning and clip to fix
            analyzer.get_logger().warn(f"Grid indices out of bounds: x_max={np.max(grid_x)}, y_max={np.max(grid_y)}, " +
                                      f"grid={grid_size_x}x{grid_size_y}")
            grid_x = np.clip(grid_x, 0, grid_size_x-1)
            grid_y = np.clip(grid_y, 0, grid_size_y-1)

        # Use np.add.at for efficient accumulation
        np.add.at(analyzer.live_heatmap_data, (grid_y, grid_x), intensity_valid)
        
        # DEBUGGING: Log occasional stats about the heatmap data
        if hasattr(analyzer, 'debug_counter'):
            analyzer.debug_counter += 1
        else:
            analyzer.debug_counter = 0
            
        if analyzer.debug_counter % 100 == 0:
            min_val = np.min(analyzer.live_heatmap_data)
            max_val = np.max(analyzer.live_heatmap_data)
            nonzero = np.count_nonzero(analyzer.live_heatmap_data)
            analyzer.get_logger().info(f"Heatmap stats: min={min_val:.4f}, max={max_val:.4f}, nonzero={nonzero}/{analyzer.live_heatmap_data.size} points")
            
    except Exception as e:
        analyzer.get_logger().error(f"Error updating live heatmap: {str(e)}")


def apply_live_heatmap_decay(analyzer) -> None:
    """
    Apply an exponential decay factor to the live heatmap.
    
    This method applies decay to the live heatmap to make more recent
    data more prominent in the visualization. The decay factor is
    configurable through the GUI.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
    """
    try:
        analyzer.live_heatmap_data *= analyzer.live_heatmap_decay_factor
    except Exception as e:
        analyzer.get_logger().error(f"Error applying heatmap decay: {str(e)}")


def compute_heatmap_metrics(analyzer) -> Dict[str, float]:
    """
    Compute scientific metrics of the live heatmap.
    
    This method calculates various metrics from the heatmap data for
    scientific analysis, including SNR, intensity statistics, and
    coverage percentages.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        
    Returns:
        Dictionary of relevant metrics including max_intensity, avg_intensity,
        snr_dB, active_cells, total_cells, and coverage_percentage.
    """
    try:
        noise_floor = 0.05
        data = analyzer.live_heatmap_data.copy()
        data[data < noise_floor] = 0

        nonzero = data[data > 0]
        max_intensity = float(np.max(data)) if nonzero.size > 0 else 0.0
        avg_intensity = float(np.mean(nonzero)) if nonzero.size > 0 else 0.0
        
        # Calculate signal-to-noise ratio in decibels
        snr_dB = 10.0 * np.log10(max_intensity / noise_floor) if max_intensity > noise_floor else 0.0
        active_cells = int(np.count_nonzero(data))
        total_cells = data.size
        coverage_percentage = 100.0 * active_cells / total_cells if total_cells > 0 else 0.0

        return {
            'max_intensity': max_intensity,
            'avg_intensity': avg_intensity,
            'snr_dB': snr_dB,
            'active_cells': float(active_cells),
            'total_cells': float(total_cells),
            'coverage_percentage': coverage_percentage
        }
    except Exception as e:
        analyzer.get_logger().error(f"Error computing heatmap metrics: {str(e)}")
        return {
            'max_intensity': 0.0,
            'avg_intensity': 0.0,
            'snr_dB': 0.0,
            'active_cells': 0.0,
            'total_cells': 1.0,
            'coverage_percentage': 0.0
        }

def update_progress(self):
    analyzer = self.main_window.analyzer
    if not analyzer or not hasattr(analyzer, 'experiment_data'):
        return
        
    # Use thread-safe access with data_lock
    if hasattr(analyzer, 'data_lock'):
        with analyzer.data_lock:
            if hasattr(analyzer, 'experiment_data') and hasattr(analyzer.experiment_data, 'x_points'):
                points = len(analyzer.experiment_data.x_points)
                self.points_collected_label.setText(f"Points: {points}")
    else:  # Properly indent this else clause
        # Fallback if no data_lock available
        points = len(analyzer.experiment_data.x_points) if hasattr(analyzer.experiment_data, 'x_points') else 0
        self.points_collected_label.setText(f"Points: {points}")

def play_rosbag(self, bag_path: str, loop: bool = True) -> bool:
    try:
        # Clear experiment data when starting a new bag playback
        if hasattr(self, 'experiment_data') and not self.collecting_data:
            self.get_logger().info("Clearing experiment data before starting bag playback")
            with self.data_lock:
                self.experiment_data.clear()
        
        # Rest of existing implementation
        play_rosbag_func(self, bag_path, loop)
        return True
    except Exception as e:
        self.get_logger().error(f"Error in play_rosbag: {str(e)}")
        return False

def _start_collection_from_bag(self):
    """Start data collection from a ROS2 bag."""
    analyzer = self.main_window.analyzer
    if not analyzer:
        return
        
    # Ensure data is cleared before starting new collection
    if hasattr(analyzer, 'experiment_data') and hasattr(analyzer.experiment_data, 'clear'):
        print("Explicitly clearing experiment data before starting collection")
        with analyzer.data_lock:
            analyzer.experiment_data.clear()
            
    # Rest of existing implementation

def on_generate_from_bag_changed(self, state):
    is_checked = state == Qt.Checked
    
    # Immediate UI feedback
    self.generate_from_bag_check.setEnabled(False)
    self.set_status(f"{'Enabling' if is_checked else 'Disabling'} data generation from bag...")
    QApplication.processEvents()
    
    # Track state changes explicitly
    if is_checked:
        # Stop any previous data collection if it was active
        if hasattr(self.main_window.analyzer, 'collecting_data') and self.main_window.analyzer.collecting_data:
            print("Stopping previous data collection before starting new one")
            self.stop_collection.emit()
            # Wait a moment for collection to stop
            QTimer.singleShot(300, lambda: self._enable_generate_from_bag(is_checked))
        else:
            self._enable_generate_from_bag(is_checked)
    else:
        # When disabling, make sure we stop data collection and reset flag
        self.bag_started_for_generation = False
        if hasattr(self.main_window.analyzer, 'collecting_data') and self.main_window.analyzer.collecting_data:
            self.stop_collection.emit()
        # Re-enable immediately
        self.generate_from_bag_check.setEnabled(True)

def _enable_generate_from_bag(self, enable):
    """Helper method to enable Generate from Bag with proper timing."""
    # Clear data before starting
    if enable and self.main_window and self.main_window.analyzer:
        with self.main_window.analyzer.data_lock:
            if hasattr(self.main_window.analyzer, 'experiment_data'):
                self.main_window.analyzer.experiment_data.clear()
    
    # Set flag and restart bag if already playing
    if enable:
        self.bag_started_for_generation = True
        # If bag is already playing, restart it to start collection
        if hasattr(self.main_window.analyzer, 'is_playing') and self.main_window.analyzer.is_playing:
            self._restart_bag_and_collection()
        # Otherwise, if a bag path is selected, play it
        elif self.bag_path_edit.text():
            self.play_rosbag.emit(self.bag_path_edit.text(), False)  # Don't loop for data generation
            QTimer.singleShot(500, self._start_collection_from_bag)
        else:
            self.set_status("No bag file selected. Please select a bag file first.")
            self.generate_from_bag_check.setChecked(False)
            self.bag_started_for_generation = False
    
    # Re-enable the checkbox
    self.generate_from_bag_check.setEnabled(True)
