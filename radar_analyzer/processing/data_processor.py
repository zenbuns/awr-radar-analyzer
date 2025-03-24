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
import os


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
        # Pre-calculate distances from origin (sensor) once for reuse
        distances_from_origin = np.sqrt(x**2 + y**2) if len(x) > 0 else np.array([], dtype=np.float32)
        
        # Create empty arrays template for reuse
        empty_float_array = np.array([], dtype=np.float32)
        empty_int_array = np.array([], dtype=np.int32)
        empty_bands = {}
            
        if len(x) == 0:
            # Fast empty initialization with reused empty arrays
            for i in range(len(analyzer.params.circles)):
                circle_key = f'circle{i+1}' if i > 0 else 'circle'
                analyzer.current_data[f'{circle_key}_x'] = empty_float_array
                analyzer.current_data[f'{circle_key}_y'] = empty_float_array
                analyzer.current_data[f'{circle_key}_intensities'] = empty_float_array
                analyzer.current_data[f'{circle_key}_indices'] = empty_int_array
                analyzer.current_data[f'{circle_key}_distances'] = empty_float_array
                analyzer.current_data[f'{circle_key}_distance_bands'] = empty_bands.copy()
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
            
            # Fast vectorized distance calculation using precomputed components
            dx = x - circle_center_x
            dy = y - circle_center_y
            dist_sq = dx * dx + dy * dy
            radius_sq = circle.radius ** 2
            
            # Find points within circle
            indices = np.where(dist_sq <= radius_sq)[0]
            
            # Store results directly without extra processing
            if len(indices) > 0:
                analyzer.current_data['circle_x'] = x[indices]
                analyzer.current_data['circle_y'] = y[indices]
                analyzer.current_data['circle_intensities'] = intensities[indices]
                analyzer.current_data['circle_indices'] = indices
                
                # Use pre-calculated distances instead of recalculating
                circle_distances = distances_from_origin[indices]
                analyzer.current_data['circle_distances'] = circle_distances
                
                # Only create minimal distance bands needed for core analysis
                dist_bands = {}
                bands = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]
                
                # Process all bands in a single loop with one calculation
                for min_dist, max_dist in bands:
                    band_mask = (circle_distances >= min_dist) & (circle_distances < max_dist)
                    band_indices = np.where(band_mask)[0]
                    dist_bands[f'{min_dist}-{max_dist}'] = {
                        'indices': band_indices,
                        'count': len(band_indices)
                    }
                analyzer.current_data['circle_distance_bands'] = dist_bands
            else:
                analyzer.current_data['circle_x'] = empty_float_array
                analyzer.current_data['circle_y'] = empty_float_array
                analyzer.current_data['circle_intensities'] = empty_float_array
                analyzer.current_data['circle_indices'] = empty_int_array
                analyzer.current_data['circle_distances'] = empty_float_array
                analyzer.current_data['circle_distance_bands'] = empty_bands.copy()
                
            return
        
        # Process each enabled circle with full processing for visualization
        analyzer.get_logger().debug(f"Processing points for {len(analyzer.params.circles)} circles, total input points: {len(x)}")
        
        # Calculate max_range and band_size once
        max_range = analyzer.params.max_range
        band_size = max_range / 10  # Divide range into 10 bands
        
        # Define standard bands once
        std_bands = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]
        
        # Precompute regular bands once
        regular_bands = [(i * band_size, (i + 1) * band_size) for i in range(10)]
        
        for i, circle in enumerate(analyzer.params.circles):
            circle_key = f'circle{i+1}' if i > 0 else 'circle'
            
            if not circle.enabled:
                # Initialize empty arrays for disabled circles - reuse templates
                analyzer.current_data[f'{circle_key}_x'] = empty_float_array
                analyzer.current_data[f'{circle_key}_y'] = empty_float_array
                analyzer.current_data[f'{circle_key}_intensities'] = empty_float_array
                analyzer.current_data[f'{circle_key}_indices'] = empty_int_array
                analyzer.current_data[f'{circle_key}_distances'] = empty_float_array
                analyzer.current_data[f'{circle_key}_distance_bands'] = empty_bands.copy()
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
            
            # Update data
            if len(indices) > 0:
                circle_x = x[indices]
                circle_y = y[indices]
                circle_intensities = intensities[indices]
                
                # Store in current data
                analyzer.current_data[f'{circle_key}_x'] = circle_x
                analyzer.current_data[f'{circle_key}_y'] = circle_y
                analyzer.current_data[f'{circle_key}_intensities'] = circle_intensities
                analyzer.current_data[f'{circle_key}_indices'] = indices
                
                # Use pre-calculated distances instead of recalculating
                circle_distances = distances_from_origin[indices]
                analyzer.current_data[f'{circle_key}_distances'] = circle_distances
                
                # Create distance bands more efficiently
                dist_bands = {}
                
                # Process all bands in one pass
                for min_dist, max_dist in regular_bands:
                    band_mask = (circle_distances >= min_dist) & (circle_distances < max_dist)
                    band_indices = np.where(band_mask)[0]
                    dist_bands[f'{min_dist:.1f}-{max_dist:.1f}'] = {
                        'indices': band_indices,
                        'count': len(band_indices)
                    }
                
                # Also add standard bands for easier comparison
                for min_dist, max_dist in std_bands:
                    band_key = f'{min_dist}-{max_dist}'
                    if band_key not in dist_bands:
                        band_mask = (circle_distances >= min_dist) & (circle_distances < max_dist)
                        band_indices = np.where(band_mask)[0]
                        dist_bands[band_key] = {
                            'indices': band_indices,
                            'count': len(band_indices)
                        }
                        
                analyzer.current_data[f'{circle_key}_distance_bands'] = dist_bands
            else:
                # Reuse empty arrays to avoid new allocations
                analyzer.current_data[f'{circle_key}_x'] = empty_float_array
                analyzer.current_data[f'{circle_key}_y'] = empty_float_array
                analyzer.current_data[f'{circle_key}_intensities'] = empty_float_array
                analyzer.current_data[f'{circle_key}_indices'] = empty_int_array
                analyzer.current_data[f'{circle_key}_distances'] = empty_float_array
                analyzer.current_data[f'{circle_key}_distance_bands'] = empty_bands.copy()
    
    except Exception as e:
        analyzer.get_logger().error(f"Error filtering points: {str(e)}")


def _prepare_grid_indices(
        analyzer,
        x: np.ndarray,
        y: np.ndarray,
        intensity: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Helper function to prepare grid indices for heatmap updates.
    
    This function handles the common calculations for both regular and live heatmap updates,
    avoiding code duplication and improving performance.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        x: X-coordinates of points.
        y: Y-coordinates of points.
        intensity: Intensity values of points.
        
    Returns:
        A tuple containing (grid_x, grid_y, intensity_valid, grid_size_x, grid_size_y)
        or (None, None, None, 0, 0) if no valid points.
    """
    if len(x) == 0:
        return None, None, None, 0, 0
        
    try:
        # Calculate these values once
        max_range = analyzer.params.max_range
        res = analyzer.params.heatmap_resolution
        grid_size_x, grid_size_y = calculate_heatmap_size(analyzer.params)

        # Optimize coordinate conversion with vectorized operations
        # Use direct array operations instead of np.floor for speed
        grid_x = ((x + max_range) / res).astype(np.int32)
        grid_y = (y / res).astype(np.int32)

        # Use single boolean mask for filtering
        valid_mask = (0 <= grid_x) & (grid_x < grid_size_x) & (0 <= grid_y) & (grid_y < grid_size_y)
        if not np.any(valid_mask):
            return None, None, None, 0, 0

        # Apply mask only once per array
        grid_x = grid_x[valid_mask]
        grid_y = grid_y[valid_mask]
        intensity_valid = intensity[valid_mask]
        
        return grid_x, grid_y, intensity_valid, grid_size_x, grid_size_y
    except Exception as e:
        analyzer.get_logger().error(f"Error preparing grid indices: {str(e)}")
        return None, None, None, 0, 0


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
        # Use shared grid preparation logic
        grid_x, grid_y, intensity_valid, grid_size_x, grid_size_y = _prepare_grid_indices(analyzer, x, y, intensity)
        if grid_x is None:
            return

        # Make sure heatmap data is properly initialized
        if analyzer.heatmap_data is None or analyzer.heatmap_data.shape != (grid_size_y, grid_size_x):
            analyzer.heatmap_data = np.zeros((grid_size_y, grid_size_x), dtype=np.float32)
            
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
        # Use shared grid preparation logic
        grid_x, grid_y, intensity_valid, grid_size_x, grid_size_y = _prepare_grid_indices(analyzer, x, y, intensity)
        if grid_x is None:
            return

        # CRITICAL FIX: Ensure heatmap array is initialized
        if analyzer.live_heatmap_data is None or analyzer.live_heatmap_data.shape != (grid_size_y, grid_size_x):
            analyzer.live_heatmap_data = np.zeros((grid_size_y, grid_size_x), dtype=np.float32)
            analyzer.get_logger().info(f"Re-initialized live heatmap with shape {grid_size_y}x{grid_size_x}")
            
        # CRITICAL FIX: Ensure intensity values are normalized - only for live heatmap
        if np.max(intensity_valid) > 100:
            # Intensity seems to be in raw format, normalize to 0-1 range
            intensity_valid = intensity_valid / np.max(intensity_valid)
            
        elif np.max(intensity_valid) < 0.001 and len(intensity_valid) > 0:
            # Intensity is too small, scale it up to be visible
            intensity_valid = np.ones_like(intensity_valid) * 0.5
        
        # Use np.add.at for efficient accumulation
        np.add.at(analyzer.live_heatmap_data, (grid_y, grid_x), intensity_valid)
        
        # DEBUGGING: Log occasional stats about the heatmap data (reduced frequency)
        if not hasattr(analyzer, 'debug_counter'):
            analyzer.debug_counter = 0
            
        analyzer.debug_counter += 1
        if analyzer.debug_counter % 500 == 0:  # Reduced frequency from 100 to 500
            min_val = np.min(analyzer.live_heatmap_data)
            max_val = np.max(analyzer.live_heatmap_data)
            nonzero = np.count_nonzero(analyzer.live_heatmap_data)
            analyzer.get_logger().debug(f"Live heatmap stats: min={min_val:.6f}, max={max_val:.6f}, nonzero={nonzero}")
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
        # Cache noise floor for reuse
        noise_floor = getattr(analyzer, '_cached_noise_floor', 0.05)
        
        # Use a view instead of a copy when possible
        data = analyzer.live_heatmap_data
        
        # Calculate metrics efficiently with vectorized operations
        # Find max value without copying the entire array
        max_intensity = float(np.max(data)) if data.size > 0 else 0.0
        
        # Calculate active cells count directly
        active_mask = data > noise_floor
        active_cells = int(np.sum(active_mask))
        total_cells = data.size
        
        # Calculate average only for active cells
        if active_cells > 0:
            # Use the mask directly instead of creating a new array
            avg_intensity = float(np.sum(data[active_mask])) / active_cells
        else:
            avg_intensity = 0.0
        
        # Calculate SNR efficiently
        if max_intensity > noise_floor:
            snr_dB = 10.0 * np.log10(max_intensity / noise_floor)
        else:
            snr_dB = 0.0
            
        # Calculate coverage percentage
        coverage_percentage = 100.0 * active_cells / total_cells if total_cells > 0 else 0.0

        # Cache these values for potential reuse elsewhere
        analyzer._cached_heatmap_metrics = {
            'max_intensity': max_intensity,
            'avg_intensity': avg_intensity,
            'snr_dB': snr_dB,
            'active_cells': float(active_cells),
            'total_cells': float(total_cells),
            'coverage_percentage': coverage_percentage
        }
        
        return analyzer._cached_heatmap_metrics
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
    """
    Update progress indicators with the current number of collected points.
    
    This method safely accesses the analyzer's experiment data to retrieve
    the current point count and updates the UI accordingly.
    """
    # Early return if analyzer is not available
    if not hasattr(self, 'main_window') or not self.main_window or not hasattr(self.main_window, 'analyzer'):
        return
        
    analyzer = self.main_window.analyzer
    
    # Early return if experiment_data is not available
    if not analyzer or not hasattr(analyzer, 'experiment_data'):
        return
    
    try:
        # Use a single thread-safe access with data_lock
        points = 0
        if hasattr(analyzer, 'data_lock'):
            with analyzer.data_lock:
                # Direct attribute access is faster than hasattr checks in the hot path
                try:
                    points = len(analyzer.experiment_data.x_points)
                except (AttributeError, TypeError):
                    # Handle the case where x_points doesn't exist or isn't a sequence
                    pass
        else:
            # Fallback if no data_lock available - with minimal attribute checking
            try:
                points = len(analyzer.experiment_data.x_points)
            except (AttributeError, TypeError):
                pass
        
        # Update UI only once with final count
        if hasattr(self, 'points_collected_label'):
            self.points_collected_label.setText(f"Points: {points}")
    except Exception as e:
        # Log error but don't crash on UI updates
        if hasattr(analyzer, 'get_logger'):
            analyzer.get_logger().debug(f"Error updating progress: {str(e)}")
        # Silent fallback if logger not available

def play_rosbag(self, bag_path: str, loop: bool = True) -> bool:
    """
    Play a ROS2 bag file containing radar data.
    
    This method handles playing back radar data from a ROS2 bag file,
    including clearing any existing experiment data and setting up
    playback parameters.
    
    Args:
        bag_path: Path to the ROS2 bag file.
        loop: Whether to loop the bag file playback.
        
    Returns:
        True if bag playback started successfully, False otherwise.
    """
    try:
        # Validate inputs first
        if not bag_path or not os.path.exists(bag_path):
            self.get_logger().error(f"Bag file doesn't exist: {bag_path}")
            return False
            
        # Cache bag path for reuse
        self.current_bag_path = bag_path
        
        # Clear experiment data only when needed (not collecting)
        # Use thread-safe operations with proper locking
        if hasattr(self, 'experiment_data') and not self.collecting_data:
            self.get_logger().info("Clearing experiment data before starting bag playback")
            
            # Ensure we have the lock before clearing data
            if hasattr(self, 'data_lock'):
                with self.data_lock:
                    # Use direct method call instead of checking hasattr first
                    try:
                        self.experiment_data.clear()
                    except (AttributeError, TypeError):
                        self.get_logger().warning("Could not clear experiment data")
            else:
                # Fallback without lock
                try:
                    self.experiment_data.clear()
                except (AttributeError, TypeError):
                    self.get_logger().warning("Could not clear experiment data")
        
        # Delegate to actual implementation with proper exception handling
        result = play_rosbag_func(self, bag_path, loop)
        
        # Update state to reflect current status
        self.is_playing = result
        
        # Log success/failure for debugging
        if result:
            self.get_logger().info(f"Started bag playback: {os.path.basename(bag_path)}")
        else:
            self.get_logger().error(f"Failed to start bag playback: {os.path.basename(bag_path)}")
            
        return result
    except Exception as e:
        self.get_logger().error(f"Error in play_rosbag: {str(e)}")
        # Ensure state is consistent even when exceptions occur
        self.is_playing = False
        return False

def _start_collection_from_bag(self):
    """
    Start data collection from a ROS2 bag.
    
    This method safely initializes data collection from a ROS2 bag file,
    ensuring proper cleanup of previous data and thread-safe operations.
    """
    try:
        # Early return if analyzer is not available
        if not hasattr(self, 'main_window') or not self.main_window:
            print("Cannot start collection: main window not available")
            return
            
        analyzer = self.main_window.analyzer
        if not analyzer:
            print("Cannot start collection: analyzer not available")
            return
            
        # Ensure data is cleared before starting new collection
        # Use a single, comprehensive lock to avoid race conditions
        if hasattr(analyzer, 'data_lock'):
            with analyzer.data_lock:
                try:
                    # Only clear if we have experiment_data with clear method
                    if hasattr(analyzer, 'experiment_data') and hasattr(analyzer.experiment_data, 'clear'):
                        print("Explicitly clearing experiment data before starting collection")
                        analyzer.experiment_data.clear()
                        
                        # Cache a reference to the start time for better performance
                        import time
                        analyzer.collection_start_time = time.time()
                        
                        # Set collection flag while still holding the lock
                        analyzer.collecting_data = True
                except Exception as e:
                    print(f"Error clearing experiment data: {str(e)}")
        else:
            # Fallback if no lock available - less safe but still functional
            try:
                if hasattr(analyzer, 'experiment_data') and hasattr(analyzer.experiment_data, 'clear'):
                    print("Clearing experiment data (without lock)")
                    analyzer.experiment_data.clear()
                    
                    import time
                    analyzer.collection_start_time = time.time()
                    analyzer.collecting_data = True
            except Exception as e:
                print(f"Error clearing experiment data: {str(e)}")
                
        # Update UI to reflect collection state
        if hasattr(self, 'set_status'):
            self.set_status("Data collection started - bag playback in progress")
            
        # Capture metadata about the collection for later use
        if hasattr(analyzer, 'current_bag_path') and analyzer.current_bag_path:
            analyzer.experiment_data.metadata = {
                'source': 'bag',
                'bag_file': analyzer.current_bag_path,
                'start_time': analyzer.collection_start_time
            }
            
    except Exception as e:
        print(f"Unexpected error in _start_collection_from_bag: {str(e)}")
        # Ensure collection flag is reset on error
        if analyzer:
            analyzer.collecting_data = False

def on_generate_from_bag_changed(self, state):
    """
    Handle checkbox state change for generating data from bag files.
    
    This method manages UI state and data collection based on the checkbox state,
    ensuring UI responsiveness while handling potentially blocking operations.
    
    Args:
        state: Qt checkbox state (Qt.Checked or Qt.Unchecked)
    """
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QApplication
    
    # Convert to boolean for clarity
    is_checked = state == Qt.Checked
    
    # Provide immediate visual feedback before processing
    self.generate_from_bag_check.setEnabled(False)
    status_message = f"{'Enabling' if is_checked else 'Disabling'} data generation from bag..."
    self.set_status(status_message)
    
    # Process UI events immediately to prevent UI freezing
    # Only do this once, not repeatedly
    QApplication.processEvents()
    
    try:
        if is_checked:
            # Handle enabling data generation
            self._handle_enable_data_generation()
        else:
            # Handle disabling data generation - simpler case
            self._handle_disable_data_generation()
    except Exception as e:
        # Log any errors and restore UI state
        print(f"Error changing data generation state: {str(e)}")
        self.set_status(f"Error: {str(e)}")
        self.generate_from_bag_check.setEnabled(True)
        
def _handle_enable_data_generation(self):
    """Helper method to handle enabling data generation from bag."""
    from PyQt5.QtCore import QTimer
    
    # Check if collection is already active
    if (hasattr(self.main_window, 'analyzer') and 
        hasattr(self.main_window.analyzer, 'collecting_data') and 
        self.main_window.analyzer.collecting_data):
        
        print("Stopping previous data collection before starting new one")
        # Emit signal to stop collection
        self.stop_collection.emit()
        
        # Use a single timer to handle the restart after stopping
        # This avoids nested callbacks and is more efficient
        QTimer.singleShot(300, self._restart_collection_after_stop)
    else:
        # No need to stop, proceed directly
        self._enable_generate_from_bag(True)

def _restart_collection_after_stop(self):
    """Helper method to restart collection after stopping previous collection."""
    self._enable_generate_from_bag(True)
    
def _handle_disable_data_generation(self):
    """Helper method to handle disabling data generation from bag."""
    # Reset flag immediately
    self.bag_started_for_generation = False
    
    # Stop collection if active
    if (hasattr(self.main_window, 'analyzer') and 
        hasattr(self.main_window.analyzer, 'collecting_data') and 
        self.main_window.analyzer.collecting_data):
        self.stop_collection.emit()
    
    # Re-enable UI immediately
    self.generate_from_bag_check.setEnabled(True)
    self.set_status("Data generation from bag disabled")

def _enable_generate_from_bag(self, enable):
    """
    Helper method to enable Generate from Bag with proper timing.
    
    Args:
        enable: Whether to enable or disable data generation
    """
    from PyQt5.QtCore import QTimer
    
    try:
        # Clear data before starting if enabled
        if enable and self.main_window and self.main_window.analyzer:
            with self.main_window.analyzer.data_lock:
                if hasattr(self.main_window.analyzer, 'experiment_data'):
                    self.main_window.analyzer.experiment_data.clear()
        
        # Set flag for tracking state
        self.bag_started_for_generation = enable
        
        if enable:
            # Check playing status first to choose appropriate action
            is_playing = (hasattr(self.main_window.analyzer, 'is_playing') and 
                         self.main_window.analyzer.is_playing)
            
            if is_playing:
                # Bag already playing - restart to start collection
                self._restart_bag_and_collection()
            elif self.bag_path_edit.text():
                # Bag not playing but path exists - play and start collection
                self.play_rosbag.emit(self.bag_path_edit.text(), False)  # Don't loop
                
                # Use timer to allow bag to start playing before starting collection
                QTimer.singleShot(500, self._start_collection_from_bag)
            else:
                # No bag path - show error
                self.set_status("No bag file selected. Please select a bag file first.")
                self.generate_from_bag_check.setChecked(False)
                self.bag_started_for_generation = False
    except Exception as e:
        print(f"Error in _enable_generate_from_bag: {str(e)}")
    finally:
        # Always re-enable the checkbox, even if errors occur
        self.generate_from_bag_check.setEnabled(True)
