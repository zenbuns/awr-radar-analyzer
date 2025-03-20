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
            # Initialize empty arrays for primary circle (for backward compatibility)
            analyzer.current_data['circle_x'] = np.array([], dtype=np.float32)
            analyzer.current_data['circle_y'] = np.array([], dtype=np.float32)
            analyzer.current_data['circle_intensities'] = np.array([], dtype=np.float32)
            analyzer.current_data['circle_indices'] = np.array([], dtype=np.int32)
            analyzer.current_data['circle_distances'] = np.array([], dtype=np.float32)
            analyzer.current_data['circle_distance_bands'] = {}
            
            # Initialize empty arrays for all circles
            for i in range(len(analyzer.params.circles)):
                circle_key = f'circle{i+1}' if i > 0 else 'circle'  # Use 'circle' for primary for backward compatibility
                analyzer.current_data[f'{circle_key}_x'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_y'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_intensities'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_indices'] = np.array([], dtype=np.int32)
                analyzer.current_data[f'{circle_key}_distances'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_distance_bands'] = {}
            return
        
        analyzer.get_logger().debug(f"Processing points for {len(analyzer.params.circles)} circles, total input points: {len(x)}")
        
        # Process each enabled circle
        for i, circle in enumerate(analyzer.params.circles):
            if not circle.enabled:
                # Initialize empty arrays for disabled circles
                circle_key = f'circle{i+1}' if i > 0 else 'circle'  # Use 'circle' for primary for backward compatibility
                analyzer.current_data[f'{circle_key}_x'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_y'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_intensities'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_indices'] = np.array([], dtype=np.int32)
                analyzer.current_data[f'{circle_key}_distances'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_distance_bands'] = {}
                analyzer.get_logger().debug(f"Circle {i} ('{circle.label}') is disabled, skipping")
                continue
                
            # Calculate circle center based on distance and angle
            angle_rad = math.radians(circle.angle)
            circle_center_x = circle.distance * math.sin(angle_rad)
            circle_center_y = circle.distance * math.cos(angle_rad)
            
            # Cache the circle center if it's the primary circle (for backward compatibility)
            if i == 0:
                analyzer._cached_circle_center = np.array([circle_center_x, circle_center_y])
            
            # Vectorized distance calculation
            dx = x - circle_center_x
            dy = y - circle_center_y
            dist_sq = dx * dx + dy * dy
            
            # Add a small tolerance factor (0.5% of radius squared) to ensure edge points are included
            # This is especially important for larger circle sizes
            radius_sq = (circle.radius ** 2) * 1.005  # 0.5% tolerance
            
            # Alternative: Add a small fixed epsilon for numerical stability
            # radius_sq = (circle.radius ** 2) + 0.001  # Fixed small epsilon

            # Find points within circle
            circle_indices = np.where(dist_sq <= radius_sq)[0]
            
            # Set the key name based on circle index
            circle_key = f'circle{i+1}' if i > 0 else 'circle'  # Use 'circle' for primary for backward compatibility
            
            # Store circle points
            analyzer.current_data[f'{circle_key}_x'] = x[circle_indices]
            analyzer.current_data[f'{circle_key}_y'] = y[circle_indices]
            analyzer.current_data[f'{circle_key}_intensities'] = intensities[circle_indices]
            analyzer.current_data[f'{circle_key}_indices'] = circle_indices
            
            # Calculate and store actual distances from origin for each point
            if len(circle_indices) > 0:
                point_distances = np.sqrt(x[circle_indices]**2 + y[circle_indices]**2)
                analyzer.current_data[f'{circle_key}_distances'] = point_distances
                
                # Categorize points by distance bands (1-meter bands by default)
                band_width = 1.0  # width of each distance band in meters
                min_dist = max(0, int(np.floor(np.min(point_distances))))
                max_dist = int(np.ceil(np.max(point_distances)))
                distance_bands = {}
                
                for band_start in range(min_dist, max_dist + 1):
                    band_end = band_start + band_width
                    band_key = f"{band_start}m-{band_end}m"
                    in_band = (point_distances >= band_start) & (point_distances < band_end)
                    band_indices = np.where(in_band)[0]
                    band_count = len(band_indices)
                    
                    if band_count > 0:
                        distance_bands[band_key] = {
                            'count': band_count,
                            'indices': band_indices,
                            'distances': point_distances[band_indices]
                        }
                
                analyzer.current_data[f'{circle_key}_distance_bands'] = distance_bands
                
                # Generate comprehensive distance statistics for logging
                band_counts = [f"{k}: {v['count']} pts" for k, v in distance_bands.items()]
                bands_str = ", ".join(band_counts)
                
                analyzer.get_logger().debug(f"Circle {i} ('{circle.label}'): center=({circle_center_x:.2f}, {circle_center_y:.2f}), " + 
                                          f"radius={circle.radius:.2f}, found {len(circle_indices)} points")
                analyzer.get_logger().debug(f"Distance bands: {bands_str}")
            else:
                analyzer.current_data[f'{circle_key}_distances'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_distance_bands'] = {}
                analyzer.get_logger().debug(f"Circle {i} ('{circle.label}'): center=({circle_center_x:.2f}, {circle_center_y:.2f}), " + 
                                          f"radius={circle.radius:.2f}, found 0 points")
            
            # Add detailed distance logging for edge cases when circle radius is large
            if circle.radius > 1.0 and len(circle_indices) > 0:
                # Get min and max distances of detected points
                distances = np.sqrt(dist_sq[circle_indices])
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                actual_circle_radius = circle.radius
                effective_circle_radius = np.sqrt(radius_sq)
                
                analyzer.get_logger().debug(f"Circle {i} distance metrics: " +
                                         f"min_dist={min_dist:.3f}m, " +
                                         f"max_dist={max_dist:.3f}m, " +
                                         f"circle_radius={actual_circle_radius:.3f}m, " +
                                         f"effective_radius_with_tolerance={effective_circle_radius:.3f}m")
            
    except Exception as e:
        analyzer.get_logger().error(f"Error filtering points in circles: {str(e)}")
        # Initialize empty arrays in case of error
        analyzer.current_data['circle_x'] = np.array([], dtype=np.float32)
        analyzer.current_data['circle_y'] = np.array([], dtype=np.float32)
        analyzer.current_data['circle_intensities'] = np.array([], dtype=np.float32)
        analyzer.current_data['circle_indices'] = np.array([], dtype=np.int32)
        analyzer.current_data['circle_distances'] = np.array([], dtype=np.float32)
        analyzer.current_data['circle_distance_bands'] = {}


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
        np.add.at(analyzer.live_heatmap_data, (grid_y, grid_x), intensity_valid)
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
