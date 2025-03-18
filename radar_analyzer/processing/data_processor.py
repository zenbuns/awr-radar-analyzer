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
    using vectorized operations.

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
            
            # Initialize empty arrays for all circles
            for i in range(len(analyzer.params.circles)):
                circle_key = f'circle{i+1}' if i > 0 else 'circle'  # Use 'circle' for primary for backward compatibility
                analyzer.current_data[f'{circle_key}_x'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_y'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_intensities'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_indices'] = np.array([], dtype=np.int32)
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
            radius_sq = circle.radius ** 2

            # Find points within circle
            circle_indices = np.where(dist_sq <= radius_sq)[0]
            
            # Set the key name based on circle index
            circle_key = f'circle{i+1}' if i > 0 else 'circle'  # Use 'circle' for primary for backward compatibility
            
            # Store circle points
            analyzer.current_data[f'{circle_key}_x'] = x[circle_indices]
            analyzer.current_data[f'{circle_key}_y'] = y[circle_indices]
            analyzer.current_data[f'{circle_key}_intensities'] = intensities[circle_indices]
            analyzer.current_data[f'{circle_key}_indices'] = circle_indices
            
            analyzer.get_logger().debug(f"Circle {i} ('{circle.label}'): center=({circle_center_x:.2f}, {circle_center_y:.2f}), " + 
                                      f"radius={circle.radius:.2f}, found {len(circle_indices)} points")
            
    except Exception as e:
        analyzer.get_logger().error(f"Error filtering points in circles: {str(e)}")
        # Initialize empty arrays in case of error
        analyzer.current_data['circle_x'] = np.array([], dtype=np.float32)
        analyzer.current_data['circle_y'] = np.array([], dtype=np.float32)
        analyzer.current_data['circle_intensities'] = np.array([], dtype=np.float32)
        analyzer.current_data['circle_indices'] = np.array([], dtype=np.int32)
        
        # Also initialize arrays for other circles
        for i in range(1, len(analyzer.params.circles)):
            analyzer.current_data[f'circle{i+1}_x'] = np.array([], dtype=np.float32)
            analyzer.current_data[f'circle{i+1}_y'] = np.array([], dtype=np.float32)
            analyzer.current_data[f'circle{i+1}_intensities'] = np.array([], dtype=np.float32)
            analyzer.current_data[f'circle{i+1}_indices'] = np.array([], dtype=np.int32)


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
