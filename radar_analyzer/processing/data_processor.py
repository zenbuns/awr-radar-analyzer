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
    # FIXED: Ensure grid size is large enough for the max_range with the given resolution
    max_range = params.max_range
    resolution = params.heatmap_resolution
    
    # Calculate the required grid size - multiply by 2 to cover negative to positive x-axis
    # and 0 to max_range on y-axis
    grid_size = int(2 * max_range / resolution)
    
    # Add a small buffer to ensure we don't lose points at the edges due to rounding
    grid_size += 2
    
    # Ensure grid size is even for better memory alignment and to avoid indexing issues
    if grid_size % 2 == 1:
        grid_size += 1
        
    # Log the calculated size for debugging
    if hasattr(params, 'get_logger'):
        params.get_logger().debug(f"Calculated heatmap grid size: {grid_size}x{grid_size} "
                                 f"for max_range={max_range}m and resolution={resolution}m")
    
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

        # Debug occasionally for validation
        if not hasattr(analyzer, 'grid_debug_counter'):
            analyzer.grid_debug_counter = 0
        
        analyzer.grid_debug_counter += 1
        should_debug = analyzer.grid_debug_counter % 500 == 0

        # CRITICAL FIX: Convert radar coordinates to grid indices EXACTLY as the scatter plot
        # Looking at the scatter plot image, we need to ensure that:
        # - Negative x (azimuth) values appear on left side
        # - Positive x (azimuth) values appear on right side
        # - Y (range) values increase from bottom to top
        #
        # The scatter plot uses a direct mapping where each point at coordinates (x,y)
        # is plotted at that exact position, so we need to do the same for our grid

        # Map from physical coordinates to grid indices
        # For a grid of size (grid_size_y, grid_size_x):
        # 1. X-axis: -max_range to +max_range maps to 0 to grid_size_x-1
        # 2. Y-axis: 0 to max_range maps to 0 to grid_size_y-1
        grid_x = np.floor(((x + max_range) / (2 * max_range) * grid_size_x)).astype(np.int32)
        grid_y = np.floor((y / max_range * grid_size_y)).astype(np.int32)

        # Check if coordinates are in valid range
        valid_mask = (0 <= grid_x) & (grid_x < grid_size_x) & (0 <= grid_y) & (grid_y < grid_size_y)
        
        # Log filtering statistics for debugging
        if should_debug:
            total_points = len(x)
            valid_points = np.sum(valid_mask)
            filtered_ratio = (total_points - valid_points) / total_points * 100 if total_points > 0 else 0
            analyzer.get_logger().debug(f"Grid mapping: x range [{np.min(x):.2f}, {np.max(x):.2f}] → grid_x [{np.min(grid_x) if len(grid_x) > 0 else -1}, {np.max(grid_x) if len(grid_x) > 0 else -1}]")
            analyzer.get_logger().debug(f"Grid mapping: y range [{np.min(y):.2f}, {np.max(y):.2f}] → grid_y [{np.min(grid_y) if len(grid_y) > 0 else -1}, {np.max(grid_y) if len(grid_y) > 0 else -1}]")
            analyzer.get_logger().debug(f"Grid size: ({grid_size_x}, {grid_size_y})")
            analyzer.get_logger().debug(f"Grid filtering: {valid_points}/{total_points} points kept, {filtered_ratio:.1f}% filtered")
            
            # Log details about filtered points if significant filtering occurs
            if filtered_ratio > 10 and total_points > 10:
                invalid_x = x[~valid_mask]
                invalid_y = y[~valid_mask]
                if len(invalid_x) > 0:
                    analyzer.get_logger().warning(
                        f"Points being filtered: "
                        f"x=[{np.min(invalid_x):.1f}, {np.max(invalid_x):.1f}], "
                        f"y=[{np.min(invalid_y):.1f}, {np.max(invalid_y):.1f}]"
                    )
        
        if not np.any(valid_mask):
            return None, None, None, 0, 0

        # Apply mask to get valid indices and intensities
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
        # Log distribution of points for debugging purposes
        if hasattr(analyzer, 'debug_counter') and analyzer.debug_counter % 500 == 0:
            # Calculate distance from origin (radar position)
            distances = np.sqrt(x**2 + y**2)
            
            # Count points in different range bands
            bands = [
                (0, 5), (5, 10), (10, 15), (15, 20), 
                (20, 25), (25, 30), (30, 35), (35, float('inf'))
            ]
            
            # Output detailed distance distribution for debugging
            band_counts = []
            for min_d, max_d in bands:
                count = np.sum((distances >= min_d) & (distances < max_d))
                band_counts.append(f"{min_d}-{max_d}m: {count}")
            
            analyzer.get_logger().debug(f"Point distance distribution: {', '.join(band_counts)}")
            
            # Also log azimuth distribution (horizontal axis)
            azimuth_bands = [
                (-35, -30), (-30, -20), (-20, -10), (-10, 0),
                (0, 10), (10, 20), (20, 30), (30, 35)
            ]
            azimuth_counts = []
            for min_a, max_a in azimuth_bands:
                count = np.sum((x >= min_a) & (x < max_a))
                azimuth_counts.append(f"{min_a}-{max_a}m: {count}")
            
            analyzer.get_logger().debug(f"Azimuth distribution: {', '.join(azimuth_counts)}")
        
        # Use shared grid preparation logic
        grid_x, grid_y, intensity_valid, grid_size_x, grid_size_y = _prepare_grid_indices(analyzer, x, y, intensity)
        if grid_x is None:
            return

        # Ensure heatmap array is initialized with proper dimensions
        if analyzer.live_heatmap_data is None or analyzer.live_heatmap_data.shape != (grid_size_y, grid_size_x):
            analyzer.live_heatmap_data = np.zeros((grid_size_y, grid_size_x), dtype=np.float32)
            analyzer.get_logger().info(f"Re-initialized live heatmap with shape {grid_size_y}x{grid_size_x}")
            
        # CRITICAL FIX: Apply intensity scaling to exactly match scatter plot brightness
        if len(intensity_valid) > 0:
            # Calculate distances from origin for each valid point
            # We need to convert grid indices back to original coordinates
            original_x = (grid_x / grid_size_x * 2 * analyzer.params.max_range) - analyzer.params.max_range
            original_y = grid_y / grid_size_y * analyzer.params.max_range
            distances = np.sqrt(original_x**2 + original_y**2)
            
            # Create distance-based scaling that matches the scatter plot's brightness
            # Looking at the scatter plot, points appear to have uniform brightness
            # regardless of distance, so we need to compensate for natural signal decay
            # The scaling should be proportional to distance from origin
            
            # Simple distance-based scaling: further = higher intensity multiplier
            # Scale from 1.0 (at origin) up to 3.0 (at max_range)
            distance_ratio = distances / analyzer.params.max_range  # 0.0 to 1.0
            scaling_factors = 1.0 + 2.0 * distance_ratio

            # Apply scaling to each point's intensity
            boosted_intensity = intensity_valid * scaling_factors
            
            # Normalize if needed
            max_intensity = np.max(boosted_intensity) if len(boosted_intensity) > 0 else 0
            
            if max_intensity > 100:
                # Scale down if values are too large
                boosted_intensity = boosted_intensity / max_intensity
            elif max_intensity < 0.001 and len(boosted_intensity) > 0:
                # Use default values if too small
                boosted_intensity = np.ones_like(boosted_intensity) * 0.5
            
            # Add to heatmap grid
            np.add.at(analyzer.live_heatmap_data, (grid_y, grid_x), boosted_intensity)
            
            # Log intensity scaling statistics occasionally
            if hasattr(analyzer, 'grid_debug_counter') and analyzer.grid_debug_counter % 500 == 0:
                min_scale = np.min(scaling_factors) if len(scaling_factors) > 0 else 0
                max_scale = np.max(scaling_factors) if len(scaling_factors) > 0 else 0
                
                analyzer.get_logger().debug(
                    f"Distance-based scaling: min={min_scale:.2f}x, max={max_scale:.2f}x, "
                    f"points={len(boosted_intensity)}"
                )
        else:
            # No valid points to add
            pass
            
        # Log heatmap statistics occasionally to verify data is being added correctly
        if not hasattr(analyzer, 'debug_counter'):
            analyzer.debug_counter = 0
            
        analyzer.debug_counter += 1
        if analyzer.debug_counter % 500 == 0:
            # Check basic statistics
            min_val = np.min(analyzer.live_heatmap_data)
            max_val = np.max(analyzer.live_heatmap_data)
            nonzero = np.count_nonzero(analyzer.live_heatmap_data)
            
            # Find the max distance where data exists
            if nonzero > 0:
                nonzero_rows = np.any(analyzer.live_heatmap_data > 0, axis=1)
                max_nonzero_row = np.max(np.where(nonzero_rows)[0]) if np.any(nonzero_rows) else 0
                max_distance = max_nonzero_row / grid_size_y * analyzer.params.max_range
                
                # Check for azimuth (x-axis) coverage
                nonzero_cols = np.any(analyzer.live_heatmap_data > 0, axis=0)
                min_col = np.min(np.where(nonzero_cols)[0]) if np.any(nonzero_cols) else 0
                max_col = np.max(np.where(nonzero_cols)[0]) if np.any(nonzero_cols) else 0
                
                # Convert to physical coordinates
                min_azimuth = (min_col / grid_size_x * 2 * analyzer.params.max_range) - analyzer.params.max_range
                max_azimuth = (max_col / grid_size_x * 2 * analyzer.params.max_range) - analyzer.params.max_range
                
                analyzer.get_logger().debug(
                    f"Heatmap stats: nonzero={nonzero}, min={min_val:.3f}, max={max_val:.3f}, "
                    f"range=[0, {max_distance:.1f}m], azimuth=[{min_azimuth:.1f}, {max_azimuth:.1f}m]"
                )
    except Exception as e:
        analyzer.get_logger().error(f"Error updating live heatmap: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        analyzer.get_logger().error(f"Detailed error: {traceback.format_exc()}")


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
        # FIXED: Prevent decay from eliminating distant points too quickly
        # Apply a position-dependent decay where distant points decay slower
        if analyzer.live_heatmap_data is not None and analyzer.live_heatmap_data.size > 0:
            # Standard decay for all points
            base_decay = analyzer.live_heatmap_decay_factor
            
            # Use more granular decay based on distance from sensor (y-axis in grid)
            # Apply less decay to points that are further away
            grid_size_y, grid_size_x = analyzer.live_heatmap_data.shape
            
            # Create a decay mask where rows further from the sensor (higher y-index) 
            # have reduced decay (values closer to 1.0)
            max_range = analyzer.params.max_range
            heatmap_resolution = analyzer.params.heatmap_resolution
            
            # Create distance-based decay factors
            # OPTIMIZATION: Avoid creating this mask on every call
            if not hasattr(analyzer, '_decay_mask') or analyzer._decay_mask.shape != analyzer.live_heatmap_data.shape:
                # Create y-coordinate array (rows)
                y_coords = np.arange(grid_size_y) * heatmap_resolution
                
                # Calculate distance-dependent decay adjustment (higher for distant points)
                # Linear scaling: 0 at origin, up to 0.1 at max range (reducing decay)
                decay_adjustment = y_coords / max_range * 0.1
                
                # Create 2D mask by broadcasting
                analyzer._decay_mask = base_decay + np.tile(
                    decay_adjustment[:, np.newaxis], (1, grid_size_x)
                )
                
                # Ensure decay values are within valid range
                analyzer._decay_mask = np.clip(analyzer._decay_mask, 0.8, 0.999)
                
                # Log the created decay mask range
                analyzer.get_logger().info(
                    f"Created distance-dependent decay mask: min={np.min(analyzer._decay_mask):.4f}, "
                    f"max={np.max(analyzer._decay_mask):.4f}"
                )
            
            # Apply the position-dependent decay
            analyzer.live_heatmap_data *= analyzer._decay_mask
            
            # OPTIMIZATION: Periodically clean up very small values to prevent numerical issues
            if hasattr(analyzer, 'debug_counter') and analyzer.debug_counter % 100 == 0:
                # Only remove extremely small values that contribute nothing to visualization
                analyzer.live_heatmap_data[analyzer.live_heatmap_data < 1e-6] = 0
        else:
            # Fallback to simple decay if heatmap not initialized
            analyzer.live_heatmap_data *= analyzer.live_heatmap_decay_factor
    except Exception as e:
        analyzer.get_logger().error(f"Error applying heatmap decay: {str(e)}")
        # Log stack trace for better debugging
        import traceback
        analyzer.get_logger().error(f"Decay error details: {traceback.format_exc()}")


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
