#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-frame processing functions for radar point clouds.

This module contains functions for combining multiple frames of radar
point cloud data to improve data quality and reduce noise.
"""

import os
import math
import time
import numpy as np
from .data_processor import calculate_heatmap_size


def process_multi_frame_data(
    analyzer, 
    x_array: np.ndarray, 
    y_array: np.ndarray, 
    z_array: np.ndarray, 
    intensities_array: np.ndarray
) -> None:
    """
    Process incoming point cloud data for multi-frame combination.
    
    This method maintains a buffer of recent frames and combines them according to
    the specified method (average, max, sum).
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        x_array: X-coordinates of current point cloud.
        y_array: Y-coordinates of current point cloud.
        z_array: Z-coordinates of current point cloud.
        intensities_array: Intensity values of current point cloud.
    """
    # Create a copy of the current frame data
    current_frame = {
        'x': x_array.copy(),
        'y': y_array.copy(),
        'z': z_array.copy(),
        'intensities': intensities_array.copy(),
        'timestamp': time.time()
    }
    
    # CRITICAL: Add memory management
    # 1. Set a hard limit on buffer growth
    MAX_BUFFER_SIZE = 100  # Absolute maximum to prevent memory issues
    
    # 2. Clear entire buffer occasionally to prevent memory fragmentation
    if hasattr(analyzer, 'frame_count'):
        analyzer.frame_count += 1
        # Every 1000 frames, do a complete buffer reset to prevent memory fragmentation
        if analyzer.frame_count > 1000:
            analyzer.frame_buffer = []
            analyzer.frame_count = 0
    else:
        analyzer.frame_count = 1
    
    # 3. Add to buffer with proper limiting
    analyzer.frame_buffer.append(current_frame)
    
    # Remove oldest frames to maintain size limit
    while len(analyzer.frame_buffer) > min(analyzer.params.multi_frame_count, MAX_BUFFER_SIZE):
        analyzer.frame_buffer.pop(0)
    
    # Only process if we have enough frames
    if len(analyzer.frame_buffer) >= analyzer.params.multi_frame_count:
        combine_multi_frames(analyzer)
        compute_multi_frame_metrics(analyzer)


def combine_multi_frames(analyzer) -> None:
    """
    Combine multiple frames according to the specified method.
    
    This method creates a combined point cloud from the buffer of frames
    using different techniques (average, max, sum).
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
    """
    # First, we need to establish a common coordinate grid
    # This is a simplified approach - in a real implementation, you might want to use 
    # 3D voxel grid or more sophisticated point cloud registration techniques
    
    max_range = analyzer.params.max_range
    res = analyzer.params.heatmap_resolution
    grid_size_x, grid_size_y = calculate_heatmap_size(analyzer.params)
    
    # Create grid accumulators
    combined_grid = np.zeros((grid_size_y, grid_size_x), dtype=np.float32)
    count_grid = np.zeros((grid_size_y, grid_size_x), dtype=np.int32)
    max_grid = np.zeros((grid_size_y, grid_size_x), dtype=np.float32)
    
    # Add each frame to the grids
    for frame in analyzer.frame_buffer:
        x = frame['x']
        y = frame['y']
        intensities = frame['intensities']
        
        # Convert coordinates to grid indices
        grid_x = np.floor((x + max_range) / res).astype(np.int32)
        grid_y = np.floor(y / res).astype(np.int32)
        
        # Filter out points outside grid bounds
        valid_idx = (0 <= grid_x) & (grid_x < grid_size_x) & (0 <= grid_y) & (grid_y < grid_size_y)
        if not np.any(valid_idx):
            continue
        
        grid_x = grid_x[valid_idx]
        grid_y = grid_y[valid_idx]
        intensity_valid = intensities[valid_idx]
        
        # Accumulate values for averaging and summing
        for i in range(len(grid_x)):
            combined_grid[grid_y[i], grid_x[i]] += intensity_valid[i]
            count_grid[grid_y[i], grid_x[i]] += 1
            max_grid[grid_y[i], grid_x[i]] = max(max_grid[grid_y[i], grid_x[i]], intensity_valid[i])
    
    # Divide by counts to get average where count > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_grid = np.divide(combined_grid, count_grid, where=count_grid > 0)
        avg_grid = np.nan_to_num(avg_grid)  # Replace NaN with 0
    
    # Different combination methods
    if analyzer.params.multi_frame_method == "average":
        result_grid = avg_grid
    elif analyzer.params.multi_frame_method == "max":
        result_grid = max_grid
    elif analyzer.params.multi_frame_method == "sum":
        result_grid = combined_grid
    else:
        # Default to average
        result_grid = avg_grid
    
    # Convert back from grid to point cloud
    non_zero = np.nonzero(result_grid)
    grid_ys, grid_xs = non_zero
    
    # Create point cloud arrays from grid
    combined_x = (grid_xs * res - max_range).astype(np.float32)
    combined_y = (grid_ys * res).astype(np.float32)
    combined_intensities = result_grid[grid_ys, grid_xs].astype(np.float32)
    
    # Store the combined point cloud
    analyzer.combined_frame = {
        'x': combined_x,
        'y': combined_y,
        'z': np.zeros_like(combined_x),  # Z is not combined in this simple implementation
        'intensities': combined_intensities
    }


def compute_multi_frame_metrics(analyzer) -> None:
    """
    Compute quality metrics for the combined multi-frame point cloud.
    
    This method calculates various metrics to evaluate the quality of the
    combined point cloud compared to individual frames.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
    """
    if not analyzer.combined_frame['x'].size:
        return
    
    metrics = {}
    
    # 1. Point density metrics
    metrics['combined_point_count'] = len(analyzer.combined_frame['x'])
    
    avg_single_frame_count = np.mean([len(frame['x']) for frame in analyzer.frame_buffer])
    metrics['avg_single_frame_count'] = float(avg_single_frame_count)
    metrics['point_density_improvement'] = float(len(analyzer.combined_frame['x']) / max(1, avg_single_frame_count))
    
    # 2. Intensity metrics
    if len(analyzer.combined_frame['intensities']) > 0:
        metrics['combined_avg_intensity'] = float(np.mean(analyzer.combined_frame['intensities']))
        metrics['combined_max_intensity'] = float(np.max(analyzer.combined_frame['intensities']))
        metrics['combined_intensity_std'] = float(np.std(analyzer.combined_frame['intensities']))
    
    # 3. Coverage metrics - percentage of voxels filled
    grid_size_x, grid_size_y = calculate_heatmap_size(analyzer.params)
    total_cells = grid_size_x * grid_size_y
    metrics['coverage_percentage'] = float(len(analyzer.combined_frame['x']) / total_cells * 100)
    
    # 4. Noise reduction metrics - using variance of intensities in each occupied voxel
    avg_frame_intensity_std = np.mean(
        [np.std(frame['intensities']) if len(frame['intensities']) > 0 else 0 
         for frame in analyzer.frame_buffer]
    )
    if avg_frame_intensity_std > 0 and 'combined_intensity_std' in metrics:
        metrics['noise_reduction_factor'] = float(avg_frame_intensity_std / metrics['combined_intensity_std'])
    
    # 5. Signal-to-noise ratio (SNR) - simplified approximation
    if 'combined_avg_intensity' in metrics and 'combined_intensity_std' in metrics and metrics['combined_intensity_std'] > 0:
        metrics['snr_dB'] = float(20 * np.log10(metrics['combined_avg_intensity'] / max(0.001, metrics['combined_intensity_std'])))
    
    # 6. 10-frame average metrics with separate ROI calculations
    # Take at most the 10 most recent frames for calculation
    recent_frames = analyzer.frame_buffer[-10:] if len(analyzer.frame_buffer) >= 10 else analyzer.frame_buffer
    
    if recent_frames:
        # Calculate average point count across 10 frames
        ten_frame_point_count = np.mean([len(frame['x']) for frame in recent_frames])
        metrics['ten_frame_avg_point_count'] = float(ten_frame_point_count)
        
        # Calculate average points across 10 frames (mean of points)
        avg_frame_points = np.mean([len(frame['x']) for frame in recent_frames])
        metrics['ten_frame_avg_points'] = float(avg_frame_points)
        
        # Calculate average intensity across 10 frames
        frame_intensities = [np.mean(frame['intensities']) if len(frame['intensities']) > 0 else 0 
                            for frame in recent_frames]
        metrics['ten_frame_avg_intensity'] = float(np.mean(frame_intensities))
        
        # Calculate stability (standard deviation of point counts across frames)
        point_counts = [len(frame['x']) for frame in recent_frames]
        if len(point_counts) > 1:  # Need at least 2 frames for std
            metrics['ten_frame_stability'] = float(1.0 - min(1.0, np.std(point_counts) / max(1, np.mean(point_counts))))
        
        # Calculate 10-frame SNR if possible
        intensity_stds = [np.std(frame['intensities']) if len(frame['intensities']) > 1 else 0 
                        for frame in recent_frames]
        avg_intensity_std = np.mean(intensity_stds) if intensity_stds else 0
        
        if avg_intensity_std > 0 and metrics['ten_frame_avg_intensity'] > 0:
            metrics['ten_frame_snr_dB'] = float(20 * np.log10(metrics['ten_frame_avg_intensity'] / max(0.001, avg_intensity_std)))
        
        # 7. Calculate metrics for all ROI circles
        # Process metrics for each of the three circles
        for circle_idx, circle in enumerate(analyzer.params.circles):
            if not circle.enabled:
                continue
                
            # Prefix for the metrics keys based on circle index
            if circle_idx == 0:
                prefix = 'roi'  # Keep original naming for backward compatibility
            else:
                prefix = f'roi{circle_idx+1}'  # roi2, roi3 for the other circles
            
            # Combine the circle points from each frame in the buffer
            circle_x_points = []
            circle_y_points = []
            circle_intensities = []
            circle_frame_counts = []
            
            # Calculate circle center based on distance and angle
            angle_rad = math.radians(circle.angle)
            circle_center_x = circle.distance * math.sin(angle_rad)
            circle_center_y = circle.distance * math.cos(angle_rad)
            
            # For each frame, filter points in this circle
            for frame in recent_frames:
                if len(frame['x']) == 0:
                    continue
                    
                # Apply ROI circle filter logic for this circle
                dx = frame['x'] - circle_center_x
                dy = frame['y'] - circle_center_y
                dist_sq = dx * dx + dy * dy
                radius_sq = circle.radius ** 2
                
                # Find points within circle
                circle_indices = np.where(dist_sq <= radius_sq)[0]
                
                if len(circle_indices) > 0:
                    circle_x_points.append(frame['x'][circle_indices])
                    circle_y_points.append(frame['y'][circle_indices])
                    circle_intensities.append(frame['intensities'][circle_indices])
                    circle_frame_counts.append(len(circle_indices))
        
            # Calculate metrics for this circle if we have data
            if circle_frame_counts:
                # Basic frame count metrics
                metrics[f'{prefix}_avg_single_frame_count'] = float(np.mean(circle_frame_counts))
                
                # Combine all points for this circle
                all_circle_x = np.concatenate(circle_x_points) if circle_x_points else np.array([])
                all_circle_y = np.concatenate(circle_y_points) if circle_y_points else np.array([])
                all_circle_intensities = np.concatenate(circle_intensities) if circle_intensities else np.array([])
                
                metrics[f'{prefix}_combined_point_count'] = len(all_circle_x)
            
                # Advanced intensity metrics
                if len(all_circle_intensities) > 0:
                    metrics[f'{prefix}_combined_avg_intensity'] = float(np.mean(all_circle_intensities))
                    metrics[f'{prefix}_combined_max_intensity'] = float(np.max(all_circle_intensities))
                    metrics[f'{prefix}_combined_min_intensity'] = float(np.min(all_circle_intensities))
                    metrics[f'{prefix}_combined_intensity_std'] = float(np.std(all_circle_intensities))
                    
                    # Coefficient of variation (measure of relative dispersion)
                    metrics[f'{prefix}_intensity_variability'] = float(metrics[f'{prefix}_combined_intensity_std'] / 
                                                                   max(0.001, metrics[f'{prefix}_combined_avg_intensity']))
                    
                    # Signal-to-noise ratio in dB
                    metrics[f'{prefix}_snr_db'] = float(20 * np.log10(metrics[f'{prefix}_combined_avg_intensity'] / 
                                                           max(0.001, metrics[f'{prefix}_combined_intensity_std'])))
            
                # Advanced point density metrics
                if metrics[f'{prefix}_avg_single_frame_count'] > 0:
                    # Circle area in m²
                    circle_area = np.pi * (circle.radius ** 2)
                    
                    # Raw improvement ratio
                    metrics[f'{prefix}_point_density_improvement'] = float(metrics[f'{prefix}_combined_point_count'] / 
                                                                      max(1, metrics[f'{prefix}_avg_single_frame_count']))
                    
                    # Logarithmic improvement scale (dB) - more scientific
                    metrics[f'{prefix}_density_gain_db'] = float(10 * np.log10(max(1.001, metrics[f'{prefix}_point_density_improvement'])))
                    
                    # Calculate actual spatial point density (points/m²)
                    metrics[f'{prefix}_spatial_density'] = float(metrics[f'{prefix}_combined_point_count'] / max(0.001, circle_area))
                    
                    # Normalized density improvement relative to single frame density
                    single_frame_density = metrics[f'{prefix}_avg_single_frame_count'] / max(0.001, circle_area)
                    metrics[f'{prefix}_normalized_density_factor'] = float(metrics[f'{prefix}_spatial_density'] /
                                                                          max(0.001, single_frame_density))
                else:
                    metrics[f'{prefix}_point_density_improvement'] = 0.0
                    metrics[f'{prefix}_density_gain_db'] = 0.0
                    metrics[f'{prefix}_spatial_density'] = 0.0
                    metrics[f'{prefix}_normalized_density_factor'] = 0.0
            
                # Spatial distribution metrics if we have enough points
                if len(all_circle_x) > 5 and len(all_circle_y) > 5:
                    try:
                        from scipy.spatial import cKDTree
                        points = np.column_stack((all_circle_x, all_circle_y))
                        tree = cKDTree(points)
                        
                        # Calculate nearest neighbor distances for spatial uniformity analysis
                        k = min(4, len(points))
                        distances, _ = tree.query(points, k=k)
                        nn_distances = distances[:, 1:] if distances.shape[1] > 1 else distances
                        
                        # Mean nearest neighbor distance (smaller values indicate better coverage)
                        metrics[f'{prefix}_mean_point_separation'] = float(np.mean(nn_distances))
                        
                        # Uniformity index based on coefficient of variation of distances
                        # (0-1 scale where higher values indicate more uniform distribution)
                        std_nn_dist = np.std(nn_distances)
                        metrics[f'{prefix}_spatial_uniformity'] = float(1.0 - min(1.0, std_nn_dist / 
                                                                      max(0.001, metrics[f'{prefix}_mean_point_separation'])))
                    except Exception as e:
                        analyzer.get_logger().warning(f"Unable to calculate spatial metrics for {prefix}: {str(e)}")
        
        # Calculate metrics for points outside any ROI circle
        # Instead of concatenating all points, calculate frame by frame for consistency with ROI calculations
        outside_roi_x_points = []
        outside_roi_y_points = []
        outside_roi_intensities = []
        outside_roi_frame_counts = []
        
        # Process each frame individually to maintain frame integrity
        for frame in recent_frames:
            if len(frame['x']) == 0:
                outside_roi_frame_counts.append(0)
                continue
            
            # Start with all points in this frame
            frame_mask = np.ones(len(frame['x']), dtype=bool)
            
            # Exclude points that are within any enabled ROI circle
            for circle_idx, circle in enumerate(analyzer.params.circles):
                if not circle.enabled:
                    continue
                
                # Calculate circle center based on distance and angle
                angle_rad = math.radians(circle.angle)
                circle_center_x = circle.distance * math.sin(angle_rad)
                circle_center_y = circle.distance * math.cos(angle_rad)
                
                # Calculate squared distances to circle center
                dx = frame['x'] - circle_center_x
                dy = frame['y'] - circle_center_y
                dist_sq = dx * dx + dy * dy
                radius_sq = circle.radius ** 2
                
                # Update frame mask to exclude points in this circle
                circle_mask = dist_sq > radius_sq  # Points OUTSIDE this circle
                frame_mask = frame_mask & circle_mask
            
            # Count points outside all ROIs in this frame
            outside_points_count = np.sum(frame_mask)
            outside_roi_frame_counts.append(outside_points_count)
            
            # Only collect points if there are any outside ROIs
            if outside_points_count > 0:
                outside_roi_x_points.append(frame['x'][frame_mask])
                outside_roi_y_points.append(frame['y'][frame_mask])
                outside_roi_intensities.append(frame['intensities'][frame_mask])
        
        # Calculate combined points (sum across all frames)
        combined_outside_x = np.concatenate(outside_roi_x_points) if outside_roi_x_points else np.array([])
        combined_outside_y = np.concatenate(outside_roi_y_points) if outside_roi_y_points else np.array([])
        combined_outside_intensities = np.concatenate(outside_roi_intensities) if outside_roi_intensities else np.array([])
        
        # Store metrics
        metrics['outside_roi_combined_point_count'] = len(combined_outside_x)
        metrics['outside_roi_avg_single_frame_count'] = float(np.mean(outside_roi_frame_counts)) if outside_roi_frame_counts else 0.0
        
        # Calculate the effective observed area more accurately
        # First, determine the actual radar coverage area based on max range
        if hasattr(analyzer.params, 'max_range'):
            # Use a sector model if we have angular information
            max_range = analyzer.params.max_range
            
            # Calculate the area excluding the ROI circles
            # For radar, typically this would be a sector or cone, but we'll approximate with a half-circle
            radar_coverage_area = (np.pi * max_range**2) / 2.0  # Half-circle
            
            # Subtract the areas of enabled ROI circles
            roi_area = 0.0
            for circle_idx, circle in enumerate(analyzer.params.circles):
                if circle.enabled:
                    roi_area += np.pi * (circle.radius ** 2)
            
            # Ensure area doesn't become negative or too small
            outside_area = max(0.01, radar_coverage_area - roi_area)
        else:
            # Fallback calculation if max_range isn't available
            grid_size_x, grid_size_y = calculate_heatmap_size(analyzer.params)
            total_area = grid_size_x * grid_size_y * (analyzer.params.heatmap_resolution ** 2)
            
            # Subtract ROI areas
            roi_area = 0.0
            for circle_idx, circle in enumerate(analyzer.params.circles):
                if circle.enabled:
                    roi_area += np.pi * (circle.radius ** 2)
            
            outside_area = max(0.01, total_area - roi_area)
        
        # Calculate spatial density (points per m²) if we have points
        if metrics['outside_roi_combined_point_count'] > 0:
            metrics['outside_roi_spatial_density'] = float(metrics['outside_roi_combined_point_count'] / outside_area)
        else:
            metrics['outside_roi_spatial_density'] = 0.0
        
        # Calculate SNR and other intensity metrics if we have enough points
        if len(combined_outside_intensities) > 1:
            metrics['outside_roi_avg_intensity'] = float(np.mean(combined_outside_intensities))
            metrics['outside_roi_max_intensity'] = float(np.max(combined_outside_intensities))
            metrics['outside_roi_min_intensity'] = float(np.min(combined_outside_intensities))
            metrics['outside_roi_intensity_std'] = float(np.std(combined_outside_intensities))
            
            # Calculate SNR in dB using proper radar signal processing approach
            # SNR = 10 * log10(Signal^2 / Noise^2) = 20 * log10(Signal / Noise)
            if metrics['outside_roi_intensity_std'] > 0:
                metrics['outside_roi_snr_db'] = float(20 * np.log10(
                    max(0.001, metrics['outside_roi_avg_intensity']) / 
                    max(0.001, metrics['outside_roi_intensity_std'])
                ))
            else:
                metrics['outside_roi_snr_db'] = 0.0
        else:
            # Set defaults for missing metrics
            metrics['outside_roi_avg_intensity'] = 0.0
            metrics['outside_roi_max_intensity'] = 0.0
            metrics['outside_roi_min_intensity'] = 0.0
            metrics['outside_roi_intensity_std'] = 0.0
            metrics['outside_roi_snr_db'] = 0.0
        
        # Log the outside ROI metrics for debugging
        analyzer.get_logger().info(
            f"Outside ROI metrics: points={metrics['outside_roi_combined_point_count']}, "
            f"avg_pts/frame={metrics['outside_roi_avg_single_frame_count']:.2f}, "
            f"spatial_density={metrics['outside_roi_spatial_density']:.3f} pts/m², "
            f"SNR={metrics['outside_roi_snr_db']:.2f} dB"
        )
        
        analyzer.get_logger().info(f"10-frame metrics computed: avg points={metrics['ten_frame_avg_point_count']:.3f}, " +
                                  f"avg intensity={metrics['ten_frame_avg_intensity']:.3f}, stability={metrics.get('ten_frame_stability', 0):.3f}")
    
    # Store metrics for data collection
    analyzer.multi_frame_metrics = metrics
    
    if analyzer.collecting_data:
        analyzer.get_logger().info(f"Multi-frame metrics updated: {len(analyzer.combined_frame['x'])} points")

    # IMPORTANT: Memory management - release large temporary data arrays
    # Only keep the combined frame results and metrics, allow other temporary arrays to be garbage collected
    temporary_arrays = []
    for frame in analyzer.frame_buffer:
        frame.pop('x_grid', None)
        frame.pop('y_grid', None)
        frame.pop('temp_grid', None)
        
    # Force garbage collection if the system supports it
    try:
        import gc
        gc.collect()
    except ImportError:
        pass


def load_latest_multi_frame_metrics(analyzer) -> bool:
    """Load the latest multi-frame metrics from available JSON files.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        
    Returns:
        bool: True if metrics were successfully loaded, False otherwise.
    """
    try:
        import glob
        import json
        
        # Find the most recent multi_frame JSON files in all config directories
        data_dir = os.path.expanduser('~/radar_experiment_data')
        multi_frame_metrics = {}
        
        # Check all configuration directories
        for config_dir in [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]:  
            pattern = os.path.join(data_dir, config_dir, 'multi_frame_*.json')
            json_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        metrics = json.load(f)
                        analyzer.get_logger().info(f"Loaded multi-frame metrics from {json_files[0]}")
                        multi_frame_metrics[config_dir] = metrics
                except Exception as e:
                    analyzer.get_logger().error(f"Error loading multi-frame metrics for {config_dir}: {e}")
        
        if multi_frame_metrics:
            analyzer.multi_frame_metrics = multi_frame_metrics
            return True
        return False
        
    except Exception as e:
        analyzer.get_logger().error(f"Error scanning for multi-frame metrics: {e}")
        return False