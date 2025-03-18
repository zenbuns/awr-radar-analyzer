#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for radar data processing.

This module provides helper functions for processing radar data,
including filtering, statistical analysis, and data transformations.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union
import scipy.ndimage


def calculate_snr(signal: np.ndarray, noise_floor: float = 0.05) -> float:
    """
    Calculate the signal-to-noise ratio in decibels.

    Args:
        signal: The signal data (1D or 2D array).
        noise_floor: The noise floor level to use for calculation.

    Returns:
        Signal-to-noise ratio in decibels.
    """
    if signal.size == 0 or np.max(signal) <= noise_floor:
        return 0.0
    
    return 10.0 * np.log10(np.max(signal) / noise_floor)


def filter_points_by_distance(
    x: np.ndarray,
    y: np.ndarray,
    intensities: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter points to find those within a specified radius from a center point.

    Args:
        x: X-coordinates of points.
        y: Y-coordinates of points.
        intensities: Intensity values of points.
        center_x: X-coordinate of circle center.
        center_y: Y-coordinate of circle center.
        radius: Radius of the circle for filtering.

    Returns:
        Tuple containing (filtered_x, filtered_y, filtered_intensities, indices)
    """
    if len(x) == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int32)
        )
    
    # Vectorized distance calculation
    dx = x - center_x
    dy = y - center_y
    dist_sq = dx * dx + dy * dy
    radius_sq = radius ** 2
    
    # Find points within circle
    indices = np.where(dist_sq <= radius_sq)[0]
    
    return x[indices], y[indices], intensities[indices], indices


def calculate_distance_bands(
    x: np.ndarray,
    y: np.ndarray,
    intensities: np.ndarray,
    max_range: float,
    band_interval: float
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for points in different distance bands from origin.

    Args:
        x: X-coordinates of points.
        y: Y-coordinates of points.
        intensities: Intensity values of points.
        max_range: Maximum range for analysis in meters.
        band_interval: Width of each distance band in meters.

    Returns:
        Dictionary mapping distance bands to statistics.
    """
    # Calculate distances from origin
    distances = np.sqrt(x ** 2 + y ** 2)
    
    # Create distance bins
    bins = np.arange(0, max_range + band_interval, band_interval)
    counts, _ = np.histogram(distances, bins=bins)
    
    distance_bands = {}
    for i in range(len(counts)):
        band_key = f"{bins[i]}-{bins[i+1]}m"
        band_mask = (distances >= bins[i]) & (distances < bins[i+1])
        band_intensities = intensities[band_mask]
        avg_intensity = float(np.mean(band_intensities)) if band_intensities.size > 0 else 0.0
        distance_bands[band_key] = {
            'count': float(counts[i]),
            'avg_intensity': avg_intensity
        }
    
    return distance_bands


def find_target_band(distance_bands: Dict[str, Dict[str, float]], target_distance: float) -> str:
    """
    Find the distance band that contains the target distance.

    Args:
        distance_bands: Dictionary of distance bands as returned by calculate_distance_bands.
        target_distance: Target distance to find in meters.

    Returns:
        The key of the band containing the target distance, or the first band if none match.
    """
    if not distance_bands:
        return "0-0m"
    
    target_band = next(
        (band for band in distance_bands
         if float(band.split('-')[0]) <= target_distance
         < float(band.split('-')[1].replace('m', ''))),
        list(distance_bands.keys())[0]
    )
    
    return target_band


def grid_heatmap_data(
    x: np.ndarray,
    y: np.ndarray,
    intensities: np.ndarray,
    max_range: float,
    resolution: float,
    grid_size: Tuple[int, int]
) -> np.ndarray:
    """
    Convert point cloud data to gridded heatmap data.

    Args:
        x: X-coordinates of points.
        y: Y-coordinates of points.
        intensities: Intensity values of points.
        max_range: Maximum range for the grid in meters.
        resolution: Size of each grid cell in meters.
        grid_size: Tuple (width, height) specifying grid dimensions.

    Returns:
        2D numpy array containing gridded intensity data.
    """
    if len(x) == 0:
        return np.zeros(grid_size, dtype=np.float32)
    
    grid_size_x, grid_size_y = grid_size
    grid_data = np.zeros(grid_size, dtype=np.float32)
    
    # Convert coordinates to grid indices
    grid_x = np.floor((x + max_range) / resolution).astype(np.int32)
    grid_y = np.floor(y / resolution).astype(np.int32)
    
    # Filter out points outside grid bounds
    valid_idx = (0 <= grid_x) & (grid_x < grid_size_x) & (0 <= grid_y) & (grid_y < grid_size_y)
    if not np.any(valid_idx):
        return grid_data
    
    grid_x = grid_x[valid_idx]
    grid_y = grid_y[valid_idx]
    intensity_valid = intensities[valid_idx]
    
    # Use np.add.at for efficient accumulation
    np.add.at(grid_data, (grid_y, grid_x), intensity_valid)
    
    return grid_data


def compute_circle_statistics(
    x: np.ndarray,
    y: np.ndarray,
    intensities: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float
) -> Dict[str, Union[int, float]]:
    """
    Compute statistics for points within a circle.

    Args:
        x: X-coordinates of points.
        y: Y-coordinates of points.
        intensities: Intensity values of points.
        center_x: X-coordinate of circle center.
        center_y: Y-coordinate of circle center.
        radius: Radius of the circle.

    Returns:
        Dictionary with point count and average intensity.
    """
    filtered_x, filtered_y, filtered_intensities, _ = filter_points_by_distance(
        x, y, intensities, center_x, center_y, radius
    )
    
    count = len(filtered_x)
    avg_intensity = float(np.mean(filtered_intensities)) if count > 0 else 0.0
    
    return {
        'count': count,
        'avg_intensity': avg_intensity
    }


def apply_gaussian_smoothing(data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to heatmap data.

    Args:
        data: 2D array of data to smooth.
        sigma: Standard deviation for Gaussian kernel.

    Returns:
        Smoothed 2D array.
    """
    try:
        return scipy.ndimage.gaussian_filter(data, sigma=sigma)
    except ImportError:
        print("Warning: scipy.ndimage not available. Smoothing skipped.")
        return data.copy()


def analyze_heatmap_region(
    heatmap_data: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    max_range: float,
    resolution: float,
    noise_floor: float = 0.05
) -> Dict[str, float]:
    """
    Analyze a circular region within a heatmap.

    Args:
        heatmap_data: 2D heatmap array.
        center_x: X-coordinate of region center (in meters).
        center_y: Y-coordinate of region center (in meters).
        radius: Radius of the region (in meters).
        max_range: Maximum range for the grid in meters.
        resolution: Size of each grid cell in meters.
        noise_floor: Threshold for noise floor.

    Returns:
        Dictionary of statistics for the region.
    """
    if heatmap_data is None or heatmap_data.size == 0:
        return {
            'mean_intensity': 0.0,
            'max_intensity': 0.0,
            'std_intensity': 0.0,
            'signal_coverage': 0.0
        }
    
    # Convert center coordinates to grid indices
    grid_size_y, grid_size_x = heatmap_data.shape
    
    # Create coordinate arrays for all grid points
    y_indices, x_indices = np.indices((grid_size_y, grid_size_x))
    x_coords = x_indices * resolution - max_range
    y_coords = y_indices * resolution
    
    # Calculate distances from each grid point to region center
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    region_mask = distances <= radius
    region_data = heatmap_data[region_mask]
    
    if region_data.size > 0:
        return {
            'mean_intensity': float(np.mean(region_data)),
            'max_intensity': float(np.max(region_data)),
            'std_intensity': float(np.std(region_data)),
            'signal_coverage': float(np.sum(region_data > noise_floor)) / region_data.size
        }
    
    return {
        'mean_intensity': 0.0,
        'max_intensity': 0.0,
        'std_intensity': 0.0,
        'signal_coverage': 0.0
    }