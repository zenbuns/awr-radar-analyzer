#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for radar visualization.

This module provides helper functions for creating and customizing
radar data visualizations using matplotlib.
"""

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, Tuple, List, Optional, Any


def setup_radar_scatter_figure(
    max_range: float,
    circle_interval: float,
    circle_configs: List[Dict[str, Any]]
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """
    Set up a radar scatter plot figure with multiple circles.
    
    Args:
        max_range: Maximum radar range in meters.
        circle_interval: Interval between range circles in meters.
        circle_configs: List of circle configurations with distance, radius, angle, etc.
        
    Returns:
        Tuple containing (figure, axes, components_dict)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # Set background colors
    ax.set_facecolor('#001A3A')  # Darker blue for better contrast
    fig.patch.set_facecolor('#001A3A')
    
    # Set plot limits
    x_limit, y_limit = max_range, 2 * max_range
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(0, y_limit)
    
    # Add range arcs
    for r in range(0, int(y_limit) + 1, int(circle_interval)):
        arc = patches.Arc(
            (0, 0),
            width=2 * r,
            height=2 * r,
            angle=0,
            theta1=0,
            theta2=180,
            fill=False,
            color='white',
            linestyle=':',
            linewidth=0.8,
            alpha=0.7
        )
        ax.add_patch(arc)
        if 0 < r <= max_range and r % (int(circle_interval) * 2) == 0:
            ax.text(0, r, f"{int(r)}m", ha='right', va='bottom', color='white', fontsize=9, weight='bold')
    
    # Create scatter plots and circles
    components = {
        'scatter': ax.scatter([], [], s=8, c=[], cmap='plasma'),
        'sampling_circles': [],
        'circle_scatters': []
    }
    
    # Add each sampling circle
    for i, config in enumerate(circle_configs):
        # Calculate position based on angle
        angle_rad = config['angle'] * (np.pi / 180.0)
        x_pos = config['distance'] * np.sin(angle_rad)
        y_pos = config['distance'] * np.cos(angle_rad)
        
        # Create circle scatter
        circle_scatter = ax.scatter([], [], s=10, c=config['color'], marker='x')
        components[f'circle_scatter_{i}'] = circle_scatter
        components['circle_scatters'].append(circle_scatter)
        
        # Create sampling circle
        sampling_circle = patches.Circle(
            (x_pos, y_pos),
            config['radius'],
            fill=False,
            color=config['color'],
            linestyle='-',
            linewidth=1.5,
            alpha=0.8 if config['enabled'] else 0.2,
            visible=config['enabled']
        )
        ax.add_patch(sampling_circle)
        components[f'sampling_circle_{i}'] = sampling_circle
        components['sampling_circles'].append({
            'circle': sampling_circle,
            'config': config,
            'x_pos': x_pos,
            'y_pos': y_pos
        })
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12, labelpad=8, color='white')
    ax.set_ylabel('Y (m)', fontsize=12, labelpad=8, color='white')
    ax.set_title('Radar Point Cloud', fontsize=14, color='white')
    
    # Configure ticks
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    
    # Set spine colors
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Add colorbar
    colorbar = fig.colorbar(
        components['scatter'],
        ax=ax,
        label='Intensity',
        fraction=0.04,
        pad=0.02
    )
    colorbar.ax.yaxis.label.set_color('white')
    colorbar.ax.tick_params(colors='white')
    
    # Add statistics text box
    components['stats_text'] = ax.text(
        0.02, 0.98, '',
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=10,
        color='white',
        bbox=dict(
            boxstyle='round',
            facecolor='#00000080',
            alpha=0.7,
            edgecolor='white'
        )
    )
    
    fig.tight_layout()
    return fig, ax, components


def setup_heatmap_figure(
    heatmap_data: np.ndarray,
    max_range: float,
    circle_interval: float,
    target_distance: float,
    circle_configs: List[Dict[str, Any]],
    colormap: str = 'plasma',
    noise_floor: float = 0.05
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """
    Set up a heatmap figure with improved visibility.
    
    Args:
        heatmap_data: 2D numpy array with intensity values.
        max_range: Maximum radar range in meters.
        circle_interval: Interval between range circles in meters.
        target_distance: Target distance for arc display.
        circle_configs: List of circle configurations.
        colormap: Matplotlib colormap name.
        noise_floor: Threshold for noise floor.
        
    Returns:
        Tuple containing (figure, axes, components_dict)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # Set background colors
    ax.set_facecolor('#001A3A')
    fig.patch.set_facecolor('#001A3A')
    
    # Apply noise floor threshold
    data_thresholded = heatmap_data.copy()
    data_thresholded[data_thresholded < noise_floor] = 0
    
    # Create heatmap
    cmap = plt.cm.get_cmap(colormap).copy()
    norm = colors.PowerNorm(gamma=0.7, vmin=noise_floor, vmax=np.max(data_thresholded) or 1.0)
    
    heatmap_img = ax.imshow(
        data_thresholded,
        extent=[-max_range, max_range, 0, 2 * max_range],
        origin='lower',
        cmap=cmap,
        norm=norm,
        aspect='auto',
        interpolation='bilinear',
        alpha=0.9
    )
    
    components = {
        'heatmap': heatmap_img,
        'norm': norm,
        'colormap': colormap,
        'sampling_circles': []
    }
    
    # Add colorbar with larger font
    colorbar = fig.colorbar(
        heatmap_img,
        ax=ax,
        label='Intensity',
        fraction=0.04,
        pad=0.02
    )
    colorbar.ax.tick_params(labelsize=10, colors='white')
    colorbar.set_label('Intensity', size=12, color='white')
    components['colorbar'] = colorbar
    
    # Define range arcs
    major_ranges = list(range(0, int(2 * max_range) + 1, int(circle_interval) * 2))
    minor_ranges = [
        r for r in range(0, int(2 * max_range) + 1, int(circle_interval))
        if r not in major_ranges
    ]
    
    # Add major range arcs
    for r in major_ranges:
        arc = patches.Arc(
            (0, 0),
            width=2 * r,
            height=2 * r,
            angle=0,
            theta1=0,
            theta2=180,
            fill=False,
            color='white',
            linestyle='-',
            linewidth=1.0,
            alpha=0.8
        )
        ax.add_patch(arc)
        if 0 < r <= max_range:
            ax.text(
                0, r, f"{int(r)}m", ha='right', va='bottom',
                color='white', fontsize=10, fontweight='bold',
                alpha=1.0,
                bbox=dict(facecolor='#00000080', edgecolor='none', pad=1)
            )
    
    # Add minor range arcs
    for r in minor_ranges:
        arc = patches.Arc(
            (0, 0),
            width=2 * r,
            height=2 * r,
            angle=0,
            theta1=0,
            theta2=180,
            fill=False,
            color='white',
            linestyle=':',
            linewidth=0.5,
            alpha=0.4
        )
        ax.add_patch(arc)
    
    # Add target arc
    target_arc = patches.Arc(
        (0, 0),
        width=2 * target_distance,
        height=2 * target_distance,
        angle=0,
        theta1=0,
        theta2=180,
        fill=False,
        color='red',
        linestyle='-',
        linewidth=1.0,
        alpha=0.8
    )
    ax.add_patch(target_arc)
    components['target_arc'] = target_arc
    
    # Add each sampling circle
    for i, config in enumerate(circle_configs):
        if config['enabled']:
            # Calculate position based on angle
            angle_rad = config['angle'] * (np.pi / 180.0)
            x_pos = config['distance'] * np.sin(angle_rad)
            y_pos = config['distance'] * np.cos(angle_rad)
            
            # Create sampling circle
            sampling_circle = patches.Circle(
                (x_pos, y_pos),
                config['radius'],
                fill=False,
                color=config['color'],
                linestyle='-',
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(sampling_circle)
            components[f'sampling_circle_{i}'] = sampling_circle
            components['sampling_circles'].append({
                'circle': sampling_circle,
                'config': config,
                'x_pos': x_pos,
                'y_pos': y_pos
            })
    
    # Add grid
    ax.grid(True, color='white', linestyle=':', linewidth=0.2, alpha=0.1)
    
    # Set plot limits
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(0, 2 * max_range)
    
    # Add labels
    ax.set_xlabel('Cross-Range (m)', fontsize=12, labelpad=8, color='white')
    ax.set_ylabel('Range (m)', fontsize=12, labelpad=8, color='white')
    ax.set_title('Radar Intensity Map', fontsize=14, color='white')
    
    # Configure ticks
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    
    # Set spine colors
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    
    # Add SNR text
    if np.max(data_thresholded) > 0:
        snr = 10.0 * np.log10(np.max(data_thresholded) / noise_floor)
        snr_text = f'SNR: {snr:.1f} dB'
    else:
        snr_text = 'SNR: N/A'
        
    components['snr_text'] = ax.text(
        0.02, 0.02, snr_text,
        transform=ax.transAxes,
        color='yellow',
        fontsize=10,
        bbox=dict(
            boxstyle='round',
            facecolor='black',
            alpha=0.7
        )
    )
    
    fig.tight_layout()
    return fig, ax, components


def add_contours_to_heatmap(
    ax: Axes,
    data: np.ndarray,
    extent: List[float],
    noise_floor: float,
    levels: int = 8,
    colormap: str = 'plasma',
    alpha: float = 0.7
) -> Any:
    """
    Add contours to an existing heatmap.
    
    Args:
        ax: Matplotlib Axes object.
        data: 2D numpy array with intensity values.
        extent: List [xmin, xmax, ymin, ymax] for contour extents.
        noise_floor: Threshold for noise floor.
        levels: Number of contour levels.
        colormap: Matplotlib colormap name.
        alpha: Opacity of contours.
        
    Returns:
        Contour object added to the axes.
    """
    from scipy.ndimage import gaussian_filter
    
    # Apply Gaussian smoothing
    smoothed_data = gaussian_filter(data, sigma=2.0)
    
    if np.max(smoothed_data) > noise_floor:
        # Create contour levels
        contour_levels = np.linspace(noise_floor, np.max(smoothed_data), levels)
        
        # Generate grid coordinates
        x_grid = np.linspace(extent[0], extent[1], smoothed_data.shape[1])
        y_grid = np.linspace(extent[2], extent[3], smoothed_data.shape[0])
        
        # Create filled contours
        contour = ax.contourf(
            x_grid, y_grid, smoothed_data,
            levels=contour_levels,
            cmap=colormap,
            alpha=alpha
        )
        
        # Add contour lines
        ax.contour(
            x_grid, y_grid, smoothed_data,
            levels=contour_levels,
            colors='white',
            alpha=0.4,
            linewidths=0.5
        )
        
        return contour
    
    return None


def save_scientific_visualization(
    filename: str,
    heatmap_data: np.ndarray,
    max_range: float,
    target_distance: float,
    circle_distance: float,
    circle_radius: float,
    circle_interval: float,
    config_name: str,
    noise_floor: float = 0.05,
    smoothing_sigma: float = 2.0,
    colormap: str = 'plasma',
    visualization_mode: str = 'heatmap'
) -> str:
    """
    Save a high-quality scientific visualization of radar data.
    
    Args:
        filename: Output filename for the visualization.
        heatmap_data: 2D array of heatmap data.
        max_range: Maximum radar range in meters.
        target_distance: Target distance for arc marker in meters.
        circle_distance: Distance for the sampling circle in meters.
        circle_radius: Radius of the sampling circle in meters.
        circle_interval: Interval between range circles in meters.
        config_name: Configuration name for the title.
        noise_floor: Noise floor threshold value.
        smoothing_sigma: Gaussian smoothing sigma parameter.
        colormap: Matplotlib colormap name.
        visualization_mode: Mode ('heatmap', 'contour' or 'combined').
        
    Returns:
        Path to the saved visualization file.
    """
    from scipy.ndimage import gaussian_filter
    
    # Create high-resolution figure
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111)
    
    # Prepare heatmap data
    heatmap_data_copy = heatmap_data.copy()
    heatmap_data_copy[heatmap_data_copy < noise_floor] = 0
    
    # Apply smoothing
    smoothed_data = gaussian_filter(heatmap_data_copy, sigma=smoothing_sigma)
    
    # Create visualization based on selected mode
    cmap = plt.cm.get_cmap(colormap)
    extent = [-max_range, max_range, 0, 2 * max_range]
    
    if visualization_mode == "contour":
        if np.max(smoothed_data) > 0:
            levels = np.linspace(noise_floor, np.max(smoothed_data), 12)
            x_grid = np.linspace(extent[0], extent[1], smoothed_data.shape[1])
            y_grid = np.linspace(extent[2], extent[3], smoothed_data.shape[0])
            
            contour = ax.contourf(
                x_grid, y_grid, smoothed_data,
                levels=levels, 
                cmap=cmap,
                alpha=0.85
            )
            ax.contour(
                x_grid, y_grid, smoothed_data,
                levels=levels, 
                colors='white',
                alpha=0.3,
                linewidths=0.5
            )
            fig.colorbar(contour, ax=ax, label='Signal Intensity')
    
    else:  # heatmap or combined
        if np.max(smoothed_data) > 0:
            vmax = np.percentile(smoothed_data[smoothed_data > 0], 98)
            if vmax < 0.1:
                vmax = 0.1
            norm = colors.PowerNorm(gamma=0.5, vmin=noise_floor, vmax=vmax)
        else:
            norm = colors.PowerNorm(gamma=0.5, vmin=noise_floor, vmax=1)
        
        heatmap = ax.imshow(
            smoothed_data,
            extent=extent,
            origin='lower',
            cmap=cmap,
            norm=norm,
            aspect='auto',
            interpolation='bicubic'
        )
        colorbar = fig.colorbar(heatmap, ax=ax, label='Signal Intensity (normalized)')
        colorbar.ax.tick_params(labelsize=10)
        
        # Add contours for combined mode
        if visualization_mode == "combined" and np.max(smoothed_data) > 0:
            levels = np.linspace(noise_floor, np.max(smoothed_data), 8)
            x_grid = np.linspace(extent[0], extent[1], smoothed_data.shape[1])
            y_grid = np.linspace(extent[2], extent[3], smoothed_data.shape[0])
            ax.contour(
                x_grid, y_grid, smoothed_data,
                levels=levels, 
                colors='white',
                alpha=0.4,
                linewidths=0.5
            )
    
    # Add range arcs
    for r in range(0, int(2 * max_range) + 1, int(circle_interval) * 2):
        arc = patches.Arc(
            (0, 0),
            width=2 * r,
            height=2 * r,
            angle=0,
            theta1=0,
            theta2=180,
            fill=False,
            color='black',
            linestyle='--',
            linewidth=0.5,
            alpha=0.5
        )
        ax.add_patch(arc)
        if 0 < r <= max_range and r % (int(circle_interval) * 2) == 0:
            ax.text(
                0, r, f"{int(r)}m", 
                ha='right', va='bottom', 
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    # Add sampling circles
    circle_configs = [
        {'distance': circle_distance, 'radius': circle_radius, 'angle': 0, 'color': 'red', 'enabled': True},
        {'distance': circle_distance + 10, 'radius': circle_radius, 'angle': -60, 'color': 'cyan', 'enabled': False},
        {'distance': circle_distance + 20, 'radius': circle_radius, 'angle': 60, 'color': 'yellow', 'enabled': False}
    ]
    
    for config in circle_configs:
        if config['enabled']:
            # Calculate position based on angle
            angle_rad = config['angle'] * (np.pi / 180.0)
            x_pos = config['distance'] * np.sin(angle_rad)
            y_pos = config['distance'] * np.cos(angle_rad)
            
            # Create sampling circle
            circle = plt.Circle(
                (x_pos, y_pos),
                config['radius'],
                fill=False,
                color=config['color'],
                linestyle='-',
                linewidth=1.5
            )
            ax.add_patch(circle)
    
    # Add target distance arc
    target_arc = patches.Arc(
        (0, 0),
        width=2 * target_distance,
        height=2 * target_distance,
        angle=0,
        theta1=0,
        theta2=180,
        fill=False,
        color='red',
        linestyle='-',
        linewidth=1.5
    )
    ax.add_patch(target_arc)
    
    # Set up axes
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(0, 2 * max_range)
    ax.set_xlabel('Cross-Range (m)', fontsize=12)
    ax.set_ylabel('Range (m)', fontsize=12)
    
    # Add title with configuration name
    ax.set_title(f'AWR1843 Radar Signal Intensity Analysis\n{config_name}', fontsize=14)
    
    # Add parameter information
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    param_text = (
        f"Timestamp: {timestamp}\nNoise floor: {noise_floor:.3f}\n"
        f"Smoothing: {smoothing_sigma:.1f}\nCircle distance: {circle_distance}m\n"
        f"Circle radius: {circle_radius:.1f}m"
    )
    fig.text(0.02, 0.02, param_text, fontsize=8)
    
    # Add grid and save figure
    ax.grid(True, linestyle=':', alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return filename


def update_statistics_text(
    stats_text: Any, 
    points_x: np.ndarray, 
    points_y: np.ndarray, 
    max_range: float, 
    circle_interval: float
) -> None:
    """
    Update the statistics text with distance band information.
    
    Args:
        stats_text: Matplotlib text object to update.
        points_x: X-coordinates of points.
        points_y: Y-coordinates of points.
        max_range: Maximum radar range in meters.
        circle_interval: Interval between distance bands in meters.
    """
    if len(points_x) > 0:
        distances = np.sqrt(np.square(points_x) + np.square(points_y))
        bins = np.arange(0, max_range + circle_interval, circle_interval)
        counts, _ = np.histogram(distances, bins=bins)
        
        stats = f"Total points: {len(points_x)}\n"
        for i in range(len(counts)):
            if counts[i] > 0:
                stats += f"{bins[i]:.0f}-{bins[i+1]:.0f}m: {counts[i]} pts\n"
        stats_text.set_text(stats)