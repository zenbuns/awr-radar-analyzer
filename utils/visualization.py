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
    file_path,
    heatmap_data,
    max_range,
    target_distance,
    circle_distance,
    circle_radius,
    circle_interval,
    config_name,
    noise_floor=0.1,
    smoothing_sigma=1.0,
    colormap='viridis',
    visualization_mode='heatmap',
    progress_callback=None
):
    """
    Save a high-quality scientific visualization of radar data.
    
    Args:
        file_path: Path where to save the visualization.
        heatmap_data: 2D numpy array with heatmap data.
        max_range: Maximum radar range in meters.
        target_distance: Target distance in meters.
        circle_distance: Distance of primary ROI circle.
        circle_radius: Radius of ROI circle.
        circle_interval: Distance between range circles.
        config_name: Configuration name (for title).
        noise_floor: Noise floor threshold.
        smoothing_sigma: Smoothing sigma for Gaussian filter.
        colormap: Colormap name.
        visualization_mode: Mode ('heatmap', 'contour', or 'combined').
        progress_callback: Optional callback function to report progress (0.0-1.0)
    
    Returns:
        None
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    # Report progress (10%)
    if progress_callback:
        progress_callback(0.1)
    
    # Process heatmap data
    if heatmap_data is None or np.sum(heatmap_data) == 0:
        # Create empty heatmap if no data is available
        grid_size = int(max_range * 2 / 0.5)  # 0.5m resolution
        heatmap_data = np.zeros((grid_size, grid_size))
    
    # Apply smoothing if needed
    if smoothing_sigma > 0:
        smoothed_data = gaussian_filter(heatmap_data, sigma=smoothing_sigma)
    else:
        smoothed_data = heatmap_data.copy()
    
    # Apply noise floor
    if noise_floor > 0:
        max_value = np.max(smoothed_data)
        smoothed_data[smoothed_data < noise_floor * max_value] = 0
    
    # Set up figure with increased size and DPI for publication quality
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    
    # Report progress (20%)
    if progress_callback:
        progress_callback(0.2)
    
    # Create X, Y meshgrid for heatmap/contour
    grid_size = smoothed_data.shape[0]
    extent = [-max_range, max_range, -max_range, max_range]
    
    # Draw different visualization types
    if visualization_mode in ['heatmap', 'combined']:
        im = ax.imshow(
            smoothed_data,
            cmap=colormap,
            origin='lower',
            extent=extent,
            interpolation='bilinear',
            aspect='equal'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Normalized Intensity', rotation=270, labelpad=20)
    
    # Report progress (40%)
    if progress_callback:
        progress_callback(0.4)
    
    if visualization_mode in ['contour', 'combined']:
        # Create contour lines
        levels = np.linspace(noise_floor * np.max(smoothed_data), np.max(smoothed_data), 10)
        contour = ax.contour(
            smoothed_data,
            levels=levels,
            extent=extent,
            colors='white' if visualization_mode == 'combined' else 'black',
            alpha=0.7,
            linewidths=1
        )
        
        # Label contours in combined mode with less frequency
        if visualization_mode == 'contour':
            ax.clabel(contour, inline=1, fontsize=8, fmt='%.2f')
    
    # Report progress (60%)
    if progress_callback:
        progress_callback(0.6)
    
    # Draw range circles
    num_circles = int(max_range / circle_interval)
    for i in range(1, num_circles + 1):
        radius = i * circle_interval
        circle = plt.Circle((0, 0), radius, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        
        # Add range labels
        ax.text(0, radius, f"{radius}m", ha='center', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Add directional angle markers
    for angle in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        rad = np.radians(angle)
        x = 0.95 * max_range * np.cos(rad)
        y = 0.95 * max_range * np.sin(rad)
        ax.text(x, y, f"{angle}°", ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Report progress (70%)
    if progress_callback:
        progress_callback(0.7)
    
    # Draw target distance circle
    target_circle = plt.Circle((0, 0), target_distance, fill=False, color='orange', linestyle='--', linewidth=1.5)
    ax.add_patch(target_circle)
    ax.text(0, target_distance, f"Target: {target_distance}m", ha='center', va='bottom', color='orange',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange', boxstyle='round,pad=0.2'))
    
    # Draw ROI circles
    roi_circle = plt.Circle((0, circle_distance), circle_radius, fill=False, color='red', linewidth=2)
    ax.add_patch(roi_circle)
    
    # Create custom legend elements
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', label='Range Circles'),
        Line2D([0], [0], color='orange', linestyle='--', label=f'Target: {target_distance}m'),
        Line2D([0], [0], color='red', label=f'ROI: {circle_distance}m @ {circle_radius}m')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # Report progress (80%)
    if progress_callback:
        progress_callback(0.8)
    
    # Set axis limits and labels
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    
    # Add title and metadata
    title = f"Radar Point Cloud Analysis: {config_name}\n"
    subtitle = f"Target: {target_distance}m, ROI: {circle_distance}m @ {circle_radius}m radius"
    ax.set_title(title + subtitle)
    
    # Add timestamp and settings
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    settings_text = (
        f"Settings: noise_floor={noise_floor:.2f}, smoothing={smoothing_sigma:.2f}, "
        f"colormap={colormap}, mode={visualization_mode}"
    )
    fig.text(0.5, 0.01, f"{timestamp}\n{settings_text}", ha='center', fontsize=8, color='gray')
    
    # Report progress (90%)
    if progress_callback:
        progress_callback(0.9)
    
    # Save the plot to file path with tight layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for timestamp text
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Report completion
    if progress_callback:
        progress_callback(1.0)


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