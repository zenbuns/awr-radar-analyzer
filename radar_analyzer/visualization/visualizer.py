#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for radar point clouds.

This module contains functions for creating and updating visualizations
of radar point cloud data, including scatter plots and heatmaps.
"""

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.artist import Artist
from typing import Tuple, List, Dict, Any, Sequence


def setup_visualization(analyzer) -> plt.Figure:
    """
    Set up the scatter plot visualization with arcs representing ranges.
    
    This method creates the main scatter plot visualization with range arcs,
    sampling circle, and statistics text.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.

    Returns:
        The created Matplotlib Figure instance for the scatter plot.
    """
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Set aspect ratio to auto to avoid aspect ratio errors
        ax.set_aspect('auto')
        
        analyzer.viz_components['fig'] = fig
        analyzer.viz_components['ax'] = ax

        # Set background colors
        ax.set_facecolor('#000040')
        fig.patch.set_facecolor('#000040')

        # Set plot limits
        x_limit, y_limit = max(1.0, analyzer.params.max_range), max(1.0, 2 * analyzer.params.max_range)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(0, y_limit)

        # Add range arcs
        for r in range(0, int(y_limit) + 1, int(analyzer.params.circle_interval)):
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
            if 0 < r <= analyzer.params.max_range and r % (int(analyzer.params.circle_interval) * 2) == 0:
                ax.text(0, r, f"{int(r)}m", ha='right', va='bottom', color='white', fontsize=8)

        # Create scatter plots
        analyzer.viz_components['scatter'] = ax.scatter([], [], s=8, c=[], cmap='viridis')
        analyzer.viz_components['circle_scatter'] = ax.scatter([], [], s=10, c='lime', marker='x')

        # Create sampling circle
        analyzer.viz_components['sampling_circle'] = plt.Circle(
            (0, analyzer.params.circle_distance),
            analyzer.params.circle_radius,
            fill=False,
            color='lime',
            linestyle='-',
            linewidth=1.5
        )
        ax.add_patch(analyzer.viz_components['sampling_circle'])

        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=10, labelpad=8, color='white')
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=8, color='white')
        ax.set_title('Radar Point Cloud', fontsize=11, color='white')

        # Configure ticks
        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)

        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color('white')

        # Add colorbar
        fig.colorbar(
            analyzer.viz_components['scatter'],
            ax=ax,
            label='Intensity',
            fraction=0.04,
            pad=0.02
        )

        # Add statistics text boxes
        analyzer.viz_components['stats_text'] = ax.text(
            0.02, 0.98, '',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=8,
            color='white',
            bbox=dict(
                boxstyle='round',
                facecolor='#000040',
                alpha=0.7,
                edgecolor='white'
            )
        )

        analyzer.viz_components['circle_stats_text'] = ax.text(
            0.02, 0.8, '',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=8,
            color='white',
            bbox=dict(
                boxstyle='round',
                facecolor='green',
                alpha=0.7,
                edgecolor='white'
            )
        )

        # Setup heatmap
        setup_heatmap_visualization(analyzer)
        fig.tight_layout()

        return fig
    except Exception as e:
        analyzer.get_logger().error(f"Error setting up visualization: {str(e)}")
        # Return a minimal figure in case of error
        return plt.figure()


def setup_heatmap_visualization(analyzer) -> plt.Figure:
    """
    Set up the heatmap visualization in a separate Matplotlib figure.
    
    This method creates the heatmap visualization with range arcs,
    sampling circle, and SNR text.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.

    Returns:
        The created Matplotlib Figure instance for the heatmap.
    """
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Set aspect ratio to auto to avoid aspect ratio errors
        ax.set_aspect('auto')
        
        analyzer.heatmap_viz['fig'] = fig
        analyzer.heatmap_viz['ax'] = ax

        # Set background colors
        ax.set_facecolor('#000020')
        fig.patch.set_facecolor('#000020')

        # Create heatmap
        cmap = plt.cm.viridis.copy()
        analyzer.heatmap_viz['norm'] = colors.PowerNorm(gamma=0.5, vmin=0.05, vmax=1.0)

        analyzer.heatmap_viz['heatmap'] = ax.imshow(
            analyzer.live_heatmap_data,
            extent=[-analyzer.params.max_range, analyzer.params.max_range, 
                   0, 2 * analyzer.params.max_range],
            origin='lower',
            cmap=cmap,
            norm=analyzer.heatmap_viz['norm'],
            aspect='auto',
            interpolation='bilinear',
            alpha=0.85
        )

        analyzer.heatmap_viz['contour'] = None

        # Add colorbar
        analyzer.heatmap_viz['colorbar'] = fig.colorbar(
            analyzer.heatmap_viz['heatmap'],
            ax=ax,
            label='Intensity',
            fraction=0.04,
            pad=0.02
        )

        # Define range arcs
        major_ranges = list(range(0, int(2 * analyzer.params.max_range) + 1, 
                                int(analyzer.params.circle_interval) * 2))
        minor_ranges = [
            r for r in range(0, int(2 * analyzer.params.max_range) + 1, 
                            int(analyzer.params.circle_interval))
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
                linewidth=0.8,
                alpha=0.7
            )
            ax.add_patch(arc)
            if 0 < r <= analyzer.params.max_range:
                ax.text(
                    0, r, f"{int(r)}m", ha='right', va='bottom',
                    color='white', fontsize=8, alpha=0.9
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
                alpha=0.3
            )
            ax.add_patch(arc)

        # Add target arc
        target_arc = patches.Arc(
            (0, 0),
            width=2 * analyzer.params.target_distance,
            height=2 * analyzer.params.target_distance,
            angle=0,
            theta1=0,
            theta2=180,
            fill=False,
            color='white',
            linestyle='-',
            linewidth=0.4
        )
        ax.add_patch(target_arc)

        # Add sampling circle
        sampling_circle = plt.Circle(
            (0, analyzer.params.circle_distance),
            analyzer.params.circle_radius,
            fill=False,
            color='lime',
            linestyle='-',
            linewidth=1.5
        )
        ax.add_patch(sampling_circle)

        analyzer.heatmap_viz['roi_indicators'] = []

        # Add grid
        ax.grid(True, color='white', linestyle=':', linewidth=0.2, alpha=0.1)

        # Set plot limits
        ax.set_xlim(-analyzer.params.max_range, analyzer.params.max_range)
        ax.set_ylim(0, 2 * analyzer.params.max_range)
        
        # Add labels
        ax.set_xlabel('Cross-Range (m)', fontsize=10, labelpad=8, color='white')
        ax.set_ylabel('Range (m)', fontsize=10, labelpad=8, color='white')
        ax.set_title('Radar Intensity Map', fontsize=11, color='white')

        # Configure ticks
        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)

        # Set spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        # Add SNR text
        analyzer.heatmap_viz['snr_text'] = ax.text(
            0.02, 0.02, 'SNR: N/A',
            transform=ax.transAxes,
            color='yellow',
            fontsize=8,
            bbox=dict(
                boxstyle='round',
                facecolor='black',
                alpha=0.7
            )
        )

        fig.tight_layout()
        return fig
    except Exception as e:
        analyzer.get_logger().error(f"Error setting up heatmap: {str(e)}")
        # Return a minimal figure in case of error
        return plt.figure()


def update_plot(analyzer, frame: int) -> Sequence[Artist]:
    """
    Update scatter and heatmap visualization on each animation frame.
    
    This method updates the visualization components based on current
    radar data. It's called by Matplotlib's FuncAnimation.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        frame: Frame index for animation updates.

    Returns:
        Sequence of updated matplotlib Artists for animation.
    """
    # Cache empty artists list for reuse when not visible
    if not hasattr(analyzer, '_empty_artists_cache'):
        analyzer._empty_artists_cache = []
        if analyzer.viz_components['scatter'] is not None:
            analyzer._empty_artists_cache.append(analyzer.viz_components['scatter'])

    if not analyzer.visible:
        # Return cached empty artists list when not visible
        return analyzer._empty_artists_cache

    # Rate limiting for better performance
    current_time = time.time()
    if (current_time - analyzer.last_update_time < analyzer.update_interval and 
            not analyzer.collecting_data):
        # Cache and return existing artists without updates
        if not hasattr(analyzer, '_existing_artists_cache'):
            analyzer._existing_artists_cache = []
            for key in ['scatter', 'stats_text', 'circle_scatter', 'sampling_circle', 'circle_stats_text']:
                if analyzer.viz_components[key] is not None:
                    analyzer._existing_artists_cache.append(analyzer.viz_components[key])
        return analyzer._existing_artists_cache

    # Update timestamp for rate limiting
    analyzer.last_update_time = current_time

    try:
        with analyzer.data_lock:
            artists = []
            x = analyzer.current_data['x']
            y = analyzer.current_data['y']
            intensities = analyzer.current_data['intensities']
            circle_x = analyzer.current_data['circle_x']
            circle_y = analyzer.current_data['circle_y']

            scatter = analyzer.viz_components['scatter']
            circle_scatter = analyzer.viz_components['circle_scatter']
            sampling_circle = analyzer.viz_components['sampling_circle']
            stats_text = analyzer.viz_components['stats_text']
            circle_stats_text = analyzer.viz_components['circle_stats_text']

            # Update scatter plot only if data exists
            if len(x) > 0 and scatter is not None:
                # Use preallocated array if possible
                if hasattr(analyzer, '_offset_array') and analyzer._offset_array.shape[0] == len(x):
                    analyzer._offset_array[:, 0] = x
                    analyzer._offset_array[:, 1] = y
                    scatter.set_offsets(analyzer._offset_array)
                else:
                    # Create new array and cache for future use
                    analyzer._offset_array = np.column_stack((x, y))
                    scatter.set_offsets(analyzer._offset_array)
                
                scatter.set_array(intensities)
                artists.append(scatter)

                # Update circle scatter - reuse existing arrays when possible
                if len(circle_x) > 0 and circle_scatter is not None:
                    if hasattr(analyzer, '_circle_offset_array') and analyzer._circle_offset_array.shape[0] == len(circle_x):
                        analyzer._circle_offset_array[:, 0] = circle_x
                        analyzer._circle_offset_array[:, 1] = circle_y
                        circle_scatter.set_offsets(analyzer._circle_offset_array)
                    else:
                        analyzer._circle_offset_array = np.column_stack((circle_x, circle_y))
                        circle_scatter.set_offsets(analyzer._circle_offset_array)
                    artists.append(circle_scatter)
                elif circle_scatter is not None:
                    # Use cached empty array
                    if not hasattr(analyzer, '_empty_offsets'):
                        analyzer._empty_offsets = np.empty((0, 2))
                    circle_scatter.set_offsets(analyzer._empty_offsets)
                    artists.append(circle_scatter)

                if sampling_circle is not None:
                    # Only update circle center if it changed
                    if not hasattr(analyzer, '_last_circle_distance') or analyzer._last_circle_distance != analyzer.params.circle_distance:
                        sampling_circle.center = (0, analyzer.params.circle_distance)
                        analyzer._last_circle_distance = analyzer.params.circle_distance
                    artists.append(sampling_circle)

                # Update statistics text only if needed (every 5 frames for performance)
                if len(x) > 0 and stats_text is not None and frame % 5 == 0:
                    # Use numpy vectorized operations for histogram calculation
                    # Optimize distance calculation - reuse existing distances if available
                    if not hasattr(analyzer, '_distances_cache') or analyzer._distances_cache.shape[0] != len(x):
                        analyzer._distances_cache = np.sqrt(np.square(x) + np.square(y))
                    
                    bins = np.arange(0, analyzer.params.max_range + analyzer.params.circle_interval, 
                                    analyzer.params.circle_interval)
                    counts, _ = np.histogram(analyzer._distances_cache, bins=bins)

                    # Only rebuild stats text when counts have changed
                    if not hasattr(analyzer, '_last_counts') or not np.array_equal(analyzer._last_counts, counts):
                        analyzer._last_counts = counts.copy()
                        
                        # Build stats text efficiently with string concatenation
                        stats = f"Total points: {len(x)}\n"
                        for i in range(len(counts)):
                            if counts[i] > 0:
                                stats += f"{bins[i]:.0f}-{bins[i+1]:.0f}m: {counts[i]} pts\n"
                        stats_text.set_text(stats)
                    
                    artists.append(stats_text)

                # Clear circle stats text
                if circle_stats_text is not None:
                    circle_stats_text.set_text("")
                    artists.append(circle_stats_text)

                # Cache artists for future use
                analyzer._existing_artists_cache = artists.copy()

            # Update live heatmap (handled in separate method)
            update_heatmap_display(analyzer, frame)

            # Return a valid sequence of artists
            return artists

    except Exception as e:
        analyzer.get_logger().error(f"Error updating plot: {str(e)}")
        # Return an empty but valid list in case of error
        if analyzer.viz_components['scatter'] is not None:
            return [analyzer.viz_components['scatter']]
        return []


def update_heatmap_display(analyzer, frame: int) -> None:
    """
    Update the heatmap visualization (separated from update_plot for thread safety).
    
    This method updates the heatmap display, including intensity values,
    contour lines, and SNR metrics.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        frame: Frame index for updates.
    """
    # Skip more frames to reduce load
    if frame % 3 != 0:  # Increased from 2 to 3
        return
        
    try:
        if analyzer.heatmap_viz['heatmap'] is not None:
            # Cache noise floor value
            noise_floor = getattr(analyzer, '_cached_noise_floor', 0.05)
            
            # Only create copy when needed
            if not hasattr(analyzer, '_heatmap_data_thresholded') or frame % 9 == 0:
                heatmap_data_thresholded = analyzer.live_heatmap_data.copy()
                heatmap_data_thresholded[heatmap_data_thresholded < noise_floor] = 0
                analyzer._heatmap_data_thresholded = heatmap_data_thresholded
            else:
                heatmap_data_thresholded = analyzer._heatmap_data_thresholded

            # Update SNR text less frequently
            if frame % 15 == 0:  # Reduced frequency
                if np.max(heatmap_data_thresholded) > 0:
                    snr = 10.0 * np.log10(np.max(heatmap_data_thresholded) / noise_floor)
                    if analyzer.heatmap_viz['snr_text'] is not None:
                        # Only update if value changed significantly
                        if not hasattr(analyzer, '_last_snr') or abs(analyzer._last_snr - snr) > 0.5:
                            analyzer._last_snr = snr
                            analyzer.heatmap_viz['snr_text'].set_text(f'SNR: {snr:.1f} dB')

            # Update colormap normalization less frequently
            if frame % 10 == 0:  # Reduced frequency
                nonzero_values = heatmap_data_thresholded[heatmap_data_thresholded > 0]
                if nonzero_values.size > 0:
                    vmax = np.percentile(nonzero_values, 98)
                    if vmax < 0.1:
                        vmax = 0.1

                    # Adjust power normalization based on point density
                    density_ratio = nonzero_values.size / heatmap_data_thresholded.size
                    if density_ratio < 0.01:
                        power = 0.3
                    elif density_ratio > 0.2:
                        power = 0.7
                    else:
                        power = 0.5

                    # Only update norm if values changed significantly
                    update_norm = False
                    if not hasattr(analyzer, '_last_norm_params'):
                        update_norm = True
                    else:
                        last_vmax, last_power = analyzer._last_norm_params
                        if abs(last_vmax - vmax) > 0.05 or abs(last_power - power) > 0.05:
                            update_norm = True
                    
                    if update_norm:
                        analyzer._last_norm_params = (vmax, power)
                        analyzer.heatmap_viz['norm'] = colors.PowerNorm(gamma=power, vmin=noise_floor, vmax=vmax)
                        analyzer.heatmap_viz['heatmap'].set_norm(analyzer.heatmap_viz['norm'])

            # Update heatmap data every time
            analyzer.heatmap_viz['heatmap'].set_data(heatmap_data_thresholded)

            # Update contour lines less frequently for performance
            if frame % 15 == 0 and np.any(heatmap_data_thresholded > 0):  # Reduced from 5 to 15
                if analyzer.heatmap_viz['contour'] is not None:
                    for coll in analyzer.heatmap_viz['contour'].collections:
                        try:
                            coll.remove()
                        except Exception:
                            pass
                    analyzer.heatmap_viz['contour'] = None

                try:
                    # Cache imported modules
                    if not hasattr(analyzer, '_gaussian_filter'):
                        from scipy.ndimage import gaussian_filter
                        analyzer._gaussian_filter = gaussian_filter
                    else:
                        gaussian_filter = analyzer._gaussian_filter
                    
                    # Use cached sigma value
                    sigma = getattr(analyzer, '_cached_gaussian_sigma', 2.0)
                    smoothed_data = gaussian_filter(heatmap_data_thresholded, sigma=sigma)

                    if np.max(smoothed_data) > noise_floor:
                        # Cache contour levels if unchanged
                        if not hasattr(analyzer, '_cached_contour_levels') or analyzer._cached_contour_levels.shape[0] != analyzer.heatmap_viz['contour_levels']:
                            levels = np.linspace(
                                noise_floor, 
                                np.max(smoothed_data), 
                                analyzer.heatmap_viz['contour_levels']
                            )
                            analyzer._cached_contour_levels = levels
                        else:
                            # Update max value in cached levels
                            analyzer._cached_contour_levels[-1] = np.max(smoothed_data)
                            levels = analyzer._cached_contour_levels
                            
                        if len(levels) > 2:
                            # Cache extent and grid values
                            if not hasattr(analyzer, '_cached_extent'):
                                extent = [
                                    -analyzer.params.max_range,
                                    analyzer.params.max_range,
                                    0,
                                    2 * analyzer.params.max_range
                                ]
                                analyzer._cached_extent = extent
                                
                                x_grid = np.linspace(extent[0], extent[1], smoothed_data.shape[1])
                                y_grid = np.linspace(extent[2], extent[3], smoothed_data.shape[0])
                                analyzer._cached_x_grid = x_grid
                                analyzer._cached_y_grid = y_grid
                            else:
                                extent = analyzer._cached_extent
                                x_grid = analyzer._cached_x_grid
                                y_grid = analyzer._cached_y_grid

                            analyzer.heatmap_viz['contour'] = analyzer.heatmap_viz['ax'].contour(
                                x_grid,
                                y_grid,
                                smoothed_data,
                                levels=levels,
                                colors='white',
                                alpha=0.4,
                                linewidths=0.5
                            )
                except Exception as e:
                    analyzer.get_logger().debug(f"Error drawing contours: {str(e)}")

            # Update circle patches positions
            for patch in analyzer.heatmap_viz['ax'].patches:
                if isinstance(patch, plt.Circle) and not isinstance(patch, patches.Arc):
                    patch.center = (0, analyzer.params.circle_distance)

            # Explicitly draw heatmap figure
            if analyzer.heatmap_viz['fig'] is not None and hasattr(analyzer.heatmap_viz['fig'].canvas, 'draw_idle'):
                try:
                    analyzer.heatmap_viz['fig'].canvas.draw_idle()
                except Exception as e:
                    analyzer.get_logger().debug(f"Error drawing heatmap: {str(e)}")
    except Exception as e:
        analyzer.get_logger().error(f"Error updating heatmap display: {str(e)}")


def _update_circle_properties(analyzer, center=None, radius=None) -> None:
    """
    Helper function to update circle properties in both scatter and heatmap plots.
    
    This internal method updates either the center, radius, or both properties
    of the sampling circles in the scatter and heatmap visualizations.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        center: New center coordinates (x, y) for the circle or None to keep current.
        radius: New radius for the circle or None to keep current.
    """
    try:
        with analyzer.data_lock:
            # Update scatter plot circle
            fig = analyzer.viz_components['fig']
            ax = analyzer.viz_components['ax']
            sampling_circle = analyzer.viz_components['sampling_circle']
            
            redraw_scatter = False
            redraw_heatmap = False
            
            # Handle scatter plot circle updates
            if sampling_circle is not None and ax is not None:
                current_center = sampling_circle.center
                current_radius = sampling_circle.radius
                
                # Create new circle with updated properties
                sampling_circle.remove()
                new_center = center if center is not None else current_center
                new_radius = radius if radius is not None else current_radius
                
                analyzer.viz_components['sampling_circle'] = plt.Circle(
                    new_center, new_radius,
                    fill=False, color='lime', linestyle='-', linewidth=2
                )
                ax.add_patch(analyzer.viz_components['sampling_circle'])
                
                # Flag for redraw
                redraw_scatter = True
            
            # Handle heatmap circle updates
            if analyzer.heatmap_viz['ax'] is not None:
                for patch in analyzer.heatmap_viz['ax'].patches:
                    if isinstance(patch, plt.Circle) and not isinstance(patch, patches.Arc):
                        if center is not None:
                            patch.center = center
                        if radius is not None:
                            patch.radius = radius
                        # Flag for redraw
                        redraw_heatmap = True
            
            # Efficiently redraw only when needed
            if redraw_scatter and fig is not None and hasattr(fig.canvas, 'draw_idle'):
                try:
                    fig.canvas.draw_idle()
                except Exception as e:
                    analyzer.get_logger().debug(f"Error updating circle in scatter: {str(e)}")
                    
            if redraw_heatmap and analyzer.heatmap_viz['fig'] is not None and hasattr(analyzer.heatmap_viz['fig'].canvas, 'draw_idle'):
                try:
                    analyzer.heatmap_viz['fig'].canvas.draw_idle()
                except Exception as e:
                    analyzer.get_logger().debug(f"Error updating circle in heatmap: {str(e)}")
                    
    except Exception as e:
        analyzer.get_logger().error(f"Error updating circle properties: {str(e)}")


def update_circle_position(analyzer, distance: float) -> None:
    """
    Update the vertical position of the sampling circle in both scatter and heatmap.
    
    This method updates the position of the sampling circle used to collect
    points at a specific distance.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        distance: New vertical position (distance) for the circle center.
    """
    try:
        # Update parameter values first
        analyzer.params.circle_distance = distance
        
        # Update primary circle in circles list
        if hasattr(analyzer.params, 'circles') and analyzer.params.circles and len(analyzer.params.circles) > 0:
            analyzer.params.circles[0].distance = distance
        
        # Cache center for efficient access
        analyzer._cached_circle_center = np.array([0, distance])
        
        # Update circle in visualization
        _update_circle_properties(analyzer, center=(0, distance))

        # Update filtered points - do this after visual updates to avoid extra locks
        from radar_analyzer.processing.data_processor import filter_points_in_circle
        filter_points_in_circle(
            analyzer,
            analyzer.current_data['x'],
            analyzer.current_data['y'],
            analyzer.current_data['intensities']
        )
    except Exception as e:
        analyzer.get_logger().error(f"Error updating circle position: {str(e)}")


def update_circle_radius(analyzer, radius: float) -> None:
    """
    Update the radius of the sampling circle in both scatter and heatmap.
    
    This method updates the radius of the sampling circle used to collect
    points at a specific distance.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        radius: New radius for the sampling circle in meters.
    """
    try:
        # Update parameter values first
        analyzer.params.circle_radius = radius
        
        # Update primary circle in circles list
        if hasattr(analyzer.params, 'circles') and analyzer.params.circles and len(analyzer.params.circles) > 0:
            analyzer.params.circles[0].radius = radius
        
        # Update circle in visualization
        _update_circle_properties(analyzer, radius=radius)

        # Update filtered points - do this after visual updates to avoid extra locks
        from radar_analyzer.processing.data_processor import filter_points_in_circle
        filter_points_in_circle(
            analyzer,
            analyzer.current_data['x'],
            analyzer.current_data['y'],
            analyzer.current_data['intensities']
        )
    except Exception as e:
        analyzer.get_logger().error(f"Error updating circle radius: {str(e)}")


def save_visualization(analyzer, config_dir: str, timestamp: str) -> None:
    """
    Save a PNG visualization of the final heatmap.
    
    This method creates a high-quality visualization of the collected
    radar data and saves it as a PNG file.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        config_dir: Directory in which to save the visualization.
        timestamp: Formatted time string for filenames.
    """
    try:
        # Prepare file path before expensive operations
        viz_file = os.path.join(config_dir, f"viz_{int(analyzer.params.target_distance)}m_{timestamp}.png")
        
        # Check if directory exists, create if needed
        os.makedirs(os.path.dirname(viz_file), exist_ok=True)
        
        # Create figure - optimize dpi for final output
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        
        # Set background colors once
        ax.set_facecolor('#000040')
        fig.patch.set_facecolor('#000040')
        
        # Prepare colormap once
        cmap = plt.cm.viridis
        
        # Calculate vmax safely
        vmax = np.max(analyzer.heatmap_data) if np.any(analyzer.heatmap_data > 0) else 1.0
        
        # Create norm with safe values
        norm = colors.LogNorm(vmin=max(0.1, vmax/1000), vmax=vmax)
        
        # Define extent once for reuse
        extent = [-analyzer.params.max_range, analyzer.params.max_range, 
                 0, 2 * analyzer.params.max_range]
        
        # Create heatmap
        heatmap = ax.imshow(
            analyzer.heatmap_data,
            extent=extent,
            origin='lower',
            cmap=cmap,
            norm=norm,
            aspect='auto'
        )
        
        # Add colorbar
        plt.colorbar(heatmap, ax=ax, label='Point Intensity')
        
        # Set limits once
        ax.set_xlim(-analyzer.params.max_range, analyzer.params.max_range)
        ax.set_ylim(0, 2 * analyzer.params.max_range)
        
        # Calculate range arc parameters once
        max_range = int(2 * analyzer.params.max_range)
        circle_interval = int(analyzer.params.circle_interval)
        range_steps = range(0, max_range + 1, circle_interval)
        
        # Common properties for arcs
        common_arc_props = {
            'angle': 0,
            'theta1': 0,
            'theta2': 180,
            'fill': False
        }
        
        # Add range arcs in a single loop
        for r in range_steps:
            # Create arc with common properties
            arc = patches.Arc(
                (0, 0),
                width=2 * r,
                height=2 * r,
                color='white',
                linestyle='--',
                linewidth=0.8,
                alpha=0.6,
                **common_arc_props
            )
            ax.add_patch(arc)
            
            # Add text for major range markers
            if 0 < r <= analyzer.params.max_range:
                ax.text(
                    0, r, f"{int(r)}m", ha='right', va='bottom',
                    color='white', fontsize=9
                )

        # Add target distance highlight
        target_arc = patches.Arc(
            (0, 0),
            width=2 * analyzer.params.target_distance,
            height=2 * analyzer.params.target_distance,
            color='red',
            linestyle='-',
            linewidth=2,
            **common_arc_props
        )
        ax.add_patch(target_arc)

        # Add sampling circle
        sampling_circle = plt.Circle(
            (0, analyzer.params.circle_distance),
            analyzer.params.circle_radius,
            fill=False,
            color='lime',
            linestyle='-',
            linewidth=2
        )
        ax.add_patch(sampling_circle)

        # Configure plot appearance - text options in a dictionary for consistency
        text_props = {
            'fontsize': 12,
            'labelpad': 10,
            'color': 'white'
        }
        
        # Set labels with common properties
        ax.set_xlabel('Distance along height axis (m)', **text_props)
        ax.set_ylabel('Doppler (m/s)', **text_props)
        
        # Set title
        ax.set_title(
            f'Radar Visualization - {analyzer.params.current_config} at {int(analyzer.params.target_distance)}m',
            fontsize=14,
            color='white'
        )
        
        # Configure tick parameters once
        tick_props = {'colors': 'white', 'labelsize': 10}
        ax.tick_params(axis='x', **tick_props)
        ax.tick_params(axis='y', **tick_props)

        # Style all spines at once
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        # Save figure with optimized settings
        plt.savefig(
            viz_file, 
            dpi=300, 
            bbox_inches='tight', 
            facecolor=fig.get_facecolor(),
            # Optimize file size with compression
            optimize=True,
            transparent=False
        )
        
        # Close figure immediately to release memory
        plt.close(fig)

        analyzer.get_logger().info(f'Saved visualization to {viz_file}')
    except Exception as e:
        analyzer.get_logger().error(f"Error saving visualization: {str(e)}")