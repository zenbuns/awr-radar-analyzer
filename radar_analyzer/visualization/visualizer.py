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
    if not analyzer.visible:
        # Return empty but valid list when not visible
        empty_artists = []
        if analyzer.viz_components['scatter'] is not None:
            empty_artists.append(analyzer.viz_components['scatter'])
        return empty_artists

    # Rate limiting for better performance
    current_time = time.time()
    if (current_time - analyzer.last_update_time < analyzer.update_interval and 
            not analyzer.collecting_data):
        # Return existing artists without updates
        artists = []
        for key in ['scatter', 'stats_text', 'circle_scatter', 'sampling_circle', 'circle_stats_text']:
            if analyzer.viz_components[key] is not None:
                artists.append(analyzer.viz_components[key])
        return artists

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

            # Update scatter plot
            if len(x) > 0 and scatter is not None:
                scatter.set_offsets(np.column_stack((x, y)))
                scatter.set_array(intensities)
                artists.append(scatter)

                # Update circle scatter
                if len(circle_x) > 0 and circle_scatter is not None:
                    circle_scatter.set_offsets(np.column_stack((circle_x, circle_y)))
                    artists.append(circle_scatter)
                elif circle_scatter is not None:
                    circle_scatter.set_offsets(np.empty((0, 2)))
                    artists.append(circle_scatter)

                if sampling_circle is not None:
                    sampling_circle.center = (0, analyzer.params.circle_distance)
                    artists.append(sampling_circle)

                # Update statistics text
                if len(x) > 0 and stats_text is not None:
                    distances = np.sqrt(np.square(x) + np.square(y))
                    bins = np.arange(0, analyzer.params.max_range + analyzer.params.circle_interval, 
                                    analyzer.params.circle_interval)
                    counts, _ = np.histogram(distances, bins=bins)

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
    import time
    # Skip frames to reduce load
    if frame % 2 != 0:
        return
        
    try:
        if analyzer.heatmap_viz['heatmap'] is not None:
            noise_floor = 0.05
            heatmap_data_thresholded = analyzer.live_heatmap_data.copy()
            heatmap_data_thresholded[heatmap_data_thresholded < noise_floor] = 0

            # Update SNR text
            if np.max(heatmap_data_thresholded) > 0:
                snr = 10.0 * np.log10(np.max(heatmap_data_thresholded) / noise_floor)
                if analyzer.heatmap_viz['snr_text'] is not None:
                    analyzer.heatmap_viz['snr_text'].set_text(f'SNR: {snr:.1f} dB')

            # Update colormap normalization
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

                analyzer.heatmap_viz['norm'] = colors.PowerNorm(gamma=power, vmin=noise_floor, vmax=vmax)
                analyzer.heatmap_viz['heatmap'].set_norm(analyzer.heatmap_viz['norm'])

            # Update heatmap data
            analyzer.heatmap_viz['heatmap'].set_data(heatmap_data_thresholded)

            # Update contour lines less frequently for performance
            if frame % 5 == 0 and np.any(heatmap_data_thresholded > 0):
                if analyzer.heatmap_viz['contour'] is not None:
                    for coll in analyzer.heatmap_viz['contour'].collections:
                        try:
                            coll.remove()
                        except Exception:
                            pass
                    analyzer.heatmap_viz['contour'] = None

                try:
                    from scipy.ndimage import gaussian_filter
                    smoothed_data = gaussian_filter(heatmap_data_thresholded, sigma=2.0)

                    if np.max(smoothed_data) > noise_floor:
                        levels = np.linspace(
                            noise_floor, 
                            np.max(smoothed_data), 
                            analyzer.heatmap_viz['contour_levels']
                        )
                        if len(levels) > 2:
                            extent = [
                                -analyzer.params.max_range,
                                analyzer.params.max_range,
                                0,
                                2 * analyzer.params.max_range
                            ]
                            x_grid = np.linspace(extent[0], extent[1], smoothed_data.shape[1])
                            y_grid = np.linspace(extent[2], extent[3], smoothed_data.shape[0])

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
        analyzer.params.circle_distance = distance
        
        # Update primary circle in circles list
        if hasattr(analyzer.params, 'circles') and analyzer.params.circles and len(analyzer.params.circles) > 0:
            analyzer.params.circles[0].distance = distance
        
        analyzer._cached_circle_center = np.array([0, distance])

        with analyzer.data_lock:
            fig = analyzer.viz_components['fig']
            ax = analyzer.viz_components['ax']
            sampling_circle = analyzer.viz_components['sampling_circle']

            if sampling_circle is not None and ax is not None:
                sampling_circle.remove()
                analyzer.viz_components['sampling_circle'] = plt.Circle(
                    (0, distance), analyzer.params.circle_radius,
                    fill=False, color='lime', linestyle='-', linewidth=2
                )
                ax.add_patch(analyzer.viz_components['sampling_circle'])
                if fig is not None and hasattr(fig.canvas, 'draw_idle'):
                    try:
                        fig.canvas.draw_idle()
                    except Exception as e:
                        analyzer.get_logger().debug(f"Error updating circle: {str(e)}")

            if analyzer.heatmap_viz['ax'] is not None:
                for patch in analyzer.heatmap_viz['ax'].patches:
                    if isinstance(patch, plt.Circle) and not isinstance(patch, patches.Arc):
                        patch.center = (0, distance)
                        if analyzer.heatmap_viz['fig'] is not None and hasattr(analyzer.heatmap_viz['fig'].canvas, 'draw_idle'):
                            try:
                                analyzer.heatmap_viz['fig'].canvas.draw_idle()
                            except Exception as e:
                                analyzer.get_logger().debug(f"Error updating heatmap: {str(e)}")

            # Update filtered points
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
        analyzer.params.circle_radius = radius
        
        # Update primary circle in circles list
        if hasattr(analyzer.params, 'circles') and analyzer.params.circles and len(analyzer.params.circles) > 0:
            analyzer.params.circles[0].radius = radius
        
        with analyzer.data_lock:
            fig = analyzer.viz_components['fig']
            ax = analyzer.viz_components['ax']
            sampling_circle = analyzer.viz_components['sampling_circle']

            if sampling_circle is not None and ax is not None:
                sampling_circle.remove()
                analyzer.viz_components['sampling_circle'] = plt.Circle(
                    (0, analyzer.params.circle_distance), radius,
                    fill=False, color='lime', linestyle='-', linewidth=2
                )
                ax.add_patch(analyzer.viz_components['sampling_circle'])
                if fig is not None and hasattr(fig.canvas, 'draw_idle'):
                    try:
                        fig.canvas.draw_idle()
                    except Exception as e:
                        analyzer.get_logger().debug(f"Error updating circle: {str(e)}")

            if analyzer.heatmap_viz['ax'] is not None:
                for patch in analyzer.heatmap_viz['ax'].patches:
                    if isinstance(patch, plt.Circle) and not isinstance(patch, patches.Arc):
                        patch.radius = radius
                        if analyzer.heatmap_viz['fig'] is not None and hasattr(analyzer.heatmap_viz['fig'].canvas, 'draw_idle'):
                            try:
                                analyzer.heatmap_viz['fig'].canvas.draw_idle()
                            except Exception as e:
                                analyzer.get_logger().debug(f"Error updating heatmap: {str(e)}")

            # Update filtered points
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
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        cmap = plt.cm.viridis
        vmax = np.max(analyzer.heatmap_data)
        norm = colors.LogNorm(vmin=0.1, vmax=vmax if vmax > 0 else 1)
        heatmap = ax.imshow(
            analyzer.heatmap_data,
            extent=[-analyzer.params.max_range, analyzer.params.max_range, 
                   0, 2 * analyzer.params.max_range],
            origin='lower',
            cmap=cmap,
            norm=norm,
            aspect='auto'
        )
        plt.colorbar(heatmap, ax=ax, label='Point Intensity')

        # Draw range arcs
        for r in range(0, int(2 * analyzer.params.max_range) + 1, int(analyzer.params.circle_interval)):
            arc = patches.Arc(
                (0, 0),
                width=2 * r,
                height=2 * r,
                angle=0,
                theta1=0,
                theta2=180,
                fill=False,
                color='white',
                linestyle='--',
                linewidth=0.8,
                alpha=0.6
            )
            ax.add_patch(arc)
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
            angle=0,
            theta1=0,
            theta2=180,
            fill=False,
            color='red',
            linestyle='-',
            linewidth=2
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

        # Configure plot appearance
        ax.set_xlim(-analyzer.params.max_range, analyzer.params.max_range)
        ax.set_ylim(0, 2 * analyzer.params.max_range)
        ax.set_xlabel('Distance along height axis (m)', fontsize=12, labelpad=10, color='white')
        ax.set_ylabel('Doppler (m/s)', fontsize=12, labelpad=10, color='white')
        ax.set_title(
            f'Radar Visualization - {analyzer.params.current_config} at {int(analyzer.params.target_distance)}m',
            fontsize=14,
            color='white'
        )
        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)

        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        ax.set_facecolor('#000040')
        fig.patch.set_facecolor('#000040')

        viz_file = os.path.join(config_dir, f"viz_{int(analyzer.params.target_distance)}m_{timestamp}.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        analyzer.get_logger().info(f'Saved visualization to {viz_file}')
    except Exception as e:
        analyzer.get_logger().error(f"Error saving visualization: {str(e)}")