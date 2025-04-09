#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for radar point clouds.

This module contains functions for creating and updating visualizations
of radar point cloud data, including scatter plots and heatmaps.
"""

import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.artist import Artist
from typing import Tuple, List, Dict, Any, Sequence
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude


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

        # Create radial gradient background for better depth perception
        import matplotlib.colors as mcolors
        from matplotlib.patches import Rectangle
        
        # Define radial gradient colors (dark at center, slightly lighter at edges)
        bg_color_center = '#020924'  # Deeper navy blue for better contrast (from scatter_view)
        bg_color_edge = '#051530'    # Slightly lighter blue at edges
        
        # Get plot dimensions
        x_limit, y_limit = max(1.0, analyzer.params.max_range), max(1.0, 2 * analyzer.params.max_range)
        
        # Create radial gradient as an image
        gradient_resolution = 100
        x = np.linspace(-x_limit, x_limit, gradient_resolution)
        y = np.linspace(0, y_limit, gradient_resolution)
        X, Y = np.meshgrid(x, y)
        
        # Create distance from center normalized to [0, 1]
        distances = np.sqrt(X**2 + Y**2)
        max_distance = np.sqrt(x_limit**2 + y_limit**2)
        normalized_distances = distances / max_distance
        
        # Create gradient array that maps distance to color
        # Convert colors to RGB
        c1 = mcolors.to_rgb(bg_color_center)
        c2 = mcolors.to_rgb(bg_color_edge)
        
        # Create gradient by interpolating between colors
        gradient = np.zeros((gradient_resolution, gradient_resolution, 3))
        for i in range(3):  # RGB channels
            gradient[:,:,i] = c1[i] + normalized_distances * (c2[i] - c1[i])
        
        # Display gradient as background
        ax.imshow(gradient, extent=[-x_limit, x_limit, 0, y_limit], 
                 origin='lower', aspect='auto', zorder=-100)  # zorder ensures it's behind everything
        
        # Still set face color as fallback
        ax.set_facecolor(bg_color_center)
        fig.patch.set_facecolor(bg_color_center)

        # Set plot limits
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(0, y_limit)

        # Define the interval for reference circles with scientific precision
        circle_interval_m = int(analyzer.params.circle_interval)
        major_ranges = list(range(0, int(y_limit) + 1, circle_interval_m * 2))
        minor_ranges = [
            r for r in range(0, int(y_limit) + 1, circle_interval_m)
            if r not in major_ranges
        ]
        
        # Add major range arcs with scientific styling
        for r in major_ranges:
            if r == 0:  # Skip the zero radius
                continue
                
            arc = patches.Arc(
                (0, 0),
                width=2 * r,
                height=2 * r,
                angle=0,
                theta1=0,
                theta2=180,
                fill=False,
                color='#99CCFF',  # Brighter blue for contrast against dark background
                linestyle='-',
                linewidth=0.6,
                alpha=0.6
            )
            ax.add_patch(arc)
            
            # Only label major circles
            if 0 < r <= analyzer.params.max_range:
                ax.text(
                    r * 0.05, r, f"{int(r)}m", ha='left', va='bottom', 
                    color='#99CCFF', fontsize=10,
                    weight='normal',
                    bbox=dict(facecolor=bg_color_center, edgecolor='none', 
                             alpha=0.7, pad=1, boxstyle='round,pad=0.1')
                )
        
        # Add minor range arcs with subtle scientific styling
        for r in minor_ranges:
            if r == 0:  # Skip the zero radius
                continue
                
            arc = patches.Arc(
                (0, 0),
                width=2 * r,
                height=2 * r,
                angle=0,
                theta1=0,
                theta2=180,
                fill=False,
                color='#7799CC',  # Lighter scientific blue
                linestyle=':',
                linewidth=0.3,
                alpha=0.4
            )
            ax.add_patch(arc)
            
        # Add angle markers every 30 degrees with scientific styling
        for angle in range(-90, 91, 30):
            if angle == 0:
                continue
                
            # Convert degrees to radians for calculations
            angle_rad = np.radians(angle)
            
            # Calculate end points using parametric form
            x_end = x_limit * np.sin(angle_rad)
            y_end = y_limit * 0.5 * np.cos(angle_rad)
            
            # Draw angle line
            ax.plot(
                [0, x_end], 
                [0, y_end], 
                linestyle='--', 
                color='#7788BB', 
                linewidth=0.5, 
                alpha=0.6
            )
            
            # Add angle label with scientific notation at 80% of max range
            label_distance = 0.4 * y_limit
            x_label = label_distance * np.sin(angle_rad)
            y_label = label_distance * np.cos(angle_rad)
            
            ax.text(
                x_label, y_label, f"{angle}°", 
                color='#99BBDD',
                fontsize=9,
                ha='center', 
                va='center',
                bbox=dict(facecolor=bg_color_center, edgecolor='none', 
                         alpha=0.7, boxstyle='round,pad=0.1')
            )

        # Create scatter plots with enhanced aesthetics
        analyzer.viz_components['scatter'] = ax.scatter(
            [], [], s=12, c=[], cmap='plasma', alpha=1.0, vmin=-64, vmax=64
        )
        analyzer.viz_components['circle_scatter'] = ax.scatter(
            [], [], s=14, c='#44DDFF', marker='x'
        )

        # Create sampling circle with scientific styling
        analyzer.viz_components['sampling_circle'] = plt.Circle(
            (0, analyzer.params.circle_distance),
            analyzer.params.circle_radius,
            fill=False,
            color='#44DDFF',
            linestyle='-',
            linewidth=1.5
        )
        ax.add_patch(analyzer.viz_components['sampling_circle'])

        # Set labels and title with scientific radar terminology
        ax.set_xlabel('Azimuth (m)', fontsize=12, labelpad=10, color='#99CCFF')
        ax.set_ylabel('Range (m)', fontsize=12, labelpad=10, color='#99CCFF') 
        ax.set_title('Radar Point Cloud', fontsize=14, color='#DDEEFF', weight='normal')

        # Configure ticks with scientific precision
        ax.tick_params(axis='x', colors='#99CCFF', labelsize=10, width=1.0, length=4)
        ax.tick_params(axis='y', colors='#99CCFF', labelsize=10, width=1.0, length=4)

        # Set spine colors for scientific border
        for spine in ax.spines.values():
            spine.set_color('#334466')
            spine.set_linewidth(0.5)

        # Add colorbar with enhanced scientific styling
        colorbar = fig.colorbar(
            analyzer.viz_components['scatter'],
            ax=ax,
            label='Signal Intensity',
            fraction=0.04,
            pad=0.02
        )
        colorbar.ax.yaxis.label.set_color('#99CCFF')
        colorbar.ax.tick_params(colors='#99CCFF')
        analyzer.viz_components['colorbar'] = colorbar

        # Add statistics text boxes with scientific styling
        analyzer.viz_components['stats_text'] = ax.text(
            0.02, 0.98, '',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            color='#99CCFF',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='#051530',
                alpha=0.8,
                edgecolor='#334488'
            )
        )

        analyzer.viz_components['circle_stats_text'] = ax.text(
            0.02, 0.8, '',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            color='#99CCFF',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='#051530',
                alpha=0.8,
                edgecolor='#334488'
            )
        )
        
        # Enable subtle grid for scientific precision
        ax.grid(True, linestyle=':', linewidth=0.2, alpha=0.3, color='#223366')

        # Setup heatmap
        # setup_heatmap_visualization(analyzer) # Removed heatmap setup
        fig.tight_layout()

        return fig
    except Exception as e:
        analyzer.get_logger().error(f"Error setting up visualization: {str(e)}")
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
                # Apply coordinate flip for display ONLY here
                display_x = -x
                if hasattr(analyzer, '_offset_array') and analyzer._offset_array.shape[0] == len(x):
                    analyzer._offset_array[:, 0] = display_x
                    analyzer._offset_array[:, 1] = y
                    scatter.set_offsets(analyzer._offset_array)
                else:
                    # Create new array and cache for future use
                    analyzer._offset_array = np.column_stack((display_x, y))
                    scatter.set_offsets(analyzer._offset_array)
                
                # Enhance intensity visualization with normalized values and improved contrast
                # Calculate point distances for dynamic sizing and color adjustment
                if not hasattr(analyzer, '_distances_cache') or analyzer._distances_cache.shape[0] != len(x):
                    analyzer._distances_cache = np.sqrt(np.square(x) + np.square(y))
                
                # Dynamic point sizing based on distance - farther points get slightly larger for better visibility
                max_distance = analyzer.params.max_range
                base_size = 12  # Base marker size
                distance_factor = 1.5  # How much to scale size by distance
                
                # Calculate sizes array (larger for more distant points)
                relative_distances = analyzer._distances_cache / max_distance
                sizes = base_size + (relative_distances * distance_factor * base_size)
                scatter.set_sizes(sizes)
                
                # Enhanced intensity normalization that emphasizes differences
                # Use histogram equalization approach for better contrast
                if len(intensities) > 10:
                    sorted_intensities = np.sort(intensities)
                    percentile_min = np.percentile(intensities, 5)  # 5th percentile as lower bound
                    percentile_max = np.percentile(intensities, 98)  # 98th percentile as upper bound
                    
                    # Ensure reasonable bounds
                    vmin = max(0.05, percentile_min)
                    vmax = max(vmin + 0.1, percentile_max)
                    
                    # Use non-linear normalization for better visual contrast
                    norm = colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax)
                else:
                    # Fallback for sparse data
                    norm = colors.Normalize(vmin=0.1, vmax=max(0.2, np.max(intensities)))
                
                scatter.set_array(intensities)
                scatter.set_norm(norm)
                
                # Update colorbar to match new normalization
                if hasattr(analyzer.viz_components, 'colorbar') and analyzer.viz_components['colorbar'] is not None:
                    analyzer.viz_components['colorbar'].update_norm(norm)
                
                artists.append(scatter)

                # Update circle scatter - reuse existing arrays when possible
                if len(circle_x) > 0 and circle_scatter is not None:
                    # Apply coordinate flip for display ONLY here
                    display_circle_x = -circle_x
                    if hasattr(analyzer, '_circle_offset_array') and analyzer._circle_offset_array.shape[0] == len(circle_x):
                        analyzer._circle_offset_array[:, 0] = display_circle_x
                        analyzer._circle_offset_array[:, 1] = circle_y
                        circle_scatter.set_offsets(analyzer._circle_offset_array)
                    else:
                        analyzer._circle_offset_array = np.column_stack((display_circle_x, circle_y))
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
            # update_heatmap_display(analyzer, frame) # Removed heatmap display update

            # Return a valid sequence of artists
            return artists

    except Exception as e:
        analyzer.get_logger().error(f"Error updating plot: {str(e)}")
        # Return an empty but valid list in case of error
        if analyzer.viz_components['scatter'] is not None:
            return [analyzer.viz_components['scatter']]
        return []


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


# Removed calculate_heatmap_size function