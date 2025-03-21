#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt-based scatter plot view for radar point cloud visualization.

This module provides a QtWidget that embeds a Matplotlib scatter plot
for visualizing radar point clouds.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Arc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import Normalize

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import pyqtSignal

from .styles import Colors
from .scatter_optimizer import ScatterOptimizer


class ScatterView(QWidget):
    """
    A PyQt widget that displays a radar point cloud scatter plot.
    
    This widget embeds a Matplotlib figure for interactive visualization
    of radar data points.
    
    Attributes:
        max_range: Maximum radar range in meters.
        circle_interval: Interval between range circles.
        figure: The Matplotlib figure instance.
        ax: The axes for the scatter plot.
        components: Dictionary of plot components.
    """
    
    # Define signals
    update_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the ScatterView widget.
        
        Args:
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        
        # Default parameters
        self.max_range = 35.0  # Ensure max range is exactly 35.0 meters
        self.circle_interval = 10.0
        self.circle_distance = 5.0
        self.circle_radius = 0.5
        
        # Plot components
        self.figure = None
        self.ax = None
        self.components = {}
        
        # Initialize the scatter optimizer for performance improvements
        self.optimizer = ScatterOptimizer()
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the widget UI components."""
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create figure and canvas with larger size and better resolution
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create minimalist toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add widgets to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)  # Give canvas all available space
        
        # Initialize the plot
        self.setup_plot()
        
    def setup_plot(self):
        """Set up the scatter plot with initial components."""
        # Create axis with a specific aspect ratio
        self.ax = self.figure.add_subplot(111)
        
        # Set background colors - deeper scientific background
        self.ax.set_facecolor('#020924')  # Even deeper navy blue for better contrast
        self.figure.patch.set_facecolor('#020924')
        
        # Set plot limits - use max_range directly for consistency
        self.ax.set_xlim(-self.max_range, self.max_range)
        self.ax.set_ylim(0, self.max_range)
        
        # Use equal aspect ratio to keep circles circular
        self.ax.set_aspect('equal')
        
        # Add polar grid with light lines for scientific precision
        theta = np.linspace(0, np.pi, 100)
        for r in range(0, int(self.max_range) + 1, 10):
            if r == 0:
                continue
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            self.ax.plot(x, y, color='#223355', linestyle='-', linewidth=0.3, alpha=0.3)
        
        # Define the interval for reference circles with scientific precision
        circle_interval_m = 5  # 5m intervals for precision
        major_ranges = list(range(0, int(self.max_range) + 1, circle_interval_m * 2))
        minor_ranges = [
            r for r in range(0, int(self.max_range) + 1, circle_interval_m)
            if r not in major_ranges
        ]
        
        # Add major range arcs with scientific styling
        for r in major_ranges:
            if r == 0:  # Skip the zero radius
                continue
                
            arc = Arc(
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
            self.ax.add_patch(arc)
            
            # Only label major circles
            if 0 < r <= self.max_range:
                self.ax.text(
                    r * 0.05, r, f"{int(r)}m", ha='left', va='bottom', 
                    color='#99CCFF', fontsize=7, weight='normal',
                    bbox=dict(facecolor='#020924', edgecolor='none', 
                             alpha=0.7, pad=1, boxstyle='round,pad=0.1')
                )
        
        # Add minor range arcs with subtle scientific styling
        for r in minor_ranges:
            if r == 0:  # Skip the zero radius
                continue
                
            arc = Arc(
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
            self.ax.add_patch(arc)
        
        # Add angle markers every 15 degrees for more precision (but skip previously excluded angles)
        for angle in range(-90, 91, 15):
            if angle == 0 or angle == -90 or angle == 90 or angle % 30 != 0:
                # Skip 0, -90, 90 degrees and keep only 15° intervals not covered by 30° intervals
                continue
                
            # Convert degrees to radians for calculations
            angle_rad = np.radians(angle)
            
            # Calculate end points using parametric form
            x_end = self.max_range * np.sin(angle_rad)
            y_end = self.max_range * np.cos(angle_rad)
            
            # Draw angle line - thinner for 15° intervals
            self.ax.plot(
                [0, x_end], 
                [0, y_end], 
                linestyle=':',  # Dotted line for minor angles
                color='#5566AA', 
                linewidth=0.3, 
                alpha=0.4
            )
        
        # Add angle markers every 30 degrees with proper scientific labels
        for angle in range(-90, 91, 30):
            if angle == 0 or angle == -90 or angle == 90:
                # Skip 0 degrees, handled with sensor indicator
                # Skip -90 and 90 degrees per user request
                continue
                
            # Convert degrees to radians for calculations
            angle_rad = np.radians(angle)
            
            # Calculate end points using parametric form
            x_end = self.max_range * np.sin(angle_rad)
            y_end = self.max_range * np.cos(angle_rad)
            
            # Draw angle line
            self.ax.plot(
                [0, x_end], 
                [0, y_end], 
                linestyle='--', 
                color='#7788BB', 
                linewidth=0.5, 
                alpha=0.6
            )
            
            # Label at 80% of max range to avoid crowding
            label_distance = 0.8 * self.max_range
            x_label = label_distance * np.sin(angle_rad)
            y_label = label_distance * np.cos(angle_rad)
            
            # Add angle label with scientific notation
            self.ax.text(
                x_label, y_label, f"{angle}°", 
                color='#99BBDD',
                fontsize=7,
                ha='center', 
                va='center',
                bbox=dict(facecolor='#050510', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.1')
            )
        
        # Add enhanced radar crosshairs with tick marks
        # Vertical line
        self.ax.plot([0, 0], [0, self.max_range], color='#00DD88', alpha=0.4, linewidth=0.5)
        
        # Horizontal line segments with ticks every 5m
        for x in range(-int(self.max_range), int(self.max_range)+1, 5):
            if x == 0:
                continue
            tick_length = 0.5 if x % 10 == 0 else 0.25
            self.ax.plot([x, x], [0, tick_length], color='#00DD88', alpha=0.3, linewidth=0.4)
        
        # Add professional sensor location indicator at origin
        sensor_circle = Circle((0, 0), 0.5, fill=True, color='#00EEFF', alpha=0.7)
        self.ax.add_patch(sensor_circle)
        sensor_ring = Circle((0, 0), 0.7, fill=False, color='#00FFFF', linewidth=0.5, alpha=0.5)
        self.ax.add_patch(sensor_ring)
        
        # Create scatter plots with enhanced aesthetics - using viridis colormap for better visualization
        self.components['scatter'] = self.ax.scatter(
            [], [], s=22, c=[], cmap='viridis', alpha=1.0, vmin=0.0, vmax=1.0
        )
        
        # Create scatter plots and circles for each sampling circle
        self.components['sampling_circles'] = []
        self.components['circle_scatters'] = []
        
        # Circle colors and positions with enhanced scientific color scheme
        circle_configs = [
            {'enabled': True, 'distance': 5.0, 'radius': 0.5, 'angle': 0, 
             'color': '#44DDFF', 'label': 'Primary'},
            {'enabled': False, 'distance': 15.0, 'radius': 0.5, 'angle': -60, 
             'color': '#77BBFF', 'label': 'Left'},
            {'enabled': False, 'distance': 25.0, 'radius': 0.5, 'angle': 60, 
             'color': '#FFDD66', 'label': 'Right'}
        ]
        
        # Create sampling circles with different positions and scientific styling
        for i, config in enumerate(circle_configs):
            # Calculate x position based on angle (convert degrees to radians)
            angle_rad = config['angle'] * (3.14159 / 180.0)
            x_pos = config['distance'] * np.sin(angle_rad)
            y_pos = config['distance'] * np.cos(angle_rad)
            
            # Create circle scatter plot with precise scientific markers
            circle_scatter = self.ax.scatter([], [], s=18, c=config['color'], marker='x', alpha=0.9)
            self.components[f'circle_scatter_{i}'] = circle_scatter
            self.components['circle_scatters'].append(circle_scatter)
            
            # Create sampling circle with scientific styling
            sampling_circle = Circle(
                (x_pos, y_pos),
                config['radius'],
                fill=False,
                color=config['color'],
                linestyle='-',
                linewidth=0.8,
                alpha=0.8 if config['enabled'] else 0.3,
                visible=config['enabled']
            )
            self.ax.add_patch(sampling_circle)
            
            # Add small targeting crosshairs at center of sampling circle
            if config['enabled']:
                crosshair_size = 0.3
                self.ax.plot(
                    [x_pos-crosshair_size, x_pos+crosshair_size], 
                    [y_pos, y_pos], 
                    color=config['color'], linewidth=0.4, alpha=0.6
                )
                self.ax.plot(
                    [x_pos, x_pos], 
                    [y_pos-crosshair_size, y_pos+crosshair_size], 
                    color=config['color'], linewidth=0.4, alpha=0.6
                )
            
            self.components[f'sampling_circle_{i}'] = sampling_circle
            self.components['sampling_circles'].append({
                'circle': sampling_circle,
                'config': config,
                'x_pos': x_pos,
                'y_pos': y_pos
            })
        
        # Set labels and title with scientific radar terminology
        self.ax.set_xlabel('Azimuth (m)', fontsize=9, labelpad=10, color='#99CCFF')
        self.ax.set_ylabel('Range (m)', fontsize=9, labelpad=10, color='#99CCFF')
        self.ax.set_title('Radar Point Cloud', fontsize=11, color='#DDEEFF', weight='normal')
        
        # Configure ticks with scientific precision
        self.ax.tick_params(axis='x', colors='#99CCFF', labelsize=8, width=1.0, length=4)
        self.ax.tick_params(axis='y', colors='#99CCFF', labelsize=8, width=1.0, length=4)
        
        # Set spine colors for scientific border
        for spine in self.ax.spines.values():
            spine.set_color('#334466')
            spine.set_linewidth(0.5)
        
        # Add colorbar with enhanced scientific styling
        self.colorbar = self.figure.colorbar(
            self.components['scatter'],
            ax=self.ax,
            label='Signal Intensity (Blue: Low, Yellow: High)',
            fraction=0.03,
            pad=0.02
        )
        self.colorbar.ax.yaxis.label.set_color('#99CCFF')
        self.colorbar.ax.tick_params(colors='#99CCFF')
        
        # Add technical statistics box with enhanced resolution information
        grid_size = int(2 * self.max_range)
        res_text = (
            f"Resolution: {1.0:.1f}m\n"
            f"Range: 0-{self.max_range:.0f}m\n"
            f"Sampling: adaptive\n"
            f"Grid: {grid_size}×{grid_size} px"
        )
        
        self.components['res_stats'] = self.ax.text(
            0.98, 0.98, res_text,
            transform=self.ax.transAxes,
            color='#99CCFF',
            fontsize=7,
            ha='right',
            va='top',
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='#051530',
                alpha=0.8,
                edgecolor='#334488'
            )
        )
        
        # Add statistics text boxes with scientific notation
        self.components['stats_text'] = self.ax.text(
            0.02, 0.98, '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=8,
            color='#99CCFF',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='#051530',
                alpha=0.8,
                edgecolor='#334488'
            )
        )
        
        self.components['circle_stats_text'] = self.ax.text(
            0.02, 0.8, '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=8,
            color='#99CCFF',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='#051530',
                alpha=0.8,
                edgecolor='#334488'
            )
        )
        
        # Enable grid for scientific precision - very subtle
        self.ax.grid(True, linestyle=':', linewidth=0.2, alpha=0.3, color='#223366')
        
        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_circle_position(self, index, distance, angle=None):
        """
        Update a sampling circle position.
        
        Args:
            index: Index of the circle to update (0-2)
            distance: New distance for circle center.
            angle: Optional new angle in degrees (if not provided, use current angle)
        """
        if index < 0 or index >= len(self.components['sampling_circles']):
            return
            
        circle_info = self.components['sampling_circles'][index]
        circle_obj = circle_info['circle']
        config = circle_info['config']
        
        # Update config
        config['distance'] = distance
        if angle is not None:
            config['angle'] = angle
        
        # Calculate new position
        angle_rad = config['angle'] * (3.14159 / 180.0)
        x_pos = distance * np.sin(angle_rad)
        y_pos = distance * np.cos(angle_rad)
        
        # Update circle position
        circle_obj.center = (x_pos, y_pos)
        
        # Update stored position
        circle_info['x_pos'] = x_pos
        circle_info['y_pos'] = y_pos
        
        self.canvas.draw_idle()
    
    def update_circle_radius(self, index, radius):
        """
        Update a sampling circle radius.
        
        Args:
            index: Index of the circle to update (0-2)
            radius: New radius for the circle.
        """
        if index < 0 or index >= len(self.components['sampling_circles']):
            return
            
        circle_info = self.components['sampling_circles'][index]
        circle_obj = circle_info['circle']
        config = circle_info['config']
        
        # Update config
        config['radius'] = radius
        
        # We need to recreate the circle patch
        circle_obj.remove()
        
        new_circle = Circle(
            (circle_info['x_pos'], circle_info['y_pos']),
            radius,
            fill=False,
            color=config['color'],
            linestyle='-',
            linewidth=1.8,  # Thicker for better visibility
            alpha=0.8 if config['enabled'] else 0.2,
            visible=config['enabled']
        )
        
        # Update references
        self.ax.add_patch(new_circle)
        circle_info['circle'] = new_circle
        self.components[f'sampling_circle_{index}'] = new_circle
        
        self.canvas.draw_idle()
    
    def toggle_circle(self, index, enabled):
        """
        Toggle a sampling circle on/off.
        
        Args:
            index: Index of the circle to toggle (0-2)
            enabled: Whether the circle should be enabled
        """
        if index < 0 or index >= len(self.components['sampling_circles']):
            return
            
        circle_info = self.components['sampling_circles'][index]
        circle_obj = circle_info['circle']
        config = circle_info['config']
        
        # Update config
        config['enabled'] = enabled
        
        # Update visibility and appearance
        circle_obj.set_visible(enabled)
        circle_obj.set_alpha(0.8 if enabled else 0.2)
        
        # Toggle scatter visibility
        if index < len(self.components['circle_scatters']):
            self.components['circle_scatters'][index].set_visible(enabled)
        
        self.canvas.draw_idle()
    
    def update_plot_data(self, x, y, intensities, circles_data):
        """
        Update the scatter plot with new data.
        
        Args:
            x: X-coordinates of all points.
            y: Y-coordinates of all points.
            intensities: Intensity values of all points.
            circles_data: List of dictionaries with circle data (x, y, intensities)
        """
        try:
            # Skip update if optimizer decides it's not necessary
            if not self.optimizer.should_update(len(x)):
                return
                
            # Apply intelligent downsampling for large datasets
            if len(x) > self.optimizer.max_points:
                x, y, intensities = self.optimizer.downsample(x, y, intensities)
                
            # Update main scatter plot with new data
            if len(x) > 0:
                # Normalize intensity values to 0-1 range
                if len(intensities) > 0:
                    min_intensity = np.min(intensities)
                    max_intensity = np.max(intensities)
                    
                    # Prevent division by zero if all intensities are the same
                    if max_intensity > min_intensity:
                        normalized_intensities = (intensities - min_intensity) / (max_intensity - min_intensity)
                    else:
                        normalized_intensities = np.zeros_like(intensities)
                else:
                    normalized_intensities = np.array([])
                
                # Set offsets and colors in one operation
                self.components['scatter'].set_offsets(np.column_stack((x, y)))
                self.components['scatter'].set_array(normalized_intensities)
                
                # Update scatter plot visibility
                self.components['scatter'].set_visible(True)
                
                # Update each circle's scatter plot
                for i, circle_data in enumerate(circles_data):
                    if i < len(self.components['circle_scatters']):
                        scatter = self.components['circle_scatters'][i]
                        circle_info = self.components['sampling_circles'][i]
                        
                        # Only update if the circle is enabled and has data
                        if circle_info['config']['enabled'] and len(circle_data['x']) > 0:
                            # Downsample circle data if needed
                            c_x, c_y = circle_data['x'], circle_data['y']
                            c_intensities = circle_data['intensities']
                            
                            if len(c_x) > 500:  # Use smaller threshold for circles
                                # Simple random sampling for circle points
                                n_keep = 500
                                indices = np.random.choice(len(c_x), n_keep, replace=False)
                                c_x, c_y = c_x[indices], c_y[indices]
                                
                            # Update offsets in one operation
                            scatter.set_offsets(np.column_stack((c_x, c_y)))
                            scatter.set_visible(True)
                        else:
                            # Use empty array for no data (faster than removing/recreating)
                            scatter.set_offsets(np.empty((0, 2)))
                            scatter.set_visible(False)
                
                # Update statistics text with enhanced scientific notation
                distances = np.sqrt(np.square(x) + np.square(y))
                bins = np.arange(0, self.max_range + self.circle_interval, self.circle_interval)
                counts, _ = np.histogram(distances, bins=bins)
                
                # Calculate standard deviation for scientific metrics
                if len(distances) > 1:
                    mean_dist = np.mean(distances)
                    std_dist = np.std(distances)
                    median_dist = np.median(distances)
                    
                    # Convert intensity to dB scale if non-zero
                    if np.max(intensities) > 0:
                        intensity_db = 10 * np.log10(np.mean(intensities) / 0.001)  # Ref: 0.001
                    else:
                        intensity_db = 0
                    
                    # Format statistics text with scientific notation
                    stats = f"n = {len(x)}"
                    if len(x) < self.optimizer.max_points:
                        stats += " (complete)"
                    else:
                        original_count = self.optimizer.last_point_count
                        percent = int(100 * len(x) / max(1, original_count))
                        stats += f" (sample: {percent}%)"
                    
                    stats += f"\nμ = {mean_dist:.2f}m, σ = {std_dist:.2f}m\n"
                    stats += f"Med. = {median_dist:.2f}m, Avg. Int. = {intensity_db:.1f}dB\n"
                    
                    # Only include non-zero bins to keep text compact (max 4 ranges)
                    non_zero_bins = 0
                    for i in range(len(counts)):
                        if counts[i] > 0:
                            if non_zero_bins < 4:  # Limit to 4 ranges for clarity
                                stats += f"R{bins[i]:.0f}-{bins[i+1]:.0f}m: {counts[i]} pts\n"
                                non_zero_bins += 1
                            elif non_zero_bins == 4:
                                stats += f"+ {sum(counts[i:])} pts in other ranges"
                                break
                else:
                    stats = f"n = {len(x)}\nInsufficient data for statistics"
                    
                self.components['stats_text'].set_text(stats)
            else:
                # Clear plots if no data - use empty arrays (more efficient)
                self.components['scatter'].set_offsets(np.empty((0, 2)))
                self.components['scatter'].set_visible(False)
                
                for scatter in self.components['circle_scatters']:
                    scatter.set_offsets(np.empty((0, 2)))
                    scatter.set_visible(False)
                
                self.components['stats_text'].set_text("No data")
            
            # Use draw_idle for more efficient rendering
            # Ensure aspect ratio is maintained
            self.ax.set_aspect('equal')
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error updating scatter plot: {e}")
    
    def update_circle_stats(self, circle_stats):
        """
        Update the circle statistics text.
        
        Args:
            circle_stats: List of dictionaries with statistics for each circle
        """
        stats_text = ""
        
        for i, stats in enumerate(circle_stats):
            if i < len(self.components['sampling_circles']):
                circle_info = self.components['sampling_circles'][i]
                config = circle_info['config']
                
                if config['enabled']:
                    if stats_text:
                        stats_text += "\n\n"
                    
                    # Compute standard deviation and SNR if possible for scientific notation
                    std_text = ""
                    snr_text = ""
                    if 'intensities' in stats and len(stats['intensities']) > 1:
                        std_intensity = np.std(stats['intensities'])
                        std_text = f", σ={std_intensity:.2f}"
                        
                        # Calculate SNR in dB if possible
                        if np.mean(stats['intensities']) > 0.001:
                            snr_db = 10 * np.log10(np.mean(stats['intensities']) / 0.001)
                            snr_text = f"\n  SNR={snr_db:.1f}dB"
                    
                    stats_text += (
                        f"• {config['label']} (r={config['radius']:.1f}m)\n"
                        f"  n={stats.get('count', 0)}, μ={stats.get('avg_intensity', 0):.2f}{std_text}{snr_text}"
                    )
        
        if not stats_text:
            stats_text = "No active sampling regions"
            
        self.components['circle_stats_text'].set_text(stats_text)
        self.canvas.draw_idle()
    
    def clear_points(self):
        """Clear all point data from the plot while preserving other elements."""
        if 'scatter' in self.components:
            self.components['scatter'].set_offsets(np.empty((0, 2)))
            for scatter in self.components['circle_scatters']:
                scatter.set_offsets(np.empty((0, 2)))
            self.components['stats_text'].set_text("")
            self.components['circle_stats_text'].set_text("")
            # Redraw the canvas
            self.canvas.draw_idle()

    # Keep the original method for backwards compatibility
    def clear_plot(self):
        """Clear all data from the plot. Legacy method - use clear_points instead."""
        self.clear_points()

    def configure_optimizer(self, update_interval=None, max_points=None, adaptive_sampling=None):
        """
        Configure the scatter optimizer settings.
        
        Args:
            update_interval: Time in seconds between updates.
            max_points: Maximum number of points to display.
            adaptive_sampling: Whether to use adaptive sampling.
        """
        if hasattr(self, 'optimizer'):
            self.optimizer.configure(
                update_interval=update_interval,
                max_points=max_points,
                adaptive_sampling=adaptive_sampling
            )
            
    def update_point_limit(self, dataset_size):
        """
        Update the point limit based on dataset size.
        
        Args:
            dataset_size: Total number of points in the dataset.
        """
        if hasattr(self, 'optimizer'):
            # Scale max points based on dataset size
            if dataset_size > 100000:
                # Very large dataset
                self.optimizer.set_max_points(3000)
                self.optimizer.set_update_interval(0.2)
            elif dataset_size > 50000:
                # Large dataset
                self.optimizer.set_max_points(5000)
                self.optimizer.set_update_interval(0.15)
            elif dataset_size > 10000:
                # Medium dataset
                self.optimizer.set_max_points(8000)
                self.optimizer.set_update_interval(0.1)
            else:
                # Small dataset - show most or all points
                self.optimizer.set_max_points(min(10000, dataset_size))
                self.optimizer.set_update_interval(0.08)