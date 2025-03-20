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
        self.max_range = 35.0
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
        
        # Set background colors
        self.ax.set_facecolor(Colors.DARK_BACKGROUND)
        self.figure.patch.set_facecolor(Colors.DARK_BACKGROUND)
        
        # Set plot limits - ensure these are positive values
        x_limit, y_limit = max(1.0, self.max_range), max(1.0, 2 * self.max_range)
        self.ax.set_xlim(-x_limit, x_limit)
        self.ax.set_ylim(0, y_limit)
        
        # Set aspect ratio to auto to avoid aspect ratio errors
        self.ax.set_aspect('auto')
        
        # Add range arcs with improved styling
        for r in range(0, int(y_limit) + 1, int(self.circle_interval)):
            arc = Arc(
                (0, 0),
                width=2 * r,
                height=2 * r,
                angle=0,
                theta1=0,
                theta2=180,
                fill=False,
                color=Colors.BORDER,
                linestyle='-',
                linewidth=1.5,
                alpha=0.8
            )
            self.ax.add_patch(arc)
            if 0 < r <= self.max_range and r % (int(self.circle_interval) * 2) == 0:
                self.ax.text(0, r, f"{int(r)}m", ha='right', va='bottom', 
                             color=Colors.TEXT, fontsize=12, weight='bold')
        
        # Create scatter plots with enhanced aesthetics
        self.components['scatter'] = self.ax.scatter(
            [], [], s=25, c=[], cmap='plasma', alpha=1.0
        )
        
        # Create scatter plots and circles for each sampling circle
        self.components['sampling_circles'] = []
        self.components['circle_scatters'] = []
        
        # Circle colors and positions with modern color scheme
        circle_configs = [
            {'enabled': True, 'distance': 5.0, 'radius': 0.5, 'angle': 0, 
             'color': Colors.ACCENT_BLUE, 'label': 'Primary'},
            {'enabled': False, 'distance': 15.0, 'radius': 0.5, 'angle': -60, 
             'color': Colors.ACCENT_LAVENDER, 'label': 'Left'},
            {'enabled': False, 'distance': 25.0, 'radius': 0.5, 'angle': 60, 
             'color': Colors.ACCENT_YELLOW, 'label': 'Right'}
        ]
        
        for i, config in enumerate(circle_configs):
            # Calculate x position based on angle (convert degrees to radians)
            angle_rad = config['angle'] * (3.14159 / 180.0)
            x_pos = config['distance'] * np.sin(angle_rad)
            y_pos = config['distance'] * np.cos(angle_rad)
            
            # Create circle scatter plot with better visibility
            circle_scatter = self.ax.scatter([], [], s=30, c=config['color'], marker='x')
            self.components[f'circle_scatter_{i}'] = circle_scatter
            self.components['circle_scatters'].append(circle_scatter)
            
            # Create sampling circle with modern styling
            sampling_circle = Circle(
                (x_pos, y_pos),
                config['radius'],
                fill=False,
                color=config['color'],
                linestyle='-',
                linewidth=2.5,  # Even thicker for better visibility
                alpha=1.0 if config['enabled'] else 0.4,
                visible=config['enabled']
            )
            self.ax.add_patch(sampling_circle)
            self.components[f'sampling_circle_{i}'] = sampling_circle
            self.components['sampling_circles'].append({
                'circle': sampling_circle,
                'config': config,
                'x_pos': x_pos,
                'y_pos': y_pos
            })
        
        # Set labels and title with modern styling
        self.ax.set_xlabel('X (m)', fontsize=14, labelpad=10, color=Colors.TEXT)
        self.ax.set_ylabel('Y (m)', fontsize=14, labelpad=10, color=Colors.TEXT)
        self.ax.set_title('Radar Point Cloud', fontsize=16, color=Colors.TEXT, weight='bold')
        
        # Configure ticks
        self.ax.tick_params(axis='x', colors=Colors.TEXT, labelsize=12, width=1.5, length=6)
        self.ax.tick_params(axis='y', colors=Colors.TEXT, labelsize=12, width=1.5, length=6)
        
        # Set spine colors
        for spine in self.ax.spines.values():
            spine.set_color(Colors.BORDER)
            spine.set_linewidth(1.5)
        
        # Add colorbar with improved aesthetics
        self.colorbar = self.figure.colorbar(
            self.components['scatter'],
            ax=self.ax,
            label='Intensity',
            fraction=0.03,  # Narrower
            pad=0.02
        )
        self.colorbar.ax.yaxis.label.set_color(Colors.TEXT)
        self.colorbar.ax.tick_params(colors=Colors.TEXT_MUTED)
        
        # Add statistics text boxes with improved styling
        self.components['stats_text'] = self.ax.text(
            0.02, 0.98, '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            color=Colors.TEXT,
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=Colors.LIGHT_BACKGROUND,
                alpha=0.8,
                edgecolor=Colors.BORDER
            )
        )
        
        self.components['circle_stats_text'] = self.ax.text(
            0.02, 0.8, '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            color=Colors.TEXT,
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=Colors.LIGHT_BACKGROUND,
                alpha=0.8,
                edgecolor=Colors.ACCENT_BLUE
            )
        )
        
        # Enable grid for better readability
        self.ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color=Colors.BORDER)
        
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
                # Set offsets and colors in one operation
                self.components['scatter'].set_offsets(np.column_stack((x, y)))
                self.components['scatter'].set_array(intensities)
                
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
                
                # Update statistics text - compute once and reuse
                distances = np.sqrt(np.square(x) + np.square(y))
                bins = np.arange(0, self.max_range + self.circle_interval, self.circle_interval)
                counts, _ = np.histogram(distances, bins=bins)
                
                # Format statistics text
                stats = f"⬤ Total points: {len(x)}"
                if len(x) < self.optimizer.max_points:
                    stats += " (showing all)"
                else:
                    original_count = self.optimizer.last_point_count
                    percent = int(100 * len(x) / max(1, original_count))
                    stats += f" (showing {percent}%)"
                    
                stats += "\n"
                
                # Only include non-zero bins to keep text compact
                for i in range(len(counts)):
                    if counts[i] > 0:
                        stats += f"⬤ {bins[i]:.0f}-{bins[i+1]:.0f}m: {counts[i]} pts\n"
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
                    
                    stats_text += (
                        f"◆ {config['label']} ({config['distance']:.1f}m, r={config['radius']:.1f}m)\n"
                        f"   Points: {stats.get('count', 0)}\n"
                        f"   Avg Intensity: {stats.get('avg_intensity', 0):.2f}"
                    )
        
        if not stats_text:
            stats_text = "No active circles"
            
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