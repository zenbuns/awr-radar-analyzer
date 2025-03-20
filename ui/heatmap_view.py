#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt-based heatmap view for radar intensity visualization.

This module provides a QtWidget that embeds a Matplotlib heatmap
for visualizing radar intensity data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Arc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import pyqtSignal

from .styles import Colors
from .heatmap_optimizer import HeatmapOptimizer


class HeatmapView(QWidget):
    """
    A PyQt widget that displays a radar intensity heatmap.
    
    This widget embeds a Matplotlib figure for interactive visualization
    of radar intensity data.
    
    Attributes:
        max_range: Maximum radar range in meters.
        circle_interval: Interval between range circles.
        figure: The Matplotlib figure instance.
        ax: The axes for the heatmap.
        components: Dictionary of plot components.
    """
    
    # Define signals
    update_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the HeatmapView widget.
        
        Args:
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        
        # Default parameters
        self.max_range = 35.0
        self.circle_interval = 10.0
        self.circle_distance = 5.0
        self.circle_radius = 0.5
        self.target_distance = 5.0
        self.noise_floor = 0.05
        self.heatmap_data = None
        self.current_colormap = 'plasma'  # Default to plasma for better contrast
        self.visualization_mode = 'heatmap'  # 'heatmap', 'contour', 'combined'
        self.roi_list = []
        
        # Plot components
        self.figure = None
        self.ax = None
        self.components = {}
        
        # Initialize the heatmap optimizer for performance improvements
        self.optimizer = HeatmapOptimizer()
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the widget UI components."""
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create figure and canvas with larger size
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create toolbar with modern styling
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add widgets to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)  # Give canvas all available space
        
        # Initialize the plot
        self.setup_plot()
        
    def setup_plot(self):
        """Set up the heatmap plot with initial components."""
        # Create initial heatmap data if not existing
        if self.heatmap_data is None:
            grid_size = int(2 * self.max_range / 0.5)
            # Ensure grid size is even for better memory alignment
            if grid_size % 2 == 1:
                grid_size += 1
            self.heatmap_data = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Create axis
        self.ax = self.figure.add_subplot(111)
        
        # Set background colors
        self.ax.set_facecolor(Colors.DARK_BACKGROUND)
        self.figure.patch.set_facecolor(Colors.DARK_BACKGROUND)
        
        # Set aspect ratio to auto to avoid aspect ratio errors
        self.ax.set_aspect('auto')
        
        # Create heatmap with improved colormap
        cmap = plt.cm.get_cmap(self.current_colormap).copy()
        
        # Adjust normalization for better visibility
        norm = colors.PowerNorm(gamma=0.7, vmin=self.noise_floor, vmax=1.0)
        
        self.components['heatmap'] = self.ax.imshow(
            self.heatmap_data,
            extent=[-self.max_range, self.max_range, 0, 2 * self.max_range],
            origin='lower',
            cmap=cmap,
            norm=norm,
            aspect='auto',
            interpolation='bilinear',
            alpha=0.9  # Slightly increased for better visibility
        )
        
        self.components['norm'] = norm
        self.components['contour'] = None
        self.components['contour_levels'] = 6
        
        # Add colorbar with modern styling
        colorbar = self.figure.colorbar(
            self.components['heatmap'],
            ax=self.ax,
            label='Intensity',
            fraction=0.03,  # Narrower
            pad=0.02
        )
        colorbar.ax.tick_params(labelsize=9, colors=Colors.TEXT_MUTED)
        colorbar.set_label('Intensity', size=11, color=Colors.TEXT)
        self.components['colorbar'] = colorbar
        
        # Define range arcs
        major_ranges = list(range(0, int(2 * self.max_range) + 1, int(self.circle_interval) * 2))
        minor_ranges = [
            r for r in range(0, int(2 * self.max_range) + 1, int(self.circle_interval))
            if r not in major_ranges
        ]
        
        # Add major range arcs with improved visibility
        for r in major_ranges:
            arc = Arc(
                (0, 0),
                width=2 * r,
                height=2 * r,
                angle=0,
                theta1=0,
                theta2=180,
                fill=False,
                color=Colors.TEXT,
                linestyle='-',
                linewidth=1.0,
                alpha=0.7
            )
            self.ax.add_patch(arc)
            if 0 < r <= self.max_range:
                self.ax.text(
                    0, r, f"{int(r)}m", ha='right', va='bottom',
                    color=Colors.TEXT, fontsize=9, fontweight='bold',
                    bbox=dict(facecolor=Colors.LIGHT_BACKGROUND, edgecolor='none', 
                             alpha=0.7, pad=1, boxstyle='round,pad=0.2')
                )
        
        # Add minor range arcs
        for r in minor_ranges:
            arc = Arc(
                (0, 0),
                width=2 * r,
                height=2 * r,
                angle=0,
                theta1=0,
                theta2=180,
                fill=False,
                color=Colors.TEXT,
                linestyle=':',
                linewidth=0.5,
                alpha=0.4
            )
            self.ax.add_patch(arc)
        
        # Add target arc with higher visibility
        self.components['target_arc'] = Arc(
            (0, 0),
            width=2 * self.target_distance,
            height=2 * self.target_distance,
            angle=0,
            theta1=0,
            theta2=180,
            fill=False,
            color=Colors.ACCENT_RED,
            linestyle='-',
            linewidth=1.5,
            alpha=0.9
        )
        self.ax.add_patch(self.components['target_arc'])
        
        # Create sampling circles with different positions and modern colors
        self.components['sampling_circles'] = []
        
        # Circle colors and positions
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
            
            # Create sampling circle with modern styling
            sampling_circle = Circle(
                (x_pos, y_pos),
                config['radius'],
                fill=False,
                color=config['color'],
                linestyle='-',
                linewidth=1.8,  # Thicker for better visibility
                alpha=0.8 if config['enabled'] else 0.2,
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
        
        # Add roi indicators list for regions of interest
        self.components['roi_indicators'] = []
        
        # Add grid for better readability
        self.ax.grid(True, color=Colors.BORDER, linestyle=':', linewidth=0.3, alpha=0.2)
        
        # Set plot limits
        self.ax.set_xlim(-self.max_range, self.max_range)
        self.ax.set_ylim(0, 2 * self.max_range)
        
        # Add labels with modern styling
        self.ax.set_xlabel('Cross-Range (m)', fontsize=11, labelpad=8, color=Colors.TEXT)
        self.ax.set_ylabel('Range (m)', fontsize=11, labelpad=8, color=Colors.TEXT)
        self.ax.set_title('Radar Intensity Map', fontsize=14, color=Colors.TEXT, weight='bold')
        
        # Configure ticks with modern styling
        self.ax.tick_params(axis='x', colors=Colors.TEXT_MUTED, labelsize=9)
        self.ax.tick_params(axis='y', colors=Colors.TEXT_MUTED, labelsize=9)
        
        # Set spine colors
        for spine in self.ax.spines.values():
            spine.set_edgecolor(Colors.BORDER)
        
        # Add SNR text with improved styling
        self.components['snr_text'] = self.ax.text(
            0.02, 0.02, 'SNR: N/A',
            transform=self.ax.transAxes,
            color=Colors.ACCENT_YELLOW,
            fontsize=10,
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor=Colors.LIGHT_BACKGROUND,
                alpha=0.8,
                edgecolor=Colors.BORDER
            )
        )
        
        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_heatmap_data(self, heatmap_data):
        """
        Update the heatmap with new data.
        
        Args:
            heatmap_data: New 2D numpy array of heatmap data.
        """
        # Store the newest data regardless of whether we update the display
        self.heatmap_data = heatmap_data
        
        # Use optimizer to determine if we should update the visualization
        if not self.optimizer.should_update_heatmap(heatmap_data):
            return
        
        # Apply noise floor threshold
        data_thresholded = heatmap_data.copy()
        data_thresholded[data_thresholded < self.noise_floor] = 0
        
        # Update display based on visualization mode - pass optimizer state for contour decisions
        self._update_visualization_mode(data_thresholded, redraw=self.optimizer.should_redraw())
        
        # Update SNR with formatting improvements
        if np.max(data_thresholded) > 0:
            snr = 10.0 * np.log10(np.max(data_thresholded) / self.noise_floor)
            self.components['snr_text'].set_text(f'SNR: {snr:.1f} dB')
        else:
            self.components['snr_text'].set_text('SNR: N/A')
        
        # Only redraw if necessary according to the optimizer
        if self.optimizer.should_redraw():
            # Use draw_idle which is more efficient than full draw
            self.canvas.draw_idle()
    
    def _update_visualization_mode(self, data, redraw=True):
        """
        Update the visualization based on the current mode.
        
        Args:
            data: Thresholded data to visualize.
            redraw: Whether to redraw the canvas.
        """
        try:
            # Update colormap normalization
            nonzero_values = data[data > 0]
            if nonzero_values.size > 0:
                vmax = np.percentile(nonzero_values, 98)
                if vmax < 0.1:
                    vmax = 0.1
                
                # Adjust power normalization based on point density
                density_ratio = nonzero_values.size / data.size
                if density_ratio < 0.01:
                    power = 0.3
                elif density_ratio > 0.2:
                    power = 0.7
                else:
                    power = 0.5
                
                norm = colors.PowerNorm(gamma=power, vmin=self.noise_floor, vmax=vmax)
                self.components['norm'] = norm
                self.components['heatmap'].set_norm(norm)
            
            # Always update the heatmap data which is relatively fast
            self.components['heatmap'].set_data(data)
            
            # Clear existing contours
            if self.components['contour'] is not None:
                for coll in self.components['contour'].collections:
                    try:
                        coll.remove()
                    except Exception:
                        pass
                self.components['contour'] = None
            
            # Only update contours if the visualization mode requires it and the optimizer allows it
            if self.visualization_mode in ['contour', 'combined'] and self.optimizer.should_update_contours():
                # Only generate contours if we have enough data
                nonzero_count = np.count_nonzero(data)
                if nonzero_count > 20:  # Skip if too few points
                    levels = np.linspace(self.noise_floor, np.max(data), 6)
                    # Create contours only if we have valid levels
                    if levels.size > 1 and levels[-1] > levels[0]:
                        self.components['contour'] = self.ax.contour(
                            data,
                            levels=levels,
                            extent=[-self.max_range, self.max_range, -self.max_range, self.max_range],
                            colors='white' if self.visualization_mode == 'combined' else 'black',
                            alpha=0.5,
                            linewidths=0.5
                        )
            
            # Set visibility of heatmap based on mode
            self.components['heatmap'].set_visible(self.visualization_mode in ['heatmap', 'combined'])
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
    
    def set_visualization_mode(self, mode):
        """
        Set the visualization mode.
        
        Args:
            mode: Visualization mode ('heatmap', 'contour', or 'combined').
        """
        if mode in ['heatmap', 'contour', 'combined']:
            self.visualization_mode = mode
            
            # Adjust optimizer settings based on visualization mode
            if mode == 'heatmap':
                # Fastest updates for heatmap-only mode
                self.optimizer.set_update_interval(0.05)
                self.optimizer.set_max_fps(30)
            elif mode == 'contour':
                # More conservative settings for contour mode which is CPU intensive
                self.optimizer.set_update_interval(0.2)
                self.optimizer.set_contour_interval(0.5)
                self.optimizer.set_max_fps(15)
            else:  # combined
                # Balanced settings for combined mode
                self.optimizer.set_update_interval(0.1)
                self.optimizer.set_contour_interval(0.5)
                self.optimizer.set_max_fps(20)
                
            # Update display with current data if available
            if self.heatmap_data is not None:
                data_thresholded = self.heatmap_data.copy()
                data_thresholded[data_thresholded < self.noise_floor] = 0
                self._update_visualization_mode(data_thresholded, True)
    
    def configure_optimizer(self, update_interval=None, contour_interval=None, max_fps=None):
        """
        Configure the heatmap optimizer settings.
        
        Args:
            update_interval: Time in seconds between heatmap updates.
            contour_interval: Time in seconds between contour updates.
            max_fps: Maximum frames per second for redrawing.
        """
        if update_interval is not None:
            self.optimizer.set_update_interval(update_interval)
        
        if contour_interval is not None:
            self.optimizer.set_contour_interval(contour_interval)
            
        if max_fps is not None:
            self.optimizer.set_max_fps(max_fps)
    
    def set_colormap(self, colormap):
        """
        Change the colormap used for the heatmap.
        
        Args:
            colormap: Name of the matplotlib colormap.
        """
        self.current_colormap = colormap
        
        if 'heatmap' in self.components:
            self.components['heatmap'].set_cmap(colormap)
            self.canvas.draw_idle()
    
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
        
        self.canvas.draw_idle()
    
    def update_target_distance(self, distance):
        """
        Update the target distance arc.
        
        Args:
            distance: New target distance in meters.
        """
        self.target_distance = distance
        if 'target_arc' in self.components:
            # Remove old arc
            old_arc = self.components['target_arc']
            old_arc.remove()
            
            # Create new arc with improved styling
            self.components['target_arc'] = Arc(
                (0, 0),
                width=2 * distance,
                height=2 * distance,
                angle=0,
                theta1=0,
                theta2=180,
                fill=False,
                color=Colors.ACCENT_RED,
                linestyle='-',
                linewidth=1.5,
                alpha=0.9
            )
            self.ax.add_patch(self.components['target_arc'])
            self.canvas.draw_idle()
    
    def set_noise_floor(self, value):
        """
        Set the noise floor threshold.
        
        Args:
            value: New noise floor value.
        """
        self.noise_floor = value
        
        if self.heatmap_data is not None:
            data_thresholded = self.heatmap_data.copy()
            data_thresholded[data_thresholded < value] = 0
            self._update_visualization_mode(data_thresholded)
    
    def add_roi(self, offset=0.5):
        """
        Add a Region of Interest circle to the plot.
        
        Args:
            offset: X-coordinate offset for the ROI.
        """
        # Place ROIs with an offset based on how many already exist
        offset = len(self.roi_list) * offset
        roi = Circle(
            (offset, self.circle_distance),
            self.circle_radius * 1.2,
            fill=False, 
            color=Colors.ACCENT_GREEN,
            linestyle='--', 
            linewidth=1.8
        )
        self.ax.add_patch(roi)
        self.roi_list.append(roi)
        self.canvas.draw_idle()
        
        return roi
    
    def clear_rois(self):
        """Remove all ROI circles from the plot."""
        for roi in self.roi_list:
            try:
                roi.remove()
            except Exception:
                pass
        self.roi_list = []
        self.canvas.draw_idle()
    
    def analyze_roi(self, roi):
        """
        Analyze data within an ROI.
        
        Args:
            roi: The ROI circle to analyze.
            
        Returns:
            Dictionary of analysis results.
        """
        if self.heatmap_data is None:
            return None
        
        try:
            center_x, center_y = roi.center
            radius = roi.radius
            
            # Get grid parameters
            max_range = self.max_range
            res = 0.5  # Assume 0.5m resolution for simplicity
            grid_size_x, grid_size_y = self.heatmap_data.shape
            
            # Create coordinate arrays for all grid points
            y_indices, x_indices = np.indices((grid_size_y, grid_size_x))
            x_coords = x_indices * res - max_range
            y_coords = y_indices * res
            
            # Calculate distances from each grid point to ROI center
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            roi_mask = distances <= radius
            roi_data = self.heatmap_data[roi_mask]
            
            if roi_data.size > 0:
                # Calculate statistics
                stats = {
                    'mean_intensity': float(np.mean(roi_data)),
                    'max_intensity': float(np.max(roi_data)),
                    'std_intensity': float(np.std(roi_data)),
                    'signal_coverage': float(np.sum(roi_data > self.noise_floor)) / roi_data.size,
                    'center': (center_x, center_y),
                    'radius': radius
                }
                
                # Create statistics text with improved styling
                stats_text = (
                    f"ROI Analysis\n"
                    f"μ={stats['mean_intensity']:.2f}  Max={stats['max_intensity']:.2f}\n"
                    f"Coverage: {stats['signal_coverage'] * 100:.1f}%"
                )
                
                # Display statistics on the heatmap with improved styling
                text = self.ax.text(
                    center_x,
                    center_y + radius + 0.5,  # Adjust position for better visibility
                    stats_text,
                    color=Colors.TEXT,
                    fontsize=9,
                    bbox=dict(
                        boxstyle='round,pad=0.4',
                        facecolor=Colors.LIGHT_BACKGROUND,
                        alpha=0.85,
                        edgecolor=Colors.ACCENT_GREEN
                    ),
                    ha='center',
                    weight='bold'
                )
                self.canvas.draw_idle()
                
                return stats
            
            return None
        
        except Exception as e:
            print(f"Error analyzing ROI: {str(e)}")
            return None
    
    def reset_heatmap(self):
        """Reset the heatmap to initial state."""
        grid_size = int(2 * self.max_range / 0.5)
        # Ensure grid size is even for better memory alignment
        if grid_size % 2 == 1:
            grid_size += 1
            
        # Create clean heatmap data
        self.heatmap_data = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Update the display
        self.update_heatmap_data(self.heatmap_data)
        
        # Reset optimizer state
        if hasattr(self, 'optimizer'):
            self.optimizer.last_update_time = 0
            self.optimizer.last_contour_time = 0
            self.optimizer.last_data_hash = None
            self.optimizer.frame_counter = 0
    
    def compute_metrics(self):
        """
        Compute scientific metrics for the heatmap.
        
        Returns:
            Dictionary of metrics including SNR, intensity statistics, etc.
        """
        if self.heatmap_data is None:
            return {
                'max_intensity': 0.0,
                'avg_intensity': 0.0,
                'snr_dB': 0.0,
                'active_cells': 0.0,
                'total_cells': 1.0,
                'coverage_percentage': 0.0
            }
        
        try:
            data = self.heatmap_data.copy()
            data[data < self.noise_floor] = 0
            
            nonzero = data[data > 0]
            max_intensity = float(np.max(data)) if nonzero.size > 0 else 0.0
            avg_intensity = float(np.mean(nonzero)) if nonzero.size > 0 else 0.0
            
            # Calculate signal-to-noise ratio in decibels
            snr_dB = 10.0 * np.log10(max_intensity / self.noise_floor) if max_intensity > self.noise_floor else 0.0
            active_cells = int(np.count_nonzero(data))
            total_cells = data.size
            coverage_percentage = 100.0 * active_cells / total_cells if total_cells > 0 else 0.0
            
            return {
                'max_intensity': max_intensity,
                'avg_intensity': avg_intensity,
                'snr_dB': snr_dB,
                'active_cells': float(active_cells),
                'total_cells': float(total_cells),
                'coverage_percentage': coverage_percentage
            }
        except Exception as e:
            print(f"Error computing heatmap metrics: {str(e)}")
            return {
                'max_intensity': 0.0,
                'avg_intensity': 0.0,
                'snr_dB': 0.0,
                'active_cells': 0.0,
                'total_cells': 1.0,
                'coverage_percentage': 0.0
            }