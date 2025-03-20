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
        self.max_range = 35.0  # Ensure max range is exactly 35.0 meters
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
            grid_size = int(2 * self.max_range / 1.0)  # Lower resolution: 1.0m instead of 0.5m
            # Ensure grid size is even for better memory alignment
            if grid_size % 2 == 1:
                grid_size += 1
            self.heatmap_data = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Create axis
        self.ax = self.figure.add_subplot(111)
        
        # Set scientific background color - deeper blue for more contrast
        self.ax.set_facecolor('#050510')  # Deeper navy blue scientific background
        self.figure.patch.set_facecolor('#050510')
        
        # Use equal aspect ratio to keep circles circular
        self.ax.set_aspect('equal')
        
        # Create heatmap with improved scientific colormap
        cmap = plt.cm.get_cmap('viridis').copy()
        
        # Adjust normalization for better visibility and contrast
        norm = colors.PowerNorm(gamma=0.5, vmin=self.noise_floor, vmax=1.0)
        
        # Ensure correct extent values using max_range of 35.0 meters
        self.components['heatmap'] = self.ax.imshow(
            self.heatmap_data,
            extent=[-self.max_range, self.max_range, 0, self.max_range],
            origin='lower',
            cmap=cmap,
            norm=norm,
            aspect='auto',
            interpolation='bilinear',
            alpha=0.95
        )
        
        self.components['norm'] = norm
        self.components['contour'] = None
        self.components['contour_levels'] = 12  # Increased precision levels
        
        # Add colorbar with enhanced scientific styling
        colorbar = self.figure.colorbar(
            self.components['heatmap'],
            ax=self.ax,
            label='Signal Intensity (dB)',
            fraction=0.03,
            pad=0.02
        )
        colorbar.ax.tick_params(labelsize=8, colors='#AACCEE')
        colorbar.set_label('Signal Intensity (dB)', size=9, color='#BBDDFF')
        self.components['colorbar'] = colorbar
        
        # Define range arcs with scientific precision
        circle_interval_m = 5  # 5-meter intervals for precision
        major_ranges = list(range(0, int(self.max_range) + 1, circle_interval_m * 2))
        minor_ranges = [
            r for r in range(0, int(self.max_range) + 1, circle_interval_m)
            if r not in major_ranges
        ]
        
        # Add polar grid with light lines for scientific precision
        theta = np.linspace(0, np.pi, 100)
        for r in range(0, int(self.max_range) + 1, 10):
            if r == 0:
                continue
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            self.ax.plot(x, y, color='#334455', linestyle='-', linewidth=0.3, alpha=0.3)
        
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
                color='#AABBDD',  # Even lighter scientific blue-gray
                linestyle='-',
                linewidth=0.6,
                alpha=0.6
            )
            self.ax.add_patch(arc)
            if 0 < r <= self.max_range:
                self.ax.text(
                    r * 0.05, r, f"{int(r)}m", ha='left', va='bottom',
                    color='#AABBDD', fontsize=7, fontweight='normal',
                    bbox=dict(facecolor='#050510', edgecolor='none', 
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
        
        # Add angle markers every 30 degrees with proper scientific labels (keep existing)
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
            
            # Add angle label
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
        sensor_circle = Circle((0, 0), 0.5, fill=True, color='#00DDCC', alpha=0.7)
        self.ax.add_patch(sensor_circle)
        sensor_ring = Circle((0, 0), 0.7, fill=False, color='#00FFEE', linewidth=0.5, alpha=0.5)
        self.ax.add_patch(sensor_ring)
        
        # Add target arc with scientific styling
        self.components['target_arc'] = Arc(
            (0, 0),
            width=2 * self.target_distance,
            height=2 * self.target_distance,
            angle=0,
            theta1=0,
            theta2=180,
            fill=False,
            color='#FF5566',  # Scientific red
            linestyle='-',
            linewidth=1.0,
            alpha=0.7
        )
        self.ax.add_patch(self.components['target_arc'])
        
        # Create sampling circles with different positions and scientific colors
        self.components['sampling_circles'] = []
        
        # Circle colors and positions - enhanced scientific color scheme
        circle_configs = [
            {'enabled': True, 'distance': 5.0, 'radius': 0.5, 'angle': 0, 
             'color': '#44DDFF', 'label': 'Primary'},
            {'enabled': False, 'distance': 15.0, 'radius': 0.5, 'angle': -60, 
             'color': '#77BBFF', 'label': 'Left'},
            {'enabled': False, 'distance': 25.0, 'radius': 0.5, 'angle': 60, 
             'color': '#FFDD66', 'label': 'Right'}
        ]
        
        for i, config in enumerate(circle_configs):
            # Calculate x position based on angle (convert degrees to radians)
            angle_rad = config['angle'] * (3.14159 / 180.0)
            x_pos = config['distance'] * np.sin(angle_rad)
            y_pos = config['distance'] * np.cos(angle_rad)
            
            # Create sampling circle with scientific styling
            sampling_circle = Circle(
                (x_pos, y_pos),
                config['radius'],
                fill=False,
                color=config['color'],
                linestyle='-',
                linewidth=0.8,
                alpha=0.8 if config['enabled'] else 0.2,
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
        
        # Add roi indicators list for regions of interest
        self.components['roi_indicators'] = []
        
        # Add grid for scientific precision - very subtle
        self.ax.grid(True, color='#223344', linestyle=':', linewidth=0.2, alpha=0.3)
        
        # Set plot limits - ensure max_range is respected
        self.ax.set_xlim(-self.max_range, self.max_range)
        self.ax.set_ylim(0, self.max_range)
        
        # Add labels with scientific radar terminology
        self.ax.set_xlabel('Azimuth (m)', fontsize=9, labelpad=8, color='#BBDDFF')
        self.ax.set_ylabel('Range (m)', fontsize=9, labelpad=8, color='#BBDDFF')
        self.ax.set_title('Radar Intensity Map', fontsize=11, color='#DDEEFF', weight='normal')
        
        # Configure ticks with scientific styling
        self.ax.tick_params(axis='x', colors='#AABBDD', labelsize=8)
        self.ax.tick_params(axis='y', colors='#AABBDD', labelsize=8)
        
        # Set spine colors for scientific border
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#223344')
            spine.set_linewidth(0.5)
        
        # Add technical statistics box with enhanced resolution information
        res_text = (
            f"Resolution: {1.0:.1f}m/px\n"
            f"Range: 0-{self.max_range:.0f}m\n"
            f"μ-threshold: {self.noise_floor:.3f}\n"
            f"Grid: {grid_size}×{grid_size} px"
        )
        
        self.components['res_text'] = self.ax.text(
            0.98, 0.98, res_text,
            transform=self.ax.transAxes,
            color='#AABBDD',
            fontsize=7,
            ha='right',
            va='top',
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='#101025',
                alpha=0.8,
                edgecolor='#334466'
            )
        )
        
        # Add SNR text with scientific notation and units
        self.components['snr_text'] = self.ax.text(
            0.02, 0.02, 'SNR: N/A',
            transform=self.ax.transAxes,
            color='#EEDD66',
            fontsize=8,
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='#101025',
                alpha=0.8,
                edgecolor='#334466'
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
        # Verify that we have valid data
        if heatmap_data is None or heatmap_data.size == 0:
            # If no data, don't update anything
            return
            
        # Store the newest data regardless of whether we update the display
        self.heatmap_data = heatmap_data
        
        # CRITICAL FIX: Always force updates during the first 25 frames
        # This ensures data appears immediately and prevents empty display
        force_update = self.optimizer.frame_counter <= 25
        
        # Use optimizer to determine if we should update the visualization
        if not force_update and not self.optimizer.should_update_heatmap(heatmap_data):
            return
        
        try:
            # Apply noise floor threshold - FIXING POTENTIAL LOGIC ERROR
            # Create a copy to avoid modifying the original
            data_thresholded = heatmap_data.copy()
            
            # CRITICAL FIX: Only apply threshold if noise floor is > 0
            # This prevents empty heatmap when noise floor is too high
            if self.noise_floor > 0:
                data_thresholded[data_thresholded < self.noise_floor] = 0
            
            # DEBUGGING: Check if we have any non-zero values
            nonzero_count = np.count_nonzero(data_thresholded)
            if nonzero_count == 0 and np.max(heatmap_data) > 0:
                # If we've zeroed out all data due to threshold, use a fraction of max instead
                self.noise_floor = max(0.01, np.max(heatmap_data) * 0.25)
                data_thresholded = heatmap_data.copy()
                data_thresholded[data_thresholded < self.noise_floor] = 0
            
            # Make sure the heatmap component exists
            if 'heatmap' not in self.components or self.components['heatmap'] is None:
                # If missing, recreate the plot
                self.setup_plot()
                
            # Update display based on visualization mode - pass optimizer state for contour decisions
            self._update_visualization_mode(data_thresholded, redraw=force_update or self.optimizer.should_redraw())
            
            # Update SNR with enhanced scientific formatting
            if np.max(data_thresholded) > 0:
                snr_value = np.max(data_thresholded) / max(0.001, self.noise_floor)  # Avoid division by zero
                snr_db = 10.0 * np.log10(snr_value)
                if 'snr_text' in self.components:
                    self.components['snr_text'].set_text(f'SNR: {snr_db:.1f} dB (ratio: {snr_value:.1f})')
            else:
                if 'snr_text' in self.components:
                    self.components['snr_text'].set_text('SNR: N/A')
            
            # Only redraw if necessary according to the optimizer
            if force_update or self.optimizer.should_redraw():
                # Ensure aspect ratio is maintained
                self.ax.set_aspect('equal')
                # Use draw_idle which is more efficient than full draw
                self.canvas.draw_idle()
                
        except Exception as e:
            print(f"Error updating heatmap: {str(e)}")
            # Try to continue with basic update in case of error
            if 'heatmap' in self.components and self.heatmap_data is not None:
                self.components['heatmap'].set_data(self.heatmap_data)
                self.canvas.draw_idle()
    
    def _update_visualization_mode(self, data, redraw=True):
        """
        Update the visualization based on the current mode.
        
        Args:
            data: Thresholded data to visualize.
            redraw: Whether to redraw the canvas.
        """
        try:
            # Make sure we have valid components before updating
            if 'heatmap' not in self.components or self.components['heatmap'] is None:
                self.setup_plot()
                return
                
            # Update colormap normalization - only if data contains nonzero values
            nonzero_values = data[data > 0]
            if nonzero_values.size > 0:
                # Use percentile instead of max to avoid outlier influence (faster)
                vmax = np.percentile(nonzero_values, 98)
                if vmax < 0.1:
                    vmax = 0.1
                
                # Skip normalization update if the new vmax is very close to previous
                update_norm = True
                if 'norm' in self.components:
                    old_vmax = self.components['norm'].vmax
                    if abs(vmax - old_vmax) / old_vmax < 0.05:  # Less than 5% change
                        update_norm = False
                
                if update_norm:
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
            
            # Ensure heatmap is visible based on visualization mode
            self.components['heatmap'].set_visible(self.visualization_mode in ['heatmap', 'combined'])
            alpha = 0.95 if self.visualization_mode == 'heatmap' else 0.6
            self.components['heatmap'].set_alpha(alpha)
            
            # Clear existing contours if needed
            if 'contour' in self.components and self.components['contour'] is not None:
                # Only remove if we're switching modes or updating contours
                if self.visualization_mode not in ['contour', 'combined'] or self.optimizer.should_update_contours():
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
                    try:
                        # Use downsampled data for contour generation on large arrays
                        contour_data = data
                        if data.size > 40000:  # Only downsample for large arrays
                            # Calculate stride based on array size
                            stride = max(1, min(4, data.shape[0] // 100))
                            contour_data = data[::stride, ::stride]
                        
                        # Use evenly distributed levels for better visualization
                        if np.max(contour_data) > self.noise_floor:
                            # Generate more contour levels for scientific precision
                            num_levels = 12
                            levels = np.linspace(self.noise_floor, np.max(contour_data), num_levels)
                            
                            # Create contours only if we have valid levels
                            if levels.size > 1 and levels[-1] > levels[0]:
                                # Set contour colors and properties based on mode
                                contour_color = '#BBCCEE' if self.visualization_mode == 'combined' else '#DDEEFF'
                                line_width = 0.4 if self.visualization_mode == 'combined' else 0.6
                                
                                self.components['contour'] = self.ax.contour(
                                    contour_data,
                                    levels=levels,
                                    extent=[-self.max_range, self.max_range, 0, self.max_range],
                                    colors=contour_color,
                                    alpha=0.7,
                                    linewidths=line_width
                                )
                    except Exception as e:
                        print(f"Error creating contours: {e}")
            
            # Only redraw if requested
            if redraw:
                # Use draw_idle for better performance
                self.canvas.draw_idle()
                
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
                color='#FF5566',  # Scientific red
                linestyle='-',
                linewidth=1.0,  # Slightly thinner but still visible
                alpha=0.7      # More transparent
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
            linewidth=1.0,  # Thinner for subtlety
            alpha=0.7      # More transparent
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
            res = 1.0  # Updated to 1.0m resolution to match grid size
            grid_size_x, grid_size_y = self.heatmap_data.shape
            
            # Create coordinate arrays for all grid points
            y_indices, x_indices = np.indices((grid_size_y, grid_size_x))
            x_coords = x_indices * res - max_range
            y_coords = y_indices * res
            
            # Calculate distances from each grid point to ROI center
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # Calculate distances from each grid point to origin (0,0) for distance bands
            distance_from_origin = np.sqrt(x_coords**2 + y_coords**2)
            
            # Create masks for inside and outside ROI
            inside_roi_mask = distances <= radius
            outside_roi_mask = ~inside_roi_mask  # Use inverse for better efficiency
            
            # Get data for inside and outside ROI
            inside_roi_data = self.heatmap_data[inside_roi_mask]
            outside_roi_data = self.heatmap_data[outside_roi_mask]
            
            # Calculate distance from origin correctly for each point
            inside_roi_distances = distance_from_origin[inside_roi_mask]
            outside_roi_distances = distance_from_origin[outside_roi_mask]
            
            # Get intensity values corresponding to each distance
            inside_roi_intensity = self.heatmap_data[inside_roi_mask]
            outside_roi_intensity = self.heatmap_data[outside_roi_mask]
            
            # Initialize stats dictionary
            stats = {}
            
            if inside_roi_data.size > 0:
                # Calculate statistics for inside ROI
                inside_stats = {
                    'mean_intensity': float(np.mean(inside_roi_data)),
                    'max_intensity': float(np.max(inside_roi_data)),
                    'std_intensity': float(np.std(inside_roi_data)),
                    'signal_coverage': float(np.sum(inside_roi_data > self.noise_floor)) / inside_roi_data.size,
                    'points_count': inside_roi_data.size,
                    'distances': inside_roi_distances,  # Store distances for band analysis
                    'intensity_by_distance': inside_roi_intensity  # Store intensity by distance
                }
                stats['inside_roi'] = inside_stats
                
            if outside_roi_data.size > 0:
                # Calculate statistics for outside ROI
                outside_stats = {
                    'mean_intensity': float(np.mean(outside_roi_data)),
                    'max_intensity': float(np.max(outside_roi_data)),
                    'std_intensity': float(np.std(outside_roi_data)),
                    'signal_coverage': float(np.sum(outside_roi_data > self.noise_floor)) / outside_roi_data.size,
                    'points_count': outside_roi_data.size,
                    'distances': outside_roi_distances,  # Store distances for band analysis
                    'intensity_by_distance': outside_roi_intensity  # Store intensity by distance
                }
                stats['outside_roi'] = outside_stats
            
            # Calculate total points correctly
            inside_count = stats['inside_roi']['points_count'] if 'inside_roi' in stats else 0
            outside_count = stats['outside_roi']['points_count'] if 'outside_roi' in stats else 0
            total_points = inside_count + outside_count
            
            # Perform distance band analysis with more precise calculations
            distance_bands = [(0, 10), (10, 20), (20, 30), (30, 40)]
            band_analysis = []
            
            for min_dist, max_dist in distance_bands:
                # Process inside ROI points in this band
                inside_band_count = 0
                if 'inside_roi' in stats:
                    inside_band_mask = (stats['inside_roi']['distances'] >= min_dist) & (stats['inside_roi']['distances'] < max_dist)
                    inside_band_count = int(np.sum(inside_band_mask))
                
                # Process outside ROI points in this band
                outside_band_count = 0
                if 'outside_roi' in stats:
                    outside_band_mask = (stats['outside_roi']['distances'] >= min_dist) & (stats['outside_roi']['distances'] < max_dist)
                    outside_band_count = int(np.sum(outside_band_mask))
                
                # Total points in this band (sum, not double counting)
                band_total = inside_band_count + outside_band_count
                
                # Calculate average intensity in this band if needed
                avg_intensity = 0.0
                if 'inside_roi' in stats and inside_band_count > 0:
                    inside_intensities = stats['inside_roi']['intensity_by_distance'][inside_band_mask]
                    inside_intensity_sum = np.sum(inside_intensities)
                else:
                    inside_intensity_sum = 0.0
                
                if 'outside_roi' in stats and outside_band_count > 0:
                    outside_intensities = stats['outside_roi']['intensity_by_distance'][outside_band_mask]
                    outside_intensity_sum = np.sum(outside_intensities)
                else:
                    outside_intensity_sum = 0.0
                
                total_band_count = inside_band_count + outside_band_count
                if total_band_count > 0:
                    avg_intensity = (inside_intensity_sum + outside_intensity_sum) / total_band_count
                
                band_analysis.append({
                    'range': f"{min_dist}-{max_dist}m",
                    'count': band_total,
                    'inside_count': inside_band_count,
                    'outside_count': outside_band_count,
                    'avg_intensity': float(avg_intensity)
                })
            
            # Add band analysis to stats
            stats['distance_bands'] = band_analysis
            
            # Add combined statistics
            stats.update({
                'center': (center_x, center_y),
                'radius': radius,
                'total_points': total_points,
                'inside_outside_ratio': inside_roi_data.size / outside_roi_data.size if outside_roi_data.size > 0 else float('inf'),
                'intensity_ratio': (float(np.mean(inside_roi_data)) / float(np.mean(outside_roi_data))) 
                                   if outside_roi_data.size > 0 and np.mean(outside_roi_data) > 0 else float('inf')
            })
            
            # Create statistics text with scientific notation and improved styling
            inside_mean = stats['inside_roi']['mean_intensity'] if 'inside_roi' in stats else 0
            inside_max = stats['inside_roi']['max_intensity'] if 'inside_roi' in stats else 0
            inside_std = stats['inside_roi']['std_intensity'] if 'inside_roi' in stats else 0
            outside_mean = stats['outside_roi']['mean_intensity'] if 'outside_roi' in stats else 0
            
            inside_count = stats['inside_roi']['points_count'] if 'inside_roi' in stats else 0
            outside_count = stats['outside_roi']['points_count'] if 'outside_roi' in stats else 0
            
            intensity_ratio = inside_mean / outside_mean if outside_mean > 0 else float('inf')
            
            stats_text = (
                f"ROI Analysis (r={radius:.1f}m)\n"
                f"Inside: n={inside_count}, μ={inside_mean:.2f}, σ={inside_std:.2f}\n"
                f"Outside: n={outside_count}, μ={outside_mean:.2f}\n"
                f"Total: {total_points} pts, I.R.={intensity_ratio:.1f}x"
            )
            
            # Add distance band summary to stats with accurate counts
            band_summary = "\nDistance Bands:\n"
            for band in band_analysis:
                band_summary += f"{band['range']}: {band['count']} pts\n"
            
            stats_text += band_summary
            
            # Display statistics on the heatmap with scientific styling
            text = self.ax.text(
                center_x,
                center_y + radius + 0.5,  # Adjust position for better visibility
                stats_text,
                color='#CCDDEE',
                fontsize=8,
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='#101025',
                    alpha=0.85,
                    edgecolor='#44AA88'
                ),
                ha='center',
                weight='normal'
            )
            self.canvas.draw_idle()
            
            return stats
        
        except Exception as e:
            print(f"Error analyzing ROI: {str(e)}")
            return None
    
    def reset_heatmap(self):
        """Reset the heatmap to initial state."""
        grid_size = int(2 * self.max_range / 1.0)  # Lower resolution: 1.0m instead of 0.5m
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