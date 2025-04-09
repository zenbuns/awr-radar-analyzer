#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt-based scatter plot view for radar point cloud visualization.

This module provides a QtWidget that embeds a Matplotlib scatter plot
for visualizing radar point clouds with enhanced scientific aesthetics,
performance optimizations, and configurable features like fading trails.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Arc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import Normalize, LinearSegmentedColormap
import time
from scipy.ndimage import gaussian_filter
import traceback  # For detailed error logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QCheckBox, QGroupBox, QHBoxLayout
from PyQt5.QtCore import pyqtSignal, QTimer

# Removed unused import: from .styles import Colors
from .scatter_optimizer import ScatterOptimizer
from .custom_toolbar import NonBlockingNavigationToolbar

# Define default colors for easier modification - dark mode color scheme
DEFAULT_BG_COLOR = '#04081A'  # Very dark blue background
DEFAULT_FIGURE_BG_COLOR = '#050A20'  # Very dark blue for figure
DEFAULT_WIDGET_BG_COLOR = '#040A1C'  # Very dark blue for widget
DEFAULT_TOOLBAR_BG_COLOR = '#0A1630'  # Slightly lighter dark blue for toolbar
DEFAULT_AXES_COLOR = '#081028'  # Dark blue for plot area
DEFAULT_GRID_MAJOR_COLOR = '#406090'  # Muted blue-grey for major grid
DEFAULT_GRID_MINOR_COLOR = '#253555'  # Darker blue-grey for minor grid
DEFAULT_GRID_TEXT_COLOR = '#80A0D0'  # Light blue-grey for text
DEFAULT_AXIS_LINE_COLOR = '#309060'  # Subtle green/teal for axis lines
DEFAULT_SPINE_COLOR = '#203050'  # Dark blue-grey for plot borders
DEFAULT_TITLE_COLOR = '#D8E8FF'  # Very light blue/white for title
DEFAULT_LABEL_COLOR = '#B0C4DE'  # Light steel blue for labels/ticks

class ScatterView(QWidget):
    """
    A PyQt widget displaying a radar point cloud scatter plot with scientific styling.
    
    Embeds a Matplotlib figure for interactive visualization of radar data points,
    featuring optional fading trails with density coloring, performance optimization,
    and configurable appearance.
    
    Attributes:
        max_range (float): Maximum radar range in meters for plot limits.
        figure (Figure): The Matplotlib figure instance.
        ax (Axes): The Matplotlib axes for the scatter plot.
        components (dict): Dictionary storing plot components (scatter plots, lines, etc.).
        optimizer (ScatterOptimizer): Optimizes point display for performance.
        latest_x (np.ndarray): Last received X coordinates.
        latest_y (np.ndarray): Last received Y coordinates.
        latest_intensities (np.ndarray): Last received intensity values.
        show_trail (bool): Flag to control the visibility of the fading trail.
        trail_duration (float): Duration in seconds for points to remain in the trail.
        use_density_coloring (bool): Flag to use density-based coloring for the trail.
    """
    
    # Define signals
    update_signal = pyqtSignal()  # General update signal (can be used by parent)
    trail_toggled_signal = pyqtSignal(bool)  # Signal emitted when trail is toggled
    density_toggled_signal = pyqtSignal(bool)  # Signal emitted when density coloring is toggled
    
    def __init__(self, parent=None):
        """
        Initialize the ScatterView widget.
        
        Args:
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        
        # Set widget background color
        self.setStyleSheet(f"background-color: {DEFAULT_WIDGET_BG_COLOR};")
        
        # --- Core Parameters ---
        self.max_range = 35.0
        # self.noise_floor = 0.05 # Unused currently

        # --- Data Storage ---
        self.latest_x = np.array([])
        self.latest_y = np.array([])
        self.latest_intensities = np.array([])
        self.latest_points = 0
        self.last_update_time = 0  # For potential debouncing/rate limiting
        
        # --- Trail Feature Parameters ---
        self.show_trail = False
        self.trail_history = []
        self.trail_duration = 20.0
        self.trail_active = False
        self.max_history_frames = 100
        self.frame_interval = 0.1
        self.last_frame_time = 0
        self.trail_decay_factor = 0.98

        # --- Trail Appearance Properties ---
        self.trail_color = '#FF3366'  # Bright pink - more visible on dark background
        self.trail_alpha_max = 0.9    # Increased for dark background
        self.trail_alpha_min = 0.3    # Increased for dark background
        self.trail_alpha_power = 1.2  # Reduced from 1.5 for slower alpha decay
        self.trail_size_max = 22      # Increased from 18
        self.trail_size_min = 6       # Increased from 3
        self.trail_size_power = 1.0   # Reduced from 1.2 for slower size reduction
        self.main_point_size = 20     # Configurable size for current points
        self.main_point_colormap = 'plasma'  # More visible on dark background

        # --- Density Coloring Parameters ---
        self.use_density_coloring = False # Changed default to False as it might be causing issues
        self.density_resolution = 80
        self.density_smoothing = 1.2
        self.density_colormap = self._create_dark_density_colormap()  # Use dark mode colormap
        
        # --- Plot Components ---
        self.figure = None
        self.ax = None
        self.components = {}
        self.colorbar = None
        
        # --- Performance ---
        self.optimizer = ScatterOptimizer()
        
        # --- UI Setup ---
        self.setup_ui()
        self._update_trail_parameters()  # Initialize dependent trail params

    def _create_density_colormap(self):
        """Creates and registers the custom density colormap."""
        try:
            # Dark mode colormap
            density_cmap_data = {
                'red':   [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.7, 1.0, 1.0), (1.0, 1.0, 1.0)],
                'green': [(0.0, 0.2, 0.2), (0.3, 0.8, 0.8), (0.7, 1.0, 1.0), (1.0, 0.0, 0.0)],
                'blue':  [(0.0, 0.6, 0.6), (0.3, 1.0, 1.0), (0.7, 0.0, 0.0), (1.0, 0.0, 0.0)]
            }
            cmap = LinearSegmentedColormap('DarkModeDensity', density_cmap_data)
            plt.cm.register_cmap(cmap=cmap)
            return cmap  # Return the colormap object
        except Exception as e:
            print(f"Warning: Could not create custom density colormap ({e}). Falling back to 'viridis'.")
            return plt.cm.get_cmap('viridis')
    
    def _create_dark_density_colormap(self):
        """Creates and registers a density colormap optimized for dark backgrounds."""
        try:
            # Bright colormap that pops on dark background
            density_cmap_data = {
                'red':   [(0.0, 0.1, 0.1), (0.3, 0.0, 0.0), (0.6, 0.9, 0.9), (0.8, 1.0, 1.0), (1.0, 1.0, 1.0)],
                'green': [(0.0, 0.3, 0.3), (0.3, 0.8, 0.8), (0.6, 0.4, 0.4), (0.8, 0.0, 0.0), (1.0, 0.7, 0.7)],
                'blue':  [(0.0, 0.8, 0.8), (0.3, 1.0, 1.0), (0.6, 0.0, 0.0), (0.8, 0.0, 0.0), (1.0, 0.7, 0.7)]
            }
            cmap = LinearSegmentedColormap('DarkModeRadarDensity', density_cmap_data)
            plt.cm.register_cmap(cmap=cmap)
            return cmap
        except Exception as e:
            print(f"Warning: Could not create dark density colormap ({e}). Falling back to 'plasma'.")
            return plt.cm.get_cmap('plasma')

    def setup_ui(self):
        """Set up the widget UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create main content area with toolbar and canvas
        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor=DEFAULT_FIGURE_BG_COLOR)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply dark background to canvas widget
        self.canvas.setStyleSheet(f"background-color: {DEFAULT_WIDGET_BG_COLOR};")
        
        # Custom Toolbar with dark styling
        self.toolbar = NonBlockingNavigationToolbar(self.canvas, self, view_type="scatter")
        self.toolbar.setStyleSheet(f"background-color: {DEFAULT_TOOLBAR_BG_COLOR}; color: white;")
        
        # Simple layout with just toolbar and canvas
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas, 1)
        
        # Initialize Plot
        self.setup_plot()

    def setup_plot(self):
        """Set up the scatter plot with dark mode styling."""
        if self.ax:  # Clear previous axes if re-setting plot
            self.figure.clear()

        # Create subplot with dark styling - adjusted position to shift right
        # Use a rect parameter to position the plot more to the right (increase left margin)
        # Default is [left, bottom, width, height] where all values are from 0-1
        self.ax = self.figure.add_subplot(111, facecolor=DEFAULT_AXES_COLOR, 
                                        position=[0.12, 0.1, 0.85, 0.85])  # Shifted right
        self.ax.set_aspect('equal', adjustable='box')  # Ensure aspect ratio maintained on resize
        
        # Add subtle border between plot area and background
        self.ax.patch.set_edgecolor(DEFAULT_SPINE_COLOR)  # Border color
        self.ax.patch.set_linewidth(0.8)      # Make edge visible

        # Plot Limits
        self._update_plot_limits()

        # Grid and Reference Lines (drawn by helper)
        self._draw_grid_elements()

        # --- Scatter Plot Components Initialization ---
        self.components['scatter'] = self.ax.scatter(
            [], [], s=self.main_point_size,
            c=[], cmap=self.main_point_colormap, vmin=-30, vmax=30,  # Initial dummy range
            alpha=0.95, edgecolor='none', zorder=10, label='Current Points'
        )

        # Create trail scatter with enhanced visibility for dark mode
        self.components['trail_scatter'] = self.ax.scatter(
            [], [], s=22,  # Increased size for visibility on dark background
            c=self.trail_color,
            alpha=0.8, edgecolor='#FFFFFF', linewidth=0.5, zorder=15, label='Trail Points'
        )

        # --- Sampling Circle Setup ---
        self._setup_sampling_circles()

        # --- Axes and Title ---
        self.ax.set_title('Radar Point Cloud Visualization', 
                         fontsize=14, color=DEFAULT_TITLE_COLOR, 
                         weight='bold', pad=10, 
                         fontname='Arial')
        self.ax.set_xticks([])  # Hide default ticks/labels
        self.ax.set_yticks([])
        
        # --- Spines ---
        for spine in self.ax.spines.values():
            spine.set_color(DEFAULT_SPINE_COLOR)
            spine.set_linewidth(0.8)
            spine.set_visible(True)

        # --- Colorbar for Intensity ---
        self._setup_colorbar()

        # Origin Marker
        self.components['origin_marker'] = self.ax.plot(0, 0, '+', markersize=8, 
                                                      color=DEFAULT_AXIS_LINE_COLOR, 
                                                      alpha=0.7, zorder=6)[0]

        # --- Final Layout Adjustment ---
        try:
            # Use constrained_layout for better automatic spacing
            self.figure.set_layout_engine('constrained')
        except Exception:  # Fallback for older matplotlib if constrained_layout fails
            print("Constrained layout engine not available, using tight_layout.")
            self.figure.tight_layout(pad=1.5)

        # Initial full draw to ensure visibility
        self.canvas.draw()

    def _update_plot_limits(self):
        """Sets the X and Y axis limits based on max_range."""
        if self.ax:
            self.ax.set_xlim(-self.max_range * 1.05, self.max_range * 1.05)
            self.ax.set_ylim(-0.02 * self.max_range, self.max_range * 1.05)

    def _draw_grid_elements(self):
        """Draws range arcs, angle lines, and crosshairs with dark mode styling."""
        if not self.ax:
            return

        # --- Clear existing grid elements before drawing new ones ---
        patches_to_remove = [p for p in self.ax.patches if isinstance(p, Arc)]
        lines_to_remove = [l for l in self.ax.lines if l.get_label() in ['_range_tick', '_angle_line', '_crosshair']]
        texts_to_remove = [t for t in self.ax.texts if t.get_label() == '_range_label']

        for p in patches_to_remove:
            p.remove()
        for l in lines_to_remove:
            l.remove()
        for t in texts_to_remove:
            t.remove()

        # --- Major Range Arcs ---
        major_range_interval = 10.0
        major_ranges = np.arange(major_range_interval, self.max_range + 1, major_range_interval)
        for r in major_ranges:
            arc = Arc((0, 0), 2 * r, 2 * r, angle=0, theta1=0, theta2=180,
                      color=DEFAULT_GRID_MAJOR_COLOR, linestyle='-', linewidth=0.8, alpha=0.7, zorder=1)
            self.ax.add_patch(arc)
            # Add range label text
            text = self.ax.text(r * 0.05, r * 0.98, f"{int(r)}m", 
                              ha='left', va='top',
                              color=DEFAULT_GRID_TEXT_COLOR, 
                              fontsize=9, weight='normal',
                              bbox=dict(facecolor=self.ax.get_facecolor(), 
                                       edgecolor='none',
                                       alpha=0.6, pad=0.1),
                              label='_range_label')

        # --- Minor Range Arcs ---
        minor_range_interval = 5.0
        for r in np.arange(minor_range_interval, self.max_range + 1, minor_range_interval):
            if r not in major_ranges:
                arc = Arc((0, 0), 2 * r, 2 * r, angle=0, theta1=0, theta2=180,
                          color=DEFAULT_GRID_MINOR_COLOR, linestyle=':', linewidth=0.6, alpha=0.6, zorder=1)
                self.ax.add_patch(arc)

        # --- Angle Lines ---
        major_angle_interval = 30
        minor_angle_interval = 15
        for angle_deg in range(-90 + minor_angle_interval, 90, minor_angle_interval):
            if angle_deg == 0:
                continue
            angle_rad = np.radians(angle_deg)
            x_end = self.max_range * 1.02 * np.sin(angle_rad)  # Extend slightly past max range
            y_end = self.max_range * 1.02 * np.cos(angle_rad)

            is_major = (angle_deg % major_angle_interval == 0)
            line_style = '--' if is_major else ':'
            line_width = 0.7 if is_major else 0.4
            line_color = DEFAULT_GRID_MAJOR_COLOR if is_major else DEFAULT_GRID_MINOR_COLOR
            line_alpha = 0.8 if is_major else 0.5

            self.ax.plot([0, x_end], [0, y_end], linestyle=line_style, color=line_color,
                         linewidth=line_width, alpha=line_alpha, zorder=1, label='_angle_line')

            if is_major:
                label_dist = 0.95 * self.max_range
                x_lab = label_dist * np.sin(angle_rad)
                y_lab = label_dist * np.cos(angle_rad)
                self.ax.text(x_lab, y_lab, f"{angle_deg}°", 
                           color=DEFAULT_GRID_TEXT_COLOR, 
                           fontsize=8, 
                           ha='center', va='center',
                           bbox=dict(facecolor=self.ax.get_facecolor(), 
                                    edgecolor='none',
                                    alpha=0.7, pad=0.1),
                           label='_range_label')

        # --- Central Axes / Crosshairs ---
        axis_alpha = 0.6
        self.ax.axvline(0, color=DEFAULT_AXIS_LINE_COLOR, linestyle='-', linewidth=0.6, alpha=axis_alpha, zorder=1, label='_crosshair')
        self.ax.axhline(0, color=DEFAULT_AXIS_LINE_COLOR, linestyle='-', linewidth=0.6, alpha=axis_alpha, zorder=1, label='_crosshair')
        
        tick_len = self.max_range * 0.01
        for tick_val in np.arange(-self.max_range, self.max_range + 1, 5.0):
            if abs(tick_val) < 1e-6:
                continue  # Skip origin
            self.ax.plot([tick_val, tick_val], [-tick_len, tick_len], 
                       color=DEFAULT_AXIS_LINE_COLOR, 
                       linewidth=0.5, alpha=axis_alpha*0.8, 
                       label='_range_tick')
        
        for tick_val in np.arange(minor_range_interval, self.max_range + 1, 5.0):
            self.ax.plot([-tick_len, tick_len], [tick_val, tick_val], 
                       color=DEFAULT_AXIS_LINE_COLOR, 
                       linewidth=0.5, alpha=axis_alpha*0.8, 
                       label='_range_tick')
            
        # Add subtle grid overlay for dark mode
        self.ax.grid(True, color=DEFAULT_GRID_MINOR_COLOR, linestyle=':', linewidth=0.3, alpha=0.15, zorder=0)

    def _setup_sampling_circles(self):
        """Initializes sampling circle patches and scatter plots."""
        if not self.ax:
            return

        # Clear previous if they exist
        if 'sampling_circle_patches' in self.components:
            for patch in self.components['sampling_circle_patches']:
                patch.remove()
        if 'sampling_circle_point_scatters' in self.components:
            for scatter in self.components['sampling_circle_point_scatters']:
                scatter.remove()
        # Clear config text if it exists (but don't recreate it)
        if 'config_text' in self.components:
            self.components['config_text'].remove()
            del self.components['config_text']

        self.components['sampling_circles_config'] = [
            {'enabled': True, 'distance': 5.0, 'radius': 0.5, 'angle': 0, 
             'color': '#44DDFF', 'marker': 'o', 'label': 'Primary', 'size': 50},
            {'enabled': False, 'distance': 15.0, 'radius': 0.5, 'angle': -45,
             'color': '#90EE90', 'marker': 's', 'label': 'Left', 'size': 40},
            {'enabled': False, 'distance': 15.0, 'radius': 0.5, 'angle': 45,
             'color': '#FFB6C1', 'marker': '^', 'label': 'Right', 'size': 40}
        ]
        self.components['sampling_circle_patches'] = []
        self.components['sampling_circle_point_scatters'] = []

        for i, config in enumerate(self.components['sampling_circles_config']):
            angle_rad = np.radians(config['angle'])
            x_pos = config['distance'] * np.sin(angle_rad)
            y_pos = config['distance'] * np.cos(angle_rad)
            config['x_pos'] = x_pos
            config['y_pos'] = y_pos

            # Circle boundary patch
            circle_patch = Arc(
                (x_pos, y_pos), 2 * config['radius'], 2 * config['radius'],
                angle=0, theta1=0, theta2=360, fill=False, color=config['color'],
                linestyle='--', linewidth=1.0, alpha=0.8 if config['enabled'] else 0.3,
                visible=config['enabled'], zorder=8, label=f"Sampling Circle {i}"
            )
            self.ax.add_patch(circle_patch)
            self.components['sampling_circle_patches'].append(circle_patch)

            # Scatter for points *within* this circle
            point_scatter = self.ax.scatter(
                [], [], s=config['size'], c=config['color'], marker=config['marker'],
                alpha=0.9 if config['enabled'] else 0.0,  # Set alpha based on enabled
                edgecolor='w', linewidth=0.5, zorder=9,
                visible=config['enabled'], label=f"Sampled Points {i}"
            )
            self.components['sampling_circle_point_scatters'].append(point_scatter)
            
        # Comment out the call to _update_config_text to remove the label
        # self._update_config_text()

    def _update_config_text(self):
        """Updates or creates the configuration text at the top of the plot."""
        # This method is disabled to remove the label from the plot
        return

    def update_circle_config(self, index, distance=None, angle=None, radius=None, enabled=None):
        """Updates a sampling circle's properties."""
        if not (0 <= index < len(self.components['sampling_circles_config'])):
            print(f"Error: Invalid sampling circle index {index}")
            return
            
        config = self.components['sampling_circles_config'][index]
        patch = self.components['sampling_circle_patches'][index]
        scatter = self.components['sampling_circle_point_scatters'][index]
        needs_redraw = False
        position_changed = False

        if distance is not None:
            config['distance'] = float(distance)
            position_changed = True
        if angle is not None:
            config['angle'] = float(angle)
            position_changed = True
        if radius is not None:
            config['radius'] = float(radius)
            needs_redraw = True
        if enabled is not None:
            config['enabled'] = bool(enabled)
            needs_redraw = True

        if position_changed:
            angle_rad = np.radians(config['angle'])
            x_pos = config['distance'] * np.sin(angle_rad)
            y_pos = config['distance'] * np.cos(angle_rad)
            config['x_pos'], config['y_pos'] = x_pos, y_pos
            patch.center = (x_pos, y_pos)
        if radius is not None:
            patch.set_width(2 * config['radius'])
            patch.set_height(2 * config['radius'])
        if enabled is not None or needs_redraw:
            is_enabled = config['enabled']
            patch.set_visible(is_enabled)
            patch.set_alpha(0.8 if is_enabled else 0.3)
            scatter.set_visible(is_enabled)
            if not is_enabled and scatter.get_offsets().shape[0] > 0:
                scatter.set_offsets(np.empty((0, 2)))

        # Update configuration text whenever circle config changes
        self._update_config_text()

        if needs_redraw or position_changed:
            self.canvas.draw_idle()

    def _setup_colorbar(self):
        """Creates or updates the colorbar with dark mode styling."""
        if not self.ax or 'scatter' not in self.components:
            return

        if self.colorbar:
            # Update existing colorbar limits and label if necessary
            scatter_norm = self.components['scatter'].norm
            self.colorbar.mappable.set_norm(scatter_norm)
        else:
            scatter_norm = self.components['scatter'].norm
            cmap = self.components['scatter'].cmap
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=scatter_norm)
            sm.set_array([])

            self.colorbar = self.figure.colorbar(
                sm, ax=self.ax,
                label='Relative Intensity (dB)',
                location='right',
                shrink=0.7,
                format='%.0f'
            )
            # Style colorbar for dark mode
            self.colorbar.ax.yaxis.label.set_color(DEFAULT_LABEL_COLOR)
            self.colorbar.ax.yaxis.label.set_size(9)
            self.colorbar.ax.tick_params(colors=DEFAULT_LABEL_COLOR, labelsize=8)
            self.colorbar.outline.set_edgecolor(DEFAULT_SPINE_COLOR)
            self.colorbar.outline.set_linewidth(0.8)

    # --- Configuration Methods ---

    def set_max_range(self, max_range):
        """Updates the maximum range and redraws relevant plot elements."""
        if max_range > 0:
            self.max_range = float(max_range)
            self._update_plot_limits()
            self._draw_grid_elements()
            self.canvas.draw_idle()
        else:
            print("Warning: Invalid max_range provided.")

    def set_main_point_size(self, size):
        """Sets the base size for the main scatter points."""
        if size > 0:
            self.main_point_size = size
            # Size will be applied in update_plot_data when new data arrives
            print(f"Main point size set to {self.main_point_size}")

    def set_intensity_colormap(self, cmap_name):
        """Sets the colormap for the main scatter points (intensity)."""
        try:
            cmap = plt.cm.get_cmap(cmap_name)
            self.main_point_colormap = cmap_name
            if 'scatter' in self.components:
                self.components['scatter'].set_cmap(cmap)
                self._setup_colorbar()  # Recreate/update colorbar for new cmap
            self.canvas.draw_idle()
            print(f"Intensity colormap set to '{cmap_name}'")
        except ValueError:
            print(f"Error: Colormap '{cmap_name}' not found.")

    def set_density_colormap(self, cmap_name):
        """Sets the colormap for density visualization in the trail."""
        try:
            self.density_colormap = plt.cm.get_cmap(cmap_name)
            print(f"Density colormap set to '{cmap_name}'")
            if self.trail_active and self.latest_points > 0:
                self.update_plot_data(self.latest_x, self.latest_y, self.latest_intensities, [])
        except ValueError:
            try:
                if isinstance(cmap_name, LinearSegmentedColormap):
                    self.density_colormap = cmap_name
                    print(f"Density colormap set to custom map '{cmap_name.name}'")
                    if self.trail_active and self.latest_points > 0:
                        self.update_plot_data(self.latest_x, self.latest_y, self.latest_intensities, [])
                else:
                    raise ValueError("Invalid Colormap")
            except ValueError:
                print(f"Error: Colormap '{cmap_name}' not found or invalid.")

    def toggle_trail(self, checked):
        """Toggle the trail feature on/off."""
        self.show_trail = checked
        
        if not self.show_trail:
            self.trail_history = []
            self.trail_active = False
            if 'trail_scatter' in self.components:
                self.components['trail_scatter'].set_visible(False)
                if self.components['trail_scatter'].get_offsets().shape[0] > 0:
                    self.components['trail_scatter'].set_offsets(np.empty((0, 2)))
            self.canvas.draw_idle()
        else:
            # Trail turned ON - force immediate update with latest data
            # First, add current data to history if we have any
            if self.latest_points > 0:
                # Add current frame twice to make trail more visible initially
                current_time = time.time()
                # Add first copy with slightly older timestamp
                self.trail_history.append({
                    'x': self.latest_x.copy(),
                    'y': self.latest_y.copy(),
                    'intensities': self.latest_intensities.copy(),
                    'timestamp': current_time - 0.5  # Slightly older for visual distinction
                })
                # Add second copy with current timestamp
                self.trail_history.append({
                    'x': self.latest_x.copy(),
                    'y': self.latest_y.copy(),
                    'intensities': self.latest_intensities.copy(),
                    'timestamp': current_time
                })
                self.last_frame_time = current_time
                self.trail_active = True  # Force trail to active state
                
                # Force trail visibility settings
                if 'trail_scatter' in self.components:
                    # Increase default size and alpha to make more visible
                    self.trail_alpha_max = 0.9  # Increase from default
                    self.trail_alpha_min = 0.2  # Increase from default
                    # Make sure scatter is visible
                    self.components['trail_scatter'].set_visible(True)
                
                # Now update the plot with latest data
                self.update_plot_data(self.latest_x, self.latest_y, self.latest_intensities, [])
                
                # Force redraw
                self.canvas.draw()  # Use full draw instead of draw_idle
            else:
                 self.canvas.draw_idle()
        
        # Emit signal for parent to update UI
        self.trail_toggled_signal.emit(checked)

    def toggle_density_coloring(self, checked):
        """Toggle density-based coloring for the trail."""
        self.use_density_coloring = checked
        if self.trail_active:
            self.canvas.draw_idle()
            
        # Emit signal for parent to update UI
        self.density_toggled_signal.emit(checked)

    def set_trail_duration(self, duration):
        """Set the trail duration and update related parameters."""
        self.trail_duration = max(0.1, float(duration))
        self._update_trail_parameters()

        current_time = time.time()
        self.trail_history = [
            h for h in self.trail_history
            if current_time - h['timestamp'] <= self.trail_duration
        ]
        if self.trail_active:
            if self.latest_points > 0:
                self.update_plot_data(self.latest_x, self.latest_y, self.latest_intensities, [])
            else:
                if 'trail_scatter' in self.components:
                    self.components['trail_scatter'].set_visible(False)
                    if self.components['trail_scatter'].get_offsets().shape[0] > 0:
                        self.components['trail_scatter'].set_offsets(np.empty((0, 2)))
        self.canvas.draw_idle()
        
    def _update_trail_parameters(self):
        """Recalculate frame interval and max history based on duration."""
        target_updates_per_sec = 15
        ideal_interval = 1.0 / target_updates_per_sec
        self.frame_interval = np.clip(ideal_interval, 0.05, 0.2)
        required_frames = int(np.ceil(self.trail_duration / self.frame_interval))
        self.max_history_frames = max(50, required_frames + 10)

    def set_trail_decay_factor(self, decay):
        """Set the decay factor (controls alpha fade speed)."""
        self.trail_decay_factor = np.clip(decay, 0.5, 0.999)
        if self.trail_active and self.latest_points > 0:
            self.update_plot_data(self.latest_x, self.latest_y, self.latest_intensities, [])

    def configure_optimizer(self, update_interval=None, max_points=None, adaptive_sampling=None):
        """Configure the scatter optimizer settings."""
        self.optimizer.configure(update_interval, max_points, adaptive_sampling)

    def update_point_limit(self, dataset_size):
        """Update optimizer's point limit based on dataset size (example logic)."""
        if dataset_size <= 20000:
            max_pts, interval = 10000, 0.05
        elif dataset_size <= 100000:
            max_pts, interval = 7000, 0.08
        else:
            max_pts, interval = 5000, 0.1
        self.optimizer.set_max_points(max_pts)
        self.optimizer.set_update_interval(interval)

    # --- Data Update and Rendering ---

    def update_plot_data(self, x, y, intensities, circles_data):
        """
        Update the plot with new data points and sampling circle contents.
        
        Args:
            x (array-like): X coordinates of current radar points.
            y (array-like): Y coordinates of current radar points.
            intensities (array-like): Intensity values of current radar points.
            circles_data (list): List of dicts, one per sampling circle.
        """
        try:
            current_time_update = time.perf_counter()
            if current_time_update - self.last_update_time < 0.05:
                return
            self.last_update_time = current_time_update

            x = np.asarray(x)
            y = np.asarray(y)
            intensities = np.asarray(intensities)
            current_point_count = len(x)

            self.latest_x, self.latest_y, self.latest_intensities = x, y, intensities
            self.latest_points = current_point_count

            # Always update trail history with original data before optimization
            self._update_trail_history(x, y, intensities)

            # Get optimized points for display
            x_display, y_display, intensities_display = self.optimizer.optimize_points(x, y, intensities)
            
            # If optimizer returned empty arrays, use original data at reduced density for current points
            if len(x_display) == 0 and len(x) > 0:
                # For display only - use a simple subsample of original data
                max_display = 1000  # Reasonable limit for display
                if len(x) > max_display:
                    indices = np.random.choice(len(x), max_display, replace=False)
                    x_display, y_display, intensities_display = x[indices], y[indices], intensities[indices]
                else:
                    x_display, y_display, intensities_display = x, y, intensities

            # Process trail visuals even if current points are optimized away
            trail_x, trail_y, trail_sizes, trail_colors = self._prepare_trail_visuals()
            self.trail_active = len(trail_x) > 0

            # Update main scatter with display points
            self._update_main_scatter(x_display, y_display, intensities_display)
            
            # Always update trail scatter, even if current points are empty
            self._update_trail_scatter(trail_x, trail_y, trail_sizes, trail_colors)

            self.update_sampling_circle_points(circles_data)

            # Force a full redraw to ensure trail is visible
            if self.show_trail and self.trail_active:
                self.canvas.draw()
            else:
                self.canvas.draw_idle()

        except Exception as e:
            print(f"Fatal Error in update_plot_data: {e}\n{traceback.format_exc()}")
            self._hide_all_points_on_error()

    def _update_trail_history(self, x, y, intensities):
        """Adds current frame to history if conditions met, prunes history."""
        if self.show_trail and len(x) > 0:
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_interval:
                self.trail_history.append({
                    'x': x.copy(), 'y': y.copy(), 'intensities': intensities.copy(),
                    'timestamp': current_time
                })
                self.last_frame_time = current_time

            self.trail_history = [
                h for h in self.trail_history 
                if current_time - h['timestamp'] <= self.trail_duration
            ]
            if len(self.trail_history) > self.max_history_frames:
                self.trail_history = self.trail_history[-self.max_history_frames:]

    def _update_main_scatter(self, x, y, intensities):
        """Safely updates the main scatter plot component."""
        if 'scatter' not in self.components:
            return
        scatter = self.components['scatter']
        try:
            if len(x) > 0:
                # Flip the x-coordinates to reverse left/right orientation
                # This ensures that negative angles (Left) appear on the left side
                # and positive angles (Right) appear on the right side of the plot
                scatter.set_offsets(np.column_stack((-x, y)))
                scatter.set_sizes(np.full(len(x), self.main_point_size))

                if len(intensities) > 20:
                    vmin, vmax = np.percentile(intensities, [2, 98])
                else:
                    vmin, vmax = np.min(intensities), np.max(intensities)
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                current_norm = scatter.norm
                if abs(current_norm.vmin - vmin) > 1 or abs(current_norm.vmax - vmax) > 1:
                    scatter.set_norm(Normalize(vmin=vmin, vmax=vmax))
                    self._setup_colorbar()
                scatter.set_array(intensities)
                scatter.set_visible(True)
            else:
                if scatter.get_visible():
                    scatter.set_visible(False)
                if scatter.get_offsets().shape[0] > 0:
                    scatter.set_offsets(np.empty((0, 2)))
        except Exception as e:
            print(f"Error updating main scatter: {e}")
            if scatter.get_visible():
                scatter.set_visible(False)

    def _update_trail_scatter(self, x, y, sizes, colors):
        """Safely updates the trail scatter plot component."""
        if 'trail_scatter' not in self.components:
            return
        scatter = self.components['trail_scatter']
        try:
            if self.show_trail and len(x) > 0:
                # Flip the x-coordinates to reverse left/right orientation
                # This ensures that negative angles (Left) appear on the left side
                # and positive angles (Right) appear on the right side of the plot
                scatter.set_offsets(np.column_stack((-x, y)))
                scatter.set_sizes(sizes)
                scatter.set_facecolors(colors)
                # Always make sure trail is visible when it should be
                scatter.set_visible(True)
                # Ensure trail is drawn above background but below main points
                scatter.set_zorder(6)  # Higher than background grid (1), lower than main points (10)
                self.trail_active = True  # Set trail active if we have points
                self.canvas.draw()
            else:
                if scatter.get_visible():
                    scatter.set_visible(False)
                if scatter.get_offsets().shape[0] > 0:
                    scatter.set_offsets(np.empty((0, 2)))
        except Exception as e:
            print(f"Error updating trail scatter: {e}")
            if scatter.get_visible():
                scatter.set_visible(False)

    def _prepare_trail_visuals(self):
        """
        Process trail history to generate positions, sizes, and colors for rendering.
        Returns:
            tuple: (trail_x, trail_y, trail_sizes, trail_colors) numpy arrays.
        """
        if not self.show_trail or not self.trail_history:
            return np.array([]), np.array([]), np.array([]), np.empty((0, 4))

        current_time = time.time()
        history_to_process = self.trail_history if self.trail_history else []
        if not history_to_process:
            return np.array([]), np.array([]), np.array([]), np.empty((0, 4))

        num_est_points = sum(len(f['x']) for f in history_to_process)
        
        all_trail_x = np.empty(num_est_points)
        all_trail_y = np.empty(num_est_points)
        all_trail_ts = np.empty(num_est_points)
        current_idx = 0

        for frame in history_to_process:
            n_points = len(frame['x'])
            if n_points == 0:
                continue

            age = current_time - frame['timestamp']
            keep_fraction = np.clip(0.05 + 0.95 * (1.0 - (age / self.trail_duration)), 0.05, 1.0)**1.5
            n_keep = max(1, int(n_points * keep_fraction))

            if n_keep == n_points:
                indices = slice(None)
            else:
                indices = np.random.choice(n_points, n_keep, replace=False)

            end_idx = current_idx + n_keep
            if end_idx > len(all_trail_x):
                new_size = int(len(all_trail_x) * 1.5 + n_keep)
                all_trail_x.resize(new_size, refcheck=False)
                all_trail_y.resize(new_size, refcheck=False)
                all_trail_ts.resize(new_size, refcheck=False)

            all_trail_x[current_idx:end_idx] = frame['x'][indices]
            all_trail_y[current_idx:end_idx] = frame['y'][indices]
            all_trail_ts[current_idx:end_idx] = frame['timestamp']
            current_idx = end_idx

        n_trail_points = current_idx
        if n_trail_points == 0:
            return np.array([]), np.array([]), np.array([]), np.empty((0, 4))

        trail_x = all_trail_x[:n_trail_points]
        trail_y = all_trail_y[:n_trail_points]
        trail_ts = all_trail_ts[:n_trail_points]

        ages = current_time - trail_ts
        age_factor = np.clip(1.0 - (ages / self.trail_duration), 0.0, 1.0)

        alpha_range = self.trail_alpha_max - self.trail_alpha_min
        base_alphas = (age_factor ** self.trail_alpha_power) * alpha_range + self.trail_alpha_min
        decay = np.clip(self.trail_decay_factor ** (ages * 5), 0.1, 1.0)
        final_alphas = np.clip(base_alphas * decay, self.trail_alpha_min, self.trail_alpha_max)
        
        size_range = self.trail_size_max - self.trail_size_min
        final_sizes = (age_factor ** self.trail_size_power) * size_range + self.trail_size_min

        if self.use_density_coloring and n_trail_points > 10:
            try:
                x_range = (-self.max_range, self.max_range)
                y_range = (0, self.max_range)
                bins = self.density_resolution
                weights = age_factor ** 1.5

                H, xedges, yedges = np.histogram2d(trail_x, trail_y, bins=bins, range=[x_range, y_range], weights=weights)
                if self.density_smoothing > 0:
                    H = gaussian_filter(H, sigma=self.density_smoothing)

                x_indices = np.clip(np.digitize(trail_x, xedges) - 1, 0, bins - 1)
                y_indices = np.clip(np.digitize(trail_y, yedges) - 1, 0, bins - 1)
                point_densities = H[x_indices, y_indices]

                log_densities = np.log1p(point_densities)
                if np.ptp(log_densities) > 1e-6:
                    d_min, d_max = np.percentile(log_densities, [5, 95])
                    if d_max <= d_min:
                        d_max = d_min + 1e-6
                    norm_densities = np.clip((log_densities - d_min) / (d_max - d_min), 0, 1)
                else:
                    norm_densities = np.zeros_like(log_densities)

                cmap = self.density_colormap if isinstance(self.density_colormap, LinearSegmentedColormap) else plt.cm.get_cmap(self.density_colormap)
                final_colors = cmap(norm_densities)
                final_colors[:, 3] = final_alphas
            except Exception as e:
                print(f"Warning: Density calculation failed ({e}). Using fixed color.")
                traceback.print_exc()
                base_color_rgba = plt.cm.colors.to_rgba(self.trail_color)
                final_colors = np.tile(base_color_rgba, (n_trail_points, 1))
                final_colors[:, 3] = final_alphas
        else:
            base_color_rgba = plt.cm.colors.to_rgba(self.trail_color)
            final_colors = np.tile(base_color_rgba, (n_trail_points, 1))
            final_colors[:, 3] = final_alphas

        return trail_x, trail_y, final_sizes, final_colors

    def update_sampling_circle_points(self, circles_data):
        """Updates the scatter plots for points within each sampling circle."""
        num_configured = len(self.components.get('sampling_circles_config', []))
        if not isinstance(circles_data, list) or len(circles_data) != num_configured:
            for i in range(num_configured):
                try:
                    scatter = self.components['sampling_circle_point_scatters'][i]
                    if scatter.get_offsets().shape[0] > 0:
                        scatter.set_offsets(np.empty((0, 2)))
                except (KeyError, IndexError):
                    pass
            return
            
        try:
            for i, data in enumerate(circles_data):
                config = self.components['sampling_circles_config'][i]
                scatter = self.components['sampling_circle_point_scatters'][i]

                if config['enabled'] and isinstance(data, dict) and 'x' in data and 'y' in data:
                    x_circ, y_circ = np.asarray(data['x']), np.asarray(data['y'])
                    n_circ_pts = len(x_circ)

                    if n_circ_pts > 0:
                        max_circle_points = 300
                        if n_circ_pts > max_circle_points:
                            indices = np.random.choice(n_circ_pts, max_circle_points, replace=False)
                            x_circ, y_circ = x_circ[indices], y_circ[indices]

                        # Flip the x-coordinates to reverse left/right orientation
                        # This ensures consistency with the main scatter and trail visualization
                        scatter.set_offsets(np.column_stack((-x_circ, y_circ)))
                    else:
                        if scatter.get_offsets().shape[0] > 0:
                            scatter.set_offsets(np.empty((0, 2)))
                else:
                    if scatter.get_offsets().shape[0] > 0:
                        scatter.set_offsets(np.empty((0, 2)))
                if scatter.get_visible() != config['enabled']:
                    scatter.set_visible(config['enabled'])
        except Exception as e:
            print(f"Error updating sampling circle points: {e}")
            self._hide_all_points_on_error()
    
    def clear_data(self):
        """Clear all plotted data and reset stored state."""
        try:
            self.latest_x = np.array([])
            self.latest_y = np.array([])
            self.latest_intensities = np.array([])
            self.latest_points = 0
            self.trail_history = []
            self.trail_active = False
            self.last_frame_time = 0
            self.last_update_time = 0
            
            self._hide_all_points_on_error()
            
            self.canvas.draw_idle()
            print("ScatterView data cleared.")
        except Exception as e:
            print(f"Error clearing scatter data: {e}")

    def _hide_all_points_on_error(self):
        """Helper to safely hide all dynamic scatter components."""
        components_to_clear = ['scatter', 'trail_scatter']
        for name in components_to_clear:
            if name in self.components:
                try:
                    scatter = self.components[name]
                    if scatter.get_visible():
                        scatter.set_visible(False)
                    if scatter.get_offsets().shape[0] > 0:
                        scatter.set_offsets(np.empty((0, 2)))
                except Exception as e:
                    print(f"Error hiding {name}: {e}")

        if 'sampling_circle_point_scatters' in self.components:
            for scatter in self.components['sampling_circle_point_scatters']:
                try:
                    if scatter.get_visible():
                        scatter.set_visible(False)
                    if scatter.get_offsets().shape[0] > 0:
                        scatter.set_offsets(np.empty((0, 2)))
                except Exception as e:
                    print(f"Error hiding sampling circle scatter: {e}")

    def clear_points(self):
        """Alias for clear_data()."""
        self.clear_data()

    def closeEvent(self, event):
        """Ensure plot resources are cleaned up on widget close."""
        if self.figure:
            plt.close(self.figure)
        super().closeEvent(event)

    def set_config_name(self, config_name, target_distance=None, collection_duration=None):
        """Updates the plot title with the configuration name and other parameters.
        
        Args:
            config_name (str): The name of the configuration
            target_distance (float, optional): Target distance in meters
            collection_duration (float, optional): Collection duration in seconds
        """
        # Create title string with configuration details
        title = config_name
        
        # Add additional details if provided
        elements = []
        if target_distance is not None:
            elements.append(f"Distance: {target_distance}m")
        if collection_duration is not None:
            elements.append(f"Duration: {collection_duration}s")
            
        if elements:
            title += f" ({', '.join(elements)})"
            
        # Update the plot title
        if self.ax:
            self.ax.set_title(title, 
                             fontsize=14, color=DEFAULT_TITLE_COLOR, 
                             weight='bold', pad=10, 
                             fontname='Arial')
            self.canvas.draw_idle()
            
        return title
