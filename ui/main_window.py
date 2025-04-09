#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main window for the Radar Point Cloud Analyzer application.

This module provides the main application window that integrates
all the UI components and connects them to the radar analyzer.
"""

import os
import csv
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QAction, QMenu, QToolBar, QStatusBar, QFileDialog, QMessageBox,
    QLabel, QFrame, QPushButton, QSizePolicy, QApplication, QProgressDialog,
    QProgressBar, QTabWidget, QMenuBar, QGroupBox, QRadioButton, QButtonGroup,
    QInputDialog, QCheckBox, QDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QSize, QUrl, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QIcon, QPixmap, QFont, QDesktopServices
import time
import math # For checking float values near zero

from ui.scatter_view import ScatterView
from ui.control_panel import ControlPanel
from utils.visualization import save_scientific_visualization
from .styles import DARK_STYLESHEET, Colors, apply_mpl_style
from ui.point_cloud_view import PointCloudView


class CombinedView(QWidget):
    """
    A combined view that shows both scatter plot and heatmap side by side.
    
    This class provides a widget that contains both visualization types
    with toggle options to show/hide each view independently.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the combined view widget.
        
        Args:
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        
        # Store reference to parent window
        self.main_window = parent
        self.scatter_view = None
        self.heatmap_view = None
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the widget UI components."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 5, 0, 0)
        
        # Create controls
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(10, 0, 10, 5)
        
        # Create scatter view toggle
        self.scatter_toggle = QCheckBox("Show Scatter Plot")
        self.scatter_toggle.setChecked(True)
        self.scatter_toggle.toggled.connect(self.toggle_scatter_view)
        controls_layout.addWidget(self.scatter_toggle)
        
        # Heatmap toggle removed to hide heatmap from UI
        
        # Add controls to main layout
        main_layout.addLayout(controls_layout)
        
        # Create splitter for the views
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter, 1)  # Give splitter all available space
    
    def set_views(self, scatter_view, heatmap_view):
        """
        Set the scatter and heatmap views.
        
        Args:
            scatter_view: ScatterView instance.
            heatmap_view: HeatmapView instance.
        """
        self.scatter_view = scatter_view
        self.heatmap_view = heatmap_view  # Store reference but don't add to UI
        
        # Add only scatter view to splitter
        if self.scatter_view and self.scatter_view.parent() != self.splitter:
            self.splitter.addWidget(self.scatter_view)
        
        # Heatmap view is kept in memory but not added to the UI
        
        # Adjust splitter
        if self.splitter.count() == 1:
            width = self.splitter.width()
            self.splitter.setSizes([width])
    
    def toggle_scatter_view(self, checked):
        """
        Toggle scatter view visibility.
        
        Args:
            checked: Whether the scatter view should be visible.
        """
        if self.scatter_view:
            self.scatter_view.setVisible(checked)
    
    def toggle_heatmap_view(self, checked):
        """
        Toggle heatmap view visibility.
        
        Args:
            checked: Whether the heatmap view should be visible.
        """
        if self.heatmap_view:
            self.heatmap_view.setVisible(checked)


class MainWindow(QMainWindow):
    """
    Main window for the Radar Point Cloud Analyzer application.
    
    This class integrates all UI components and handles the connections
    between them and the radar analyzer.
    
    Attributes:
        analyzer: RadarPointCloudAnalyzer instance.
        scatter_view: ScatterView widget for scatter plot.
        heatmap_view: HeatmapView widget for heatmap.
        control_panel: ControlPanel widget for UI controls.
        update_timer: Timer for periodic UI updates.
    """
    
    def __init__(self, analyzer=None):
        """
        Initialize the main window.
        
        Args:
            analyzer: RadarPointCloudAnalyzer instance (optional).
        """
        super().__init__()
        
        # Initialize analyzer
        self.analyzer = analyzer
        
        # UI components
        self.scatter_view = None
        self.control_panel = None
        self.status_bar = None
        self.progress_bar = None
        self.combined_view = None
        
        # State tracking
        self.collection_active = False
        self.export_thread = None
        self.export_worker = None
        self.export_timer = None
        self.export_cancelled = False
        
        # Session report path - store CSV path for the session
        self.session_report_path = None
        
        # Counter for generate report actions in this session
        self.generate_counter = 0
        
        # Initialize UI
        self.init_ui()
        
        # Initialize the combined view with the scatter and heatmap views
        self.combined_view.set_views(self.scatter_view, None)
        
        # Connect signals and slots
        self.connect_signals()
        
        # Initialize the control panel checkboxes to match the scatter view settings
        if self.scatter_view and self.control_panel:
            self.control_panel.trail_visibility_checkbox.setChecked(self.scatter_view.show_trail)
            self.control_panel.density_coloring_checkbox.setChecked(self.scatter_view.use_density_coloring)
        
        # Setup periodic update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_timer.start(100)  # 10 Hz update rate
        
        # Apply automatic optimization to the visualization pipeline
        # Use QTimer to delay this until after UI is fully initialized
        QTimer.singleShot(1000, self.optimize_visualization_pipeline)
        
        # Set window title, size, and style
        self.setWindowTitle("AWR1843 Radar Analyzer")
        self.resize(1600, 900)  # Larger default size
        self.setStyleSheet(DARK_STYLESHEET)
    
    def init_ui(self):
        """Initialize the main UI layout."""
        # Create main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add splitter for control panel and visualization area
        splitter = QSplitter(Qt.Horizontal)
        
        # Create and add control panel to splitter
        self.control_panel = ControlPanel(self)
        splitter.addWidget(self.control_panel)
        
        # Create visualization container
        viz_container = QWidget()
        viz_layout = QHBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabbed interface for visualizations
        tabs = QTabWidget()
        
        # Create and add 3D view tab
        self.point_cloud_view = PointCloudView(self)
        tabs.addTab(self.point_cloud_view, "3D View")
        
        # Create scatter view and heatmap view instances without adding them to tabs
        self.scatter_view = ScatterView(self)
        # self.heatmap_view = HeatmapView(self) # Removed heatmap view instantiation
        
        # Create and add combined view tab
        self.combined_view = CombinedView(self)
        tabs.addTab(self.combined_view, "2D View")
        
        # Connect tab change signal to handle view reparenting
        tabs.currentChanged.connect(self.handle_tab_change)
        
        # Set current tab to combined view
        tabs.setCurrentIndex(1)
        
        # Add tabs to visualization layout
        viz_layout.addWidget(tabs)
        
        # Add visualization container to splitter
        splitter.addWidget(viz_container)
        
        # Set splitter sizes
        splitter.setSizes([250, 750])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create progress bar for long operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Additional UI setup as needed
        self.create_menu_bar()
        self.create_toolbar()
        
        # Set window properties
        self.setGeometry(100, 100, 1280, 800)
        self.setWindowTitle('AWR Radar Analyzer')
        
        # Set up update timer for real-time visualizations
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_timer.start(100)  # Update 10 times per second initially
        
        # Schedule auto-optimization to run after the UI is fully initialized
        QTimer.singleShot(2000, self.optimize_visualization_pipeline)
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        save_heatmap_action = QAction("&Save Heatmap", self)
        save_heatmap_action.setShortcut("Ctrl+S")
        save_heatmap_action.triggered.connect(self.save_heatmap)
        # file_menu.addAction(save_heatmap_action) # Removed save heatmap action
        
        export_plot_action = QAction("&Export Plot", self)
        export_plot_action.setShortcut("Ctrl+E")
        export_plot_action.triggered.connect(self.export_scientific_plot)
        file_menu.addAction(export_plot_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Data menu
        data_menu = menu_bar.addMenu("&Data")
        
        start_collection_action = QAction("&Start Collection", self)
        start_collection_action.setShortcut("Ctrl+Space")
        start_collection_action.triggered.connect(self.start_data_collection_with_params)
        data_menu.addAction(start_collection_action)
        
        stop_collection_action = QAction("Sto&p Collection", self)
        stop_collection_action.setShortcut("Ctrl+Shift+Space")
        stop_collection_action.triggered.connect(self.stop_data_collection)
        data_menu.addAction(stop_collection_action)
        
        data_menu.addSeparator()
        
        reset_data_action = QAction("&Reset Data", self)
        reset_data_action.setShortcut("Ctrl+R")
        reset_data_action.triggered.connect(self.handle_data_reset)
        data_menu.addAction(reset_data_action)
        
        data_menu.addSeparator()
        
        gen_report_action = QAction("&Generate Report", self)
        gen_report_action.setShortcut("Ctrl+G")
        gen_report_action.triggered.connect(self.generate_report)
        data_menu.addAction(gen_report_action)
        
        # View menu
        view_menu = menu_bar.addMenu("&View")
        
        # Colormap submenu
        colormap_menu = view_menu.addMenu("&Colormap")
        
        colormaps = {
            'plasma': "Plasma (Default)",
            'viridis': "Viridis", 
            'inferno': "Inferno",
            'magma': "Magma",
            'cividis': "Cividis",
            'turbo': "Turbo"
        }
        
        for cmap_name, cmap_label in colormaps.items():
            cmap_action = QAction(cmap_label, self)
            cmap_action.triggered.connect(lambda checked, c=cmap_name: self.set_colormap(c))
            colormap_menu.addAction(cmap_action)
        
        view_menu.addSeparator()
        
        reset_heatmap_action = QAction("Reset &Heatmap", self)
        reset_heatmap_action.setShortcut("Ctrl+Shift+R")
        reset_heatmap_action.triggered.connect(self.reset_heatmap)
        # view_menu.addAction(reset_heatmap_action) # Removed reset heatmap action
        
        # Visualization performance submenu
        perf_menu = view_menu.addMenu("&Performance Settings")
        
        perf_menu.addSeparator()
        
        auto_perf_action = QAction("&Auto-Optimize", self)
        auto_perf_action.triggered.connect(self.optimize_visualization_pipeline)
        # perf_menu.addAction(auto_perf_action) # Auto-optimize likely tied to heatmap, remove for now
        
        view_menu.addSeparator()
        
        # Visualization mode submenu
        mode_menu = view_menu.addMenu("Visualization &Mode")
        
        # Removed heatmap/contour/combined mode actions
        # heatmap_mode_action = QAction("&Heatmap", self)
        # heatmap_mode_action.triggered.connect(lambda: self.set_visualization_mode("heatmap"))
        # mode_menu.addAction(heatmap_mode_action)
        
        # contour_mode_action = QAction("&Contour", self)
        # contour_mode_action.triggered.connect(lambda: self.set_visualization_mode("contour"))
        # mode_menu.addAction(contour_mode_action)
        
        # combined_mode_action = QAction("Co&mbined", self)
        # combined_mode_action.triggered.connect(lambda: self.set_visualization_mode("combined"))
        # mode_menu.addAction(combined_mode_action)
        
        # Colormap submenu
        colormap_menu = view_menu.addMenu("&Colormap")
        
        for cmap in ["viridis", "plasma", "inferno", "magma", "jet"]:
            cmap_action = QAction(cmap.capitalize(), self)
            cmap_action.triggered.connect(lambda checked, cm=cmap: self.set_colormap(cm))
            colormap_menu.addAction(cmap_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the application toolbar with modern icons."""
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add controls with icons (placeholder paths - would need actual icons)
        start_action = QAction(QIcon(":/icons/start.png"), "Start Collection", self)
        start_action.triggered.connect(self.start_data_collection)
        start_action.setStatusTip("Start data collection")
        toolbar.addAction(start_action)
        
        stop_action = QAction(QIcon(":/icons/stop.png"), "Stop Collection", self)
        stop_action.triggered.connect(self.stop_data_collection)
        stop_action.setStatusTip("Stop data collection")
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        reset_action = QAction(QIcon(":/icons/reset.png"), "Reset Heatmap", self)
        reset_action.triggered.connect(self.reset_heatmap)
        reset_action.setStatusTip("Reset the heatmap data")
        # toolbar.addAction(reset_action) # Removed reset heatmap toolbar action
        
        export_action = QAction(QIcon(":/icons/export.png"), "Export Plot", self)
        export_action.triggered.connect(self.export_scientific_plot)
        export_action.setStatusTip("Export scientific visualization")
        toolbar.addAction(export_action)
        
        toolbar.addSeparator()
        
        report_action = QAction(QIcon(":/icons/report.png"), "Generate Report", self)
        report_action.triggered.connect(self.generate_report)
        report_action.setStatusTip("Generate a comparison report")
        toolbar.addAction(report_action)
    
    def connect_signals(self):
        """Connect signals between UI components and analyzer."""
        if self.analyzer is None:
            return
        
        # Connect control panel signals to methods
        self.control_panel.circle_distance_changed.connect(self.update_circle_distance)
        self.control_panel.circle_radius_changed.connect(self.update_circle_radius)
        self.control_panel.circle_angle_changed.connect(self.update_circle_angle)
        self.control_panel.circle_toggled.connect(self.toggle_circle)
        
        self.control_panel.start_collection.connect(self.start_data_collection_with_params)
        self.control_panel.stop_collection.connect(self.stop_data_collection)
        # self.control_panel.reset_heatmap.connect(self.reset_heatmap) # Removed heatmap signal
        # self.control_panel.colormap_changed.connect(self.set_colormap) # Removed heatmap signal
        # self.control_panel.decay_factor_changed.connect(self.set_decay_factor) # Removed heatmap signal
        # self.control_panel.visualization_mode_changed.connect(self.set_visualization_mode) # Removed heatmap signal
        # self.control_panel.noise_floor_changed.connect(self.set_noise_floor) # Removed heatmap signal
        # self.control_panel.smoothing_changed.connect(self.set_smoothing) # Removed heatmap signal
        # self.control_panel.add_roi.connect(self.add_roi) # Removed heatmap signal
        # self.control_panel.clear_rois.connect(self.clear_rois) # Removed heatmap signal
        # self.control_panel.save_heatmap.connect(self.save_heatmap) # Removed heatmap signal
        self.control_panel.export_plot.connect(self.export_scientific_plot)
        self.control_panel.generate_report.connect(self.generate_report)
        self.control_panel.trail_duration_changed.connect(self.set_trail_duration)
        self.control_panel.trail_decay_factor_changed.connect(self.set_trail_decay_factor)
        self.control_panel.trail_visibility_changed.connect(self.set_trail_visibility)
        self.control_panel.density_coloring_changed.connect(self.set_density_coloring)
        
        # Connect ROS2 bag playback and recording signals
        self.control_panel.play_rosbag.connect(self.play_rosbag)
        self.control_panel.record_rosbag.connect(self.record_rosbag)
        self.control_panel.stop_rosbag.connect(self.stop_rosbag)
        self.control_panel.timeline_position_changed.connect(self.seek_rosbag)
        self.control_panel.visualize_pointcloud.connect(self.visualize_pointcloud)
        
        # Connect analyzer signals for playback progress updates
        if hasattr(self.analyzer, 'signals') and hasattr(self.analyzer.signals, 'update_playback_position_signal'):
            # Connect analyzer signal to ControlPanel's specific update slot
            self.analyzer.signals.update_playback_position_signal.connect(self.control_panel.update_playback_position)
        
        # Connect to data reset signal for handling PCL resets
        if hasattr(self.analyzer, 'signals') and hasattr(self.analyzer.signals, 'data_reset_signal'):
            self.analyzer.signals.data_reset_signal.connect(self.handle_data_reset)
    
    @pyqtSlot()
    def update_visualizations(self):
        """Update visualizations with current data from analyzer."""
        if self.analyzer is None:
            return
        
        try:
            with self.analyzer.data_lock:
                # Update scatter plot data
                x = self.analyzer.current_data['x']
                y = self.analyzer.current_data['y']
                intensities = self.analyzer.current_data['intensities']
                
                # Get circle data for all circles
                circles_data = []
                circle_stats = []
                
                # For now we're only using the primary circle from the analyzer until we update it
                # to handle multiple circles
                circle_x = self.analyzer.current_data['circle_x']
                circle_y = self.analyzer.current_data['circle_y']
                circle_intensities = self.analyzer.current_data['circle_intensities']
                
                # Mock data for additional circles - in a real implementation, the analyzer would
                # provide data for all circles
                for i in range(3):
                    if i == 0:
                        # Primary circle (use actual data)
                        circles_data.append({
                            'x': circle_x,
                            'y': circle_y,
                            'intensities': circle_intensities
                        })
                        
                        # Calculate circle statistics
                        circle_count = len(circle_x)
                        circle_avg_intensity = (
                            float(np.mean(circle_intensities))
                            if len(circle_intensities) > 0
                            else 0.0
                        )
                        
                        circle_stats.append({
                            'count': circle_count,
                            'avg_intensity': circle_avg_intensity
                        })
                    else:
                        # Mock data for other circles (empty for now)
                        circles_data.append({
                            'x': np.array([], dtype=np.float32),
                            'y': np.array([], dtype=np.float32),
                            'intensities': np.array([], dtype=np.float32)
                        })
                        
                        circle_stats.append({
                            'count': 0,
                            'avg_intensity': 0.0
                        })
                
                # Update views with all circle data
                self.scatter_view.update_plot_data(x, y, intensities, circles_data)
                self.scatter_view.update_circle_stats(circle_stats)
                
                # Get a reference to heatmap data (avoid copying the large array if possible)
                # heatmap_data = self.analyzer.live_heatmap_data # Removed heatmap data reference
                
                # Update heatmap data through the improved, optimized pipeline
                # self.heatmap_view.update_heatmap_data(heatmap_data) # Removed heatmap update
                
                # Update analysis metrics - only do this periodically as it's CPU intensive
                # Use a counter to update every 10 frames to reduce CPU load
                if not hasattr(self, '_metrics_update_counter'):
                    self._metrics_update_counter = 0
                
                if self._metrics_update_counter % 10 == 0:
                    # metrics = self.analyzer.compute_heatmap_metrics() # Removed heatmap metrics calculation
                    # self.control_panel.update_metrics(metrics) # Removed metrics update
                    pass # Placeholder if no other metrics are updated
                
                self._metrics_update_counter += 1
        except Exception as e:
            # Silently handle errors to avoid crashing the UI
            pass
    
    @pyqtSlot(int, float)
    def update_circle_distance(self, index, distance):
        """
        Update circle distance in views and analyzer.
        
        Args:
            index: Index of the circle to update (0-2)
            distance: New distance from origin in meters.
        """
        # Update analyzer
        if self.analyzer is not None:
            if index == 0:
                self.analyzer.update_circle_position(distance)
            # For all circles, update the params directly
            if hasattr(self.analyzer, 'params'):
                self.analyzer.params.update_circle_distance(index, distance)
        
        # Get the CURRENT angle for this circle from the parameters
        current_angle = 0.0 # Default fallback
        try:
            if self.analyzer and hasattr(self.analyzer, 'params') and 0 <= index < len(self.analyzer.params.circles):
                current_angle = self.analyzer.params.circles[index].angle
            else:
                print(f"Warning: Could not get current angle for circle {index}. Using fallback {current_angle}.")
        except (AttributeError, IndexError) as e:
             print(f"Error getting angle for circle {index}: {e}. Using fallback {current_angle}.")

        # Update views with the new distance and the CURRENT angle
        if self.scatter_view:
            self.scatter_view.update_circle_config(index, distance=distance, angle=current_angle)
        # self.heatmap_view.update_circle_position(index, distance, angle) # Removed heatmap view update
    
    @pyqtSlot(int, float)
    def update_circle_radius(self, index, radius):
        """
        Update circle radius in views and analyzer.
        
        Args:
            index: Index of the circle to update (0-2)
            radius: New circle radius in meters.
        """
        # Update analyzer
        if self.analyzer is not None:
            if index == 0:
                self.analyzer.update_circle_radius(radius)
            # For all circles, update the params directly
            if hasattr(self.analyzer, 'params'):
                self.analyzer.params.update_circle_radius(index, radius)
        
        # Update views
        self.scatter_view.update_circle_config(index, radius=radius)
        # self.heatmap_view.update_circle_radius(index, radius) # Removed heatmap view update
    
    @pyqtSlot(int, float)
    def update_circle_angle(self, index, angle):
        """
        Update circle angle in views and analyzer.

        Args:
            index: Index of the circle to update (0-2)
            angle: New angle in degrees.
        """
        # Update analyzer parameters first
        if self.analyzer is not None and hasattr(self.analyzer, 'params'):
            self.analyzer.params.update_circle_angle(index, angle)
        else:
            print("Warning: Analyzer or params not found, cannot update backend angle.")
            return # Don't update visualization if backend wasn't updated

        # Get current distance for this circle from analyzer parameters
        distance = 5.0 # Default fallback
        try:
            if self.analyzer and hasattr(self.analyzer, 'params') and 0 <= index < len(self.analyzer.params.circles):
                distance = self.analyzer.params.circles[index].distance
            else:
                 print(f"Warning: Could not get current distance for circle {index}. Using fallback {distance}.")
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error getting distance for circle {index}: {e}. Using fallback {distance}.")

        # Get the angle that was actually set (potentially clamped)
        final_angle = angle # Fallback to input angle
        try:
            if self.analyzer and hasattr(self.analyzer, 'params') and 0 <= index < len(self.analyzer.params.circles):
                final_angle = self.analyzer.params.circles[index].angle
            else:
                print(f"Warning: Could not get updated angle for circle {index}. Using input value {angle}.")
        except (AttributeError, IndexError) as e:
             print(f"Error getting updated angle for circle {index}: {e}. Using input value {angle}.")

        # Update views with current distance and the FINAL (clamped) angle
        if self.scatter_view:
            self.scatter_view.update_circle_config(index, distance=distance, angle=final_angle)
        # self.heatmap_view.update_circle_position(index, distance, angle) # Removed heatmap view update
    
    @pyqtSlot(int, bool)
    def toggle_circle(self, index, enabled):
        """
        Toggle circle visibility.
        
        Args:
            index: Index of the circle to toggle (0-2)
            enabled: Whether the circle should be visible
        """
        # Update views
        self.scatter_view.update_circle_config(index, enabled=enabled)
        # self.heatmap_view.toggle_circle(index, enabled) # Removed heatmap view update
        
        # Update analyzer if it exists
        if self.analyzer is not None and hasattr(self.analyzer, 'params'):
            self.analyzer.params.toggle_circle(index, enabled)
    
    def start_data_collection_with_params(self, config_name, target_distance, duration):
        """Start data collection with provided parameters."""
        if not hasattr(self, 'analyzer') or self.analyzer is None:
            QMessageBox.warning(self, "Error", "Analyzer not initialized")
            return

        # Convert target distance to float
        try:
            target_dist = float(target_distance)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid target distance")
            return

        # Start data collection
        success = self.analyzer.start_data_collection(
            config_name, target_dist, duration
        )

        if success:
            self.status_bar.showMessage(
                f"Started data collection for {config_name} at {target_dist}m for {duration}s"
            )
            # Update UI to reflect collection state
            self.collection_active = True
            # Additional UI updates can be added here
        else:
            QMessageBox.warning(
                self, "Error", "Failed to start data collection"
            )
    
    @pyqtSlot()
    def start_data_collection(self):
        """Open dialog to configure and start data collection."""
        if not hasattr(self, 'analyzer') or self.analyzer is None:
            QMessageBox.warning(self, "Error", "Analyzer not initialized")
            return
            
        # Ensure the distance calculation mode is set correctly before starting collection
        if hasattr(self, 'directional_radio') and hasattr(self, 'euclidean_radio'):
            use_directional = self.directional_radio.isChecked()
            self.analyzer.params.use_directional_distance = use_directional
            self.analyzer.get_logger().info(
                f"Distance calculation set to: {'directional' if use_directional else 'Euclidean'}"
            )
        
        # Get collection parameters from UI
        config_name, ok = QInputDialog.getText(
            self, "Start Collection", "Configuration Name:"
        )
        if not ok or not config_name:
            return
            
        target_distance, ok = QInputDialog.getText(
            self, "Start Collection", "Target Distance (m):"
        )
        if not ok or not target_distance:
            return
            
        try:
            # Convert target distance to float
            target_dist = float(target_distance)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid target distance")
            return
            
        # Set duration (default to 60 seconds)
        duration = 60
            
        # Call the actual collection method with parameters
        self.start_data_collection_with_params(config_name, target_distance, duration)
    
    @pyqtSlot()
    def stop_data_collection(self):
        """Stop data collection."""
        if self.analyzer is not None:
            self.analyzer.stop_data_collection()
            self.status_bar.showMessage("Data collection stopped")
    
    @pyqtSlot()
    def reset_heatmap(self):
        """Reset the heatmap visualization."""
        if self.analyzer is not None:
            # self.analyzer.reset_live_heatmap() # Removed analyzer heatmap reset
            pass # Placeholder
        
        # self.heatmap_view.reset_heatmap() # Removed heatmap view reset
        self.status_bar.showMessage("Heatmap functionality removed")
    
    @pyqtSlot(str)
    def set_colormap(self, colormap):
        """
        Set the colormap for visualizations.
        
        Args:
            colormap: Name of the colormap to use.
        """
        # self.heatmap_view.set_colormap(colormap) # Removed heatmap view update
        self.status_bar.showMessage(f"Colormap setting removed (heatmap disabled)")
    
    @pyqtSlot(float)
    def set_decay_factor(self, decay):
        """
        Set the decay factor for live heatmap.
        
        Args:
            decay: New decay factor value.
        """
        if self.analyzer is not None:
            # self.analyzer.live_heatmap_decay_factor = decay # Removed analyzer decay factor setting
            pass # Placeholder
        
        # If synchronize_trail_decay option is enabled on the control panel, also update the trail
        if hasattr(self.control_panel, 'sync_trail_decay') and self.control_panel.sync_trail_decay.isChecked():
            self.set_trail_decay_factor(decay)
            
        self.status_bar.showMessage(f"Decay factor set to {decay:.3f}")
    
    @pyqtSlot(float)
    def set_trail_decay_factor(self, decay):
        """
        Set the decay factor for point trail visualization.
        
        Args:
            decay: New decay factor value (0-1, higher values = slower decay)
        """
        if self.scatter_view:
            self.scatter_view.set_trail_decay_factor(decay)
            # Don't update status bar here, as this is often called from set_decay_factor
    
    @pyqtSlot(str)
    def set_visualization_mode(self, mode):
        """
        Set the visualization mode.
        
        Args:
            mode: Visualization mode ('heatmap', 'contour', or 'combined').
        """
        # self.heatmap_view.set_visualization_mode(mode) # Removed heatmap view update
        self.status_bar.showMessage(f"Visualization mode setting removed (heatmap disabled)")
    
    @pyqtSlot(float)
    def set_noise_floor(self, value):
        """
        Set the noise floor threshold.
        
        Args:
            value: New noise floor value.
        """
        # self.heatmap_view.set_noise_floor(value) # Removed heatmap view update
        self.status_bar.showMessage(f"Noise floor setting removed (heatmap disabled)")
    
    @pyqtSlot(float)
    def set_smoothing(self, value):
        """
        Set the smoothing factor.
        
        Args:
            value: New smoothing factor value.
        """
        # This doesn't directly affect anything in real-time,
        # it will be used when exporting visualizations
        self.status_bar.showMessage(f"Smoothing factor set to {value:.1f}")
    
    @pyqtSlot()
    def add_roi(self):
        """Add a Region of Interest to the heatmap."""
        # roi = self.heatmap_view.add_roi() # Removed ROI addition
        # if roi is not None:
        #     stats = self.heatmap_view.analyze_roi(roi)
        #     if stats:
        #         self.status_bar.showMessage(
        #             f"ROI added: avg={stats['mean_intensity']:.2f}, max={stats['max_intensity']:.2f}, "
        #             f"coverage={stats['signal_coverage']*100:.1f}%"
        #         )
        self.status_bar.showMessage("ROI functionality removed (heatmap disabled)")
    
    @pyqtSlot()
    def clear_rois(self):
        """Clear all Regions of Interest from the heatmap."""
        # self.heatmap_view.clear_rois() # Removed ROI clearing
        self.status_bar.showMessage("ROI functionality removed (heatmap disabled)")
    
    @pyqtSlot()
    def save_heatmap(self):
        """Save the current heatmap data and visualization."""
        # if self.analyzer is None or self.analyzer.live_heatmap_data is None: # Removed heatmap check
        #     self.status_bar.showMessage("No heatmap data to save")
        #     return
        self.status_bar.showMessage("Save heatmap functionality removed")
        # Removed all saving logic
    
    @pyqtSlot()
    def export_scientific_plot(self):
        """Export a high-quality scientific visualization of the radar data."""
        # Check if we have data to export
        if self.analyzer is None or self.analyzer.live_heatmap_data is None:
            self.status_bar.showMessage("No data to export")
            QMessageBox.warning(self, "No Data", "No radar data available to export.")
            return
        
        try:
            # Use state manager to lock UI during export if available
            if hasattr(self.control_panel, 'state_manager'):
                self.control_panel.state_manager.transition('lock_ui')
            
            # Ensure the export directory exists
            export_dir = os.path.join(os.path.expanduser("~"), "radar_experiment_data", "exports")
            try:
                os.makedirs(export_dir, exist_ok=True)
            except Exception as e:
                print(f"Error creating export directory: {e}")
                # If default export directory creation fails, try using the home directory
                export_dir = os.path.expanduser("~")
            
            # Generate default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = "default_config"
            if hasattr(self.control_panel, 'config_entry') and self.control_panel.config_entry.text().strip():
                config_name = self.control_panel.config_entry.text().strip()
                
            default_filename = f"radar_plot_{config_name}_{timestamp}.png"
            default_path = os.path.join(export_dir, default_filename)
            
            # Use a non-blocking file dialog approach
            dialog = QFileDialog(self, "Save Scientific Plot", default_path)
            dialog.setAcceptMode(QFileDialog.AcceptSave)
            dialog.setNameFilter("PNG Image (*.png);;PDF Document (*.pdf);;SVG Image (*.svg)")
            dialog.setDefaultSuffix("png")
            dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # Use Qt's dialog instead of native for better control
            dialog.setWindowModality(Qt.WindowModal)  # Make dialog modal to main window only
            
            # Process events to keep UI responsive
            QApplication.processEvents()
            
            # Show dialog and wait for result
            if dialog.exec_() == QDialog.Accepted:
                selected_files = dialog.selectedFiles()
                if not selected_files:
                    # User canceled or no file selected
                    if hasattr(self.control_panel, 'state_manager'):
                        self.control_panel.state_manager.transition('unlock_ui')
                    return
                    
                file_path = selected_files[0]
                
                # Save the directory for next time if needed
                if hasattr(self.control_panel, 'save_last_used_directory'):
                    try:
                        self.control_panel.save_last_used_directory(os.path.dirname(file_path))
                    except Exception as e:
                        print(f"Error saving last used directory: {e}")
                
                # Update status message
                self.status_bar.showMessage("Preparing to export plot...")
                QApplication.processEvents()  # Force UI update
                
                # Get visualization parameters with defaults and error handling
                try:
                    # Get colormap
                    colormap = "viridis"  # Default
                    if hasattr(self.control_panel, 'colormap_combo'):
                        colormap = self.control_panel.colormap_combo.currentText()
                    
                    # Get visualization mode
                    visualization_mode = "heatmap"  # Default
                    if hasattr(self.control_panel, 'vis_mode_group'):
                        for button in self.control_panel.vis_mode_group.buttons():
                            if button.isChecked():
                                mode_text = button.text().lower()
                                if mode_text == "heat":
                                    visualization_mode = "heatmap"
                                elif mode_text == "contour":
                                    visualization_mode = "contour"
                                elif mode_text == "combined":
                                    visualization_mode = "combined"
                                break
                    
                    # Get noise floor and smoothing parameters
                    noise_floor = 0.1  # Default
                    smoothing_sigma = 1.0  # Default
                    
                    if hasattr(self.control_panel, 'noise_value'):
                        try:
                            noise_floor = float(self.control_panel.noise_value.text())
                        except (ValueError, AttributeError):
                            print("Error parsing noise floor, using default")
                            
                    if hasattr(self.control_panel, 'smooth_value'):
                        try:
                            smoothing_sigma = float(self.control_panel.smooth_value.text())
                        except (ValueError, AttributeError):
                            print("Error parsing smoothing value, using default")
                except Exception as param_error:
                    print(f"Error getting parameters: {param_error}")
                    # Continue with defaults if there was an error
                
                # Create a progress dialog to show export progress
                self.progress_dialog = QProgressDialog("Initializing export...", "Cancel", 0, 100, self)
                self.progress_dialog.setWindowTitle("Exporting Plot")
                self.progress_dialog.setWindowModality(Qt.WindowModal)
                self.progress_dialog.setMinimumDuration(0)  # Show immediately
                self.progress_dialog.setValue(0)
                self.progress_dialog.setAutoClose(False)
                self.progress_dialog.setAutoReset(False)
                
                # Connect cancel signal to handler
                cancel_button = self.progress_dialog.findChild(QPushButton)
                if cancel_button:
                    cancel_button.clicked.disconnect()  # Disconnect default behavior
                    cancel_button.clicked.connect(self.cancel_export)
                
                # Show the progress dialog
                self.progress_dialog.show()
                QApplication.processEvents()  # Force UI update
                
                # Clone the heatmap data to avoid threading issues
                try:
                    heatmap_data_copy = None
                    if self.analyzer.live_heatmap_data is not None:
                        # Make a copy but handle potential memory issues
                        heatmap_data_copy = self.analyzer.live_heatmap_data.copy()
                    
                    if heatmap_data_copy is None or heatmap_data_copy.size == 0:
                        QMessageBox.warning(self, "No Data", "Heatmap data is empty. Export may not succeed.")
                except Exception as data_error:
                    print(f"Error copying heatmap data: {data_error}")
                    QMessageBox.critical(
                        self, 
                        "Data Error", 
                        "Failed to copy heatmap data for export. Try reducing heatmap resolution."
                    )
                    
                    # Cleanup and return
                    if hasattr(self, 'progress_dialog') and self.progress_dialog:
                        self.progress_dialog.close()
                    if hasattr(self.control_panel, 'state_manager'):
                        self.control_panel.state_manager.transition('unlock_ui')
                    return
                
                # Prepare parameters for the export thread
                try:
                    export_params = {
                        'max_range': self.analyzer.params.max_range,
                        'target_distance': self.analyzer.params.target_distance,
                        'circle_distance': self.analyzer.params.circle_distance,
                        'circle_radius': self.analyzer.params.circle_radius,
                        'circle_interval': self.analyzer.params.circle_interval,
                        'config_name': config_name,
                        'noise_floor': noise_floor,
                        'smoothing_sigma': smoothing_sigma,
                        'colormap': colormap,
                        'visualization_mode': visualization_mode
                    }
                except Exception as param_error:
                    print(f"Error preparing export parameters: {param_error}")
                    QMessageBox.critical(self, "Parameter Error", f"Failed to prepare export parameters: {param_error}")
                    
                    # Cleanup and return
                    if hasattr(self, 'progress_dialog') and self.progress_dialog:
                        self.progress_dialog.close()
                    if hasattr(self.control_panel, 'state_manager'):
                        self.control_panel.state_manager.transition('unlock_ui')
                    return
                
                # Create worker object and thread
                try:
                    self.export_thread = QThread()
                    self.export_worker = self.ExportWorker(file_path, heatmap_data_copy, export_params)
                    self.export_worker.moveToThread(self.export_thread)
                    
                    # Connect signals
                    self.export_thread.started.connect(self.export_worker.run)
                    self.export_worker.progress.connect(self.update_export_progress)
                    self.export_worker.finished.connect(self.on_export_completed)
                    self.export_worker.finished.connect(self.export_thread.quit)
                    self.export_thread.finished.connect(self.cleanup_export_thread)
                except Exception as thread_error:
                    print(f"Error setting up export thread: {thread_error}")
                    QMessageBox.critical(self, "Thread Error", f"Failed to set up export thread: {thread_error}")
                    
                    # Cleanup and return
                    if hasattr(self, 'progress_dialog') and self.progress_dialog:
                        self.progress_dialog.close()
                    if hasattr(self.control_panel, 'state_manager'):
                        self.control_panel.state_manager.transition('unlock_ui')
                    return
                
                # Set a timeout timer
                try:
                    self.export_timeout_timer = QTimer(self)
                    self.export_timeout_timer.setSingleShot(True)
                    self.export_timeout_timer.timeout.connect(self.check_export_timeout)
                    self.export_timeout_timer.start(20000)  # 20 second timeout - reduced from 30
                except Exception as timer_error:
                    print(f"Error setting up timeout timer: {timer_error}")
                    # Continue without the timer if it fails
                
                # Start the thread
                try:
                    self.export_thread.start()
                    self.status_bar.showMessage("Export thread started...")
                except Exception as start_error:
                    print(f"Error starting export thread: {start_error}")
                    QMessageBox.critical(self, "Thread Error", f"Failed to start export thread: {start_error}")
                    
                    # Cleanup and return
                    if hasattr(self, 'progress_dialog') and self.progress_dialog:
                        self.progress_dialog.close()
                    if hasattr(self.control_panel, 'state_manager'):
                        self.control_panel.state_manager.transition('unlock_ui')
                    if hasattr(self, 'export_worker'):
                        self.export_worker.deleteLater()
                    if hasattr(self, 'export_thread'):
                        self.export_thread.deleteLater()
                    return
            else:
                # Dialog was canceled
                if hasattr(self.control_panel, 'state_manager'):
                    self.control_panel.state_manager.transition('unlock_ui')
        except Exception as e:
            self.status_bar.showMessage(f"Error exporting plot: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export plot: {str(e)}")
            
            # Ensure UI is unlocked
            if hasattr(self.control_panel, 'state_manager'):
                self.control_panel.state_manager.transition('unlock_ui')
    
    # Worker class inside the MainWindow class
    class ExportWorker(QObject):
        finished = pyqtSignal(bool, str)
        progress = pyqtSignal(float)
        
        def __init__(self, file_path, heatmap_data, params):
            super().__init__()
            self.file_path = file_path
            self.heatmap_data = heatmap_data
            self.params = params
            self.cancelled = False
        
        def run(self):
            try:
                from utils.visualization import save_scientific_visualization
                
                # Define a cancellation check function
                def check_cancelled():
                    # Return True if export should be cancelled
                    return self.cancelled
                
                # Call the visualization function with cancellation support
                success = save_scientific_visualization(
                    self.file_path,
                    self.heatmap_data,
                    self.params['max_range'],
                    self.params['target_distance'],
                    self.params['circle_distance'],
                    self.params['circle_radius'],
                    self.params['circle_interval'],
                    self.params['config_name'],
                    self.params['noise_floor'],
                    self.params['smoothing_sigma'],
                    self.params['colormap'],
                    self.params['visualization_mode'],
                    progress_callback=lambda p: self.progress.emit(p),
                    cancellation_check=check_cancelled
                )
                
                # Check if we were cancelled during execution
                if self.cancelled:
                    self.finished.emit(False, "Operation cancelled by user")
                    return
                
                # Check if visualization failed
                if not success:
                    self.finished.emit(False, "Visualization process failed")
                    return
                
                self.finished.emit(True, self.file_path)
            except Exception as e:
                print(f"Error in export thread: {e}")
                import traceback
                traceback.print_exc()
                self.finished.emit(False, str(e))
    
    def cancel_export(self):
        """Handle user cancellation of export process."""
        if hasattr(self, 'export_worker'):
            self.export_worker.cancelled = True
            self.status_bar.showMessage("Cancelling export...")
        self.terminate_export_thread()
    
    @pyqtSlot(float)
    def update_export_progress(self, progress):
        """Update the export progress dialog.
        
        Args:
            progress: Progress value between 0.0 and 1.0
        """
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setValue(int(progress * 100))
            
            # Update the label text to provide more feedback
            if progress < 0.1:
                self.progress_dialog.setLabelText("Initializing export...")
            elif progress < 0.3:
                self.progress_dialog.setLabelText("Preparing data...")
            elif progress < 0.6:
                self.progress_dialog.setLabelText("Generating visualization...")
            elif progress < 0.9:
                self.progress_dialog.setLabelText("Applying finishing touches...")
            else:
                self.progress_dialog.setLabelText("Saving file...")
    
    def check_export_timeout(self):
        """Check if the export operation has timed out."""
        if hasattr(self, 'export_thread') and self.export_thread.isRunning():
            reply = QMessageBox.question(
                self, "Export Taking Too Long",
                "The export operation is taking longer than expected. Wait longer?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            
            if reply == QMessageBox.No:
                self.terminate_export_thread()
            else:
                # User wants to wait longer
                self.export_timeout_timer.start(30000)  # 30 more seconds
    
    def terminate_export_thread(self):
        """Forcefully terminate the export thread."""
        if hasattr(self, 'export_thread') and self.export_thread.isRunning():
            # Try to quit normally first
            self.export_thread.quit()
            
            # If thread doesn't quit within 3 seconds, terminate it
            if not self.export_thread.wait(3000):
                self.export_thread.terminate()
            
            self.cleanup_export_thread()
            
            # Show message and clean up UI
            self.status_bar.showMessage("Export operation canceled")
            
            if hasattr(self.control_panel, 'state_manager'):
                self.control_panel.state_manager.transition('unlock_ui')
            
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.progress_dialog.close()
    
    def cleanup_export_thread(self):
        """Clean up export thread resources."""
        # Stop timeout timer if it's running
        if hasattr(self, 'export_timeout_timer') and self.export_timeout_timer.isActive():
            self.export_timeout_timer.stop()
        
        # Clean up worker and thread
        if hasattr(self, 'export_worker'):
            self.export_worker.deleteLater()
        
        if hasattr(self, 'export_thread'):
            self.export_thread.deleteLater()
    
    def on_export_completed(self, success, result):
        """
        Handle completion of the export thread.
        
        Args:
            success: Whether the export was successful
            result: File path if successful, error message if not
        """
        # Close progress dialog
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
        
        # Stop timeout timer if it's running
        if hasattr(self, 'export_timeout_timer') and self.export_timeout_timer.isActive():
            self.export_timeout_timer.stop()
        
        # Unlock the UI
        if hasattr(self.control_panel, 'state_manager'):
            self.control_panel.state_manager.transition('unlock_ui')
        
        if success:
            file_path = result
            self.status_bar.showMessage(f"Plot exported to: {file_path}")
            
            # Ask if user wants to view the saved file
            reply = QMessageBox.question(
                self, "Export Complete",
                f"Scientific plot saved to:\n{file_path}\n\nWould you like to open it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Open file with default system application
                import subprocess
                import sys
                try:
                    if sys.platform == 'win32':
                        os.startfile(file_path)
                    elif sys.platform == 'darwin':  # macOS
                        subprocess.call(['open', file_path])
                    else:  # Linux
                        subprocess.call(['xdg-open', file_path])
                except Exception as e:
                    QMessageBox.warning(
                        self, 
                        "Open Error", 
                        f"Could not open the exported file: {str(e)}"
                    )
        else:
            error_message = result
            self.status_bar.showMessage(f"Error exporting plot: {error_message}")
            QMessageBox.critical(self, "Export Error", f"Failed to export plot: {error_message}")
    
    @pyqtSlot()
    def generate_report(self):
        """Generate a report of collected radar data automatically without UI interactions."""
        # Increment the session generate counter
        self.generate_counter += 1
        
        if self.analyzer is None:
            self.status_bar.showMessage("No analyzer instance available")
            if hasattr(self.control_panel, 'on_report_completed'):
                self.control_panel.on_report_completed(success=False)
            return
        
        if not self.analyzer.config_results:
            self.status_bar.showMessage("No configuration results available for report")
            QMessageBox.information(self, "No Data", "No configuration results available for report")
            if hasattr(self.control_panel, 'on_report_completed'):
                self.control_panel.on_report_completed(success=False)
            return
        
        # Lock the UI during report generation
        if hasattr(self.control_panel, 'state_manager'):
            self.control_panel.state_manager.transition('lock_ui')
        
        try:
            # Automatically create report directory if it doesn't exist
            default_dir = os.path.expanduser('~/radar_experiment_data/reports')
            os.makedirs(default_dir, exist_ok=True)
            
            # Check if we already have a session report file, and use it if available
            if not hasattr(self, 'session_report_path') or self.session_report_path is None:
                # First time generating report in this session - create a new file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.session_report_path = os.path.join(default_dir, f"radar_report_{timestamp}.csv")
            
            # Use the session report path
            file_path = self.session_report_path
            
            # Before generating the report, initialize the tracking set by reading existing entries
            self._initialize_report_tracking(file_path)
            
            # Generate the report at the specified location without asking for location
            self.status_bar.showMessage("Generating CSV report, please wait...")
            QApplication.processEvents()  # Process UI events to update status
            
            # Generate the report without user interaction
            success = self.generate_custom_report(file_path)
            
            # Show success message
            if success:
                self.status_bar.showMessage(f"CSV report saved to {file_path}", 5000)
                if hasattr(self.control_panel, 'on_report_completed'):
                    self.control_panel.on_report_completed(success=True)
            else:
                self.status_bar.showMessage("Failed to generate CSV report", 5000)
                if hasattr(self.control_panel, 'on_report_completed'):
                    self.control_panel.on_report_completed(success=False)
        
            # Unlock UI after completion
            if hasattr(self.control_panel, 'state_manager'):
                self.control_panel.state_manager.transition('unlock_ui')
                
        except Exception as e:
            self.status_bar.showMessage(f"Error generating CSV report: {str(e)}", 5000)
            QMessageBox.warning(self, "Report Error", f"Failed to generate CSV report: {str(e)}")
            if hasattr(self.control_panel, 'on_report_completed'):
                self.control_panel.on_report_completed(success=False)
            if hasattr(self.control_panel, 'state_manager'):
                self.control_panel.state_manager.transition('unlock_ui')
                
    def _initialize_report_tracking(self, file_path):
        """
        Initialize report tracking set by reading existing entries from a CSV file.
        This prevents duplicating rows when appending to the same file.
        
        Args:
            file_path: Path to the CSV file to read from.
        """
        # Create a fresh tracking set
        self.report_data_tracked = set()
        
        # Check if the file exists
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return  # Nothing to read, use empty tracking set
            
        try:
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                # Skip header
                headers = next(reader, None)
                if not headers:
                    return  # Empty file or no headers
                    
                # Read each row and add tracking keys
                for row in reader:
                    if len(row) >= 4:  # Need at least config, distance, ROI type, and angle
                        config = row[0]
                        distance_str = row[1]
                        roi_label = row[2]
                        # Create tracking key format that matches what's used in _generate_report_rows
                        tracking_key = f"{config}_{distance_str}_{roi_label}"
                        self.report_data_tracked.add(tracking_key)
                        
                        # Also add the "Outside ROI" context key if applicable
                        if "ROI" in roi_label and "Outside" not in roi_label:
                            context_label = "Primary" if "Primary" in roi_label else roi_label.replace(" ROI", "")
                            outside_key = f"{config}_{distance_str}_Outside ROI_Context_{context_label}"
                            self.report_data_tracked.add(outside_key)
                            
        except Exception as e:
            print(f"Error initializing report tracking from file: {e}")
            # On error, use a new empty tracking set (may cause some duplicates but won't crash)
    
    def _enhance_distance_band_metrics(self):
        """
        Enhance the distance band metrics with advanced scientific measurements.
        
        This method directly calculates point density, intensity SNR, and temporal consistency
        metrics for each distance band without relying on frame-by-frame data.
        """
        if not self.analyzer or not hasattr(self.analyzer, 'config_results'):
            return
            
        try:
            # Loop through all configurations and distances
            for config, distances in self.analyzer.config_results.items():
                for distance, results in distances.items():
                    # Only process if we have distance_bands data
                    if 'distance_bands' not in results:
                        continue
                        
                    updated_bands = {}
                    for band, band_data in results['distance_bands'].items():
                        # Support both dictionary and scalar formats
                        if not isinstance(band_data, dict):
                            # If it's just a count, convert to dict
                            band_data = {'count': float(band_data), 'avg_intensity': 0.0}
                        
                        # Create a copy of the original data
                        updated_band = band_data.copy()
                        
                        # Extract the distance range
                        band_range = band.split('-')
                        min_dist = float(band_range[0])
                        max_dist = float(band_range[1].replace('m', ''))
                        
                        # 1. Calculate point density (points per cubic meter)
                        # Volume of spherical segment between min_dist and max_dist (front hemisphere)
                        if min_dist == 0:
                            # Special case for first band to avoid division by zero
                            volume = (2*np.pi/3) * (max_dist**3) / 2
                        else:
                            volume = (2*np.pi/3) * (max_dist**3 - min_dist**3) / 2
                        
                        count = band_data.get('count', 0)
                        point_density = count / volume if volume > 0 else 0.0
                        updated_band['point_density'] = point_density
                        
                        # 2. Calculate intensity SNR
                        # Simple approach: Use avg_intensity / sqrt(10) as a reasonable SNR estimate
                        # We use a fixed variance estimate since we don't have actual intensity values
                        avg_intensity = band_data.get('avg_intensity', 0.0)
                        estimated_variance = 10.0  # Typical variance in radar intensity data
                        intensity_snr = avg_intensity / np.sqrt(estimated_variance) if estimated_variance > 0 else 0.0
                        updated_band['intensity_snr'] = intensity_snr
                        
                        # 3. Calculate temporal consistency
                        # Since we don't have actual frame-by-frame data, estimate based on distance
                        # Closer distances typically have more consistent detections
                        # This is a heuristic: consistency decreases with distance
                        if count > 0:
                            # Estimate consistency as a function of distance and count
                            # Closer distances with more points -> higher consistency
                            avg_dist = (min_dist + max_dist) / 2
                            max_reliable_dist = 20.0  # Radar is less reliable beyond this
                            
                            # Base consistency on distance (decreases with distance)
                            base_consistency = max(0.0, 1.0 - (avg_dist / max_reliable_dist))
                            
                            # Also factor in point count (more points = more stable)
                            count_factor = min(1.0, count / 50.0)  # Normalize up to 50 points
                            
                            # Combine the factors - higher count and lower distance = better consistency
                            temporal_consistency = base_consistency * (0.5 + 0.5 * count_factor)
                        else:
                            temporal_consistency = 0.0
                            
                        updated_band['temporal_consistency'] = temporal_consistency
                        
                        # Add some variance for intensity for SNR calculations
                        updated_band['intensity_variance'] = estimated_variance
                        
                        # Store the updated band data
                        updated_bands[band] = updated_band
                    
                    # Replace the distance_bands with our enhanced version
                    if updated_bands:
                        results['distance_bands'] = updated_bands
                
            # Log completion
            self.status_bar.showMessage("Enhanced distance band metrics calculated")
                
        except Exception as e:
            print(f"Error enhancing distance band metrics: {str(e)}")
    
    def _generate_secondary_roi_table_headers(self, circles: list) -> str:
        """
        Generate HTML table headers for secondary ROI circles
        
        Args:
            circles: List of circle objects from radar_params
            
        Returns:
            HTML string for secondary ROI circle table headers
        """
        headers = ""
        for i in range(1, 3):  # Circles 1 and 2 (secondary circles)
            if i < len(circles) and circles[i].enabled:
                headers += f"<th colspan=\"3\">{circles[i].label} ROI</th>"
        
        return headers
    
    def _generate_secondary_roi_subheaders(self, circles: list) -> str:
        """
        Generate HTML table subheaders for secondary ROI circles
        
        Args:
            circles: List of circle objects from radar_params
            
        Returns:
            HTML string for secondary ROI circle table subheaders
        """
        subheaders = ""
        for i in range(1, 3):  # Circles 1 and 2 (secondary circles)
            if i < len(circles) and circles[i].enabled:
                subheaders += "<th>Points</th><th>Density</th><th>SNR</th>"
        
        return subheaders
    
    def _generate_secondary_roi_table_cells(self, metrics: dict, circles: list) -> str:
        """
        Generate HTML table cells for secondary ROI circles
        
        Args:
            metrics: Dictionary of metrics from the analyzer
            circles: List of circle objects from radar_params
            
        Returns:
            HTML string for secondary ROI circle table cells
        """
        cells = ""
        for i in range(1, 3):  # Circles 1 and 2 (secondary circles)
            if i < len(circles) and circles[i].enabled:
                prefix = f'roi{i+1}'
                # Only add cells if we have metrics for this circle
                if f'{prefix}_combined_point_count' in metrics:
                    points = metrics.get(f'{prefix}_combined_point_count', 0)
                    density = metrics.get(f'{prefix}_spatial_density', 0)
                    snr = metrics.get(f'{prefix}_snr_db', 0)
                    cells += f"<td>{points}</td><td>{density:.2f}</td><td>{snr:.2f}</td>"
                else:
                    cells += "<td>-</td><td>-</td><td>-</td>"
        
        return cells
    
    def _generate_secondary_roi_metrics_html(self, metrics: dict, circle_index: int, circles: list) -> str:
        """
        Generate HTML section for secondary ROI circle metrics
        
        Args:
            metrics: Dictionary of metrics from the analyzer
            circle_index: Index of the ROI circle (1 or 2 for secondary circles)
            circles: List of circle objects from radar_params
            
        Returns:
            HTML string for the secondary ROI circle metrics section, or empty string if circle is disabled
        """
        # Check if this circle is enabled
        if circle_index >= len(circles) or not circles[circle_index].enabled:
            return ""
            
        # Get the prefix for the metrics keys
        prefix = f'roi{circle_index+1}'
        circle = circles[circle_index]
        
        # Return empty string if no metrics are available for this circle
        if f'{prefix}_combined_point_count' not in metrics:
            return ""
        
        return f"""
        <h4 class="card-title">{circle.label} ROI Circle Metrics</h4>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-label">Combined Points</div>
                <div class="metric-value">{metrics.get(f'{prefix}_combined_point_count', 0)}</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Avg Points/Frame</div>
                <div class="metric-value">{metrics.get(f'{prefix}_avg_single_frame_count', 0):.1f}</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Spatial Density</div>
                <div class="metric-value">{metrics.get(f'{prefix}_spatial_density', 0):.2f}<span class="metric-units">pts/m²</span></div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Density Gain</div>
                <div class="metric-value">{metrics.get(f'{prefix}_density_gain_db', 0):.2f}<span class="metric-units">dB</span></div>
            </div>
            
            <div class="metric">
                <div class="metric-label">SNR</div>
                <div class="metric-value">{metrics.get(f'{prefix}_snr_db', 0):.2f}<span class="metric-units">dB</span></div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Intensity Range</div>
                <div class="metric-value">{metrics.get(f'{prefix}_combined_min_intensity', 0):.1f} - {metrics.get(f'{prefix}_combined_max_intensity', 0):.1f}</div>
            </div>
        </div>
        """

    def _calculate_roi_distance_bands(self, data, roi_prefix):
        """
        Calculate distance band metrics for a specific ROI.
        
        Args:
            data: Data dictionary from processed_configs
            roi_prefix: Prefix for the ROI type ('roi', 'roi2', 'roi3', 'outside_roi')
            
        Returns:
            Tuple of (band_0_10, band_10_20, band_20_30) point counts
        """
        band_0_10 = 0
        band_10_20 = 0
        band_20_30 = 0
        
        # Check if we have the direct metrics available (preferred method)
        if 'multi_frame_metrics' in data:
            metrics = data['multi_frame_metrics']
            
            # For outside ROI, use the total band counts if available (which include both inside and outside points)
            if roi_prefix == 'outside_roi':
                # Try using the total counts first (preferred)
                if 'total_band_0_10m_count' in metrics:
                    band_0_10 = metrics.get('total_band_0_10m_count', 0)
                    band_10_20 = metrics.get('total_band_10_20m_count', 0)
                    band_20_30 = metrics.get('total_band_20_30m_count', 0)
                    
                    # Return early if we have all the data
                    if band_0_10 != 0 or band_10_20 != 0 or band_20_30 != 0:
                        return band_0_10, band_10_20, band_20_30
                
                # Fallback to just outside counts if totals aren't available
                band_0_10 = metrics.get('outside_roi_band_0_10m_count', 0)
                band_10_20 = metrics.get('outside_roi_band_10_20m_count', 0)
                band_20_30 = metrics.get('outside_roi_band_20_30m_count', 0)
                
                # Return early if we have all the data
                if band_0_10 != 0 or band_10_20 != 0 or band_20_30 != 0:
                    return band_0_10, band_10_20, band_20_30
        
        # If we don't have direct metrics or they're not for outside ROI, try the distance_bands data
        if 'distance_bands' not in data:
            return band_0_10, band_10_20, band_20_30
        
        # Get all band keys from the distance_bands (fallback method)
        try:
            # Assume band keys are in the format "min-maxm" (e.g., "0-10m") or similar
            for band_key, band_data in data['distance_bands'].items():
                # Clean up the band key to handle different formats
                clean_key = band_key.replace('m', '').strip()
                if '-' not in clean_key:
                    continue  # Skip if not in expected format
                
                # Parse min and max distances
                parts = clean_key.split('-')
                if len(parts) != 2:
                    continue
                
                try:
                    min_dist = float(parts[0])
                    max_dist = float(parts[1])
                    
                    # Get the count value from the band data
                    if isinstance(band_data, dict):
                        count = band_data.get('count', 0)
                    else:
                        # If it's not a dict, assume it's a numeric value representing count
                        count = float(band_data)
                    
                    # Categorize the band based on distance range
                    if min_dist < 10 and max_dist <= 10:
                        # This is a 0-10m band (fully or partially)
                        band_0_10 += count
                    elif min_dist >= 10 and min_dist < 20 and max_dist <= 20:
                        # This is a 10-20m band (fully or partially)
                        band_10_20 += count
                    elif min_dist >= 20 and min_dist < 30 and max_dist <= 30:
                        # This is a 20-30m band (fully or partially)
                        band_20_30 += count
                    elif min_dist < 10 and max_dist > 10:
                        # This spans the 0-10m and beyond bands, distribute proportionally
                        # Simplified approach: assign based on the midpoint
                        midpoint = (min_dist + max_dist) / 2
                        if midpoint < 10:
                            band_0_10 += count
                        elif midpoint < 20:
                            band_10_20 += count
                        else:
                            band_20_30 += count
                    elif min_dist < 20 and max_dist > 20:
                        # This spans the 10-20m and beyond bands, distribute proportionally
                        # Simplified approach: assign based on the midpoint
                        midpoint = (min_dist + max_dist) / 2
                        if midpoint < 20:
                            band_10_20 += count
                        else:
                            band_20_30 += count
                except (ValueError, TypeError):
                    # Skip if we can't parse the distances
                    continue
        except Exception as e:
            # If anything goes wrong, return zeros
            return 0, 0, 0
            
        return band_0_10, band_10_20, band_20_30

    def generate_custom_report(self, file_path: str) -> bool:
        """
        Generate or update a CSV report containing radar metrics for the current session.
        Appends new data (not previously seen in this session) to the session file.

        Args:
            file_path: Path to save/update the CSV report for the session.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.analyzer or not hasattr(self.analyzer, 'config_results'):
            self.statusBar().showMessage("Analyzer or results not available.", 5000)
            return False

        try:
            # Import necessary modules
            import csv
            import os
            import math 
            
            # --- 1. File path validation and directory creation ---
            # Ensure file_path has .csv extension
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'

            # Create directory if it doesn't exist
            dir_path = os.path.dirname(file_path)
            if dir_path:  # Only if there's a directory part
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except PermissionError:
                    self.statusBar().showMessage(f"Permission denied: Cannot create directory {dir_path}", 5000)
                    return False
                except Exception as e:
                    self.statusBar().showMessage(f"Error creating directory for CSV file: {str(e)}", 5000)
                    return False

            # --- 2. Initialize tracking data if needed ---
            if not hasattr(self, 'report_data_tracked'):
                self.report_data_tracked = set()

            # --- 3. File existence check ---
            file_exists = os.path.exists(file_path)
            needs_header = not file_exists or os.path.getsize(file_path) == 0
            
            # --- 4. CSV Header definition ---
            header_row = [
                'Config', 'Target_Distance', 'ROI_Type', 'Angle_Degrees',
                'Total_Points_All_Frames', 'Avg_Points_Per_Frame',
                'SNR_dB', 'Min_Intensity', 'Max_Intensity', 'Avg_Intensity',
                'Total_0_10m_Points', 'Total_10_20m_Points', 'Total_20_30m_Points'
            ]

            # --- 5. Data preprocessing ---
            processed_configs = self._preprocess_report_data()
            
            # --- 6. Generate rows to append ---
            rows_to_append = []
            if needs_header:
                rows_to_append.append(header_row)
            
            new_rows_added_count = self._generate_report_rows(processed_configs, rows_to_append)
            
            # --- 7. Write to file only if we have data to write ---
            if needs_header or new_rows_added_count > 0:
                try:
                    # Write in a single operation
                    if self._write_csv_rows(file_path, rows_to_append):
                        # Success message
                        if needs_header:
                            status_message = f"Report created: {new_rows_added_count} rows added to {os.path.basename(file_path)}"
                        else:
                            status_message = f"Report updated: {new_rows_added_count} new rows added to {os.path.basename(file_path)}"
                        self.statusBar().showMessage(status_message, 5000)
                    else:
                        return False  # _write_csv_rows already showed error message
                except Exception as write_error:
                    self.statusBar().showMessage(f"Error writing to CSV file: {str(write_error)}", 5000)
                    return False
            else:
                self.statusBar().showMessage("No new data to add to the report.", 5000)

            return True

        except Exception as e:
            self.statusBar().showMessage(f"Error generating report: {str(e)}", 5000)
            import traceback
            traceback.print_exc()  # Print detailed error for debugging
            return False
    
    def _preprocess_report_data(self):
        """
        Preprocess the analyzer data for the report.
        
        Returns:
            dict: Processed configuration data.
        """
        processed_configs = {}
        for config, distances in self.analyzer.config_results.items():
            processed_configs[config] = {}
            for distance, results in distances.items():
                if 'multi_frame_metrics' in results:
                    processed_configs[config][distance] = {
                        'multi_frame_metrics': results.get('multi_frame_metrics', {}),
                        'density': results.get('point_density', 0.0),
                        'avg_intensity': results.get('avg_intensity', 0.0),
                        'distance_bands': results.get('distance_bands', {})
                    }
        return processed_configs
    
    def _create_report_row(self, config, distance_str, roi_label, angle_str, metrics, 
                           metric_prefix, is_outside_roi=False, outside_bands=None):
        """Helper function to create a single formatted row for the CSV report."""
        if is_outside_roi:
            point_count = metrics.get(f'{metric_prefix}_combined_point_count', 0)
            band_0_10, band_10_20, band_20_30 = outside_bands if outside_bands else (0, 0, 0)
            return [
                config, distance_str, roi_label, angle_str, int(point_count),
                self._safe_float_format(metrics.get(f'{metric_prefix}_avg_single_frame_count'), 1), 
                self._safe_float_format(metrics.get(f'{metric_prefix}_snr_db'), 2),
                self._safe_float_format(metrics.get(f'{metric_prefix}_min_intensity'), 1), 
                self._safe_float_format(metrics.get(f'{metric_prefix}_max_intensity'), 1),
                self._safe_float_format(metrics.get(f'{metric_prefix}_avg_intensity'), 1), 
                int(band_0_10), int(band_10_20), int(band_20_30)
            ]
        else:
            point_count = metrics.get(f'{metric_prefix}_combined_point_count', 0)
            return [
                config, distance_str, roi_label, angle_str,
                int(point_count),
                self._safe_float_format(metrics.get(f'{metric_prefix}_avg_single_frame_count'), 1), 
                self._safe_float_format(metrics.get(f'{metric_prefix}_snr_db'), 2),
                self._safe_float_format(metrics.get(f'{metric_prefix}_combined_min_intensity'), 1), 
                self._safe_float_format(metrics.get(f'{metric_prefix}_combined_max_intensity'), 1),
                self._safe_float_format(metrics.get(f'{metric_prefix}_combined_avg_intensity'), 1), 
                "", "", "" # No bands for specific ROIs
            ]
            
    def _generate_report_rows(self, processed_configs, rows_to_append):
        """
        Generate rows for the CSV report based on processed data.
        Iterates through all ROI circles, adding specific ROI rows and 
        corresponding 'Outside ROI' context rows.
        
        Args:
            processed_configs (dict): Preprocessed configuration data.
            rows_to_append (list): List to append rows to.
            
        Returns:
            int: Count of new rows added.
        """
        new_rows_added_count = 0
        
        for config, distances in processed_configs.items():
            for distance, data in distances.items():
                multi_frame_metrics = data['multi_frame_metrics']
                distance_str = f"{distance}m"

                # --- Calculate Outside ROI metrics once per config/distance ---
                outside_roi_points = multi_frame_metrics.get('outside_roi_combined_point_count', 0)
                outside_band_0_10, outside_band_10_20, outside_band_20_30 = \
                    self._calculate_roi_distance_bands(data, 'outside_roi')
                outside_bands_tuple = (outside_band_0_10, outside_band_10_20, outside_band_20_30)

                # --- Loop through ALL configured circles (Primary + Secondary) ---
                for i, circle in enumerate(self.analyzer.params.circles):
                    if not circle.enabled:
                        continue # Skip disabled circles
                        
                    # Determine ROI specifics
                    is_primary = (i == 0)
                    roi_prefix = 'roi' if is_primary else f'roi{i+1}'
                    roi_label = "Primary ROI" if is_primary else getattr(circle, 'label', f"Secondary {i}") + " ROI"
                    angle_str = "0" if is_primary else self._safe_float_format(getattr(circle, 'angle', 0), 1)
                    
                    # --- Add Specific ROI Row (if not tracked) ---
                    specific_roi_key = f"{config}_{distance_str}_{roi_label}"
                    if specific_roi_key not in self.report_data_tracked:
                        self.report_data_tracked.add(specific_roi_key)
                        
                        # Create and append the specific ROI row
                        roi_row = self._create_report_row(
                            config, distance_str, roi_label, angle_str, 
                            multi_frame_metrics, roi_prefix, is_outside_roi=False
                        )
                        rows_to_append.append(roi_row)
                        new_rows_added_count += 1

                        # --- Add Corresponding Outside ROI Row (if not tracked for this context) ---
                        outside_context_label = "Primary" if is_primary else getattr(circle, 'label', f"Secondary_{i}")
                        outside_key_context = f"{config}_{distance_str}_Outside ROI_Context_{outside_context_label}"
                        if outside_key_context not in self.report_data_tracked:
                            self.report_data_tracked.add(outside_key_context)
                            
                            # Create and append the Outside ROI row for this context
                            outside_row = self._create_report_row(
                                config, distance_str, "Outside ROI", "N/A",
                                multi_frame_metrics, 'outside_roi', 
                                is_outside_roi=True, outside_bands=outside_bands_tuple
                            )
                            rows_to_append.append(outside_row)
                            new_rows_added_count += 1
                            
        return new_rows_added_count
    
    def _write_csv_rows(self, file_path, rows):
        """
        Write rows to a CSV file in append mode.
        
        Args:
            file_path: Path to the CSV file.
            rows: Rows to write.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with open(file_path, 'a', newline='') as f:  # Use 'a' (append) mode to add to existing file
                writer = csv.writer(f, dialect='excel', lineterminator='\n')
                writer.writerows(rows)
            return True
        except PermissionError:
            self.statusBar().showMessage(f"Permission denied: Cannot write to {file_path}", 5000)
            return False
        except IOError as e:
            self.statusBar().showMessage(f"I/O error writing CSV file: {str(e)}", 5000)
            return False
        except Exception as e:
            self.statusBar().showMessage(f"Error writing to CSV file: {str(e)}", 5000)
            return False
    
    def _safe_float_format(self, value, decimal_places=1):
        """
        Safely format a value as a float with specified decimal places.
        
        Args:
            value: Value to format.
            decimal_places (int): Number of decimal places.
            
        Returns:
            str: Formatted float string.
        """
        try:
            if value is None:
                return f"0.{'0' * decimal_places}"
            f_value = float(value)
            if not math.isfinite(f_value):
                return f"0.{'0' * decimal_places}"  # Represent non-finite as zero
            return f"{f_value:.{decimal_places}f}"
        except (TypeError, ValueError):
            return f"0.{'0' * decimal_places}"  # Default value
    
    def _validate_csv_file(self, file_path: str) -> None:
        """
        Validate a CSV file to ensure it's properly formatted.
        
        Args:
            file_path: Path to the CSV file to validate.
        """
        try:
            import csv
            import os
            
            # Check if the file exists and has content
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                self.statusBar().showMessage(f"Warning: CSV file {file_path} is empty or doesn't exist", 5000)
                return
                
            # Read the CSV file to verify its structure
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                # Check if we have at least a header row
                if len(rows) < 1:
                    self.statusBar().showMessage(f"Warning: CSV file {file_path} has no rows", 5000)
                    return
                    
                # Check if the header has the expected number of columns
                expected_columns = 13  # Based on the CSV structure
                if len(rows[0]) != expected_columns:
                    self.statusBar().showMessage(
                        f"Warning: CSV header has {len(rows[0])} columns, expected {expected_columns}", 5000
                    )
                    
                # Check if data rows have the same number of columns as the header
                for i, row in enumerate(rows[1:], 1):
                    if len(row) != expected_columns:
                        self.statusBar().showMessage(
                            f"Warning: Row {i} has {len(row)} columns, expected {expected_columns}", 5000
                        )
                        break
                
                # Check for empty or problematic fields in numerical columns
                numerical_columns = [4, 5, 6, 7, 8, 9]  # Indices of columns that should contain numbers
                for i, row in enumerate(rows[1:], 1):
                    if len(row) != expected_columns:
                        continue  # Skip rows with incorrect column count
                    
                    for col_idx in numerical_columns:
                        if col_idx < len(row):
                            value = row[col_idx]
                            # Check if it's a valid number
                            try:
                                float(value)
                            except ValueError:
                                self.statusBar().showMessage(
                                    f"Warning: Row {i}, column {col_idx+1} has non-numeric value: '{value}'", 5000
                                )
                                break
                
                # If we get here, the CSV is likely well-formed
                self.statusBar().showMessage(f"CSV validation successful: {len(rows)} rows, {expected_columns} columns", 3000)
                
        except Exception as e:
            self.statusBar().showMessage(f"Error validating CSV file: {str(e)}", 5000)
    
    def show_about_dialog(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About Radar Point Cloud Analyzer",
            "<h1>Radar Point Cloud Analyzer</h1>"
            "<p>Version 1.0</p>"
            "<p>A comprehensive tool for analyzing radar point clouds from "
            "an AWR1843 mmWave radar using ROS 2.</p>"
            "<p>Features include real-time visualization, data collection, "
            "and scientific analysis.</p>"
        )
    
    @pyqtSlot(str)
    def play_rosbag(self, bag_path):
        """Start playback of a ROS2 bag file.
        
        Args:
            bag_path: Path to the bag file to play.
        """
        if self.analyzer is None:
            QMessageBox.warning(self, "Not Available", "ROS2 bag playback not available in visualization-only mode.")
            return
        
        try:
            # Ensure the analyzer's signals object is properly connected to our slot
            if hasattr(self.analyzer, 'signals') and hasattr(self.analyzer.signals, 'update_playback_position_signal'):
                # Disconnect any existing connections to avoid duplicates
                try:
                    self.analyzer.signals.update_playback_position_signal.disconnect(self.update_playback_position)
                except:
                    pass  # No existing connection
                # Connect the signal
                self.analyzer.signals.update_playback_position_signal.connect(self.update_playback_position)
            
            # Call analyzer method to play the bag file with explicit loop=False
            self.analyzer.play_rosbag(bag_path, loop=False)
            self.status_bar.showMessage(f"Playing ROS2 bag: {bag_path}")
        except Exception as e:
            QMessageBox.critical(self, "Playback Error", f"Failed to play ROS2 bag: {str(e)}")
    
    @pyqtSlot(str, list, int)
    def record_rosbag(self, save_path, topics, duration_minutes=0):
        """Start recording a ROS2 bag file.
        
        Args:
            save_path: Directory to save the bag file.
            topics: List of topics to record.
            duration_minutes: Duration in minutes to record (0 = unlimited).
        """
        if self.analyzer is None:
            QMessageBox.warning(self, "Not Available", "ROS2 bag recording not available in visualization-only mode.")
            return
        
        try:
            # Call analyzer method to record the bag file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"radar_recording_{timestamp}"
            full_path = os.path.join(save_path, filename)
            
            # Pass duration to the analyzer's record_rosbag method
            self.analyzer.record_rosbag(full_path, topics, duration_minutes)
            
            duration_text = "" if duration_minutes == 0 else f" for {duration_minutes} minutes"
            self.status_bar.showMessage(f"Recording ROS2 bag to: {full_path}{duration_text}")
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to start recording: {str(e)}")
    
    @pyqtSlot()
    def stop_rosbag(self):
        """Stop ROS2 bag playback or recording."""
        if self.analyzer is not None:
            self.analyzer.stop_rosbag()
            self.status_bar.showMessage("Stopped ROS2 bag playback/recording")
    
    @pyqtSlot(float)
    def seek_rosbag(self, position):
        """Seek to a specific position in the currently playing ROS2 bag.
        
        Args:
            position: Normalized position in the bag (0.0-1.0)
        """
        if self.analyzer is None:
            return
        
        try:
            # Call analyzer method to seek in the bag file
            if hasattr(self.analyzer, 'seek_rosbag'):
                self.analyzer.seek_rosbag(position)
                self.status_bar.showMessage(f"Seeking to {position:.1%} of ROS2 bag")
            else:
                # If the method doesn't exist yet, just show a temporary message
                self.status_bar.showMessage(f"Seeking to {position:.1%} (not implemented yet)")
        except Exception as e:
            self.status_bar.showMessage(f"Error seeking in bag: {str(e)}")
    
    @pyqtSlot(float)
    def update_playback_position(self, position):
        """Slot to receive playback position updates from the analyzer and forward to control panel."""
        # This slot might now be redundant if the connection is made directly in connect_signals
        # Keeping it for potential future use or debugging
        # self.control_panel.update_playback_position(position)
        pass

    @pyqtSlot(str)
    def visualize_pointcloud(self, topic):
        """Visualize point cloud data from a specific topic.
        
        Args:
            topic: ROS2 topic to visualize point cloud from.
        """
        if self.analyzer is None:
            QMessageBox.warning(self, "Not Available", "Point cloud visualization not available in visualization-only mode.")
            return
        
        try:
            # Call analyzer method to visualize point cloud
            self.analyzer.visualize_pointcloud(topic)
            self.status_bar.showMessage(f"Visualizing point cloud from: {topic}")
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Failed to visualize point cloud: {str(e)}")
    
    @pyqtSlot()
    def handle_data_reset(self):
        """
        Handle data reset signal from the analyzer.
        
        This method updates the UI to reflect that all data has been reset
        after a bag playback ends.
        """
        try:
            # Reset visualization components
            if self.scatter_view is not None:
                self.scatter_view.clear_points()
            
            # if self.heatmap_view is not None: # Removed heatmap view check
            #     self.heatmap_view.reset_heatmap()
            
            # Reset control panel UI state
            if self.control_panel is not None:
                # Reset timeline slider
                self.control_panel.timeline_slider.setValue(0)
                # Reset any UI state related to playback
                self.control_panel.set_status("Data reset after bag playback ended")
            
            self.status_bar.showMessage("Point cloud data has been reset")
        except Exception as e:
            # Just log silently to avoid disturbing the user
            print(f"Error handling data reset: {e}")
    
    def closeEvent(self, event):
        """
        Handle window close event.
        
        Args:
            event: Close event.
        """
        # Stop any ongoing data collection
        if self.analyzer is not None and self.analyzer.collecting_data:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Data collection is in progress. Stop and exit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.analyzer.stop_data_collection()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
    
    @pyqtSlot(float, float, int)
    def configure_heatmap_update_params(self, min_time_interval, max_time_interval, threshold_percent):
        """
        Configure heatmap update parameters for performance optimization.
        
        Args:
            min_time_interval: Minimum time between updates (seconds)
            max_time_interval: Maximum time between updates (seconds)
            threshold_percent: Data change threshold percentage to trigger update
        """
        # if hasattr(self.heatmap_view, 'optimizer'): # Removed optimizer configuration
        #     self.heatmap_view.optimizer.configure(
        #         min_time_interval=min_time_interval,
        #         max_time_interval=max_time_interval,
        #         change_threshold_percent=threshold_percent
        #     )
        #     self.status_bar.showMessage(
        #         f"Visualization performance set to: {min_time_interval}s min, "
        #         f"{max_time_interval}s max, {threshold_percent}% threshold", 
        #         3000
        #     )
        pass # Heatmap optimizer removed
    
    def optimize_visualization_pipeline(self):
        """
        Automatically optimize visualization pipeline based on system capabilities and dataset sizes.
        
        This function analyzes the current dataset size and system performance
        to configure the optimal visualization settings.
        """
        try:
            # Get dataset size indicators
            data_size = 0
            heatmap_size = 0
            
            if hasattr(self, 'analyzer') and hasattr(self.analyzer, 'live_heatmap_data'):
                heatmap_data = self.analyzer.live_heatmap_data
                if heatmap_data is not None:
                    heatmap_size = heatmap_data.size
            
            if hasattr(self, 'analyzer') and hasattr(self.analyzer, 'point_cloud_data'):
                point_cloud = self.analyzer.point_cloud_data
                if point_cloud is not None and hasattr(point_cloud, 'x'):
                    data_size = len(point_cloud.x)
            
            # Configure based on dataset size
            if heatmap_size > 500000 or data_size > 10000:
                # Very large dataset - use conservative settings
                min_time = 0.25  # 4 FPS for heatmap
                max_time = 1.0   # 1 FPS for contours
                threshold = 10   # 10% change threshold
                mode = "heatmap"  # Use simpler visualization mode
                
                # Set additional scatter view optimizations
                if hasattr(self, 'scatter_view'):
                    self.scatter_view.configure_optimizer(
                        update_interval=0.2,  # 5 FPS
                        max_points=3000,      # Aggressive downsampling
                        adaptive_sampling=True
                    )
                    
            elif heatmap_size > 200000 or data_size > 5000:
                # Medium-large dataset
                min_time = 0.15  # ~7 FPS
                max_time = 0.6   # ~1.7 FPS for contours
                threshold = 5    # 5% change threshold
                mode = "combined"  # Can use combined mode with medium datasets
                
                # Medium settings for scatter view
                if hasattr(self, 'scatter_view'):
                    self.scatter_view.configure_optimizer(
                        update_interval=0.15,  # ~7 FPS
                        max_points=5000,       # Medium downsampling
                        adaptive_sampling=True
                    )
                
            else:
                # Small dataset - can use more detailed visualization
                min_time = 0.1   # 10 FPS
                max_time = 0.3   # 3.3 FPS for contours
                threshold = 2    # 2% change threshold (more frequent updates)
                mode = "combined"  # Full combined mode for small datasets
                
                # Higher fidelity for scatter view with small datasets
                if hasattr(self, 'scatter_view'):
                    self.scatter_view.configure_optimizer(
                        update_interval=0.1,   # 10 FPS
                        max_points=10000,      # Show more points
                        adaptive_sampling=True
                    )
            
            # Apply the configuration
            if hasattr(self, 'heatmap_view') and hasattr(self.heatmap_view, 'optimizer'):
                self.heatmap_view.optimizer.configure(
                    min_time_interval=min_time,
                    max_time_interval=max_time,
                    change_threshold_percent=threshold
                )
                
                # Update visualization mode if needed
                self.set_visualization_mode(mode)
                
                # Update status message
                self.status_bar.showMessage(
                    f"Auto-optimized for {data_size} points and {heatmap_size} heatmap cells: "
                    f"{min_time:.2f}s min, {max_time:.2f}s max, {threshold}% threshold", 
                    5000
                )
                
                # Return configuration for testing
                return {
                    'min_time': min_time,
                    'max_time': max_time,
                    'threshold': threshold,
                    'mode': mode
                }
                
        except Exception as e:
            # Log error and fall back to medium settings
            print(f"Error optimizing visualization pipeline: {e}")
            if hasattr(self, 'heatmap_view') and hasattr(self.heatmap_view, 'optimizer'):
                self.heatmap_view.optimizer.configure(
                    min_time_interval=0.2,
                    max_time_interval=0.7,
                    change_threshold_percent=5
                )
            if hasattr(self, 'scatter_view') and hasattr(self.scatter_view, 'optimizer'):
                self.scatter_view.configure_optimizer(
                    update_interval=0.15,
                    max_points=5000,
                    adaptive_sampling=True
                )
            self.status_bar.showMessage("Error optimizing visualization. Using default settings.", 3000)
            return None
    
    def open_data_file(self):
        """Open a saved radar data file."""
        # Get default directory from last used directory if available
        default_dir = os.path.expanduser("~")
        if hasattr(self.control_panel, 'last_used_directory'):
            last_dir = self.control_panel.last_used_directory()
            if last_dir and os.path.exists(last_dir):
                default_dir = last_dir
        
        # Create a non-blocking file dialog
        dialog = QFileDialog(self, "Open Radar Data", default_dir)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Radar Data Files (*.dat);;CSV Files (*.csv);;All Files (*)")
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # Use Qt's dialog instead of native for better control
        dialog.setWindowModality(Qt.WindowModal)  # Make dialog modal to main window only
        
        # Process events to keep UI responsive
        QApplication.processEvents()
        
        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files and selected_files[0]:
                file_path = selected_files[0]
                
                # Save the last used directory if available
                if hasattr(self.control_panel, 'save_last_used_directory'):
                    try:
                        self.control_panel.save_last_used_directory(os.path.dirname(file_path))
                    except Exception as e:
                        print(f"Error saving last used directory: {e}")
                
                # Load the data file
                if hasattr(self, 'analyzer') and self.analyzer is not None:
                    try:
                        # Use a status message to indicate loading
                        self.status_bar.showMessage(f"Loading data from {os.path.basename(file_path)}...")
                        QApplication.processEvents()  # Force UI update
                        
                        # Use the analyzer to load the data
                        success = self.analyzer.load_data_from_file(file_path)
                        
                        if success:
                            self.status_bar.showMessage(f"Loaded data from {os.path.basename(file_path)}", 3000)
                            self.update_visualizations()
                        else:
                            QMessageBox.warning(self, "Error", "Failed to load data file.")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
                else:
                    QMessageBox.warning(self, "Error", "Analyzer not initialized.")
    
    def save_data_file(self):
        """Save radar data to a file."""
        # Check if we have data to save
        if not hasattr(self, 'analyzer') or self.analyzer is None or not hasattr(self.analyzer, 'live_heatmap_data') or self.analyzer.live_heatmap_data is None:
            QMessageBox.warning(self, "No Data", "No radar data available to save.")
            return
        
        # Get default directory and filename
        default_dir = os.path.expanduser("~")
        if hasattr(self.control_panel, 'last_used_directory'):
            last_dir = self.control_panel.last_used_directory()
            if last_dir and os.path.exists(last_dir):
                default_dir = last_dir
        
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"radar_data_{timestamp}.dat"
        default_path = os.path.join(default_dir, default_filename)
        
        # Create a non-blocking file dialog
        dialog = QFileDialog(self, "Save Radar Data", default_path)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("Radar Data Files (*.dat);;CSV Files (*.csv);;All Files (*)")
        dialog.setDefaultSuffix("dat")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # Use Qt's dialog instead of native for better control
        dialog.setWindowModality(Qt.WindowModal)  # Make dialog modal to main window only
        
        # Process events to keep UI responsive
        QApplication.processEvents()
        
        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files and selected_files[0]:
                file_path = selected_files[0]
                
                # Save the last used directory if available
                if hasattr(self.control_panel, 'save_last_used_directory'):
                    try:
                        self.control_panel.save_last_used_directory(os.path.dirname(file_path))
                    except Exception as e:
                        print(f"Error saving last used directory: {e}")
                
                # Save the data file
                if hasattr(self, 'analyzer') and self.analyzer is not None:
                    try:
                        # Use a status message to indicate saving
                        self.status_bar.showMessage(f"Saving data to {os.path.basename(file_path)}...")
                        QApplication.processEvents()  # Force UI update
                        
                        # Use the analyzer to save the data
                        success = self.analyzer.save_data_to_file(file_path)
                        
                        if success:
                            self.status_bar.showMessage(f"Saved data to {os.path.basename(file_path)}", 3000)
                        else:
                            QMessageBox.warning(self, "Error", "Failed to save data file.")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Error saving data: {str(e)}")
                else:
                    QMessageBox.warning(self, "Error", "Analyzer not initialized.")
    
    def on_distance_calculation_changed(self, checked):
        """Handle change in distance calculation method radio button."""
        if not checked:  # Only process when a button is checked (not unchecked)
            return
            
        # Update the analyzer parameter
        if hasattr(self.analyzer, 'params'):
            if self.directional_radio.isChecked():
                self.analyzer.params.use_directional_distance = True
                self.status_bar.showMessage("Using directional (forward) distance calculation", 3000)
            else:
                self.analyzer.params.use_directional_distance = False
                self.status_bar.showMessage("Using Euclidean (radial) distance calculation", 3000)
            
            # Log the change
            self.analyzer.get_logger().info(
                f"Distance calculation method changed to: {'directional' if self.analyzer.params.use_directional_distance else 'Euclidean'}"
            )
    
    @pyqtSlot(int)
    def handle_tab_change(self, tab_index):
        """
        Handle tab change event to manage views.
        
        When the combined view tab is selected, ensure both 
        scatter and heatmap views are properly set in the combined view.
        
        Args:
            tab_index: Index of the selected tab.
        """
        # Get tab widget
        tabs = self.central_widget.findChild(QTabWidget)
        if not tabs:
            return
        
        # Get the selected tab
        selected_tab = tabs.widget(tab_index)
        
        # Check if the combined view tab is selected
        if isinstance(selected_tab, CombinedView):
            # Ensure views are set in combined view
            self.combined_view.set_views(self.scatter_view, self.heatmap_view)
            self.status_bar.showMessage("2D scatter view mode active")
    
    @pyqtSlot(float)
    def set_trail_duration(self, duration):
        """
        Set the duration for which points should remain visible in the trail.
        
        Args:
            duration: Duration in seconds
        """
        if self.scatter_view:
            self.scatter_view.set_trail_duration(duration)
    
    @pyqtSlot(bool)
    def set_trail_visibility(self, visible):
        """
        Set the visibility of the trail.
        
        Args:
            visible: Whether the trail should be visible
        """
        if self.scatter_view:
            self.scatter_view.toggle_trail(visible)
    
    @pyqtSlot(bool)
    def set_density_coloring(self, enabled):
        """
        Set whether to use enhanced density coloring for the trail.
        
        Args:
            enabled: Whether enhanced density coloring should be enabled
        """
        if self.scatter_view:
            self.scatter_view.toggle_density_coloring(enabled)
    
    def initiate_report_generation(self):
        """Handles the actual report generation process after checks."""
        # --- Point to add data clearing ---
        # <<<<<<< SEARCH - REMOVE THIS MARKER
        # ======= - REMOVE THIS MARKER
        # --- Clear existing experiment data before generating report --- 
        self.get_logger().info("Clearing existing experiment data before report generation.")
        try:
            if hasattr(self.analyzer, 'data_lock'):
                with self.analyzer.data_lock:
                    if hasattr(self.analyzer, 'experiment_data'):
                        # Reset lists directly (mirroring ControlPanel.reset_point_counter)
                        if hasattr(self.analyzer.experiment_data, 'x_points'): self.analyzer.experiment_data.x_points = []
                        if hasattr(self.analyzer.experiment_data, 'y_points'): self.analyzer.experiment_data.y_points = []
                        if hasattr(self.analyzer.experiment_data, 'z_points'): self.analyzer.experiment_data.z_points = []
                        if hasattr(self.analyzer.experiment_data, 'intensity'): self.analyzer.experiment_data.intensity = []
                        if hasattr(self.analyzer.experiment_data, 'snr'): self.analyzer.experiment_data.snr = []
                        if hasattr(self.analyzer.experiment_data, 'noise'): self.analyzer.experiment_data.noise = []
                        # Reset metrics that might be recalculated
                        if hasattr(self.analyzer.experiment_data, 'multi_frame_metrics'):
                            self.analyzer.experiment_data.multi_frame_metrics = {
                                'total_frames': 0,
                                'roi_combined_point_count': 0,
                                'outside_roi_combined_point_count': 0,
                                'roi_avg_single_frame_count': 0,
                                'outside_roi_avg_single_frame_count': 0,
                                'ten_frame_avg_points': 0
                            }
                        # Reset distance bands 
                        if hasattr(self.analyzer.experiment_data, 'metadata'):
                            metadata = getattr(self.analyzer.experiment_data, 'metadata', {})
                            if metadata:
                                metadata['distance_bands'] = {}
                                metadata['target_band'] = ''
                                metadata['target_band_count'] = 0
                                self.analyzer.experiment_data.metadata = metadata
                        self.get_logger().info("Cleared experiment data lists and metrics.")
                    else:
                        self.get_logger().warning("Analyzer has no 'experiment_data' attribute to clear.")
            else:
                self.get_logger().warning("Analyzer has no 'data_lock', cannot safely clear data.")
            
            # Also reset the point counter display in the control panel
            if hasattr(self.control_panel, 'reset_point_counter'):
                self.control_panel.reset_point_counter()
                self.get_logger().info("Called control panel reset_point_counter.")

        except Exception as e:
            self.get_logger().error(f"Error clearing data before report generation: {e}", exc_info=True)
            QMessageBox.warning(self, "Warning", f"Could not fully clear previous data before generating report: {e}")
        # --- End data clearing ---
        # >>>>>>> REPLACE - REMOVE THIS MARKER