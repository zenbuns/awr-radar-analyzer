#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main window for the Radar Point Cloud Analyzer application.

This module provides the main application window that integrates
all the UI components and connects them to the radar analyzer.
"""

import os
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QAction, QMenu, QToolBar, QStatusBar, QFileDialog, QMessageBox,
    QLabel, QFrame, QPushButton, QSizePolicy, QApplication, QProgressDialog,
    QProgressBar, QTabWidget, QMenuBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QSize, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QFont, QDesktopServices
import time

from ui.scatter_view import ScatterView
from ui.heatmap_view import HeatmapView
from ui.control_panel import ControlPanel
from utils.visualization import save_scientific_visualization
from .styles import DARK_STYLESHEET, Colors, apply_mpl_style
from ui.point_cloud_view import PointCloudView


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
        
        # Store analyzer reference
        self.analyzer = analyzer
        
        # UI components
        self.scatter_view = None
        self.heatmap_view = None
        self.control_panel = None
        
        # Apply dark style to matplotlib
        apply_mpl_style()
        
        # Set up the UI
        self.init_ui()
        
        # Connect signals and slots
        self.connect_signals()
        
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
        
        # Create and add scatter view tab
        self.scatter_view = ScatterView(self)
        tabs.addTab(self.scatter_view, "Scatter Plot")
        
        # Create and add heatmap view tab
        self.heatmap_view = HeatmapView(self)
        tabs.addTab(self.heatmap_view, "Heatmap")
        
        # Set current tab to heatmap
        tabs.setCurrentIndex(2)
        
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
        file_menu.addAction(save_heatmap_action)
        
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
        start_collection_action.triggered.connect(self.start_data_collection)
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
        view_menu.addAction(reset_heatmap_action)
        
        # Visualization performance submenu
        perf_menu = view_menu.addMenu("&Performance Settings")
        
        high_perf_action = QAction("&High Performance", self)
        high_perf_action.triggered.connect(lambda: self.configure_heatmap_update_params(0.1, 0.5, 30))
        perf_menu.addAction(high_perf_action)
        
        medium_perf_action = QAction("&Medium Performance", self)
        medium_perf_action.triggered.connect(lambda: self.configure_heatmap_update_params(0.2, 0.7, 20))
        perf_menu.addAction(medium_perf_action)
        
        low_perf_action = QAction("&Low Performance", self)
        low_perf_action.triggered.connect(lambda: self.configure_heatmap_update_params(0.3, 1.0, 15))
        perf_menu.addAction(low_perf_action)
        
        perf_menu.addSeparator()
        
        auto_perf_action = QAction("&Auto-Optimize", self)
        auto_perf_action.triggered.connect(self.optimize_visualization_pipeline)
        perf_menu.addAction(auto_perf_action)
        
        view_menu.addSeparator()
        
        # Visualization mode submenu
        mode_menu = view_menu.addMenu("Visualization &Mode")
        
        heatmap_mode_action = QAction("&Heatmap", self)
        heatmap_mode_action.triggered.connect(lambda: self.set_visualization_mode("heatmap"))
        mode_menu.addAction(heatmap_mode_action)
        
        contour_mode_action = QAction("&Contour", self)
        contour_mode_action.triggered.connect(lambda: self.set_visualization_mode("contour"))
        mode_menu.addAction(contour_mode_action)
        
        combined_mode_action = QAction("Co&mbined", self)
        combined_mode_action.triggered.connect(lambda: self.set_visualization_mode("combined"))
        mode_menu.addAction(combined_mode_action)
        
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
        toolbar.addAction(reset_action)
        
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
        self.control_panel.reset_heatmap.connect(self.reset_heatmap)
        self.control_panel.colormap_changed.connect(self.set_colormap)
        self.control_panel.decay_factor_changed.connect(self.set_decay_factor)
        self.control_panel.visualization_mode_changed.connect(self.set_visualization_mode)
        self.control_panel.noise_floor_changed.connect(self.set_noise_floor)
        self.control_panel.smoothing_changed.connect(self.set_smoothing)
        self.control_panel.add_roi.connect(self.add_roi)
        self.control_panel.clear_rois.connect(self.clear_rois)
        self.control_panel.save_heatmap.connect(self.save_heatmap)
        self.control_panel.export_plot.connect(self.export_scientific_plot)
        self.control_panel.generate_report.connect(self.generate_report)
        
        # Connect ROS2 bag playback and recording signals
        self.control_panel.play_rosbag.connect(self.play_rosbag)
        self.control_panel.record_rosbag.connect(self.record_rosbag)
        self.control_panel.stop_rosbag.connect(self.stop_rosbag)
        self.control_panel.timeline_position_changed.connect(self.seek_rosbag)
        self.control_panel.visualize_pointcloud.connect(self.visualize_pointcloud)
        
        # Connect analyzer signals for playback progress updates
        if hasattr(self.analyzer, 'signals') and hasattr(self.analyzer.signals, 'update_playback_position_signal'):
            self.analyzer.signals.update_playback_position_signal.connect(self.update_playback_position)
        
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
                heatmap_data = self.analyzer.live_heatmap_data
                
                # Update heatmap data through the improved, optimized pipeline
                self.heatmap_view.update_heatmap_data(heatmap_data)
                
                # Update analysis metrics - only do this periodically as it's CPU intensive
                # Use a counter to update every 10 frames to reduce CPU load
                if not hasattr(self, '_metrics_update_counter'):
                    self._metrics_update_counter = 0
                
                if self._metrics_update_counter % 10 == 0:
                    metrics = self.analyzer.compute_heatmap_metrics()
                    self.control_panel.update_metrics(metrics)
                
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
            distance: New circle distance in meters.
        """
        # Update analyzer
        if self.analyzer is not None:
            if index == 0:
                self.analyzer.update_circle_position(distance)
            # For all circles, update the params directly
            if hasattr(self.analyzer, 'params'):
                self.analyzer.params.update_circle_distance(index, distance)
        
        # Get the angle for this circle
        angle = 0
        if index == 1:
            angle = -60
        elif index == 2:
            angle = 60
        
        # Update views
        self.scatter_view.update_circle_position(index, distance, angle)
        self.heatmap_view.update_circle_position(index, distance, angle)
    
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
        self.scatter_view.update_circle_radius(index, radius)
        self.heatmap_view.update_circle_radius(index, radius)
    
    @pyqtSlot(int, float)
    def update_circle_angle(self, index, angle):
        """
        Update circle angle in views.
        
        Args:
            index: Index of the circle to update (0-2)
            angle: New angle in degrees.
        """
        # Get current distance for this circle
        distance = 5.0
        if index == 1:
            distance = 15.0
        elif index == 2:
            distance = 25.0
            
        # Update views with both distance and new angle
        self.scatter_view.update_circle_position(index, distance, angle)
        self.heatmap_view.update_circle_position(index, distance, angle)
    
    @pyqtSlot(int, bool)
    def toggle_circle(self, index, enabled):
        """
        Toggle circle visibility.
        
        Args:
            index: Index of the circle to toggle (0-2)
            enabled: Whether the circle should be visible
        """
        # Update views
        self.scatter_view.toggle_circle(index, enabled)
        self.heatmap_view.toggle_circle(index, enabled)
        
        # Update analyzer if it exists
        if self.analyzer is not None and hasattr(self.analyzer, 'params'):
            self.analyzer.params.toggle_circle(index, enabled)
    
    @pyqtSlot(str, str, int)
    def start_data_collection_with_params(self, config_name, target_distance, duration):
        """
        Start data collection with specified parameters.
        
        Args:
            config_name: Configuration name.
            target_distance: Target distance as string.
            duration: Collection duration in seconds.
        """
        if self.analyzer is not None:
            success = self.analyzer.start_data_collection(config_name, target_distance, duration)
            if success:
                self.status_bar.showMessage(f"Collecting data for configuration: {config_name}")
                self.heatmap_view.update_target_distance(float(target_distance))
            else:
                self.status_bar.showMessage("Failed to start data collection")
                self.control_panel.on_stop_collection()  # Reset UI state
    
    @pyqtSlot()
    def start_data_collection(self):
        """Start data collection using current UI parameters."""
        config_name = self.control_panel.config_entry.text().strip()
        target_distance = self.control_panel.target_combo.currentText()
        duration = self.control_panel.duration_spin.value()
        
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
            self.analyzer.reset_live_heatmap()
        
        self.heatmap_view.reset_heatmap()
        self.status_bar.showMessage("Heatmap reset")
    
    @pyqtSlot(str)
    def set_colormap(self, colormap):
        """
        Set the colormap for visualizations.
        
        Args:
            colormap: Name of the colormap to use.
        """
        self.heatmap_view.set_colormap(colormap)
        self.status_bar.showMessage(f"Colormap set to {colormap}")
    
    @pyqtSlot(float)
    def set_decay_factor(self, decay):
        """
        Set the decay factor for live heatmap.
        
        Args:
            decay: New decay factor value.
        """
        if self.analyzer is not None:
            self.analyzer.live_heatmap_decay_factor = decay
        self.status_bar.showMessage(f"Decay factor set to {decay:.3f}")
    
    @pyqtSlot(str)
    def set_visualization_mode(self, mode):
        """
        Set the visualization mode.
        
        Args:
            mode: Visualization mode ('heatmap', 'contour', or 'combined').
        """
        self.heatmap_view.set_visualization_mode(mode)
        self.status_bar.showMessage(f"Visualization mode set to {mode}")
    
    @pyqtSlot(float)
    def set_noise_floor(self, value):
        """
        Set the noise floor threshold.
        
        Args:
            value: New noise floor value.
        """
        self.heatmap_view.set_noise_floor(value)
        self.status_bar.showMessage(f"Noise floor set to {value:.2f}")
    
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
        roi = self.heatmap_view.add_roi()
        if roi is not None:
            stats = self.heatmap_view.analyze_roi(roi)
            if stats:
                self.status_bar.showMessage(
                    f"ROI added: avg={stats['mean_intensity']:.2f}, max={stats['max_intensity']:.2f}, "
                    f"coverage={stats['signal_coverage']*100:.1f}%"
                )
    
    @pyqtSlot()
    def clear_rois(self):
        """Clear all Regions of Interest from the heatmap."""
        self.heatmap_view.clear_rois()
        self.status_bar.showMessage("All ROIs cleared")
    
    @pyqtSlot()
    def save_heatmap(self):
        """Save the current heatmap data and visualization."""
        if self.analyzer is None or self.analyzer.live_heatmap_data is None:
            self.status_bar.showMessage("No heatmap data to save")
            return
        
        try:
            # Create directory if it doesn't exist
            data_dir = os.path.expanduser('~/radar_experiment_data/heatmaps')
            os.makedirs(data_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save heatmap data as NumPy array
            heatmap_file = os.path.join(data_dir, f"live_heatmap_{timestamp}.npz")
            np.savez_compressed(heatmap_file, data=self.analyzer.live_heatmap_data)
            
            # Save heatmap visualization as PNG
            viz_file = os.path.join(data_dir, f"live_heatmap_viz_{timestamp}.png")
            self.heatmap_view.figure.savefig(
                viz_file,
                dpi=300,
                bbox_inches='tight',
                facecolor=self.heatmap_view.figure.get_facecolor()
            )
            
            self.status_bar.showMessage(f"Saved heatmap to {os.path.basename(viz_file)}")
            
            # Ask if user wants to view the saved file
            reply = QMessageBox.question(
                self, "Save Complete",
                f"Heatmap saved to:\n{viz_file}\n\nWould you like to open it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Open file with default system application
                import subprocess
                import sys
                if sys.platform == 'win32':
                    os.startfile(viz_file)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.call(['open', viz_file])
                else:  # Linux
                    subprocess.call(['xdg-open', viz_file])
        
        except Exception as e:
            self.status_bar.showMessage(f"Error saving heatmap: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"Failed to save heatmap: {str(e)}")
    
    @pyqtSlot()
    def export_scientific_plot(self):
        """Export a high-quality scientific visualization of the radar data."""
        if self.analyzer is None or self.analyzer.live_heatmap_data is None:
            self.status_bar.showMessage("No data to export")
            return
        
        try:
            # Use state manager to lock UI during export if available
            if hasattr(self.control_panel, 'state_manager'):
                self.control_panel.state_manager.transition('lock_ui')
            
            # Ensure the export directory exists
            export_dir = os.path.join(os.path.expanduser("~"), "radar_experiment_data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            # Get last used directory or default to exports directory
            default_dir = export_dir
            if hasattr(self.control_panel, 'last_used_directory'):
                last_dir = self.control_panel.last_used_directory()
                if last_dir and os.path.exists(last_dir):
                    default_dir = last_dir
            
            # Generate default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = self.control_panel.config_entry.text().strip() or "default_config"
            default_filename = f"radar_plot_{config_name}_{timestamp}.png"
            
            # Show directory selection dialog first
            export_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Export Directory",
                default_dir,
                QFileDialog.ShowDirsOnly
            )
            
            if not export_dir:
                # User canceled, unlock UI and return
                if hasattr(self.control_panel, 'state_manager'):
                    self.control_panel.state_manager.transition('unlock_ui')
                return
            
            # Now show file name dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Scientific Plot",
                os.path.join(export_dir, default_filename),
                "PNG Image (*.png);;PDF Document (*.pdf);;SVG Image (*.svg)"
            )
            
            if not file_path:
                # User canceled, unlock UI and return
                if hasattr(self.control_panel, 'state_manager'):
                    self.control_panel.state_manager.transition('unlock_ui')
                return
            
            # Save the directory for next time
            if hasattr(self.control_panel, 'save_last_used_directory'):
                self.control_panel.save_last_used_directory(os.path.dirname(file_path))
            
            # Update status message
            self.status_bar.showMessage("Exporting plot, please wait...")
            QApplication.processEvents()  # Force UI update
            
            # Get visualization parameters
            colormap = self.control_panel.colormap_combo.currentText()
            
            # Get visualization mode
            visualization_mode = "heatmap"  # Default
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
            try:
                noise_floor = float(self.control_panel.noise_value.text())
            except ValueError:
                noise_floor = 0.1  # Default
                
            try:
                smoothing_sigma = float(self.control_panel.smooth_value.text())
            except ValueError:
                smoothing_sigma = 1.0  # Default
            
            # Create a progress dialog to show export progress
            progress_dialog = QProgressDialog("Exporting plot...", "Cancel", 0, 100, self)
            progress_dialog.setWindowTitle("Exporting")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)  # Show immediately
            progress_dialog.setValue(0)
            
            # Clone the heatmap data to avoid threading issues
            heatmap_data_copy = self.analyzer.live_heatmap_data.copy() if self.analyzer.live_heatmap_data is not None else None
            
            # Create a callback function for progress updates
            def progress_callback(progress):
                # Only update if not canceled
                if not progress_dialog.wasCanceled():
                    progress_dialog.setValue(int(progress * 100))
                    QApplication.processEvents()  # Allow UI to update
            
            # Run the export in a separate thread to avoid freezing UI
            class ExportThread(QThread):
                finished = pyqtSignal(bool, str)
                progress = pyqtSignal(float)
                
                def __init__(self, file_path, heatmap_data, params):
                    super().__init__()
                    self.file_path = file_path
                    self.heatmap_data = heatmap_data
                    self.params = params
                
                def run(self):
                    try:
                        from utils.visualization import save_scientific_visualization
                        save_scientific_visualization(
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
                            progress_callback=lambda p: self.progress.emit(p)
                        )
                        self.finished.emit(True, self.file_path)
                    except Exception as e:
                        print(f"Error in export thread: {e}")
                        self.finished.emit(False, str(e))
            
            # Prepare parameters for the export thread
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
            
            # Create and start the export thread
            self.export_thread = ExportThread(file_path, heatmap_data_copy, export_params)
            
            # Connect signals
            self.export_thread.progress.connect(progress_callback)
            self.export_thread.finished.connect(self.on_export_completed)
            
            # Store file path for later use
            self.current_export_path = file_path
            
            # Start the thread
            self.export_thread.start()
            
        except Exception as e:
            self.status_bar.showMessage(f"Error exporting plot: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export plot: {str(e)}")
            
            # Ensure UI is unlocked in case of error
            if hasattr(self.control_panel, 'state_manager'):
                self.control_panel.state_manager.transition('unlock_ui')
    
    def on_export_completed(self, success, result):
        """
        Handle completion of the export thread.
        
        Args:
            success: Whether the export was successful
            result: File path if successful, error message if not
        """
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
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(default_dir, f"radar_report_{timestamp}.html")
            
            # Generate the report at the specified location without asking for location
            self.status_bar.showMessage("Generating report, please wait...")
            QApplication.processEvents()  # Process UI events to update status
            
            # Generate the report without user interaction
            success = self.generate_custom_report(file_path)
            
            # Unlock UI after completion
            if hasattr(self.control_panel, 'state_manager'):
                self.control_panel.state_manager.transition('unlock_ui')
            
            if success:
                self.status_bar.showMessage(f"Report generated: {os.path.basename(file_path)}")
                
                # Notify the control panel that report generation is complete
                if hasattr(self.control_panel, 'on_report_completed'):
                    self.control_panel.on_report_completed(success=True, report_path=file_path)
                
                # Automatically open the report without asking
                try:
                    # Open file with default system application
                    if os.path.exists(file_path):
                        QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
                        QMessageBox.information(
                            self, "Report Generated",
                            f"Report saved to:\n{file_path}"
                        )
                    else:
                        QMessageBox.warning(
                            self, "Report Generated",
                            f"Report was generated but cannot be found at:\n{file_path}"
                        )
                except Exception as e:
                    QMessageBox.information(
                        self, "Report Generated",
                        f"Report saved to:\n{file_path}\n\n(Unable to open automatically: {str(e)})"
                    )
                    import subprocess
                    import sys
                    if sys.platform == 'win32':
                        os.startfile(file_path)
                    elif sys.platform == 'darwin':  # macOS
                        subprocess.call(['open', file_path])
                    else:  # Linux
                        subprocess.call(['xdg-open', file_path])
            else:
                self.status_bar.showMessage("Failed to generate report")
                QMessageBox.warning(self, "Report Error", "Failed to generate report")
                # Notify the control panel that report generation failed
                if hasattr(self.control_panel, 'on_report_completed'):
                    self.control_panel.on_report_completed(success=False)
        
        except Exception as e:
            self.status_bar.showMessage(f"Error generating report: {str(e)}")
            QMessageBox.critical(self, "Report Error", f"Error generating report: {str(e)}")
            
            # Ensure UI is unlocked in case of error
            if hasattr(self.control_panel, 'state_manager'):
                self.control_panel.state_manager.transition('unlock_ui')
                
            # Notify the control panel about the error
            if hasattr(self.control_panel, 'on_report_completed'):
                self.control_panel.on_report_completed(success=False)
    
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

    def generate_custom_report(self, file_path: str) -> bool:
        """
        Generate enhanced HTML report with better formatting and visualization.
        
        Args:
            file_path: Path where the report should be saved.
            
        Returns:
            True if report generation was successful, False otherwise.
        """
        if not self.analyzer or not self.analyzer.config_results:
            return False
            
        try:
            # Process UI events to keep the UI responsive during report generation
            QApplication.processEvents()
            
            # Generate current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Pre-process all data needed for the report to minimize UI freezing
            processed_configs = {}
            for config, distances in self.analyzer.config_results.items():
                processed_configs[config] = {}
                for distance, results in distances.items():
                    # Pre-calculate all metrics to avoid performance bottlenecks
                    points = results.get('circle_points', 0)
                    avg_intensity = results.get('circle_avg_intensity', 0)
                    circle_area = np.pi * self.analyzer.params.circle_radius**2
                    density = points / circle_area if circle_area > 0 else 0
                    
                    # Store pre-calculated values
                    processed_configs[config][distance] = {
                        'points': points,
                        'avg_intensity': avg_intensity,
                        'density': density,
                        'multi_frame_metrics': results.get('multi_frame_metrics', {})
                    }
                    
                    # Process UI events periodically to keep the app responsive
                    QApplication.processEvents()
            
            # Generate HTML content with enhanced styling
            with open(file_path, 'w') as f:
                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <title>Radar Experiment Comparison Report</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        :root {{
                            --primary-color: #2563eb;
                            --primary-light: #3b82f6;
                            --primary-dark: #1d4ed8;
                            --secondary-color: #64748b;
                            --accent-color: #f59e0b;
                            --success-color: #10b981;
                            --danger-color: #ef4444;
                            --gray-100: #f1f5f9;
                            --gray-200: #e2e8f0;
                            --gray-300: #cbd5e1;
                            --gray-800: #1e293b;
                            --white: #ffffff;
                        }}
                        
                        * {{
                            box-sizing: border-box;
                            margin: 0;
                            padding: 0;
                        }}
                        
                        body {{ 
                            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; 
                            color: var(--gray-800);
                            line-height: 1.6;
                            background-color: #f8fafc;
                            padding: 0;
                            margin: 0;
                        }}
                        
                        .container {{
                            max-width: 1200px;
                            margin: 0 auto;
                            padding: 0 20px;
                        }}
                        
                        h1, h2, h3, h4, h5, h6 {{ 
                            color: var(--gray-800);
                            margin-top: 1.5rem;
                            margin-bottom: 1rem;
                            font-weight: 600;
                            line-height: 1.25;
                        }}
                        
                        h1 {{ 
                            font-size: 1.875rem;
                            border-bottom: 2px solid var(--primary-color);
                            padding-bottom: 0.5rem;
                        }}
                        
                        h2 {{ 
                            font-size: 1.5rem;
                            border-bottom: 1px solid var(--primary-light);
                            padding-bottom: 0.25rem;
                        }}
                        
                        h3 {{
                            font-size: 1.25rem;
                        }}
                        
                        .timestamp {{ 
                            color: var(--secondary-color);
                            font-style: italic;
                            margin-bottom: 2rem;
                        }}
                        
                        .header {{
                            background-color: var(--white);
                            padding: 2rem 0;
                            border-bottom: 1px solid var(--gray-200);
                            margin-bottom: 2rem;
                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        }}
                        
                        table {{ 
                            border-collapse: collapse; 
                            width: 100%; 
                            margin: 1.5rem 0;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
                            background-color: var(--white);
                            border-radius: 0.5rem;
                            overflow: hidden;
                        }}
                        
                        th, td {{ 
                            padding: 1rem 0.75rem; 
                            text-align: left; 
                        }}
                        
                        th {{ 
                            background-color: var(--primary-color); 
                            color: var(--white);
                            font-weight: 600;
                            position: sticky;
                            top: 0;
                        }}
                        
                        tr:nth-child(even) {{ 
                            background-color: var(--gray-100); 
                        }}
                        
                        tr:hover {{
                            background-color: #dbeafe;
                        }}
                        
                        .data-card {{
                            background-color: var(--white);
                            border-radius: 0.5rem;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
                            margin-bottom: 1.5rem;
                            overflow: hidden;
                        }}
                        
                        .card-header {{
                            background-color: var(--primary-color);
                            color: var(--white);
                            padding: 1rem;
                            font-weight: 600;
                            font-size: 1.25rem;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        }}
                        
                        .card-body {{
                            padding: 1.5rem;
                        }}
                        
                        .card-title {{
                            font-size: 1.25rem;
                            font-weight: 600;
                            margin-bottom: 1rem;
                            color: var(--primary-dark);
                        }}
                        
                        .metrics-grid {{
                            display: grid;
                            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                            gap: 1rem;
                            margin: 1rem 0;
                        }}
                        
                        .metric {{
                            background-color: var(--gray-100);
                            padding: 1rem;
                            border-radius: 0.375rem;
                            border-left: 3px solid var(--primary-color);
                            transition: transform 0.2s ease-in-out;
                        }}
                        
                        .metric:hover {{
                            transform: translateY(-2px);
                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        }}
                        
                        .metric-label {{
                            font-weight: 500;
                            color: var(--secondary-color);
                            margin-bottom: 0.25rem;
                            font-size: 0.875rem;
                        }}
                        
                        .metric-value {{
                            font-size: 1.5rem;
                            font-weight: 600;
                            color: var(--gray-800);
                        }}
                        
                        .metric-units {{
                            font-size: 0.75rem;
                            color: var(--secondary-color);
                            margin-left: 0.25rem;
                        }}
                        
                        .section {{
                            margin-bottom: 3rem;
                        }}
                        
                        .section-title {{
                            display: flex;
                            align-items: center;
                            margin-bottom: 1rem;
                        }}
                        
                        .section-title::before {{
                            content: "";
                            width: 4px;
                            height: 1.5rem;
                            background-color: var(--primary-color);
                            margin-right: 0.5rem;
                            border-radius: 2px;
                        }}
                        
                        .tabs {{
                            display: flex;
                            border-bottom: 1px solid var(--gray-300);
                            margin-bottom: 1rem;
                        }}
                        
                        .tab {{
                            padding: 0.75rem 1rem;
                            font-weight: 500;
                            cursor: pointer;
                            border-bottom: 2px solid transparent;
                            transition: all 0.2s ease-in-out;
                        }}
                        
                        .tab.active {{
                            border-bottom: 2px solid var(--primary-color);
                            color: var(--primary-color);
                        }}
                        
                        .tab:hover {{
                            color: var(--primary-color);
                        }}
                        
                        .tab-content {{
                            display: none;
                        }}
                        
                        .tab-content.active {{
                            display: block;
                        }}
                        
                        .highlight {{
                            background-color: #fef3c7 !important;
                            font-weight: 600;
                        }}
                        
                        .header-row {{
                            background-color: #dbeafe !important;
                        }}
                        
                        footer {{
                            background-color: var(--gray-800);
                            color: var(--white);
                            padding: 2rem 0;
                            margin-top: 3rem;
                            text-align: center;
                        }}
                        
                        @media (max-width: 768px) {{
                            .metrics-grid {{
                                grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
                            }}
                            
                            .metric-value {{
                                font-size: 1.25rem;
                            }}
                            
                            table {{
                                display: block;
                                overflow-x: auto;
                            }}
                        }}
                        
                        /* Add basic JavaScript interaction for tabs */
                        .js-tabs {{
                            margin-top: 1rem;
                        }}
                    </style>
                    <script>
                        document.addEventListener('DOMContentLoaded', function() {{
                            // Tab functionality
                            const tabs = document.querySelectorAll('.tab');
                            tabs.forEach(tab => {{
                                tab.addEventListener('click', function() {{
                                    // Remove active class from all tabs
                                    tabs.forEach(t => t.classList.remove('active'));
                                    
                                    // Add active class to clicked tab
                                    this.classList.add('active');
                                    
                                    // Hide all tab content
                                    const tabContents = document.querySelectorAll('.tab-content');
                                    tabContents.forEach(content => content.classList.remove('active'));
                                    
                                    // Show selected tab content
                                    const targetContent = document.getElementById(this.dataset.target);
                                    if (targetContent) {{
                                        targetContent.classList.add('active');
                                    }}
                                }});
                            }});
                            
                            // Activate first tab by default
                            const firstTab = document.querySelector('.tab');
                            if (firstTab) {{
                                firstTab.click();
                            }}
                        }});
                    </script>
                </head>
                <body>
                    <div class="header">
                        <div class="container">
                            <h1>Radar Experiment Comparison Report</h1>
                            <p class="timestamp">Generated on: {timestamp}</p>
                        </div>
                    </div>
                    
                    <div class="container">
                """
                
                f.write(html_content)

                # Summary section
                summary_html = """
                    <div class="section">
                        <div class="section-title">
                            <h2>Configuration Summary</h2>
                        </div>
                        <div class="data-card">
                            <div class="card-body">
                                <table>
                                    <thead>
                                        <tr class="header-row">
                                            <th>Configuration</th>
                                            <th>Target Distance</th>
                                            <th>Total Points</th>
                                            <th>10-Frame Avg</th>
                                            <th>Point Density</th>
                                            <th>Avg. Intensity</th>
                                            <th>ROI Type</th>
                                            <th>Points</th>
                                            <th>Avg Pts/Frame</th>
                                            <th>Density</th>
                                            <th>SNR</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                """
                
                f.write(summary_html)

                # Use pre-processed data to generate the table rows more efficiently
                table_rows = []
                for config, distances in processed_configs.items():
                    for distance, data in distances.items():
                        multi_frame_metrics = data['multi_frame_metrics']
                        
                        # Get inside ROI metrics
                        roi_points = multi_frame_metrics.get('roi_combined_point_count', 0)
                        roi_spatial_density = multi_frame_metrics.get('roi_spatial_density', 0)
                        roi_snr = multi_frame_metrics.get('roi_snr_db', 0)
                        
                        # Primary ROI row
                        roi_avg_points = multi_frame_metrics.get('roi_avg_single_frame_count', 0)
                        
                        # Calculate correct total points (sum of inside and outside ROI)
                        outside_points = multi_frame_metrics.get('outside_roi_combined_point_count', 0)
                        total_points = roi_points + outside_points
                        
                        # Calculate correct 10-frame average
                        outside_avg = multi_frame_metrics.get('outside_roi_avg_single_frame_count', 0)
                        ten_frame_avg = roi_avg_points + outside_avg
                        
                        table_rows.append(f"""
                            <tr>
                                <td rowspan="{1 + len([c for c in self.analyzer.params.circles[1:] if c.enabled]) + 1}">{config}</td>
                                <td rowspan="{1 + len([c for c in self.analyzer.params.circles[1:] if c.enabled]) + 1}">{distance}m</td>
                                <td rowspan="{1 + len([c for c in self.analyzer.params.circles[1:] if c.enabled]) + 1}">{total_points}</td>
                                <td rowspan="{1 + len([c for c in self.analyzer.params.circles[1:] if c.enabled]) + 1}">{ten_frame_avg:.1f}</td>
                                <td rowspan="{1 + len([c for c in self.analyzer.params.circles[1:] if c.enabled]) + 1}">{data['density']:.3f} pts/m²</td>
                                <td rowspan="{1 + len([c for c in self.analyzer.params.circles[1:] if c.enabled]) + 1}">{data['avg_intensity']:.3f}</td>
                                <td>Primary ROI</td>
                                <td>{roi_points}</td>
                                <td>{roi_avg_points:.1f}</td>
                                <td>{roi_spatial_density:.2f}</td>
                                <td>{roi_snr:.2f}</td>
                            </tr>
                        """)
                        
                        # Secondary ROI rows
                        for i in range(1, 3):  # Add rows for secondary circles
                            if i < len(self.analyzer.params.circles) and self.analyzer.params.circles[i].enabled:
                                prefix = f'roi{i+1}'
                                sec_points = multi_frame_metrics.get(f'{prefix}_combined_point_count', 0)
                                sec_avg_points = multi_frame_metrics.get(f'{prefix}_avg_single_frame_count', 0)
                                sec_density = multi_frame_metrics.get(f'{prefix}_spatial_density', 0)
                                sec_snr = multi_frame_metrics.get(f'{prefix}_snr_db', 0)
                                table_rows.append(f"""
                                    <tr>
                                        <td>{self.analyzer.params.circles[i].label} ROI</td>
                                        <td>{sec_points}</td>
                                        <td>{sec_avg_points:.1f}</td>
                                        <td>{sec_density:.2f}</td>
                                        <td>{sec_snr:.2f}</td>
                                    </tr>
                                """)
                        
                        # Outside ROI row
                        outside_avg_points = multi_frame_metrics.get('outside_roi_avg_single_frame_count', 0)
                        table_rows.append(f"""
                            <tr>
                                <td>Outside ROI</td>
                                <td>{multi_frame_metrics.get('outside_roi_combined_point_count', 0)}</td>
                                <td>{outside_avg_points:.1f}</td>
                                <td>{multi_frame_metrics.get('outside_roi_spatial_density', 0):.2f}</td>
                                <td>{multi_frame_metrics.get('outside_roi_snr_db', 0):.2f}</td>
                            </tr>
                        """)
                        
                        # Process UI events periodically during HTML generation
                        QApplication.processEvents()
                        
                # Write all table rows at once for better performance
                f.write(''.join(table_rows))

                # End of summary section
                f.write("""
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                """)

                # Detailed results - build all HTML in memory first then write at once
                f.write("""
                    <div class="section">
                        <div class="section-title">
                            <h2>Detailed Results</h2>
                        </div>
                """)
                
                # Pre-build all detailed results sections
                all_details = []
                
                for config, distances in processed_configs.items():
                    # Process UI events to keep app responsive
                    QApplication.processEvents()
                    
                    section_html = [f"""
                        <div class="data-card">
                            <div class="card-header">
                                Configuration: {config}
                            </div>
                            <div class="card-body">
                    """]
                    
                    # Add tabs for each distance
                    section_html.append('<div class="js-tabs">')
                    section_html.append('<div class="tabs">')
                    
                    # Generate tab buttons
                    for i, (distance, _) in enumerate(distances.items()):
                        active_class = 'active' if i == 0 else ''
                        section_html.append(f"""
                            <div class="tab {active_class}" data-target="tab-{config}-{distance}">
                                {distance}m
                            </div>
                        """)
                    
                    section_html.append('</div>') # End tabs
                    
                    # Generate tab content
                    for distance, data in distances.items():
                        # Process UI events periodically
                        QApplication.processEvents()
                        
                        # Get original results for fields not pre-processed
                        results = self.analyzer.config_results[config][distance]
                        
                        # Calculate correct total points and averages
                        roi_points = data['multi_frame_metrics'].get('roi_combined_point_count', 0)
                        outside_points = data['multi_frame_metrics'].get('outside_roi_combined_point_count', 0)
                        total_points = roi_points + outside_points
                        
                        roi_avg = data['multi_frame_metrics'].get('roi_avg_single_frame_count', 0)
                        outside_avg = data['multi_frame_metrics'].get('outside_roi_avg_single_frame_count', 0)
                        ten_frame_avg = roi_avg + outside_avg
                        
                        section_html.append(f"""
                            <div id="tab-{config}-{distance}" class="tab-content">
                                <h3 class="card-title">Target Distance: {distance}m</h3>
                                
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <div class="metric-label">Total Points</div>
                                        <div class="metric-value">{total_points}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">10-Frame Average Points</div>
                                        <div class="metric-value">{ten_frame_avg:.1f}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Average Intensity</div>
                                        <div class="metric-value">{results.get('avg_intensity', 0):.3f}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Points in Target Band</div>
                                        <div class="metric-value">{results.get('target_band_points', 0)}</div>
                                    </div>
                                </div>

                                <h4 class="card-title">10-Frame Average Metrics</h4>
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <div class="metric-label">Combined Points</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('combined_point_count', 0)}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Avg Points/Frame</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('avg_single_frame_count', 0):.1f}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Density Improvement</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('point_density_improvement', 0):.2f}<span class="metric-units">x</span></div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Avg Intensity</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('combined_avg_intensity', 0):.2f}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Max Intensity</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('combined_max_intensity', 0):.2f}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Stability Score</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('ten_frame_stability', 0):.2f}</div>
                                    </div>
                                </div>
                                
                                <h4 class="card-title">Primary ROI Circle Metrics</h4>
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <div class="metric-label">Combined Points</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('roi_combined_point_count', 0)}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Avg Points/Frame</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('roi_avg_single_frame_count', 0):.1f}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Spatial Density</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('roi_spatial_density', 0):.2f}<span class="metric-units">pts/m²</span></div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Density Gain</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('roi_density_gain_db', 0):.2f}<span class="metric-units">dB</span></div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">SNR</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('roi_snr_db', 0):.2f}<span class="metric-units">dB</span></div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Intensity Range</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('roi_combined_min_intensity', 0):.1f} - {data['multi_frame_metrics'].get('roi_combined_max_intensity', 0):.1f}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Point Separation</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('roi_mean_point_separation', 0):.3f}<span class="metric-units">m</span></div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Spatial Uniformity</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('roi_spatial_uniformity', 0):.3f}</div>
                                    </div>
                                </div>
                                
                                <!-- Secondary ROI metrics -->
                                {self._generate_secondary_roi_metrics_html(data['multi_frame_metrics'], 1, self.analyzer.params.circles)}
                                {self._generate_secondary_roi_metrics_html(data['multi_frame_metrics'], 2, self.analyzer.params.circles)}
                                
                                <!-- Outside ROI metrics (in a collapsible section) -->
                                <h4 class="card-title">Outside ROI Metrics</h4>
                                <div class="metrics-grid">
                                    <div class="metric">
                                        <div class="metric-label">Total Points</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('outside_roi_combined_point_count', 0)}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Avg Points/Frame</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('outside_roi_avg_single_frame_count', 0):.1f}</div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">Spatial Density</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('outside_roi_spatial_density', 0):.2f}<span class="metric-units">pts/m²</span></div>
                                    </div>
                                    
                                    <div class="metric">
                                        <div class="metric-label">SNR</div>
                                        <div class="metric-value">{data['multi_frame_metrics'].get('outside_roi_snr_db', 0):.2f}<span class="metric-units">dB</span></div>
                                    </div>
                                </div>

                                <h4 class="card-title">Distance Band Analysis</h4>
                                <table>
                                    <thead>
                                        <tr class="header-row">
                                            <th>Distance Band</th>
                                            <th>Points</th>
                                            <th>Avg. Intensity</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                        """)

                        # Process distance bands more efficiently by building all HTML at once
                        band_rows = []
                        distance_bands = results.get('distance_bands', {})
                        
                        for band, band_data in distance_bands.items():
                            highlight = 'class="highlight"' if band == results.get('target_band', '') else ''
                            band_rows.append(f"""
                                <tr {highlight}>
                                    <td>{band}</td>
                                    <td>{band_data.get('count', 0):.0f}</td>
                                    <td>{band_data.get('avg_intensity', 0):.3f}</td>
                                </tr>
                            """)
                            
                            # Process UI events periodically during band processing
                            if len(band_rows) % 5 == 0:  # Process every 5 rows
                                QApplication.processEvents()
                        
                        # Append all band rows at once
                        section_html.append(''.join(band_rows))
                        section_html.append("""
                                    </tbody>
                                </table>
                            </div>
                        """)
                    
                    # Complete this configuration section
                    section_html.append("</div>") # End js-tabs
                    section_html.append("</div>") # End card-body
                    section_html.append("</div>") # End data-card
                    
                    # Add this completed section to all details
                    all_details.append(''.join(section_html))

                # Write all sections at once
                f.write(''.join(all_details))
                
                # Close the details section
                f.write("</div>")
                
                # Add summary statistics and footer
                footer_html = f"""
                    <div class="section">
                        <div class="section-title">
                            <h2>Report Summary</h2>
                        </div>
                        <div class="data-card">
                            <div class="card-body">
                                <p>This report contains data for {len(processed_configs)} radar configurations and 
                                {sum(len(distances) for distances in processed_configs.values())} distance measurements.</p>
                                <p>Data was processed using multi-frame analysis over 10 frames to provide enhanced statistics.</p>
                            </div>
                        </div>
                    </div>
                    
                    <footer>
                        <div class="container">
                            <p>AWR Radar Analyzer &copy; {datetime.now().year}</p>
                        </div>
                    </footer>
                </div>
                </body>
                </html>
                """
                
                f.write(footer_html)

            self.statusBar().showMessage(f"Report saved to {file_path}", 5000)
            return True
            
        except Exception as e:
            self.statusBar().showMessage(f"Error generating report: {str(e)}", 5000)
            return False
    
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
            
            # Call analyzer method to play the bag file
            self.analyzer.play_rosbag(bag_path)
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
        if self.analyzer is None:
            return
        
        try:
            # Call analyzer method to stop bag operations
            self.analyzer.stop_rosbag()
            self.status_bar.showMessage("Stopped ROS2 bag playback/recording")
        except Exception as e:
            QMessageBox.critical(self, "Stop Error", f"Failed to stop ROS2 bag: {str(e)}")
    
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
        """Update the timeline slider position during bag playback.
        
        This method is called from the analyzer's update_playback_position_signal
        to update the UI timeline as the bag plays.
        
        Args:
            position: Normalized position in the bag (0.0-1.0)
        """
        try:
            if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'timeline_slider'):
                # Check if the user is currently dragging the slider
                if not getattr(self.control_panel, 'timeline_dragging', False):
                    # Store the last position and current time if not already set
                    if not hasattr(self, '_last_position'):
                        self._last_position = 0.0
                        self._last_position_time = time.time()
                        self._target_position = position
                    else:
                        # Store the target position for smoother updates
                        self._target_position = position
                    
                    # Calculate elapsed time since last update
                    current_time = time.time()
                    elapsed = current_time - self._last_position_time
                    
                    # Only update at most 30 times per second for smoother appearance
                    if elapsed >= 1.0 / 30.0:
                        # Calculate how much to move toward the target position
                        # Higher smoothing factor = faster movement
                        smoothing_factor = min(elapsed * 5.0, 1.0)  # Adjust smoothing based on time passed
                        
                        # Interpolate between current and target position
                        new_position = self._last_position + (self._target_position - self._last_position) * smoothing_factor
                        
                        # Convert position to slider value (0-100)
                        slider_val = int(new_position * 100)
                        self.control_panel.timeline_slider.setValue(slider_val)
                        
                        # Update stored position
                        self._last_position = new_position
                        self._last_position_time = current_time
                    
                    # If the analyzer has bag duration info, update that in the control panel
                    if hasattr(self.analyzer, 'bag_duration') and self.analyzer.bag_duration > 0:
                        self.control_panel.bag_duration_seconds = self.analyzer.bag_duration
        except Exception as e:
            # Silently handle errors in the UI update
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
            
            if self.heatmap_view is not None:
                self.heatmap_view.reset_heatmap()
            
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
        if hasattr(self.heatmap_view, 'optimizer'):
            self.heatmap_view.optimizer.configure(
                min_time_interval=min_time_interval,
                max_time_interval=max_time_interval,
                change_threshold_percent=threshold_percent
            )
            self.status_bar.showMessage(
                f"Visualization performance set to: {min_time_interval}s min, "
                f"{max_time_interval}s max, {threshold_percent}% threshold", 
                3000
            )
    
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
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Radar Data", "", 
            "Radar Data Files (*.dat);;CSV Files (*.csv);;All Files (*)",
            options=options
        )
        
        if file_path:
            if hasattr(self, 'analyzer') and self.analyzer is not None:
                try:
                    # Use the analyzer to load the data
                    success = self.analyzer.load_data_from_file(file_path)
                    
                    if success:
                        self.status_bar.showMessage(f"Loaded data from {file_path}", 3000)
                        self.update_visualizations()
                    else:
                        QMessageBox.warning(self, "Error", "Failed to load data file.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
            else:
                QMessageBox.warning(self, "Error", "Analyzer not initialized.")
    
    def save_data_file(self):
        """Save radar data to a file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Radar Data", "", 
            "Radar Data Files (*.dat);;CSV Files (*.csv);;All Files (*)",
            options=options
        )
        
        if file_path:
            if hasattr(self, 'analyzer') and self.analyzer is not None:
                try:
                    # Use the analyzer to save the data
                    success = self.analyzer.save_data_to_file(file_path)
                    
                    if success:
                        self.status_bar.showMessage(f"Saved data to {file_path}", 3000)
                    else:
                        QMessageBox.warning(self, "Error", "Failed to save data file.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error saving data: {str(e)}")
            else:
                QMessageBox.warning(self, "Error", "Analyzer not initialized or no data to save.")