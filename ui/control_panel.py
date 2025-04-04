#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt-based control panel for radar analysis configuration.

This module provides control widgets for configuring radar data
collection, visualization, and analysis parameters.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QSlider, QComboBox, QLineEdit, QSpinBox, QCheckBox,
    QPushButton, QProgressBar, QFormLayout, QButtonGroup,
    QRadioButton, QFileDialog, QDoubleSpinBox, QTabWidget,
    QGridLayout, QFrame, QDialog, QListWidget, QListWidgetItem,
    QInputDialog, QDialogButtonBox, QMessageBox
)
from PyQt5.QtGui import QPixmap, QColor, QPainter, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize, QDateTime, QThread
import os
import json
from radar_analyzer.processing.data_processor import filter_points_in_circle, calculate_heatmap_size
from radar_analyzer.visualization.visualizer import update_plot, update_heatmap_display
from radar_analyzer.utils.ros_bag_handler import play_rosbag, record_rosbag, stop_rosbag
import time
import glob
from PyQt5.QtWidgets import QApplication
import numpy as np
import math
from datetime import datetime


class ControlPanel(QWidget):
    """
    Control panel for radar data collection and visualization settings.
    
    This widget provides UI controls for configuring data collection,
    visualization parameters, and analysis options.
    
    Signals:
        circle_distance_changed: Emitted when the circle distance changes.
        circle_radius_changed: Emitted when the circle radius changes.
        circle_angle_changed: Emitted when the circle angle changes.
        circle_toggled: Emitted when a circle is enabled/disabled.
        play_rosbag: Emitted when a ROS2 bag playback is started.
        record_rosbag: Emitted when ROS2 bag recording is started.
        stop_rosbag: Emitted when ROS2 bag playback/recording is stopped.
        visualize_pointcloud: Emitted when point cloud visualization is requested.
        start_collection: Emitted when data collection should start.
        stop_collection: Emitted when data collection should stop.
        reset_heatmap: Emitted when the heatmap should be reset.
        colormap_changed: Emitted when the colormap is changed.
        decay_factor_changed: Emitted when the decay factor is changed.
        visualization_mode_changed: Emitted when the visualization mode changes.
        noise_floor_changed: Emitted when the noise floor value changes.
        smoothing_changed: Emitted when the smoothing factor changes.
        add_roi: Emitted when an ROI should be added.
        clear_rois: Emitted when ROIs should be cleared.
        save_heatmap: Emitted when the heatmap should be saved.
        export_plot: Emitted when the scientific plot should be exported.
        generate_report: Emitted when a report should be generated.
    """
    
    # Define signals
    circle_distance_changed = pyqtSignal(int, float)  # (circle_index, distance)
    circle_radius_changed = pyqtSignal(int, float)    # (circle_index, radius)
    circle_angle_changed = pyqtSignal(int, float)     # (circle_index, angle)
    circle_toggled = pyqtSignal(int, bool)            # (circle_index, enabled)
    start_collection = pyqtSignal(str, str, int)
    stop_collection = pyqtSignal()
    reset_heatmap = pyqtSignal()
    colormap_changed = pyqtSignal(str)
    decay_factor_changed = pyqtSignal(float)
    visualization_mode_changed = pyqtSignal(str)
    noise_floor_changed = pyqtSignal(float)
    smoothing_changed = pyqtSignal(float)
    add_roi = pyqtSignal()
    clear_rois = pyqtSignal()
    save_heatmap = pyqtSignal()
    export_plot = pyqtSignal()
    generate_report = pyqtSignal()
    
    # ROS2 Bag and Point Cloud signals
    play_rosbag = pyqtSignal(str)  # Path to bag file
    record_rosbag = pyqtSignal(str, list, int)  # Path, topics, and duration in seconds to record
    stop_rosbag = pyqtSignal()  # Stop recording or playback
    timeline_position_changed = pyqtSignal(float)  # Bag playback position (0.0-1.0)
    visualize_pointcloud = pyqtSignal(str)  # Point cloud topic
    
    def __init__(self, parent=None):
        """
        Initialize the ControlPanel widget.
        
        Args:
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        
        # Store reference to main window
        self.main_window = parent
        
        # Default values
        self.collection_start_time = None
        self.collection_duration = 60
        self.last_point_count = 0
        self.collection_frames = 0
        self.point_update_counter = 0
        self.point_history = []
        self.bag_started_for_generation = False
        self.manual_stop_requested = False  # Track if collection was manually stopped
        
        # Initialize the state manager
        from ui.state_manager import ApplicationStateManager
        self.state_manager = ApplicationStateManager(self)
        
        # Set up progress update timer
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        
        # Create a folder to store application settings
        self.settings_dir = os.path.join(os.path.expanduser("~"), ".radar_analyzer")
        if not os.path.exists(self.settings_dir):
            os.makedirs(self.settings_dir, exist_ok=True)
        self.settings_file = os.path.join(self.settings_dir, "ui_settings.json")
        
        # Initialize status bar (must be before setup_ui)
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("color: #cccccc; font-style: italic;")
        self.status_timer = QTimer()
        self.status_timer.setSingleShot(True)
        self.status_timer.timeout.connect(self.clear_status)
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the widget UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Setup the control sections
        self.create_scatter_controls(main_layout)
        self.create_data_collection_controls(main_layout)
        self.create_rosbag_controls(main_layout)
        self.create_pointcloud_controls(main_layout)
        self.create_heatmap_controls(main_layout)
        self.create_analysis_controls(main_layout)
        
        # Add status bar at the bottom
        main_layout.addWidget(self.status_bar)
        
        # Set layout properties
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Initialize state of all UI elements
        self.initialize_ui_state()
    
    def initialize_ui_state(self):
        """Initialize the UI state based on current application state."""
        # Initialize ROS2 bag controls state
        if hasattr(self, 'rosbag_group'):
            # Check if we have an analyzer and it's in a recording/playing state
            if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'analyzer'):
                analyzer = self.main_window.analyzer
                
                if hasattr(analyzer, 'is_recording') and analyzer.is_recording:
                    self.rosbag_group.setTitle("ROS2 Bag Controls - Recording")
                    if hasattr(self, 'recording_status_label'):
                        self.recording_status_label.setText("Recording in Progress")
                        self.recording_status_label.setStyleSheet("font-weight: bold; color: #F44336;")
                elif hasattr(analyzer, 'is_playing') and analyzer.is_playing:
                    self.rosbag_group.setTitle("ROS2 Bag Controls - Playing")
                else:
                    self.rosbag_group.setTitle("ROS2 Bag Controls")
                    if hasattr(self, 'recording_status_label'):
                        self.recording_status_label.setText("Record Settings")
                        self.recording_status_label.setStyleSheet("font-weight: bold; color: #757575;")
            else:
                # No analyzer or not initialized yet, set default state
                self.rosbag_group.setTitle("ROS2 Bag Controls")
                if hasattr(self, 'recording_status_label'):
                    self.recording_status_label.setText("Record Settings")
                    self.recording_status_label.setStyleSheet("font-weight: bold; color: #757575;")
    
    def create_scatter_controls(self, parent_layout):
        """
        Create the scatter plot control section with multiple circle controls.
        
        Args:
            parent_layout: Parent layout to add widgets to.
        """
        scatter_group = QGroupBox("Sampling Circles")
        # Apply modern UI principles with clean layout and proper spacing
        scatter_layout = QVBoxLayout(scatter_group)
        scatter_layout.setSpacing(15)  # Increased spacing for better readability
        scatter_layout.setContentsMargins(15, 15, 15, 15)  # Consistent margins
        
        # Circle selection tabs - modern tab design
        self.circle_tabs = QTabWidget()
        self.circle_tabs.setMaximumHeight(200)  # Provide enough space for properly spaced controls
        self.circle_tabs.setStyleSheet("QTabBar::tab { height: 30px; min-width: 80px; }")
        
        # Create a tab for each circle - applying consistency principle
        circle_configs = [
            {'name': 'Primary', 'color': 'lime', 'distance': 5, 'angle': 0, 'enabled': True},
            {'name': 'Left', 'color': 'cyan', 'distance': 15, 'angle': -60, 'enabled': False},
            {'name': 'Right', 'color': 'yellow', 'distance': 25, 'angle': 60, 'enabled': False}
        ]
        
        self.circle_controls = []
        
        for i, config in enumerate(circle_configs):
            tab = QWidget()
            # Use a simplified two-column layout for better clarity and reduced clutter
            tab_layout = QGridLayout(tab)
            tab_layout.setContentsMargins(15, 20, 15, 20)  # More generous margins for breathing room
            tab_layout.setHorizontalSpacing(20)  # Wide spacing between columns
            tab_layout.setVerticalSpacing(15)  # Good spacing between rows
            
            # Circle enable checkbox - made more prominent
            enable_check = QCheckBox("Enable")
            enable_check.setChecked(config['enabled'])
            enable_check.setStyleSheet("QCheckBox { font-weight: bold; }")
            enable_check.stateChanged.connect(lambda state, idx=i: self.on_circle_toggle(idx, state))
            tab_layout.addWidget(enable_check, 0, 0, 1, 2)  # Span across both columns
            
            # Distance control - more intuitive labeling and layout
            dist_label = QLabel("Distance:")
            dist_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            distance_spin = QSpinBox()
            distance_spin.setRange(1, 35)
            distance_spin.setValue(config['distance'])
            distance_spin.setSuffix(" m")
            distance_spin.setMinimumWidth(100)  # Wider for better usability
            distance_spin.valueChanged.connect(lambda value, idx=i: self.on_circle_distance_changed(idx, value))
            
            # Organized layout with right-aligned labels
            tab_layout.addWidget(dist_label, 1, 0)
            tab_layout.addWidget(distance_spin, 1, 1)
            
            # Radius control
            radius_label = QLabel("Radius:")
            radius_slider = QSlider(Qt.Horizontal)
            radius_slider.setMinimumWidth(150)  # Wider slider for better control
            radius_slider.setRange(10, 300)  # 0.1 to 3.0 m (scale by 100)
            radius_slider.setValue(50)       # 0.5 m default
            radius_value = QLabel("0.5 m")
            radius_value.setMinimumWidth(50)  # Fixed width for value
            radius_slider.valueChanged.connect(lambda value, idx=i, lbl=radius_value: self.on_circle_radius_changed(idx, value, lbl))
            
            # Place each widget in grid layout with proper spacing
            tab_layout.addWidget(radius_label, 0, 1)
            tab_layout.addWidget(radius_slider, 0, 2)
            tab_layout.addWidget(radius_value, 0, 3)
            
            # Angle control
            angle_label = QLabel("Angle:")
            angle_spin = QSpinBox()
            angle_spin.setRange(-90, 90)
            angle_spin.setValue(config['angle'])
            angle_spin.setSuffix("°")
            angle_spin.setMinimumWidth(80)  # Ensure enough width
            angle_spin.valueChanged.connect(lambda value, idx=i: self.on_circle_angle_changed(idx, value))
            
            # Place each widget in grid layout with proper spacing
            tab_layout.addWidget(angle_label, 1, 2)
            tab_layout.addWidget(angle_spin, 1, 3)
            
            # Store controls for this circle
            self.circle_controls.append({
                'enable': enable_check,
                'distance': distance_spin,
                'radius': radius_slider,
                'radius_label': radius_value,
                'angle': angle_spin
            })
            
            # Add tab with color styling
            self.circle_tabs.addTab(tab, config['name'])
            
            # Apply color styling to tab
            self.circle_tabs.setTabIcon(i, self.create_color_icon(config['color']))
        
        scatter_layout.addWidget(self.circle_tabs)
        
        # Add global controls with proper spacing
        global_controls = QHBoxLayout()
        global_controls.setSpacing(15)  # More spacing between buttons
        global_controls.setContentsMargins(5, 10, 5, 5)  # Add some top margin
        
        add_all_button = QPushButton("Enable All")
        add_all_button.setMinimumHeight(30)  # Taller buttons
        add_all_button.clicked.connect(self.enable_all_circles)
        global_controls.addWidget(add_all_button)
        
        clear_all_button = QPushButton("Disable All")
        clear_all_button.setMinimumHeight(30)  # Taller buttons
        clear_all_button.clicked.connect(self.disable_all_circles)
        global_controls.addWidget(clear_all_button)
        
        scatter_layout.addLayout(global_controls)
        
        # Add group to parent layout
        parent_layout.addWidget(scatter_group)
    
    def create_color_icon(self, color_name):
        """Create a colored icon for tab display."""
        # Map color names to QColor
        color_map = {
            'lime': QColor(50, 205, 50),
            'cyan': QColor(0, 255, 255),
            'yellow': QColor(255, 255, 0),
            'red': QColor(255, 0, 0)
        }
        
        # Get color or default to white
        color = color_map.get(color_name, QColor(255, 255, 255))
        
        # Create pixmap and fill with color
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        
        return QIcon(pixmap)
    
    def create_data_collection_controls(self, parent_layout):
        """
        Create the data collection control section.
        
        Args:
            parent_layout: Parent layout to add widgets to.
        """
        collection_group = QGroupBox("Data Collection")
        collection_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        collection_layout = QHBoxLayout(collection_group)
        collection_layout.setContentsMargins(15, 20, 15, 15)  # More generous margins
        collection_layout.setSpacing(15)  # Better spacing between elements
        
        # Config and target distance controls with improved section header
        params_group = QGroupBox("Configuration")
        params_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                color: #2196F3;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(10)  # Increase spacing between form rows
        params_layout.setContentsMargins(15, 20, 15, 15)
        params_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Better input styling for config entry
        self.config_entry = QLineEdit("default_config")
        self.config_entry.setMinimumHeight(30)
        self.config_entry.setStyleSheet("""
            QLineEdit {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: #FFFFFF;
                color: #212121;
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
        """)
        config_label = QLabel("Config:")
        config_label.setStyleSheet("font-weight: bold;")
        params_layout.addRow(config_label, self.config_entry)
        
        # Better styling for target dropdown
        self.target_combo = QComboBox()
        self.target_combo.addItems([str(d) for d in range(5, 40, 5)])
        self.target_combo.setCurrentIndex(0)
        self.target_combo.setMinimumHeight(30)
        self.target_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: #FFFFFF;
                color: #212121;
                font-weight: bold;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox::down-arrow {
                image: url(/home/zen/Pictures/Radar_stuff/icons/dropdown.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #BDBDBD;
                background-color: #F5F5F5;
                color: #212121;
                selection-background-color: #E0E0E0;
                selection-color: #212121;
                padding: 2px;
            }
        """)
        target_label = QLabel("Target Distance:")
        target_label.setStyleSheet("font-weight: bold;")
        params_layout.addRow(target_label, self.target_combo)
        
        # Better styling for duration spinner
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(10, 300)
        self.duration_spin.setValue(60)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setMinimumHeight(30)
        self.duration_spin.setStyleSheet("""
            QSpinBox {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: #FFFFFF;
                color: #212121;
                font-weight: bold;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                border-radius: 2px;
                background-color: #E0E0E0;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #BDBDBD;
            }
        """)
        duration_label = QLabel("Collection Duration:")
        duration_label.setStyleSheet("font-weight: bold;")
        params_layout.addRow(duration_label, self.duration_spin)
        
        collection_layout.addWidget(params_group)
        
        # Create a visual separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #E0E0E0;")
        collection_layout.addWidget(separator)
        
        # Collection action buttons in a group with improved header
        action_group = QGroupBox("Actions")
        action_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                color: #2196F3;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        buttons_layout = QVBoxLayout(action_group)
        buttons_layout.setSpacing(12)  # Increased spacing between buttons
        buttons_layout.setContentsMargins(15, 20, 15, 15)
        
        # More prominent, better styled action buttons
        button_style = """
            QPushButton {
                background-color: #F5F5F5;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                color: #424242;
                font-weight: bold;
                padding: 8px;
                text-align: left;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
                border: 1px solid #9E9E9E;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                color: #9E9E9E;
            }
        """
        
        self.start_button = QPushButton("  Start Collection")
        self.start_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/start.png"))
        self.start_button.setIconSize(QSize(24, 24))
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setStyleSheet(button_style + """
            QPushButton {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        self.start_button.clicked.connect(self.on_start_collection)
        buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("  Stop Collection")
        self.stop_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/stop.png"))
        self.stop_button.setIconSize(QSize(24, 24))
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.setStyleSheet(button_style + """
            QPushButton {
                background-color: #F44336;
                color: white;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)
        self.stop_button.clicked.connect(self.on_stop_collection)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        self.report_button = QPushButton("  Generate Report")
        self.report_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/report.png"))
        self.report_button.setIconSize(QSize(24, 24))
        self.report_button.setCursor(Qt.PointingHandCursor)
        self.report_button.setStyleSheet(button_style + """
            QPushButton {
                background-color: #2196F3;
                color: white;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.report_button.clicked.connect(self.on_generate_report)
        buttons_layout.addWidget(self.report_button)
        
        collection_layout.addWidget(action_group)
        
        # Create another visual separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet("background-color: #E0E0E0;")
        collection_layout.addWidget(separator2)
        
        # Status and progress in a group with improved header
        status_group = QGroupBox("Status")
        status_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                color: #2196F3;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(10)  # Increased spacing
        status_layout.setContentsMargins(15, 20, 15, 15)
        
        # Better styled status label with icon
        status_hlayout = QHBoxLayout()
        status_icon = QLabel()
        status_icon.setObjectName("status_icon")  # Set object name for later reference
        status_pixmap = QPixmap(16, 16)
        status_pixmap.fill(QColor(76, 175, 80))  # Green color for "Ready"
        status_icon.setPixmap(status_pixmap)
        status_hlayout.addWidget(status_icon)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel { 
                font-weight: bold; 
                color: #424242;
                font-size: 14px;
            }
        """)
        status_hlayout.addWidget(self.status_label)
        status_hlayout.addStretch()
        status_layout.addLayout(status_hlayout)
        
        # Modern progress bar
        progress_label = QLabel("Collection Progress:")
        progress_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        status_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                text-align: center;
                background-color: #F5F5F5;
                color: #000000;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
        """)
        status_layout.addWidget(self.progress_bar)
        
        # Collection Stats
        stats_label = QLabel("Collection Stats:")
        stats_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        status_layout.addWidget(stats_label)
        
        self.points_collected_label = QLabel("Density: 0 pts/frame")
        self.points_collected_label.setStyleSheet("font-family: monospace;")
        status_layout.addWidget(self.points_collected_label)
        
        self.collection_time_label = QLabel("Time: 00:00")
        self.collection_time_label.setStyleSheet("font-family: monospace;")
        status_layout.addWidget(self.collection_time_label)
        
        collection_layout.addWidget(status_group)
        
        # Add to parent layout with spacing
        parent_layout.addWidget(collection_group)
        parent_layout.addSpacing(15)  # Space before next section
    
    def create_heatmap_controls(self, parent_layout):
        """
        Create the heatmap control section.
        
        Args:
            parent_layout: Parent layout to add widgets to.
        """
        heatmap_group = QGroupBox("Heatmap Controls")
        heatmap_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        heatmap_layout = QHBoxLayout(heatmap_group)
        heatmap_layout.setContentsMargins(15, 20, 15, 15)  # More generous margins
        
        # Reset and visualization controls - with improved section header
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(12)  # Increased spacing for better readability
        
        # Add control parameters header
        controls_header = QLabel("Display Settings")
        controls_header.setStyleSheet("font-weight: bold; color: #2196F3;")
        controls_header.setAlignment(Qt.AlignLeft)
        controls_layout.addWidget(controls_header)
        controls_layout.addSpacing(5)  # Spacer after header
        
        # Colormap and reset controls - improved layout
        reset_layout = QHBoxLayout()
        reset_layout.setSpacing(10)
        
        # More prominent reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/reset.png"))
        self.reset_button.setIconSize(QSize(24, 24))
        self.reset_button.setMinimumHeight(30)
        self.reset_button.setCursor(Qt.PointingHandCursor)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.reset_button.clicked.connect(self.on_reset_heatmap)
        reset_layout.addWidget(self.reset_button)
        
        # Better labeled color map selector
        map_label = QLabel("Color Map:")
        map_label.setStyleSheet("font-weight: bold;")
        reset_layout.addWidget(map_label)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "jet"])
        self.colormap_combo.setCurrentIndex(0)
        self.colormap_combo.setMinimumHeight(30)
        self.colormap_combo.setMinimumWidth(100)
        self.colormap_combo.setStyleSheet("QComboBox { padding: 4px; }")
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        reset_layout.addWidget(self.colormap_combo)
        
        controls_layout.addLayout(reset_layout)
        
        # Decay factor controls - improved with better feedback
        decay_layout = QHBoxLayout()
        decay_layout.setSpacing(8)
        
        decay_label = QLabel("Decay:")
        decay_label.setStyleSheet("font-weight: bold;")
        decay_label.setMinimumWidth(50)
        decay_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        decay_layout.addWidget(decay_label)
        
        self.decay_slider = QSlider(Qt.Horizontal)
        self.decay_slider.setRange(800, 999)  # 0.8 to 0.999 (scale by 1000)
        self.decay_slider.setValue(980)       # 0.98 default
        self.decay_slider.setMinimumHeight(20)
        self.decay_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #E0E0E0;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.decay_slider.valueChanged.connect(self.on_decay_changed)
        decay_layout.addWidget(self.decay_slider)
        
        self.decay_value = QLabel("0.980")
        self.decay_value.setStyleSheet("color: #2196F3; font-family: monospace;")
        self.decay_value.setMinimumWidth(50)  # Ensure consistent width
        decay_layout.addWidget(self.decay_value)
        
        controls_layout.addLayout(decay_layout)
        
        heatmap_layout.addLayout(controls_layout)
        
        # Visualization mode controls - improved with visual section header
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(10)
        
        # Add mode header
        mode_header = QLabel("Display Mode")
        mode_header.setStyleSheet("font-weight: bold; color: #2196F3;")
        mode_header.setAlignment(Qt.AlignLeft)
        mode_layout.addWidget(mode_header)
        mode_layout.addSpacing(5)  # Spacer after header
        
        self.vis_mode_group = QButtonGroup(self)
        
        # Better styled radio buttons
        radio_style = "QRadioButton { min-height: 25px; }"
        
        heat_radio = QRadioButton("Heat")
        heat_radio.setChecked(True)
        heat_radio.setStyleSheet(radio_style)
        heat_radio.setCursor(Qt.PointingHandCursor)
        heat_radio.toggled.connect(lambda checked: checked and self.on_vis_mode_changed("heatmap"))
        self.vis_mode_group.addButton(heat_radio)
        mode_layout.addWidget(heat_radio)
        
        contour_radio = QRadioButton("Contour")
        contour_radio.setStyleSheet(radio_style)
        contour_radio.setCursor(Qt.PointingHandCursor)
        contour_radio.toggled.connect(lambda checked: checked and self.on_vis_mode_changed("contour"))
        self.vis_mode_group.addButton(contour_radio)
        mode_layout.addWidget(contour_radio)
        
        combined_radio = QRadioButton("Combined")
        combined_radio.setStyleSheet(radio_style)
        combined_radio.setCursor(Qt.PointingHandCursor)
        combined_radio.toggled.connect(lambda checked: checked and self.on_vis_mode_changed("combined"))
        self.vis_mode_group.addButton(combined_radio)
        mode_layout.addWidget(combined_radio)
        
        heatmap_layout.addLayout(mode_layout)
        
        # Noise and smoothing controls - with improved section header
        params_layout = QVBoxLayout()
        params_layout.setSpacing(12)
        
        # Add parameters header
        params_header = QLabel("Parameters")
        params_header.setStyleSheet("font-weight: bold; color: #2196F3;")
        params_header.setAlignment(Qt.AlignLeft)
        params_layout.addWidget(params_header)
        params_layout.addSpacing(5)  # Spacer after header
        
        # Noise floor control - improved with better visual feedback
        noise_layout = QHBoxLayout()
        noise_layout.setSpacing(8)
        
        noise_label = QLabel("Noise:")
        noise_label.setStyleSheet("font-weight: bold;")
        noise_label.setMinimumWidth(50)
        noise_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        noise_layout.addWidget(noise_label)
        
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(1, 20)  # 0.01 to 0.2 (scale by 100)
        self.noise_slider.setValue(5)      # 0.05 default
        self.noise_slider.setMinimumHeight(20)
        self.noise_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #E0E0E0;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.noise_slider.valueChanged.connect(self.on_noise_changed)
        noise_layout.addWidget(self.noise_slider)
        
        self.noise_value = QLabel("0.05")
        self.noise_value.setStyleSheet("color: #2196F3; font-family: monospace;")
        self.noise_value.setMinimumWidth(50)  # Ensure consistent width
        noise_layout.addWidget(self.noise_value)
        
        params_layout.addLayout(noise_layout)
        
        # Smoothing control - improved with consistent styling
        smooth_layout = QHBoxLayout()
        smooth_layout.setSpacing(8)
        
        smooth_label = QLabel("Smooth:")
        smooth_label.setStyleSheet("font-weight: bold;")
        smooth_label.setMinimumWidth(50)
        smooth_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        smooth_layout.addWidget(smooth_label)
        
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(5, 50)  # 0.5 to 5.0 (scale by 10)
        self.smooth_slider.setValue(20)     # 2.0 default
        self.smooth_slider.setMinimumHeight(20)
        self.smooth_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #E0E0E0;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.smooth_slider.valueChanged.connect(self.on_smoothing_changed)
        smooth_layout.addWidget(self.smooth_slider)
        
        self.smooth_value = QLabel("2.0")
        self.smooth_value.setStyleSheet("color: #2196F3; font-family: monospace;")
        self.smooth_value.setMinimumWidth(50)  # Ensure consistent width
        smooth_layout.addWidget(self.smooth_value)
        
        params_layout.addLayout(smooth_layout)
        
        # Add vertical separator between sections
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.VLine)
        separator1.setFrameShadow(QFrame.Sunken)
        separator1.setStyleSheet("background-color: #E0E0E0;")
        
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet("background-color: #E0E0E0;")
        
        # Add all layouts with proper separators and spacing
        heatmap_layout.addLayout(controls_layout, 1)
        heatmap_layout.addWidget(separator1)
        heatmap_layout.addLayout(mode_layout, 1)
        heatmap_layout.addWidget(separator2)
        heatmap_layout.addLayout(params_layout, 1)
        
        # Add to parent layout with spacing
        parent_layout.addWidget(heatmap_group)
        parent_layout.addSpacing(15)  # Space before next section
    
    def create_rosbag_controls(self, parent_layout):
        """
        Create the ROS2 bag playback and recording controls.
        
        Args:
            parent_layout: Parent layout to add widgets to.
        """
        self.rosbag_group = QGroupBox("ROS2 Bag Controls")
        self.rosbag_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        rosbag_layout = QHBoxLayout(self.rosbag_group)
        rosbag_layout.setContentsMargins(15, 20, 15, 15)  # More generous margins
        
        # Left side - Playback controls with clear section header
        playback_layout = QVBoxLayout()
        playback_layout.setSpacing(12)  # Increased spacing for better readability
        
        # Store the last used directory as an instance variable
        self.last_directory = ""
        
        # Add playback header
        playback_header = QLabel("Playback")
        playback_header.setStyleSheet("font-weight: bold; color: #2196F3;")
        playback_header.setAlignment(Qt.AlignLeft)
        playback_layout.addWidget(playback_header)
        
        # Bag file selection - improved with better spacing and visual cues
        bag_select_layout = QHBoxLayout()
        bag_select_layout.setSpacing(8)
        
        self.bag_path_edit = QLineEdit()
        self.bag_path_edit.setPlaceholderText("Select ROS2 bag file...")
        self.bag_path_edit.setToolTip("Path to ROS2 bag file for playback")
        self.bag_path_edit.setMinimumHeight(28)  # Slightly taller for better touch targets
        
        # Discover button to find available ROS2 bags
        refresh_button = QPushButton()
        refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_button.setToolTip("Discover ROS2 bags in common directories")
        refresh_button.setMinimumHeight(28)
        refresh_button.setMaximumWidth(32)
        refresh_button.setCursor(Qt.PointingHandCursor)
        refresh_button.clicked.connect(self.discover_rosbags)
        
        bag_select_button = QPushButton("Browse")
        bag_select_button.setMinimumHeight(28)
        bag_select_button.setCursor(Qt.PointingHandCursor)  # Change cursor on hover
        bag_select_button.setIcon(QIcon.fromTheme("folder-open"))
        bag_select_button.clicked.connect(self.on_select_bagfile)
        
        bag_select_layout.addWidget(self.bag_path_edit, 5)  # Proportional sizing
        bag_select_layout.addWidget(refresh_button, 1)      # Refresh button
        bag_select_layout.addWidget(bag_select_button, 2)   # Browse button
        
        playback_layout.addLayout(bag_select_layout)
        playback_layout.addSpacing(8)  # Add spacing between sections
        
        # Timeline slider with improved visual design
        timeline_layout = QVBoxLayout()
        timeline_layout.setSpacing(6)
        
        # Timeline header with clearer labeling
        timeline_header = QHBoxLayout()
        timeline_label = QLabel("Timeline:")
        timeline_label.setStyleSheet("font-weight: bold;")
        timeline_header.addWidget(timeline_label)
        timeline_header.addStretch(1)
        
        self.timestamp_label = QLabel("00:00 / 00:00")
        self.timestamp_label.setMinimumWidth(100)  # Ensure enough space for timestamp
        self.timestamp_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # Right aligned
        self.timestamp_label.setStyleSheet("color: #4285F4; font-family: monospace;")
        timeline_header.addWidget(self.timestamp_label)
        timeline_layout.addLayout(timeline_header)
        
        # Timeline slider with better visual feedback
        # Track whether the user is dragging the timeline slider
        self.timeline_dragging = False
        
        # Create a custom slider that can track mouse press/release events
        class TimelineSlider(QSlider):
            def __init__(self, orientation, parent=None):
                super().__init__(orientation, parent)
                self.parent_panel = parent
                self.setPageStep(5)  # Smaller page step for smoother navigation
                self.setSingleStep(1)  # Smaller single step for finer control
                # Track whether we're currently updating to avoid feedback loops
                self.is_programmatic_update = False
                
            def mousePressEvent(self, event):
                if self.parent_panel:
                    self.parent_panel.timeline_dragging = True
                super().mousePressEvent(event)
            
            def mouseReleaseEvent(self, event):
                if self.parent_panel:
                    # Delay setting dragging to False to avoid immediate overwrite by signal
                    QTimer.singleShot(150, lambda: setattr(self.parent_panel, 'timeline_dragging', False))
                super().mouseReleaseEvent(event)
                
            def setValue(self, value):
                """Override setValue to track programmatic updates."""
                self.is_programmatic_update = True
                super().setValue(value)
                self.is_programmatic_update = False
        
        self.timeline_slider = TimelineSlider(Qt.Horizontal, self)
        self.timeline_slider.setEnabled(False)  # Initially disabled until bag is loaded
        self.timeline_slider.setRange(0, 100)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setMinimumHeight(24)  # Taller for easier interaction
        self.timeline_slider.setTickPosition(QSlider.TicksBelow)
        self.timeline_slider.setTickInterval(10)  # Ticks at 10% intervals
        self.timeline_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #E0E0E0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4285F4;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)
        self.timeline_slider.valueChanged.connect(self.on_timeline_changed)
        timeline_layout.addWidget(self.timeline_slider)
        
        # Add "Generate from Bag" checkbox for data collection during playback
        self.generate_from_bag_check = QCheckBox("Generate from bag")
        self.generate_from_bag_check.setEnabled(False)  # Initially disabled until bag is loaded
        self.generate_from_bag_check.setStyleSheet("""
            QCheckBox {
                color: #4CAF50;
                font-weight: bold;
                margin-top: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #BDBDBD;
                border-radius: 3px;
            }
        """)
        self.generate_from_bag_check.setToolTip("Enable to collect data from bag playback")
        self.generate_from_bag_check.stateChanged.connect(self.on_generate_from_bag_changed)
        playback_layout.addWidget(self.generate_from_bag_check)
        
        playback_layout.addLayout(timeline_layout)
        
        # Playback controls - larger buttons with better visual emphasis
        playback_buttons = QHBoxLayout()
        playback_buttons.setSpacing(10)  # More space between buttons
        
        self.play_button = QPushButton("Play")
        self.play_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/playback.png"))
        self.play_button.setIconSize(QSize(24, 24))
        self.play_button.setMinimumHeight(36)  # Taller button for easier interaction
        self.play_button.setCursor(Qt.PointingHandCursor)  # Change cursor on hover
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.play_button.clicked.connect(self.on_play_rosbag)
        playback_buttons.addWidget(self.play_button)
        
        self.stop_playback_button = QPushButton("Stop")
        self.stop_playback_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/stop.png"))
        self.stop_playback_button.setIconSize(QSize(24, 24))
        self.stop_playback_button.setMinimumHeight(36)  # Consistent height
        self.stop_playback_button.setCursor(Qt.PointingHandCursor)  # Change cursor on hover
        self.stop_playback_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.stop_playback_button.clicked.connect(self.on_stop_rosbag)
        self.stop_playback_button.setEnabled(False)
        playback_buttons.addWidget(self.stop_playback_button)
        
        # Add the playback buttons layout to the main playback layout
        playback_layout.addLayout(playback_buttons)
        
        # Right side - Recording controls with clear section header
        recording_layout = QVBoxLayout()
        recording_layout.setSpacing(12)  # Increased spacing for better readability
        
        # Add recording header for visual separation - using a status label that will update
        self.recording_status_label = QLabel("Record Settings")
        self.recording_status_label.setStyleSheet("font-weight: bold; color: #757575;")
        self.recording_status_label.setAlignment(Qt.AlignLeft)
        recording_layout.addWidget(self.recording_status_label)
        
        # Record path selection - improved with better visual cues
        record_path_layout = QHBoxLayout()
        record_path_layout.setSpacing(8)
        
        self.record_path_edit = QLineEdit()
        self.record_path_edit.setPlaceholderText("Select where to save ROS2 bag...")
        self.record_path_edit.setToolTip("Path where recorded ROS2 bag will be saved")
        self.record_path_edit.setMinimumHeight(28)  # Slightly taller for better touch targets
        
        record_path_button = QPushButton("Browse")
        record_path_button.setMinimumHeight(28)
        record_path_button.setCursor(Qt.PointingHandCursor)  # Change cursor on hover
        record_path_button.setIcon(QIcon.fromTheme("folder-open"))
        record_path_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border: 1px solid #aaa;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        record_path_button.clicked.connect(self.on_select_record_path)
        
        record_path_layout.addWidget(self.record_path_edit, 3)  # Proportional sizing
        record_path_layout.addWidget(record_path_button, 1)
        
        # Recording duration controls
        duration_layout = QHBoxLayout()
        duration_layout.setSpacing(8)
        
        duration_label = QLabel("Duration:")
        duration_label.setStyleSheet("font-weight: bold;")
        duration_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        duration_label.setMinimumWidth(50)
        
        # Create time unit selection combobox
        self.duration_unit_combo = QComboBox()
        self.duration_unit_combo.addItems(["sec", "min"])
        self.duration_unit_combo.setCurrentIndex(0)  # Default to seconds
        self.duration_unit_combo.setMaximumWidth(60)
        self.duration_unit_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 2px 5px;
                background-color: #FFFFFF;
                color: #212121;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #BDBDBD;
                background-color: #F5F5F5;
                color: #212121;
                selection-background-color: #E0E0E0;
                selection-color: #212121;
            }
        """)
        self.duration_unit_combo.currentIndexChanged.connect(self.on_duration_unit_changed)
        
        self.record_duration_spin = QSpinBox()
        self.record_duration_spin.setRange(0, 3600)  # 0 means no time limit, up to 3600 seconds
        self.record_duration_spin.setValue(0)
        self.record_duration_spin.setSpecialValueText("No limit")  # Show "No limit" when value is 0
        self.record_duration_spin.setToolTip("Recording duration (0 = no time limit)")
        self.record_duration_spin.setMinimumHeight(28)
        self.record_duration_spin.setStyleSheet("""
            QSpinBox {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: #FFFFFF;
                color: #212121;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                border-radius: 2px;
                background-color: #E0E0E0;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #BDBDBD;
            }
        """)
        self.record_duration_spin.valueChanged.connect(self.on_record_duration_changed)
        
        duration_layout.addWidget(duration_label, 1)
        duration_layout.addWidget(self.record_duration_spin, 3)
        duration_layout.addWidget(self.duration_unit_combo, 1)
        recording_layout.addLayout(duration_layout)
        
        # Add estimated bag size label
        self.bag_size_label = QLabel("Est. size: Unknown (unlimited duration)")
        self.bag_size_label.setStyleSheet("color: #777777; font-style: italic;")
        self.bag_size_label.setAlignment(Qt.AlignRight)
        recording_layout.addWidget(self.bag_size_label)
        recording_layout.addSpacing(8)
        
        # Topics selection - improved with clearer labeling
        topics_layout = QHBoxLayout()
        topics_layout.setSpacing(8)
        
        topics_label = QLabel("Topics:")
        topics_label.setStyleSheet("font-weight: bold;")
        topics_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        topics_label.setMinimumWidth(50)  # Ensure consistent alignment with other labels
        
        self.topics_edit = QLineEdit("/radar/data,/radar/status")
        self.topics_edit.setToolTip("Comma-separated list of ROS2 topics to record")
        self.topics_edit.setMinimumHeight(28)  # Consistent height with other inputs
        
        topics_layout.addWidget(topics_label, 1)
        topics_layout.addWidget(self.topics_edit, 3)
        recording_layout.addLayout(topics_layout)
        recording_layout.addSpacing(10)  # Visual separation before buttons
        
        # Record buttons - larger with better visual emphasis
        record_buttons = QHBoxLayout()
        record_buttons.setSpacing(10)  # More space between buttons
        
        self.record_button = QPushButton("Record")
        self.record_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/record.png"))
        self.record_button.setIconSize(QSize(24, 24))
        self.record_button.setMinimumHeight(36)  # Taller button for easier interaction
        self.record_button.setCursor(Qt.PointingHandCursor)  # Change cursor on hover
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.record_button.clicked.connect(self.on_record_rosbag)
        record_buttons.addWidget(self.record_button)
        
        self.stop_record_button = QPushButton("Stop")
        self.stop_record_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/stop.png"))
        self.stop_record_button.setIconSize(QSize(24, 24))
        self.stop_record_button.setMinimumHeight(36)  # Consistent height
        self.stop_record_button.setCursor(Qt.PointingHandCursor)  # Change cursor on hover
        self.stop_record_button.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        self.stop_record_button.clicked.connect(self.on_stop_rosbag)
        self.stop_record_button.setEnabled(False)
        record_buttons.addWidget(self.stop_record_button)
        
        recording_layout.addLayout(record_buttons)
        
        # Add layouts to main layout
        rosbag_layout.addLayout(playback_layout)
        rosbag_layout.addLayout(recording_layout)
        
        # Add to parent layout
        parent_layout.addWidget(self.rosbag_group)
    
    def create_pointcloud_controls(self, parent_layout):
        """
        Create the point cloud visualization controls.
        
        Args:
            parent_layout: Parent layout to add widgets to.
        """
        pointcloud_group = QGroupBox("Point Cloud Visualization")
        pointcloud_layout = QHBoxLayout(pointcloud_group)
        
        # Point cloud topic selection
        self.pcl_topic_combo = QComboBox()
        self.pcl_topic_combo.addItems(["/radar/pointcloud", "/lidar/points", "/camera/depth/points"])
        pointcloud_layout.addWidget(QLabel("Topic:"))
        pointcloud_layout.addWidget(self.pcl_topic_combo)
        
        # Visualization button
        self.pcl_visualize_button = QPushButton("Visualize")
        self.pcl_visualize_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/pointcloud.png"))
        self.pcl_visualize_button.setIconSize(QSize(24, 24))
        self.pcl_visualize_button.clicked.connect(self.on_visualize_pointcloud)
        pointcloud_layout.addWidget(self.pcl_visualize_button)
        
        # Options dropdown for visualization type
        self.pcl_viz_type = QComboBox()
        self.pcl_viz_type.addItems(["Points", "Heatmap", "Surface"])
        pointcloud_layout.addWidget(QLabel("View:"))
        pointcloud_layout.addWidget(self.pcl_viz_type)
        
        # Add to parent layout
        parent_layout.addWidget(pointcloud_group)
    
    def on_select_bagfile(self):
        """Open file dialog to select a ROS2 bag file."""
        # Get the last used directory or use home directory as default
        last_dir = self.last_used_directory() or os.path.expanduser("~")
        
        # Create a non-modal file dialog for better UI responsiveness
        dialog = QFileDialog(self, "Select ROS2 Bag File", last_dir,
                           "ROS2 Bag Files (*.db3 *.mcap);;All Files (*)")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setOptions(QFileDialog.DontUseNativeDialog | QFileDialog.ReadOnly)
        
        # Make the dialog more efficient
        dialog.setMinimumSize(800, 500)  # Larger dialog for better browsing
        
        if dialog.exec_() == QFileDialog.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                self.bag_path_edit.setText(file_path)
                self.save_last_used_directory(os.path.dirname(file_path))
                return True
        return False
    
    def on_select_record_path(self):
        """Open dialog to select where to save the ROS2 bag."""
        # Get the last used directory or use home directory as default
        last_dir = self.last_used_directory() or os.path.expanduser("~")
        
        # Create a default filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_bag_name = f"radar_data_{timestamp}"
        
        # Create a non-modal directory dialog for better UI responsiveness
        dialog = QFileDialog(self, "Select Location to Save ROS2 Bag", last_dir)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setMinimumSize(800, 500)  # Larger dialog for better browsing
        
        if dialog.exec_() == QFileDialog.Accepted:
            selected_dirs = dialog.selectedFiles()
            if selected_dirs:
                base_dir = selected_dirs[0]
                
                # Suggest a full path including the timestamped subdirectory
                full_path = os.path.join(base_dir, default_bag_name)
                
                # Ask if the user wants to use the suggested subdirectory
                from PyQt5.QtWidgets import QInputDialog
                bag_name, ok = QInputDialog.getText(
                    self,
                    "Bag Name",
                    "Enter a name for the ROS2 bag directory:",
                    text=default_bag_name
                )
                
                if ok and bag_name:
                    # Create the full path with the user's chosen name
                    full_path = os.path.join(base_dir, bag_name)
                
                # Update the UI and save the last used directory
                self.record_path_edit.setText(full_path)
                self.save_last_used_directory(base_dir)
                
                # Try to create the directory now
                try:
                    os.makedirs(full_path, exist_ok=True)
                    self.set_status(f"Ready to record to {os.path.basename(full_path)}")
                except Exception as e:
                    self.set_status(f"Error creating directory: {e}")
                
                return True
        return False
    
    def on_generate_from_bag_changed(self, state):
        """Handle changes to the 'Generate from Bag' checkbox."""
        is_checked = state == Qt.Checked
        
        # Prevent rapid toggling of the checkbox
        self.generate_from_bag_check.setEnabled(False)
        
        # Immediate UI feedback
        QApplication.processEvents()  # Force UI update
        self.set_status(f"{'Enabling' if is_checked else 'Disabling'} data generation from bag...")
        
        # Re-enable after a short delay to prevent accidental clicks
        QTimer.singleShot(500, lambda: self.generate_from_bag_check.setEnabled(True))
        
        # Update the UI to show the checkbox state
        if is_checked:
            print("Generate from bag enabled, preparing to start playback...")
            
            # First, make sure any previous data collection is stopped
            if self.main_window and self.main_window.analyzer and hasattr(self.main_window.analyzer, 'collecting_data'):
                if self.main_window.analyzer.collecting_data:
                    print("Stopping previous data collection before starting new one")
                    # Use try/except in case the signal emission fails
                    try:
                        self.stop_collection.emit()
                        # Wait to ensure collection stops before proceeding
                        print("Waiting for previous collection to stop...")
                        QApplication.processEvents()
                        QTimer.singleShot(500, lambda: self._enable_generate_from_bag())
                        return
                    except Exception as e:
                        print(f"Error stopping previous collection: {e}")
            
            # Acquire bag file path
            bag_path = self.bag_path_edit.text().strip()
            if not bag_path:
                self.set_status("Please select a bag file first")
                self.generate_from_bag_check.setChecked(False)
                return
            
            # Start playing the bag automatically if not already playing
            if self.main_window and self.main_window.analyzer:
                if hasattr(self.main_window.analyzer, 'is_playing'):
                    if not self.main_window.analyzer.is_playing:
                        # Set flag to track that bag was started for generation
                        self.bag_started_for_generation = True
                        
                        # Play bag once (no loop) for data generation
                        print(f"Starting bag playback without looping: {bag_path}")
                        # First, stop any existing playback to ensure clean state
                        try:
                            self.stop_rosbag.emit()
                            # Small wait to ensure playback is stopped
                            QApplication.processEvents()
                            time.sleep(0.2)
                        except Exception as e:
                            print(f"Error stopping existing playback: {e}")
                        
                        # Now start the new playback
                        success = self.play_bag_with_options(bag_path, loop=False)
                        
                        if success:
                            # Small delay to ensure playback starts before collection
                            print("Scheduling data collection to start in 1 second...")
                            QTimer.singleShot(1000, self._start_collection_from_bag)
                        else:
                            self.set_status("Failed to start bag playback")
                            self.generate_from_bag_check.setChecked(False)
                    else:  # Bag is already playing
                        # Set flag that bag is being used for generation
                        print("Bag is already playing, reusing for data generation")
                        self.bag_started_for_generation = True
                        
                        # If bag is already playing, restart it without looping
                        if self.bag_path_edit.text().strip():
                            # Stop current playback
                            try:
                                self.stop_rosbag.emit()
                                # Wait for playback to stop completely before restarting
                                print("Stopping current playback to restart without looping...")
                                QApplication.processEvents()
                                QTimer.singleShot(500, lambda: self._restart_bag_and_collection())
                            except Exception as e:
                                print(f"Error stopping playback: {e}")
                        else:
                            # If bag is already playing but we don't have a path, start collection immediately
                            print("Starting collection from currently playing bag...")
                            self._start_collection_from_bag()
                else:
                    self.set_status("Analyzer doesn't support playback")
                    self.generate_from_bag_check.setChecked(False)
            else:
                self.set_status("Analyzer not available")
                self.generate_from_bag_check.setChecked(False)
        else:  # Checkbox unchecked - disable data generation
            self.bag_started_for_generation = False
            self.set_status("Data generation from bag disabled", timeout=3000)
            
            # If data collection is active, stop it
            if self.main_window and self.main_window.analyzer and hasattr(self.main_window.analyzer, 'collecting_data'):
                if self.main_window.analyzer.collecting_data:
                    try:
                        print("Stopping data collection due to generate_from_bag being disabled")
                        self.stop_collection.emit()
                        # Update UI elements related to data collection
                        self.start_button.setEnabled(True)
                        self.stop_button.setEnabled(False)
                    except Exception as e:
                        print(f"Error stopping collection: {e}")
                
        print(f"Generate from bag state changed to: {is_checked}")
    
    def _enable_generate_from_bag(self):
        """Helper method to retry enabling Generate from Bag after stopping collection."""
        self.generate_from_bag_check.setChecked(True)
        self.on_generate_from_bag_changed(Qt.Checked)
    
    def _restart_bag_and_collection(self):
        """Helper method to restart bag playback and start collection."""
        bag_path = self.bag_path_edit.text().strip()
        if bag_path:
            self.play_bag_with_options(bag_path, loop=False)
            QTimer.singleShot(500, self._start_collection_from_bag)
    
    def play_bag_with_options(self, bag_path, loop=False):
        """
        Start bag playback with the specified options and error handling.
        
        Args:
            bag_path: Path to the bag file to play
            loop: Whether to loop playback (defaults to False)
            
        Returns:
            bool: Whether playback was started successfully
        """
        try:
            # Lock UI during the operation
            self.state_manager.transition('lock_ui')
            
            # Emit signal to start playback
            self.play_rosbag.emit(bag_path)
            
            # Set appropriate status message
            loop_text = "looping enabled" if loop else "no loop"
            self.set_status(f"Playing {os.path.basename(bag_path)} ({loop_text})")
            
            # Update state based on result
            self.state_manager.transition('unlock_ui')
            return True
        except Exception as e:
            # Handle failure cases
            print(f"Error starting bag playback: {e}")
            self.set_status(f"Failed to play bag: {str(e)}")
            
            # Make sure UI is unlocked and reset to appropriate state
            self.state_manager.transition('unlock_ui')
            self.state_manager.transition('stop_playback')
            return False
    
    def _start_collection_from_bag(self):
        """Start data collection when playing a bag with 'Generate from Bag' checked."""
        # Reset manual stop flag since we're explicitly starting collection
        self.manual_stop_requested = False
        
        # Set flag to track that bag playback was started for data generation
        self.bag_started_for_generation = True
        
        # Make sure we have a main window reference
        if not hasattr(self, 'main_window') or self.main_window is None:
            print("Error: No main window reference available")
            self.set_status("Error: Cannot start collection from bag")
            return
        
        # Get parameters for data collection
        config_name = self.config_entry.text().strip() or "bag_auto"
        target_distance = self.target_combo.currentText()
        
        # Use the state manager to update UI state
        self.state_manager.transition('start_collection')
        
        # Try to get the bag duration for better progress reporting
        try:
            from PyQt5.QtCore import QDateTime
            duration = 60  # Default duration in seconds
            
            if self.main_window and self.main_window.analyzer:
                analyzer = self.main_window.analyzer
                
                # Try to get the duration from the bag file
                if hasattr(analyzer, 'ros_bag_handler') and hasattr(analyzer.ros_bag_handler, 'get_bag_duration'):
                    duration = analyzer.ros_bag_handler.get_bag_duration()
                    self.set_status(f"Bag duration detected: {duration} seconds")
                    print(f"Using bag duration: {duration} seconds")
            
            print(f"Starting collection with bag duration: {duration} seconds")
        except Exception as e:
            # Fallback to the value in the duration spinner
            duration = int(self.duration_spin.value())
            self.set_status(f"Using default duration: {duration} seconds")
            print(f"Starting collection with default duration: {duration} seconds (bag duration unknown)")
        
        # Ensure the duration is reasonable
        if duration < 5:
            duration = 5  # Minimum 5 seconds
            print(f"Adjusted duration to minimum value: {duration} seconds")
        
        # Start the data collection
        try:
            # Emit a clear signal first if it exists
            if hasattr(self, 'reset_heatmap'):
                self.reset_heatmap.emit()
                print("Emitted reset_heatmap signal to ensure clean state")
                QApplication.processEvents()  # Process pending events before continuing
            
            self.set_status(f"Starting data collection from bag with config: {config_name} for {duration}s")
            self.start_collection.emit(config_name, target_distance, duration)
            
            # Update the progress bar and its maximum value
            from PyQt5.QtCore import QDateTime
            self.collection_start_time = QDateTime.currentDateTime()
            self.collection_duration = duration
            
            # Display the collection time in the UI
            if hasattr(self, 'collection_time_label'):
                self.collection_time_label.setText(f"Time: 00:00 / {duration//60:02d}:{duration%60:02d}")
            
            # Start the timer for progress updates
            if hasattr(self, 'progress_timer') and not self.progress_timer.isActive():
                self.progress_timer.start(100)  # Update every 0.1 seconds
            
            # Register for end-of-bag notification to stop collection
            if hasattr(self.main_window, 'analyzer') and hasattr(self.main_window.analyzer, 'signals'):
                analyzer = self.main_window.analyzer
                try:
                    # If possible, connect to the bag end signal using our safe helper
                    if hasattr(analyzer.signals, 'bag_playback_ended'):
                        self._safely_connect_signal(analyzer.signals.bag_playback_ended, self.on_bag_playback_ended)
                except Exception as e:
                    print(f"Error setting up bag end signal: {e}")
                
        except Exception as e:
            print(f"Error starting collection from bag: {e}")
            self.set_status(f"Error: {str(e)}")
            
            # Reset UI state on error
            self.state_manager.transition('stop_collection')
    
    def on_bag_playback_ended(self):
        """
        Handle the end of ROS2 bag playback.
        
        This method is called when the ROS2 bag playback ends.
        It also handles when recording ends (either normally or unexpectedly).
        """
        # Reset UI state for playback
        self.state_manager.transition('stop_playback')
        
        # Also reset recording state in case it was a recording that ended
        if self.state_manager.get_state('recording_bag'):
            self.state_manager.transition('stop_recording')
            self.set_status("Recording stopped")
        
        # Stop recording timer if active
        if hasattr(self, 'recording_timer') and self.recording_timer.isActive():
            self.recording_timer.stop()
        
        # Reset bag group title explicitly
        if hasattr(self, 'rosbag_group'):
            self.rosbag_group.setTitle("ROS2 Bag Controls")
        
        # Reset recording status label explicitly
        if hasattr(self, 'recording_status_label'):
            self.recording_status_label.setText("Record Settings")
            self.recording_status_label.setStyleSheet("font-weight: bold; color: #757575;")
        
        # Check if we should restart the bag playback for data generation
        # Don't restart if collection was manually stopped
        if (self.bag_started_for_generation and 
            hasattr(self, 'generate_from_bag_check') and 
            self.generate_from_bag_check.isChecked() and
            not self.manual_stop_requested):
            
            # Reset the heatmap for a new iteration
            self.reset_heatmap.emit()
            
            # Restart bag playback
            QTimer.singleShot(500, self._restart_bag_and_collection)
        else:
            self.bag_started_for_generation = False
            self.set_status("Bag playback complete")
    
    def reset_point_counter(self):
        """Reset the point counter display and state tracking."""
        # Reset displayed values and counters
        self.points_collected_label.setText("Density: 0 pts/frame")
        self.progress_bar.setValue(0)
        self.collection_time_label.setText("Time: 00:00")
        
        # Reset internal tracking variables
        self.collection_start_time = None
        self.last_point_count = 0
        self.collection_frames = 0
        self.point_update_counter = 0
        self.point_history = []  # Clear point history
        
        # Reset experiment data in analyzer if available
        analyzer = getattr(self, 'main_window', None)
        if analyzer:
            analyzer = getattr(analyzer, 'analyzer', None)
            if analyzer:
                try:
                    with analyzer.data_lock:
                        if hasattr(analyzer, 'experiment_data'):
                            # Reset all points lists
                            if hasattr(analyzer.experiment_data, 'x_points'):
                                analyzer.experiment_data.x_points = []
                            if hasattr(analyzer.experiment_data, 'y_points'):
                                analyzer.experiment_data.y_points = []
                            if hasattr(analyzer.experiment_data, 'z_points'):
                                analyzer.experiment_data.z_points = []
                            if hasattr(analyzer.experiment_data, 'intensity'):
                                analyzer.experiment_data.intensity = []
                            if hasattr(analyzer.experiment_data, 'snr'):
                                analyzer.experiment_data.snr = []
                            if hasattr(analyzer.experiment_data, 'noise'):
                                analyzer.experiment_data.noise = []
                                
                            # Reset all multi-frame metrics
                            if hasattr(analyzer.experiment_data, 'multi_frame_metrics'):
                                analyzer.experiment_data.multi_frame_metrics = {
                                    'total_frames': 0,
                                    'roi_combined_point_count': 0,
                                    'outside_roi_combined_point_count': 0,
                                    'roi_avg_single_frame_count': 0,
                                    'outside_roi_avg_single_frame_count': 0,
                                    'ten_frame_avg_points': 0
                                }
                                
                            # Reset distance band metadata
                            if hasattr(analyzer.experiment_data, 'metadata'):
                                metadata = getattr(analyzer.experiment_data, 'metadata', {})
                                if metadata:
                                    metadata['distance_bands'] = {}
                                    metadata['target_band'] = ''
                                    metadata['target_band_count'] = 0
                                    analyzer.experiment_data.metadata = metadata
                except Exception as e:
                    print(f"Error resetting point data: {e}")
        
        # Print a debug confirmation
        print("All collection stats reset to zero.")
    
    def on_timeline_changed(self, value):
        """Handle timeline slider value change.
        
        Args:
            value: Slider position (0-100)
        """
        # Convert to normalized position (0.0-1.0)
        position = value / 100.0
        
        # Update timestamp display based on bag duration if available
        if hasattr(self, 'bag_duration_seconds') and self.bag_duration_seconds > 0:
            current_time = position * self.bag_duration_seconds
            self.timestamp_label.setText(f"{current_time:.1f}s / {self.bag_duration_seconds:.1f}s")
        else:
            # Fallback to placeholder format
            minutes = int(position * 60)
            seconds = int((position * 60 - minutes) * 60)
            self.timestamp_label.setText(f"{minutes:02d}:{seconds:02d} / 60:00")
        
        # Check if slider update is from user interaction (not programmatic)
        if self.timeline_dragging:
            # Throttle seek events during dragging by using a timer
            if not hasattr(self, '_last_seek_time'):
                self._last_seek_time = 0
                self._pending_seek_position = None
                
            current_time = time.time()
            elapsed = current_time - self._last_seek_time
            
            # Only send a seek event every 100ms during dragging to avoid overwhelming the system
            if elapsed >= 0.1:  # 100ms
                self._last_seek_time = current_time
                self.timeline_position_changed.emit(position)
                self._pending_seek_position = None
            else:
                # Store the position for sending when the timer elapses
                self._pending_seek_position = position
                
                # If we haven't set up a timer yet, set one up to send the final position
                if not hasattr(self, '_seek_timer'):
                    self._seek_timer = QTimer()
                    self._seek_timer.setSingleShot(True)
                    self._seek_timer.timeout.connect(self._send_pending_seek)
                
                # Restart the timer for 100ms
                self._seek_timer.start(100)
    
    def _send_pending_seek(self):
        """Send the pending seek position if available."""
        if hasattr(self, '_pending_seek_position') and self._pending_seek_position is not None:
            self.timeline_position_changed.emit(self._pending_seek_position)
            self._pending_seek_position = None
            self._last_seek_time = time.time()
    
    def on_visualize_pointcloud(self):
        """Handle visualize point cloud button."""
        topic = self.pcl_topic_combo.currentText()
        self.visualize_pointcloud.emit(topic)
        self.set_status(f"Visualizing point cloud from: {topic}")
    
    def set_status(self, message, timeout=5000):
        """Set status bar message with auto-clear timeout."""
        self.status_bar.setText(message)
        self.status_timer.start(timeout)  # Auto-clear after timeout ms
        
    def clear_status(self):
        """Clear the status message."""
        self.status_bar.setText("Ready")
        
    def create_analysis_controls(self, parent_layout):
        """
        Create the analysis control section.
        
        Args:
            parent_layout: Parent layout to add widgets to.
        """
        analysis_group = QGroupBox("Analysis Controls")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Action buttons in horizontal layout
        buttons_layout = QHBoxLayout()
        
        self.save_heatmap_button = QPushButton("Save Heatmap")
        self.save_heatmap_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/save.png"))
        self.save_heatmap_button.setIconSize(QSize(24, 24))
        self.save_heatmap_button.clicked.connect(lambda: self.save_heatmap.emit())
        buttons_layout.addWidget(self.save_heatmap_button)
        
        self.export_plot_button = QPushButton("Export Plot")
        self.export_plot_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/export.png"))
        self.export_plot_button.setIconSize(QSize(24, 24))
        self.export_plot_button.clicked.connect(self.on_export_plot)
        buttons_layout.addWidget(self.export_plot_button)
        
        self.add_roi_button = QPushButton("Add ROI")
        self.add_roi_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/add_roi.png"))
        self.add_roi_button.setIconSize(QSize(24, 24))
        self.add_roi_button.clicked.connect(lambda: self.add_roi.emit())
        buttons_layout.addWidget(self.add_roi_button)
        
        self.clear_rois_button = QPushButton("Clear ROIs")
        self.clear_rois_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/clear.png"))
        self.clear_rois_button.setIconSize(QSize(24, 24))
        self.clear_rois_button.clicked.connect(lambda: self.clear_rois.emit())
        buttons_layout.addWidget(self.clear_rois_button)
        
        analysis_layout.addLayout(buttons_layout)
        
        # Generate report button
        button_style = """
            QPushButton {
                background-color: #0D47A1;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0A2C6B;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #757575;
            }
        """
        
        report_layout = QHBoxLayout()
        self.report_button = QPushButton("Generate Report")
        self.report_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/report.png"))
        self.report_button.setIconSize(QSize(24, 24))
        self.report_button.setCursor(Qt.PointingHandCursor)
        self.report_button.setStyleSheet(button_style)
        
        # Assign the button to generate_report_button for state manager compatibility
        self.generate_report_button = self.report_button
        
        self.report_button.clicked.connect(self.on_generate_report)
        report_layout.addWidget(self.report_button)
        analysis_layout.addLayout(report_layout)
        
        # Analysis metrics display
        metrics_header = QLabel("Analysis Metrics")
        metrics_header.setStyleSheet("font-weight: bold;")
        analysis_layout.addWidget(metrics_header)
        
        self.metrics_label = QLabel("No analysis data available")
        self.metrics_label.setStyleSheet("padding: 10px; background-color: rgba(30, 30, 30, 0.5); border-radius: 5px;")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.metrics_label.setMinimumHeight(80)
        analysis_layout.addWidget(self.metrics_label)
        
        # Add to parent layout
        parent_layout.addWidget(analysis_group)
    
    def on_circle_toggle(self, index, state):
        """
        Handle circle enable/disable toggle.
        
        Args:
            index: Index of circle being toggled (0-2)
            state: Qt.Checked or Qt.Unchecked
        """
        enabled = (state == Qt.Checked)
        self.circle_toggled.emit(index, enabled)
    
    def on_circle_distance_changed(self, index, value):
        """
        Handle circle distance change.
        
        Args:
            index: Index of circle being modified (0-2)
            value: New distance value in meters
        """
        self.circle_distance_changed.emit(index, float(value))
    
    def on_circle_radius_changed(self, index, value, label):
        """
        Handle circle radius slider change.
        
        Args:
            index: Index of circle being modified (0-2)
            value: Slider value (scaled by 100)
            label: QLabel to update with formatted value
        """
        radius = value / 100.0
        label.setText(f"{radius:.1f} m")
        self.circle_radius_changed.emit(index, radius)
    
    def on_circle_angle_changed(self, index, value):
        """
        Handle circle angle change.
        
        Args:
            index: Index of circle being modified (0-2)
            value: New angle value in degrees
        """
        self.circle_angle_changed.emit(index, float(value))
    
    def enable_all_circles(self):
        """Enable all sampling circles."""
        for i, controls in enumerate(self.circle_controls):
            controls['enable'].setChecked(True)
            self.circle_toggled.emit(i, True)
    
    def disable_all_circles(self):
        """Disable all sampling circles except the primary one."""
        for i, controls in enumerate(self.circle_controls):
            if i > 0:  # Keep the primary circle enabled
                controls['enable'].setChecked(False)
                self.circle_toggled.emit(i, False)
    
    def on_start_collection(self):
        """Handle start collection button click."""
        # Use state manager to handle UI updates if defined
        if hasattr(self, 'state_manager'):
            self.state_manager.transition('start_collection')
        
        # Reset manual stop flag when starting a new collection
        self.manual_stop_requested = False
        
        # Get parameters
        config_name = self.config_entry.text().strip() or "unnamed_config"
        target_distance = self.target_combo.currentText()
        duration = int(self.duration_spin.value())
        
        # Emit signal to start collection
        self.start_collection.emit(config_name, target_distance, duration)
        
        # Update UI
        self.collection_start_time = QDateTime.currentDateTime()
        self.collection_duration = duration
        
        # Start the timer for progress updates
        if self.progress_timer.isActive():
            self.progress_timer.stop()  # Ensure any previous timer is stopped
        self.progress_timer.start(100)  # Update every 0.1 seconds
        
        self.set_status(f"Collecting: {config_name}@{target_distance}m")
    
    def on_stop_collection(self):
        """Handle stop collection button click."""
        # Mark that collection was manually stopped
        self.manual_stop_requested = True
        
        self.stop_collection.emit()
        
        # Use state manager to handle UI updates
        self.state_manager.transition('stop_collection')
        
        self.progress_timer.stop()
        
        # Update UI to indicate collection stopped
        self.progress_bar.setValue(0)
        self.reset_point_counter()
        
        # Create a red status indicator
        status_pixmap = QPixmap(16, 16)
        status_pixmap.fill(QColor(244, 67, 54))
        status_icon_widgets = self.findChildren(QLabel, "status_icon")
        if status_icon_widgets:
            status_icon_widgets[0].setPixmap(status_pixmap)
        
        # If bag playback was started for data generation, stop it too
        if self.bag_started_for_generation:
            # Stop the bag playback
            print("Stopping bag playback that was started for data generation")
            self.stop_rosbag.emit()
            self.bag_started_for_generation = False
            self.generate_from_bag_check.setChecked(False)
            self.set_status("Collection and bag playback stopped")
        else:
            self.set_status("Collection stopped")
    
    def update_progress(self):
        """Update the progress bar during data collection."""
        if self.collection_start_time is None:
            return
        
        # Calculate elapsed time in seconds correctly
        from PyQt5.QtCore import QDateTime
        current_time = QDateTime.currentDateTime()
        elapsed_msecs = self.collection_start_time.msecsTo(current_time)
        elapsed = elapsed_msecs / 1000.0
        
        # Calculate progress percentage
        progress = min(100, int((elapsed / self.collection_duration) * 100))
        
        # Update progress bar with current percentage
        self.progress_bar.setValue(progress)
        
        # Update time display
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        self.collection_time_label.setText(f"Time: {minutes:02d}:{seconds:02d}")
        
        # Thread-safe access to point count
        display_text = None  # Only set if value changes
        points = -1  # Invalid value to detect if it was set
        target_band_points = -1  # Count points at the target distance band
        distance_bands = {}
        roi_points = 0
        outside_roi_points = 0
        
        try:
            # Use the direct reference to the main window
            analyzer = getattr(self, 'main_window', None)
            if analyzer:
                analyzer = getattr(analyzer, 'analyzer', None)
                
            if analyzer and hasattr(analyzer, 'data_lock'):
                # Use the analyzer's data lock for thread safety
                with analyzer.data_lock:
                    # Get total points count - correctly count inside and outside ROI
                    if hasattr(analyzer, 'experiment_data') and hasattr(analyzer.experiment_data, 'x_points'):
                        # Old way - this is the total of all points
                        total_raw_points = len(analyzer.experiment_data.x_points)
                        
                        # New way - count ROI points and outside ROI points
                        # Get the multi-frame metrics if available
                        if hasattr(analyzer.experiment_data, 'multi_frame_metrics'):
                            metrics = analyzer.experiment_data.multi_frame_metrics
                            roi_points = metrics.get('roi_combined_point_count', 0)
                            outside_roi_points = metrics.get('outside_roi_combined_point_count', 0)
                            # Calculate total points as sum of both
                            points = roi_points + outside_roi_points
                        else:
                            # Fall back to old method if we don't have multi-frame metrics
                            points = total_raw_points
                        
                        # Get distance band information if available in metadata
                        if hasattr(analyzer.experiment_data, 'metadata'):
                            metadata = getattr(analyzer.experiment_data, 'metadata', {})
                            if 'distance_bands' in metadata:
                                distance_bands = metadata.get('distance_bands', {})
                                target_band = metadata.get('target_band', '')
                                target_band_points = metadata.get('target_band_count', 0)
                        
                        # Only access current_data inside the lock
                        if hasattr(analyzer, 'current_data'):
                            # Get points in the current frame
                            current_frame_points = len(analyzer.current_data.get('x', []))
                            
                            # Add to point history (keep last 10 frames)
                            self.point_history.append(current_frame_points)
                            if len(self.point_history) > 10:
                                self.point_history.pop(0)
                            
                            # Calculate average points per frame
                            avg_points_per_frame = sum(self.point_history) / len(self.point_history)
                            
                            circle_points = len(analyzer.current_data.get('circle_x', []))
                            current_bands = analyzer.current_data.get('circle_distance_bands', {})
                            
                            if current_bands:
                                band_str = ", ".join([f"{k}: {v['count']}" for k, v in current_bands.items()])
                                print(f"DEBUG - Current frame bands: {band_str}")
                                
                            # Get target distance and corresponding band
                            if hasattr(analyzer, 'params'):
                                target_dist = analyzer.params.target_distance
                                target_band_width = 1.0  # 1-meter bands
                                target_band_start = int(np.floor(target_dist))
                                target_band_key = f"{target_band_start}m-{target_band_start + target_band_width}m"
                                
                                # Check for current frame target band count
                                target_band_current = 0
                                if target_band_key in current_bands:
                                    target_band_current = current_bands[target_band_key]['count']
                                
                                print(f"DEBUG - Circle points in current frame: {circle_points}, " +
                                      f"Target band ({target_band_key}): {target_band_current}, " +
                                      f"Total points: ROI({roi_points}) + Outside({outside_roi_points}) = {points}, " +
                                      f"Progress: {progress}%")
            else:
                print("WARNING: No data_lock available for thread safety! Cannot reliably read point count.")
        except Exception as e:
            print(f"Error getting point count: {e}")
        
        # Update point counter with throttling to reduce UI load and race conditions
        self.point_update_counter += 1
        
        # Check if the point count has changed or is -1 (error case)
        if points != -1:
            # Every 10th update or when count changes
            if points != self.last_point_count or self.point_update_counter >= 10:
                # Calculate average points per frame if history is available
                avg_text = ""
                if self.point_history:
                    avg_points = sum(self.point_history) / len(self.point_history)
                    avg_text = f"Density: {avg_points:.1f} pts/frame"
                else:
                    avg_text = "Density: 0 pts/frame"
                
                # Update UI text with the average density
                self.points_collected_label.setText(avg_text)
                
                # Show total point count and target band points if available in debug output
                if target_band_points > 0 and analyzer:
                    try:
                        with analyzer.data_lock:
                            target_dist = analyzer.params.target_distance
                    except Exception:
                        target_dist = 0
                    target_band_width = 1.0
                    target_band = f"{int(target_dist)}m-{int(target_dist)+target_band_width}m"
                    
                    # Log detailed point information for debugging
                    debug_text = f"Total pts: {points} (ROI: {roi_points}, Outside: {outside_roi_points})"
                    debug_text += f" (Target {target_band}: {target_band_points})"
                    print(debug_text)
                else:
                    # Log detailed point information for debugging
                    if roi_points > 0 or outside_roi_points > 0:
                        debug_text = f"Total pts: {points} (ROI: {roi_points}, Outside: {outside_roi_points})"
                        print(debug_text)
                
                # Add distance band breakdown if we have it
                if distance_bands and self.point_update_counter >= 20:
                    band_info = []
                    for band, count in sorted(distance_bands.items()):
                        if count > 0:
                            band_info.append(f"{band}: {count}")
                    if band_info:
                        print(f"Distance bands: {', '.join(band_info)}")
                
                self.last_point_count = points
                self.point_update_counter = 0
        else:
            # Error case - only update occasionally
            if self.point_update_counter >= 20:
                self.points_collected_label.setText("Density: -- pts/frame")
                self.point_update_counter = 0
    
        # Handle completion
        if progress >= 100:
            self.progress_bar.setValue(100)
            self.progress_timer.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.set_status("Collection complete")
            return
    
    def on_reset_heatmap(self):
        """Handle reset heatmap button click."""
        self.reset_heatmap.emit()
        self.reset_point_counter()  # Also reset the point counter
        self.set_status("Heatmap reset")
    
    def on_colormap_changed(self, colormap):
        """
        Handle colormap change.
        
        Args:
            colormap: New colormap name.
        """
        self.colormap_changed.emit(colormap)
    
    def on_decay_changed(self, value):
        """
        Handle decay factor slider change.
        
        Args:
            value: Slider value (scaled by 1000).
        """
        decay = value / 1000.0
        self.decay_value.setText(f"{decay:.3f}")
        self.decay_factor_changed.emit(decay)
    
    def on_vis_mode_changed(self, mode):
        """
        Handle visualization mode change.
        
        Args:
            mode: New visualization mode.
        """
        self.visualization_mode_changed.emit(mode)
    
    def on_noise_changed(self, value):
        """
        Handle noise floor slider change.
        
        Args:
            value: Slider value (scaled by 100).
        """
        noise = value / 100.0
        self.noise_value.setText(f"{noise:.2f}")
        self.noise_floor_changed.emit(noise)
    
    def on_smoothing_changed(self, value):
        """
        Handle smoothing slider change.
        
        Args:
            value: Slider value (scaled by 10).
        """
        smoothing = value / 10.0
        self.smooth_value.setText(f"{smoothing:.1f}")
        self.smoothing_changed.emit(smoothing)
    
    def on_generate_report(self):
        """Handle generate report button click."""
        # Use state manager to handle UI state
        self.state_manager.transition('start_generating_report')
        
        try:
            # Ensure bag playback doesn't restart automatically
            self.bag_started_for_generation = False
            self.manual_stop_requested = True
            
            # If the generate_from_bag_check is checked, uncheck it to prevent restart
            if hasattr(self, 'generate_from_bag_check') and self.generate_from_bag_check.isChecked():
                self.generate_from_bag_check.setChecked(False)
            
            # Emit signal to generate report
            self.generate_report.emit()
            self.set_status("Generating report...")
        except Exception as e:
            print(f"Error generating report: {e}")
            self.set_status(f"Error generating report: {str(e)}")
            
            # Reset state on error
            self.state_manager.transition('stop_generating_report')
    
    def on_report_completed(self, success=True, report_path=None):
        """
        Handle report generation completion.
        
        Args:
            success: Whether report generation was successful
            report_path: Path to the generated report if successful
        """
        # Update UI state
        self.state_manager.transition('stop_generating_report')
        
        if success and report_path:
            self.set_status(f"Report generated successfully: {os.path.basename(report_path)}")
        else:
            self.set_status("Report generation failed or was cancelled")
    
    def update_metrics(self, metrics):
        """
        Update the displayed analysis metrics.
        
        Args:
            metrics: Dictionary of metric values.
        """
        if metrics is None:
            self.metrics_label.setText("No analysis data available")
            return
        
        # Check if we have 10-frame metrics available
        has_10frame = 'ten_frame_avg_intensity' in metrics
        
        basic_text = (
            f"Max: {metrics.get('max_intensity', 0):.3f}  "
            f"Avg: {metrics.get('avg_intensity', 0):.3f}  "
            f"SNR: {metrics.get('snr_dB', 0):.1f}dB  "
            f"Coverage: {metrics.get('coverage_percentage', 0):.1f}%"
        )
        
        if has_10frame:
            ten_frame_text = (
                f" | 10-Frame Avg: {metrics.get('ten_frame_avg_intensity', 0):.3f}  "
                f"Stability: {metrics.get('ten_frame_stability', 0):.3f}  "
                f"10F-SNR: {metrics.get('ten_frame_snr_dB', 0):.1f}dB"
            )
            text = basic_text + ten_frame_text
        else:
            text = basic_text
            
        self.metrics_label.setText(text)

    def on_record_duration_changed(self, value):
        """Handle changes to the recording duration spinner."""
        # Update the estimated bag size based on the duration
        self.update_bag_size_estimate()

    def update_bag_size_estimate(self):
        """Update the estimated bag size label based on the recording duration."""
        duration_value = self.record_duration_spin.value()
        
        # If duration is 0 (unlimited), show as "unknown"
        if duration_value == 0:
            self.bag_size_label.setText("Est. size: Unknown (unlimited duration)")
            return
        
        # Convert to minutes for size estimation
        duration_unit = "sec"
        if hasattr(self, 'duration_unit_combo'):
            duration_unit = self.duration_unit_combo.currentText()
        
        duration_minutes = duration_value
        if duration_unit == "sec":
            duration_minutes = duration_value / 60.0
        
        # Estimate bag size based on common datatypes
        # Assumptions:
        # - PointCloud2 messages: ~50KB each, ~10Hz = 500KB/s
        # - MarkerArray messages: ~5KB each, ~10Hz = 50KB/s
        # - Other topics: ~100KB/s total
        # Total: ~650KB/s or ~39MB/min
        
        # Count the number of topics
        topics = [t.strip() for t in self.topics_edit.text().split(',') if t.strip()]
        topic_count = len(topics)
        
        # Base size + per-topic overhead
        size_per_min = 39 * 1024 * 1024  # 39MB per minute
        
        # Adjust for topic count
        if topic_count > 3:
            size_per_min = size_per_min * (1 + (topic_count - 3) * 0.2)  # Add 20% per extra topic
        
        # Calculate total estimated size
        total_size_bytes = size_per_min * duration_minutes
        
        # Format size for display
        if total_size_bytes < 1024 * 1024:
            size_str = f"{total_size_bytes / 1024:.1f} KB"
        elif total_size_bytes < 1024 * 1024 * 1024:
            size_str = f"{total_size_bytes / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{total_size_bytes / (1024 * 1024 * 1024):.1f} GB"
        
        # Create a formatted duration text for the label
        if duration_unit == "min":
            duration_text = f"{duration_value} min"
        else:
            # For seconds, convert to min:sec format if over 60 seconds
            if duration_value >= 60:
                minutes = duration_value // 60
                seconds = duration_value % 60
                duration_text = f"{minutes}:{seconds:02d} min"
            else:
                duration_text = f"{duration_value} sec"
        
        self.bag_size_label.setText(f"Est. size: {size_str} ({duration_text})")
    
    def discover_rosbags(self):
        """Find available ROS2 bag files in common directories without freezing the UI."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QLabel, QPushButton, QHBoxLayout, QProgressBar
        from PyQt5.QtCore import QThread, pyqtSignal, Qt
        import os
        import glob
        import subprocess
        import time
        
        # Check if there's already a finder thread running
        if hasattr(self, 'bag_finder_thread') and self.bag_finder_thread is not None:
            if self.bag_finder_thread.isRunning():
                # Thread is already running, don't start another one
                print("A bag finder thread is already running, not starting another one")
                return
        
        # Create a worker thread for scanning directories
        class BagFinderThread(QThread):
            # Define signals for thread communication
            found_bag = pyqtSignal(str, str)  # description, path
            update_status = pyqtSignal(str)   # status message
            progress_update = pyqtSignal(int, int)  # current, total
            search_complete = pyqtSignal()
            
            def __init__(self, search_paths):
                super().__init__()
                self.search_paths = search_paths
                self.stop_requested = False
                self.max_runtime = 30  # Maximum runtime in seconds to prevent hanging
                
            def run(self):
                start_time = time.time()
                total_paths = len(self.search_paths)
                for i, path in enumerate(self.search_paths):
                    # Check for timeout or stop request
                    if self.stop_requested or (time.time() - start_time) > self.max_runtime:
                        self.update_status.emit("Search stopped (timeout or requested)")
                        break
                        
                    if not os.path.isdir(path):
                        continue
                        
                    self.update_status.emit(f"Searching in {path}...")
                    self.progress_update.emit(i, total_paths)
                    
                    # More efficient approach: first find .db3 files in top directories
                    top_db3_files = glob.glob(os.path.join(path, "*.db3"))
                    for db3_file in top_db3_files:
                        if self.stop_requested or (time.time() - start_time) > self.max_runtime:
                            break
                        bag_dir = os.path.dirname(db3_file)
                        self._process_potential_bag(bag_dir)
                    
                    # Check stop condition again
                    if self.stop_requested or (time.time() - start_time) > self.max_runtime:
                        break
                    
                    # Then check immediate subdirectories (depth=1)
                    try:
                        # Use os.listdir which is faster than glob for this purpose
                        for subdir in os.listdir(path):
                            if self.stop_requested or (time.time() - start_time) > self.max_runtime:
                                break
                                
                            subdir_path = os.path.join(path, subdir)
                            if not os.path.isdir(subdir_path):
                                continue
                                
                            # Check for typical ROS2 bag directory names
                            if "bag" in subdir.lower() or "ros" in subdir.lower() or "data" in subdir.lower():
                                self.update_status.emit(f"Checking {subdir_path}...")
                                
                                # First check if this directory itself is a bag
                                if os.path.exists(os.path.join(subdir_path, "metadata.yaml")):
                                    self._process_potential_bag(subdir_path)
                                    continue
                                    
                                # Look for db3 files in this likely directory
                                try:
                                    for item in os.listdir(subdir_path):
                                        if self.stop_requested or (time.time() - start_time) > self.max_runtime:
                                            break
                                        if item.endswith(".db3"):
                                            nested_dir = os.path.join(subdir_path, os.path.dirname(item))
                                            self._process_potential_bag(nested_dir)
                                            # Only process a few per directory to avoid hanging
                                            break
                                except (PermissionError, OSError):
                                    continue
                    except (PermissionError, OSError):
                        continue
                
                # Send completion signal unless we were stopped
                if not self.stop_requested and (time.time() - start_time) <= self.max_runtime:
                    self.search_complete.emit()
                    self.update_status.emit("Search complete")
                else:
                    self.update_status.emit("Search stopped")
                
            def _process_potential_bag(self, bag_dir):
                # Check if this is a valid ROS2 bag (has metadata.yaml)
                if os.path.exists(os.path.join(bag_dir, "metadata.yaml")):
                    # Get basic info without subprocess for speed
                    bag_name = os.path.basename(bag_dir)
                    
                    try:
                        # Fast metadata check without full subprocess call
                        metadata_size = os.path.getsize(os.path.join(bag_dir, "metadata.yaml"))
                        if metadata_size > 0:
                            # Calculate rough size from .db3 files 
                            total_size = 0
                            db_files = glob.glob(os.path.join(bag_dir, "*.db3"))
                            for db_file in db_files:
                                if self.stop_requested:
                                    break
                                try:
                                    total_size += os.path.getsize(db_file)
                                except (OSError, PermissionError):
                                    pass
                                
                                # Format human-readable size
                                if total_size < 1024 * 1024:
                                    size_str = f"{total_size / 1024:.1f} KB"
                                elif total_size < 1024 * 1024 * 1024:
                                    size_str = f"{total_size / (1024 * 1024):.1f} MB"
                                else:
                                    size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
                                    
                                description = f"{bag_name} (Size: {size_str})"
                                self.found_bag.emit(description, bag_dir)
                    except Exception:
                        # Fallback to basic info
                        description = f"{bag_name}"
                        self.found_bag.emit(description, bag_dir)
                
                self.update_status.emit(f"Checked {bag_dir}")
            
            def stop(self):
                self.stop_requested = True
        
        # Search paths for ROS2 bags - prioritize likely locations
        search_paths = [
            self.last_used_directory() or os.path.expanduser("~"),  # Last used directory or home
            os.path.expanduser("~/ros2_bags"),                      # Common ROS2 bag location
            os.path.expanduser("~/rosbags"),                        # Another common location
            os.path.expanduser("~/bags"),                           # Another variant
            os.path.join(os.path.expanduser("~"), "Pictures"),      # Pictures directory
            os.path.expanduser("~/data"),                           # Data directory
        ]
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Discover ROS2 Bags")
        dialog.setMinimumSize(700, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Status display with progress
        status_layout = QHBoxLayout()
        status_label = QLabel("Initializing search...")
        status_layout.addWidget(status_label, 1)
        
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("Searching...")
        status_layout.addWidget(progress_bar)
        
        layout.addLayout(status_layout)
        
        # Bag list
        bag_list = QListWidget()
        bag_list.setAlternatingRowColors(True)
        bag_list.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(bag_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        select_button = QPushButton("Select")
        select_button.setDefault(True)
        select_button.setEnabled(False)  # Disable until bags are found
        
        stop_button = QPushButton("Stop Search")
        cancel_button = QPushButton("Cancel")
        
        button_layout.addWidget(stop_button)
        button_layout.addStretch()
        button_layout.addWidget(select_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        # Create and set up worker thread
        worker = BagFinderThread(search_paths)
        
        # Handle results dynamically
        bags_found = []
        
        def on_found_bag(description, path):
            bags_found.append((description, path))
            item = QListWidgetItem(description)
            item.setData(Qt.UserRole, path)
            bag_list.addItem(item)
            select_button.setEnabled(True)
            
        def on_status_update(message):
            status_label.setText(message)
            
        def on_progress_update(current, total):
            if total > 0:
                progress_bar.setValue(int(100 * current / total))
            
        def on_search_complete():
            status_label.setText(f"Found {len(bags_found)} ROS2 bags")
            progress_bar.setValue(100)
            progress_bar.setFormat("Complete")
            stop_button.setEnabled(False)
            
        def on_stop_search():
            worker.stop()
            status_label.setText("Search stopped")
            stop_button.setEnabled(False)
        
        # Connect signals
        worker.found_bag.connect(on_found_bag)
        worker.update_status.connect(on_status_update)
        worker.progress_update.connect(on_progress_update)
        worker.search_complete.connect(on_search_complete)
        stop_button.clicked.connect(on_stop_search)
        
        # Set up dialog buttons
        select_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        # Make sure to clean up thread when dialog closes
        dialog.finished.connect(lambda: self._clean_up_bag_finder(worker))
        
        # Store reference to the thread for management
        self.bag_finder_thread = worker
        
        # Start the worker thread
        worker.start()
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted and bag_list.currentItem():
            # Get selected bag path
            selected_path = bag_list.currentItem().data(Qt.UserRole)
            self.bag_path_edit.setText(selected_path)
            self.save_last_used_directory(os.path.dirname(selected_path))
            
            # Enable the Generate from Bag checkbox now that a bag is loaded
            self.generate_from_bag_check.setEnabled(True)
            return True
        
        return False

    def _clean_up_bag_finder(self, thread):
        """Ensure bag finder thread is properly cleaned up."""
        if thread and thread.isRunning():
            thread.stop()
            # Give the thread a moment to stop
            thread.wait(1000)  # Wait up to 1 second for the thread to finish
            if thread.isRunning():
                print("Warning: BagFinderThread did not stop properly")
        
        # Clear the reference to avoid memory leaks
        if hasattr(self, 'bag_finder_thread'):
            self.bag_finder_thread = None

    def on_play_rosbag(self):
        """Handle play bag button."""
        bag_path = self.bag_path_edit.text().strip()
        if not bag_path:
            self.set_status("Please select a bag file first")
            return
        
        # Check if we have a valid main window reference
        if not hasattr(self, 'main_window') or self.main_window is None:
            print("Warning: No main window reference available. Using direct signal.")
            # Just emit the signal and let the connected slot handle it
            self.play_rosbag.emit(bag_path)
            self.set_status(f"Playing {os.path.basename(bag_path)}")
            return
            
        # If there's any active data collection, stop it first
        if self.main_window and self.main_window.analyzer:
            analyzer = self.main_window.analyzer
            if hasattr(analyzer, 'collecting_data') and analyzer.collecting_data:
                try:
                    print("Stopping current data collection before starting bag playback")
                    self.stop_collection.emit()
                    # Allow time for collection to stop
                    QApplication.processEvents()
                    time.sleep(0.2)
                except Exception as e:
                    print(f"Error stopping data collection: {e}")
                    
            # Reset experiment data if it exists
            if hasattr(analyzer, 'experiment_data') and hasattr(analyzer.experiment_data, 'clear'):
                try:
                    # Use thread-safe access to clear the data
                    if hasattr(analyzer, 'data_lock'):
                        with analyzer.data_lock:
                            analyzer.experiment_data.clear()
                            print("Experiment data cleared before bag playback")
                    else:
                        analyzer.experiment_data.clear()
                        print("Warning: Experiment data cleared without lock")
                except Exception as e:
                    print(f"Error clearing experiment data: {e}")
        
        # Reset timeline UI
        self.timeline_slider.setValue(0)
        
        # Use state manager to handle UI transitions
        self.state_manager.transition('start_playback')
        
        # Enable the "Generate from Bag" checkbox now that a bag is loaded
        self.generate_from_bag_check.setEnabled(True)
        
        # Start playback - loop by default for normal playback
        success = self.play_bag_with_options(bag_path, loop=True)
        if not success:
            # Revert state if operation failed
            self.state_manager.transition('start_playback', success=False)
            self.set_status("Failed to start bag playback")
            return
        
        # Reset point counter display
        if hasattr(self, 'points_collected_label'):
            self.points_collected_label.setText("Points: 0")
        
        # If "Generate from Bag" is checked, restart with no loop and start data collection
        if self.generate_from_bag_check.isChecked():
            self.stop_rosbag.emit()
            QTimer.singleShot(500, lambda: self.play_bag_with_options(bag_path, loop=False))
            QTimer.singleShot(1000, self._start_collection_from_bag)

    def on_record_rosbag(self):
        """Handle record bag button - show setup dialog and start recording if confirmed."""
        # Show recording setup dialog
        params = self.prepare_recording_dialog()
        if not params:
            return  # User canceled
        
        # Extract parameters
        folder = params["folder"]
        duration = params["duration"]
        topics = params["topics"]
        all_topics = params["all_topics"]
        
        # Create a unique subfolder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_path = os.path.join(folder, f"radar_recording_{timestamp}")
        
        try:
            # Create the directory
            os.makedirs(record_path, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to create recording directory: {str(e)}")
            return
        
        # Stop any ongoing data collection if needed
        if self.main_window and self.main_window.analyzer and hasattr(self.main_window.analyzer, 'collecting_data'):
            if self.main_window.analyzer.collecting_data:
                try:
                    self.stop_collection.emit()
                    QApplication.processEvents()
                    time.sleep(0.2)
                except Exception as e:
                    print(f"Error stopping data collection: {e}")
        
        # Format topics list for display
        if all_topics:
            topics_display = "all topics"
        else:
            topics_display = ", ".join(topics)
        
        # Update UI state
        self.state_manager.transition('start_recording')
        
        # Format duration text for status message
        if duration == 0:
            duration_text = "without time limit"
        else:
            minutes = duration // 60
            seconds = duration % 60
            if minutes > 0:
                duration_text = f"for {minutes} min {seconds} sec"
            else:
                duration_text = f"for {seconds} sec"
        
        # Start recording
        self.record_rosbag.emit(record_path, topics, duration)
        self.set_status(f"Recording to {os.path.basename(record_path)} {duration_text}: {topics_display}")
        
        # Set timer if duration is limited
        if duration > 0:
            duration_ms = duration * 1000
            QTimer.singleShot(duration_ms, self.on_recording_timeout)
            self.recording_end_time = time.time() + duration
            
            # Update the recording status label to show countdown
            self.start_recording_timer()

    def on_recording_timeout(self):
        """Called when the recording timer expires."""
        # Only stop if recording is still active
        if self.main_window and self.main_window.analyzer and hasattr(self.main_window.analyzer, 'is_recording'):
            if self.main_window.analyzer.is_recording:
                self.set_status("Recording time limit reached, stopping...")
                self.stop_rosbag.emit()
                
                # Update state using state manager
                self.state_manager.transition('stop_recording')

    def on_stop_rosbag(self):
        """Handle stop bag button."""
        # Check if we have a valid main window reference before accessing analyzer
        if not hasattr(self, 'main_window') or self.main_window is None:
            print("Warning: No main window reference available. Using direct signal.")
            # Just emit the signal and let the connected slot handle it
            self.stop_rosbag.emit()
            self.set_status("Stopping ROS2 bag operations")
            return
            
        # Clear any state related to collection from bag
        self.bag_started_for_generation = False
        
        # Stop recording timer if active
        if hasattr(self, 'recording_timer') and self.recording_timer.isActive():
            self.recording_timer.stop()
        
        # Emit the signal to stop bag operations
        self.stop_rosbag.emit()
        
        # Use state manager to handle UI updates for both playback and recording
        if self.state_manager.get_state('playing_bag'):
            self.state_manager.transition('stop_playback')
        
        if self.state_manager.get_state('recording_bag'):
            self.state_manager.transition('stop_recording')
        
        # Disable the "Generate from Bag" checkbox and uncheck it
        self.generate_from_bag_check.setEnabled(False)
        self.generate_from_bag_check.setChecked(False)
        
        # If data collection is active, stop it
        if self.main_window and self.main_window.analyzer and hasattr(self.main_window.analyzer, 'collecting_data'):
            if self.main_window.analyzer.collecting_data:
                self.stop_collection.emit()
        
        self.set_status("Stopped ROS2 bag operation")

    def last_used_directory(self):
        """Get the last used directory for file dialogs.
        
        Returns:
            str: Path to the last used directory, or empty string if not set.
        """
        try:
            if hasattr(self, 'settings_file') and os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    return settings.get('last_directory', '')
        except Exception as e:
            print(f"Error loading last directory: {e}")
        return ''

    def save_last_used_directory(self, directory):
        """Save the last used directory for file dialogs.
        
        Args:
            directory (str): Directory path to save.
        """
        try:
            settings = {}
            if hasattr(self, 'settings_file'):
                if os.path.exists(self.settings_file):
                    try:
                        with open(self.settings_file, 'r') as f:
                            settings = json.load(f)
                    except:
                        # If file exists but is invalid, start with empty settings
                        settings = {}
                
                settings['last_directory'] = directory
                
                with open(self.settings_file, 'w') as f:
                    json.dump(settings, f)
        except Exception as e:
            print(f"Error saving last directory: {e}")

    def _safely_connect_signal(self, signal, slot):
        """Helper method to safely connect a signal to a slot, avoiding duplicates.
        
        Args:
            signal: The Qt signal to connect
            slot: The method to call when the signal is emitted
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if not signal:
            print("Cannot connect: Signal is None")
            return False
            
        # First try to disconnect any existing connection to avoid duplicates
        try:
            signal.disconnect(slot)
            print(f"Disconnected existing signal connection to {slot.__name__}")
        except (TypeError, RuntimeError):
            # It's okay if there was no connection
            pass
            
        # Now connect the signal
        try:
            signal.connect(slot)
            print(f"Connected signal to {slot.__name__}")
            return True
        except Exception as e:
            print(f"Error connecting signal: {e}")
            return False

    def on_export_plot(self):
        """
        Handle export plot button click.
        
        This method forwards the export plot request to the main window
        with proper state management.
        """
        # Use state manager to update UI state
        if hasattr(self, 'state_manager'):
            self.state_manager.transition('lock_ui')
            
        try:
            # Emit signal to export plot
            self.export_plot.emit()
            self.set_status("Preparing to export plot...")
        except Exception as e:
            print(f"Error starting plot export: {e}")
            self.set_status(f"Error: {str(e)}")
            
            # Reset state on error
            if hasattr(self, 'state_manager'):
                self.state_manager.transition('unlock_ui')

    def check_for_analyzer(self):
        """Check if the analyzer is available and update UI accordingly."""
        has_analyzer = False
        
        if hasattr(self, 'main_window') and self.main_window is not None:
            if hasattr(self.main_window, 'analyzer') and self.main_window.analyzer is not None:
                has_analyzer = True
        
        # Enable/disable controls based on analyzer availability
        self.start_button.setEnabled(has_analyzer)
        
        ros_buttons_enabled = has_analyzer
        if has_analyzer and hasattr(self.main_window.analyzer, 'ros2_available'):
            ros_buttons_enabled = self.main_window.analyzer.ros2_available
        
        self.play_button.setEnabled(ros_buttons_enabled)
        self.stop_button.setEnabled(ros_buttons_enabled)
        self.record_button.setEnabled(ros_buttons_enabled)
        
        # Update status message
        if not has_analyzer:
            self.set_status("No analyzer connected - visualization only")
        elif not ros_buttons_enabled:
            self.set_status("ROS2 not available - only collection mode available")
        else:
            self.set_status("Ready")

    def on_duration_unit_changed(self, index):
        """Handle change in duration unit (seconds or minutes).
        
        Args:
            index: Index of the selected unit (0=seconds, 1=minutes)
        """
        current_value = self.record_duration_spin.value()
        if current_value == 0:  # No limit case
            return
            
        # Convert between units while preserving the actual duration
        if index == 0:  # Switched to seconds
            # If previously in minutes, convert to seconds
            self.record_duration_spin.setRange(0, 3600)  # Up to 1 hour in seconds
            if hasattr(self, '_previous_unit') and self._previous_unit == 1:
                self.record_duration_spin.setValue(current_value * 60)
        else:  # Switched to minutes
            # If previously in seconds, convert to minutes
            self.record_duration_spin.setRange(0, 60)  # Up to 1 hour in minutes
            if hasattr(self, '_previous_unit') and self._previous_unit == 0:
                self.record_duration_spin.setValue(max(1, current_value // 60))
                
        # Store current unit for next change
        self._previous_unit = index
        
        # Update the bag size estimate
        self.update_bag_size_estimate()

    def prepare_recording_dialog(self):
        """Show dialog to configure ROS2 bag recording settings.
        
        Returns:
            dict: Recording parameters if confirmed, None if canceled
        """
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                                   QLabel, QPushButton, QLineEdit, QSpinBox, 
                                   QComboBox, QRadioButton, QDialogButtonBox, QFileDialog)
        
        # Create dialog and layout
        dialog = QDialog(self)
        dialog.setWindowTitle("Setup ROS2 Bag Recording")
        dialog.setMinimumSize(600, 400)
        layout = QVBoxLayout(dialog)
        
        # 1. Folder selection section
        folder_group = QGroupBox("1. Select Recording Folder")
        folder_layout = QVBoxLayout(folder_group)
        
        folder_path_layout = QHBoxLayout()
        folder_path_edit = QLineEdit()
        folder_path_edit.setPlaceholderText("Select where to save ROS2 bag...")
        
        # Define a non-blocking browse function
        def browse_folder():
            # Temporarily disable the button to prevent multiple clicks
            browse_button.setEnabled(False)
            QApplication.processEvents()
            
            try:
                self._select_recording_folder(folder_path_edit)
            finally:
                # Re-enable the button
                browse_button.setEnabled(True)
                # Update validation
                validate_inputs()
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(browse_folder)
        
        folder_path_layout.addWidget(folder_path_edit, 3)
        folder_path_layout.addWidget(browse_button, 1)
        folder_layout.addLayout(folder_path_layout)
        
        # Pre-populate with last used directory if available
        last_dir = self.last_used_directory()
        if last_dir:
            folder_path_edit.setText(last_dir)
        
        # 2. Recording duration section
        duration_group = QGroupBox("2. Set Recording Duration")
        duration_layout = QHBoxLayout(duration_group)
        
        duration_spin = QSpinBox()
        duration_spin.setRange(0, 3600)
        duration_spin.setValue(60)
        duration_spin.setSpecialValueText("No limit")
        
        unit_combo = QComboBox()
        unit_combo.addItems(["seconds", "minutes"])
        unit_combo.setCurrentIndex(1)  # Default to minutes
        
        duration_layout.addWidget(QLabel("Duration:"), 1)
        duration_layout.addWidget(duration_spin, 2)
        duration_layout.addWidget(unit_combo, 1)
        
        # 3. Topics selection section
        topics_group = QGroupBox("3. Select Topics to Record")
        topics_layout = QVBoxLayout(topics_group)
        
        all_topics_radio = QRadioButton("Record all topics")
        all_topics_radio.setChecked(True)
        
        custom_topics_radio = QRadioButton("Custom topics:")
        custom_topics_edit = QLineEdit()
        custom_topics_edit.setPlaceholderText("Enter comma-separated topic names...")
        custom_topics_edit.setEnabled(False)
        
        # Connect radio buttons to enable/disable custom topics field
        all_topics_radio.toggled.connect(lambda checked: custom_topics_edit.setEnabled(not checked))
        
        topics_layout.addWidget(all_topics_radio)
        topics_layout.addWidget(custom_topics_radio)
        topics_layout.addWidget(custom_topics_edit)
        
        # Storage estimation
        storage_label = QLabel("Estimated storage: Unknown")
        
        # Update storage estimation when values change
        def update_storage_estimate():
            try:
                # Basic estimation logic - just a rough calculation
                duration_val = duration_spin.value()
                if duration_val == 0:
                    storage_label.setText("Estimated storage: Unknown (no time limit)")
                    return
                    
                if unit_combo.currentText() == "minutes":
                    duration_val *= 60  # Convert to seconds
                    
                # Rough estimate: 5MB per minute of recording (adjust based on your data)
                est_mb = (duration_val / 60) * 5
                
                if est_mb < 1000:
                    storage_label.setText(f"Estimated storage: {est_mb:.1f} MB")
                else:
                    storage_label.setText(f"Estimated storage: {est_mb/1000:.2f} GB")
            except Exception:
                storage_label.setText("Estimated storage: Unknown")
            
            # Process events to keep UI responsive
            QApplication.processEvents()
        
        # Connect signals to update storage estimate
        duration_spin.valueChanged.connect(update_storage_estimate)
        unit_combo.currentIndexChanged.connect(update_storage_estimate)
        
        # Initially calculate estimate
        update_storage_estimate()
        
        # Add sections to main layout
        layout.addWidget(folder_group)
        layout.addWidget(duration_group)
        layout.addWidget(topics_group)
        layout.addWidget(storage_label)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        ok_button = button_box.button(QDialogButtonBox.Ok)
        ok_button.setText("Start Recording")
        ok_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        layout.addWidget(button_box)
        
        # Validation function
        def validate_inputs():
            folder = folder_path_edit.text().strip()
            ok_button.setEnabled(bool(folder))
            
            # Process events to ensure button state updates immediately
            QApplication.processEvents()
        
        # Connect signals for validation
        folder_path_edit.textChanged.connect(validate_inputs)
        
        # Set initial state
        validate_inputs()
        
        # Set to non-modal and process events to keep UI fluid during dialog execution
        dialog.setWindowModality(Qt.WindowModal)
        QApplication.processEvents()
        
        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            # Gather parameters
            folder = folder_path_edit.text().strip()
            duration = duration_spin.value()
            if unit_combo.currentText() == "minutes":
                duration *= 60  # Convert to seconds
                
            if all_topics_radio.isChecked():
                topics = ["-a"]  # Special flag for all topics
            else:
                topics = [t.strip() for t in custom_topics_edit.text().split(',') if t.strip()]
                if not topics:
                    topics = ["-a"]  # Default to all if none specified
            
            return {
                "folder": folder,
                "duration": duration,
                "topics": topics,
                "all_topics": all_topics_radio.isChecked()
            }
        
        return None  # Canceled

    def _select_recording_folder(self, edit_field):
        """Helper to select a recording folder and update the edit field."""
        default_dir = self.last_used_directory() or os.path.expanduser("~")
        
        # Create a dialog instance instead of using the static method
        # This allows us to make it non-modal and use proper options
        dialog = QFileDialog(self, "Select Folder for ROS2 Bag Recording", default_dir)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # Use Qt's dialog instead of native for better control
        
        # Use a non-blocking approach
        dialog.setWindowModality(Qt.WindowModal)  # Modal to parent window only
        
        # Process events during dialog execution to keep UI responsive
        result = dialog.exec_()
        QApplication.processEvents()
        
        if result == QDialog.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files:
                folder = selected_files[0]
                edit_field.setText(folder)
                self.save_last_used_directory(folder)
                
                # Process events again to ensure UI updates
                QApplication.processEvents()

    def start_recording_timer(self):
        """Start a timer to update the recording status with a countdown."""
        # Create timer if it doesn't exist
        if not hasattr(self, 'recording_timer'):
            self.recording_timer = QTimer()
            self.recording_timer.timeout.connect(self.update_recording_status)
        else:
            # Stop any existing timer
            self.recording_timer.stop()
        
        # Start timer to update every second
        self.recording_timer.start(1000)  # 1000ms = 1 second
        
        # Update immediately
        self.update_recording_status()

    def update_recording_status(self):
        """Update the recording status label with remaining time."""
        if not hasattr(self, 'recording_end_time') or not self.state_manager.get_state('recording_bag'):
            # Not recording or no end time set
            if hasattr(self, 'recording_timer'):
                self.recording_timer.stop()
            return
        
        # Calculate remaining time
        remaining = max(0, self.recording_end_time - time.time())
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        
        # Update status label with countdown
        if hasattr(self, 'recording_status_label'):
            if remaining > 0:
                self.recording_status_label.setText(f"Recording in Progress - {minutes:02d}:{seconds:02d} remaining")
            else:
                self.recording_status_label.setText("Recording - Finishing up...")
                
        # If recording finished, update UI
        if remaining <= 0 and self.state_manager.get_state('recording_bag'):
            if hasattr(self, 'recording_timer'):
                self.recording_timer.stop()