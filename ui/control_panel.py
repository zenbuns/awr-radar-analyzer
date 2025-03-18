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
    QGridLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QColor, QPainter, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize
import os
import json
from radar_analyzer.processing.data_processor import filter_points_in_circle, calculate_heatmap_size
from radar_analyzer.visualization.visualizer import update_plot, update_heatmap_display
from radar_analyzer.utils.ros_bag_handler import play_rosbag, record_rosbag, stop_rosbag


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
    record_rosbag = pyqtSignal(str, list)  # Path and topics to record
    stop_rosbag = pyqtSignal()  # Stop recording or playback
    timeline_position_changed = pyqtSignal(float)  # Bag playback position (0.0-1.0)
    visualize_pointcloud = pyqtSignal(str)  # Point cloud topic
    
    def __init__(self, parent=None):
        # Create a folder to store application settings
        self.settings_dir = os.path.join(os.path.expanduser("~"), ".radar_analyzer")
        if not os.path.exists(self.settings_dir):
            os.makedirs(self.settings_dir, exist_ok=True)
        self.settings_file = os.path.join(self.settings_dir, "ui_settings.json")
        """
        Initialize the ControlPanel widget.
        
        Args:
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        
        # Initialize status bar (must be before setup_ui)
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("color: #cccccc; font-style: italic;")
        self.status_timer = QTimer()
        self.status_timer.setSingleShot(True)
        self.status_timer.timeout.connect(self.clear_status)
        
        # Create UI components
        self.setup_ui()
        
        # Initialize timer for progress updates
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.collection_start_time = None
        self.collection_duration = 60  # seconds
        
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
        
        # Add some additional stats placeholders for future use
        stats_label = QLabel("Collection Stats:")
        stats_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        status_layout.addWidget(stats_label)
        
        self.points_collected_label = QLabel("Points: 0")
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
        rosbag_group = QGroupBox("ROS2 Bag Controls")
        rosbag_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        rosbag_layout = QHBoxLayout(rosbag_group)
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
        
        bag_select_button = QPushButton("Browse")
        bag_select_button.setMinimumHeight(28)
        bag_select_button.setCursor(Qt.PointingHandCursor)  # Change cursor on hover
        bag_select_button.setIcon(QIcon.fromTheme("folder-open"))
        bag_select_button.clicked.connect(self.on_select_bagfile)
        
        bag_select_layout.addWidget(self.bag_path_edit, 3)  # Proportional sizing
        bag_select_layout.addWidget(bag_select_button, 1)
        
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
            
            def mousePressEvent(self, event):
                if self.parent_panel:
                    self.parent_panel.timeline_dragging = True
                super().mousePressEvent(event)
            
            def mouseReleaseEvent(self, event):
                if self.parent_panel:
                    # Keep dragging true briefly to avoid immediate overwrite by signal
                    QTimer.singleShot(100, lambda: setattr(self.parent_panel, 'timeline_dragging', False))
                super().mouseReleaseEvent(event)
        
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
        
        # Add recording header for visual separation
        recording_header = QLabel("Recording")
        recording_header.setStyleSheet("font-weight: bold; color: #F44336;")
        recording_header.setAlignment(Qt.AlignLeft)
        recording_layout.addWidget(recording_header)
        
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
        
        recording_layout.addLayout(record_path_layout)
        recording_layout.addSpacing(8)  # Add spacing between sections
        
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
        parent_layout.addWidget(rosbag_group)
    
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
        
        # Create a non-modal directory dialog for better UI responsiveness
        dialog = QFileDialog(self, "Select Directory to Save Bag", last_dir)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setMinimumSize(800, 500)  # Larger dialog for better browsing
        
        if dialog.exec_() == QFileDialog.Accepted:
            selected_dirs = dialog.selectedFiles()
            if selected_dirs:
                dir_path = selected_dirs[0]
                self.record_path_edit.setText(dir_path)
                self.save_last_used_directory(dir_path)
                return True
        return False
    
    def on_play_rosbag(self):
        """Handle play bag button."""
        bag_path = self.bag_path_edit.text().strip()
        if not bag_path:
            self.set_status("Please select a bag file first")
            return
        
        # Reset timeline UI
        self.timeline_slider.setValue(0)
        self.timeline_slider.setEnabled(True)  # Enable timeline slider
        
        # Start playback
        self.play_rosbag.emit(bag_path)
        self.play_button.setEnabled(False)
        self.stop_playback_button.setEnabled(True)
        self.set_status(f"Playing bag: {bag_path}")
    
    def on_record_rosbag(self):
        """Handle record bag button."""
        record_path = self.record_path_edit.text().strip()
        if not record_path:
            self.set_status("Please select a directory to save the bag")
            return
        
        topics = [t.strip() for t in self.topics_edit.text().split(',') if t.strip()]
        if not topics:
            self.set_status("Please specify at least one topic to record")
            return
        
        self.record_rosbag.emit(record_path, topics)
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.set_status(f"Recording to {record_path}: {', '.join(topics)}")
    
    def on_stop_rosbag(self):
        """Handle stop bag button (works for both playback and recording)."""
        self.stop_rosbag.emit()
        
        # Enable/disable appropriate buttons
        self.play_button.setEnabled(True)
        self.stop_playback_button.setEnabled(False)
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        
        # Reset timeline slider
        self.timeline_slider.setValue(0)
        self.timeline_slider.setEnabled(False)
        self.timeline_dragging = False
        self.timestamp_label.setText("00:00 / 00:00")
        
        self.set_status("Stopped ROS2 bag playback/recording")
    
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
        
        # Only emit the seek signal if the user is actually dragging the slider
        # This prevents feedback loops when the slider is updated programmatically
        if self.timeline_dragging:
            self.timeline_position_changed.emit(position)
    
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
        analysis_layout = QHBoxLayout(analysis_group)
        
        # Action buttons
        self.save_heatmap_button = QPushButton("Save Heatmap")
        self.save_heatmap_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/save.png"))
        self.save_heatmap_button.setIconSize(QSize(24, 24))
        self.save_heatmap_button.clicked.connect(lambda: self.save_heatmap.emit())
        analysis_layout.addWidget(self.save_heatmap_button)
        
        self.export_plot_button = QPushButton("Export Plot")
        self.export_plot_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/export.png"))
        self.export_plot_button.setIconSize(QSize(24, 24))
        self.export_plot_button.clicked.connect(lambda: self.export_plot.emit())
        analysis_layout.addWidget(self.export_plot_button)
        
        self.add_roi_button = QPushButton("Add ROI")
        self.add_roi_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/add_roi.png"))
        self.add_roi_button.setIconSize(QSize(24, 24))
        self.add_roi_button.clicked.connect(lambda: self.add_roi.emit())
        analysis_layout.addWidget(self.add_roi_button)
        
        self.clear_rois_button = QPushButton("Clear ROIs")
        self.clear_rois_button.setIcon(QIcon("/home/zen/Pictures/Radar_stuff/icons/clear.png"))
        self.clear_rois_button.setIconSize(QSize(24, 24))
        self.clear_rois_button.clicked.connect(lambda: self.clear_rois.emit())
        analysis_layout.addWidget(self.clear_rois_button)
        
        # Analysis metrics display
        self.metrics_label = QLabel("No analysis data available")
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
        config_name = self.config_entry.text().strip()
        if not config_name:
            self.set_status("Please enter a configuration name")
            return
        
        target_distance = self.target_combo.currentText()
        duration = self.duration_spin.value()
        
        self.start_collection.emit(config_name, target_distance, duration)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Reset all collection stats
        self.progress_bar.setValue(0)
        self.points_collected_label.setText("Points: 0")
        self.collection_time_label.setText("Time: 00:00")
        
        # Create a green status indicator
        status_pixmap = QPixmap(16, 16)
        status_pixmap.fill(QColor(76, 175, 80))  # Green color
        status_icon_widgets = self.findChildren(QLabel, "status_icon")
        if status_icon_widgets:
            status_icon_widgets[0].setPixmap(status_pixmap)
        
        # Store current time and duration for progress calculations
        from PyQt5.QtCore import QDateTime
        self.collection_start_time = QDateTime.currentDateTime()
        self.collection_duration = duration
        
        # Start the timer for progress updates
        if self.progress_timer.isActive():
            self.progress_timer.stop()  # Ensure any previous timer is stopped
        self.progress_timer.start(100)  # Update every 0.1 seconds
        
        self.set_status(f"Collecting: {config_name}@{target_distance}m")
    
    def on_stop_collection(self):
        """Handle stop collection button click."""
        self.stop_collection.emit()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_timer.stop()
        
        # Update UI to indicate collection stopped
        self.progress_bar.setValue(0)
        
        # Create a red status indicator
        status_pixmap = QPixmap(16, 16)
        status_pixmap.fill(QColor(244, 67, 54))  # Red color
        status_icon_widgets = self.findChildren(QLabel, "status_icon")
        if status_icon_widgets:
            status_icon_widgets[0].setPixmap(status_pixmap)
        
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
        
        # Get the actual points count from the analyzer - ONLY trust actual data
        points = 0
        try:
            # Use the direct reference to the main window
            if hasattr(self, 'main_window') and hasattr(self.main_window, 'analyzer') and self.main_window.analyzer:
                points = len(self.main_window.analyzer.experiment_data.x_points)
                self.points_collected_label.setText(f"Points: {points}")
                print(f"Using ACTUAL point count from analyzer: {points}")
            else:
                # Use placeholder only if we can't get real data
                self.points_collected_label.setText("Points: --")
                print("No analyzer available, can't get point count")
        except Exception as e:
            # Fall back to simple display if there's an error
            print(f"Error getting point count: {e}")
            self.points_collected_label.setText("Points: --")
        
        # Print brief debug info
        print(f"Progress: {progress}%, Time: {minutes:02d}:{seconds:02d}, Actual Points: {points}")
        
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
        self.generate_report.emit()
        self.set_status("Generating report...")
    
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