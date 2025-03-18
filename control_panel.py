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
    QScrollArea, QSizePolicy, QGridLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QColor, QPainter, QIcon, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize

from styles import Colors


class ModernSlider(QWidget):
    """A modernized slider widget with improved visuals."""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, label, min_val, max_val, default_val, scale=1.0, suffix="", precision=2, parent=None):
        """Initialize the ModernSlider widget."""
        super().__init__(parent)
        self.scale = scale
        self.precision = precision
        self.suffix = suffix
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add label above slider
        self.label = QLabel(label)
        self.label.setStyleSheet(f"color: {Colors.TEXT}; font-weight: bold;")
        layout.addWidget(self.label)
        
        # Create slider and value display in horizontal layout
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val / scale))
        self.slider.setMaximum(int(max_val / scale))
        self.slider.setValue(int(default_val / scale))
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.slider, 7)  # slider gets more space
        
        # Value display
        self.value_label = QLabel(f"{default_val:.{precision}f}{suffix}")
        self.value_label.setFixedWidth(60)  # Fixed width for consistent layout
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setStyleSheet(f"color: {Colors.ACCENT_BLUE}; font-weight: bold;")
        slider_layout.addWidget(self.value_label, 3)  # value gets less space
        
        layout.addLayout(slider_layout)
    
    def _on_slider_changed(self, value):
        # Scale the value and update label
        scaled_value = value * self.scale
        self.value_label.setText(f"{scaled_value:.{self.precision}f}{self.suffix}")
        # Emit the signal with scaled value
        self.valueChanged.emit(scaled_value)
    
    def value(self):
        """Get the current scaled value."""
        return self.slider.value() * self.scale
    
    def setValue(self, value):
        """Set slider to the given value (will be scaled internally)."""
        self.slider.setValue(int(value / self.scale))


class ColorCircleIcon(QIcon):
    """Create a circular color icon for UI elements."""
    
    def __init__(self, color_name):
        """Initialize with a color name that maps to the modern palette."""
        super().__init__()
        
        # Map color names to Colors class
        color_map = {
            'lime': Colors.ACCENT_GREEN,
            'cyan': Colors.ACCENT_BLUE_HOVER,
            'yellow': Colors.ACCENT_YELLOW,
            'red': Colors.ACCENT_RED,
            'blue': Colors.ACCENT_BLUE,
            'lavender': Colors.ACCENT_LAVENDER
        }
        
        # Get color or default to accent blue
        color_str = color_map.get(color_name, Colors.ACCENT_BLUE)
        
        # Create pixmap and fill with color
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        
        # Draw circle on pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(color_str))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 12, 12)
        painter.end()
        
        # Set the pixmap as the icon
        self.addPixmap(pixmap)


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
    
    def __init__(self, parent=None):
        """
        Initialize the ControlPanel widget.
        
        Args:
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        
        # Create UI components
        self.setup_ui()
        
        # Initialize timer for progress updates
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.collection_start_time = None
        self.collection_duration = 60  # seconds
        
    def setup_ui(self):
        """Set up the widget UI components."""
        # Main layout - vertical scrollable container
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create a scroll area to handle potential overflow
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # Container widget for all controls
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(15)
        
        # Setup the control sections
        self.create_data_collection_controls(scroll_layout)
        self.create_scatter_controls(scroll_layout)
        self.create_heatmap_controls(scroll_layout)
        self.create_analysis_controls(scroll_layout)
        
        # Add a spacer at the bottom for aesthetics
        scroll_layout.addStretch()
        
        # Set the scroll area widget
        scroll_area.setWidget(scroll_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
    
    def create_scatter_controls(self, parent_layout):
        """
        Create the scatter plot control section with multiple circle controls.
        
        Args:
            parent_layout: Parent layout to add widgets to.
        """
        scatter_group = QGroupBox("Sampling Circles")
        scatter_group.setObjectName("scatter-group")
        scatter_layout = QVBoxLayout(scatter_group)
        
        # Circle selection tabs
        self.circle_tabs = QTabWidget()
        
        # Create a tab for each circle
        circle_configs = [
            {'name': 'Primary', 'color': 'lime', 'distance': 5, 'angle': 0, 'enabled': True},
            {'name': 'Left', 'color': 'cyan', 'distance': 15, 'angle': -60, 'enabled': False},
            {'name': 'Right', 'color': 'yellow', 'distance': 25, 'angle': 60, 'enabled': False}
        ]
        
        self.circle_controls = []
        
        for i, config in enumerate(circle_configs):
            tab = QWidget()
            tab_layout = QGridLayout(tab)
            tab_layout.setContentsMargins(10, 15, 10, 10)
            tab_layout.setVerticalSpacing(15)
            
            # Circle enable checkbox in first row
            enable_check = QCheckBox("Enable circle")
            enable_check.setChecked(config['enabled'])
            enable_check.setStyleSheet("font-weight: bold;")
            enable_check.stateChanged.connect(lambda state, idx=i: self.on_circle_toggle(idx, state))
            tab_layout.addWidget(enable_check, 0, 0, 1, 2)
            
            # Circle distance controls - using SpinBox for precise input
            distance_layout = QHBoxLayout()
            distance_label = QLabel("Distance:")
            distance_spin = QSpinBox()
            distance_spin.setRange(1, 35)
            distance_spin.setValue(config['distance'])
            distance_spin.setSuffix(" m")
            distance_spin.setFixedWidth(80)
            distance_spin.valueChanged.connect(lambda value, idx=i: self.on_circle_distance_changed(idx, value))
            
            distance_layout.addWidget(distance_label)
            distance_layout.addWidget(distance_spin)
            distance_layout.addStretch()
            tab_layout.addLayout(distance_layout, 1, 0, 1, 2)
            
            # Circle radius controls with modern slider
            radius_slider = ModernSlider(
                label="Radius:",
                min_val=0.1,
                max_val=3.0,
                default_val=0.5,
                scale=0.01,
                suffix=" m",
                precision=2
            )
            radius_slider.valueChanged.connect(lambda value, idx=i: self.on_circle_radius_changed(idx, value))
            tab_layout.addWidget(radius_slider, 2, 0, 1, 2)
            
            # Circle angle controls
            angle_layout = QHBoxLayout()
            angle_label = QLabel("Angle:")
            angle_spin = QSpinBox()
            angle_spin.setRange(-90, 90)
            angle_spin.setValue(config['angle'])
            angle_spin.setSuffix("°")
            angle_spin.setFixedWidth(80)
            angle_spin.valueChanged.connect(lambda value, idx=i: self.on_circle_angle_changed(idx, value))
            
            angle_layout.addWidget(angle_label)
            angle_layout.addWidget(angle_spin)
            angle_layout.addStretch()
            tab_layout.addLayout(angle_layout, 3, 0, 1, 2)
            
            # Add vertical spacer
            tab_layout.setRowStretch(4, 1)
            
            # Store controls for this circle
            self.circle_controls.append({
                'enable': enable_check,
                'distance': distance_spin,
                'radius': radius_slider,
                'angle': angle_spin
            })
            
            # Add tab with color styling
            self.circle_tabs.addTab(tab, config['name'])
            
            # Apply color styling to tab
            self.circle_tabs.setTabIcon(i, ColorCircleIcon(config['color']))
        
        scatter_layout.addWidget(self.circle_tabs)
        
        # Add global controls in a separate frame with horizontal layout
        global_controls_frame = QFrame()
        global_controls_frame.setObjectName("global-controls")
        global_controls_layout = QHBoxLayout(global_controls_frame)
        global_controls_layout.setContentsMargins(0, 10, 0, 0)
        
        enable_all_button = QPushButton("Enable All")
        enable_all_button.clicked.connect(self.enable_all_circles)
        enable_all_button.setMinimumHeight(30)
        global_controls_layout.addWidget(enable_all_button)
        
        disable_all_button = QPushButton("Disable All")
        disable_all_button.clicked.connect(self.disable_all_circles)
        disable_all_button.setMinimumHeight(30)
        global_controls_layout.addWidget(disable_all_button)
        
        scatter_layout.addWidget(global_controls_frame)
        
        parent_layout.addWidget(scatter_group)
    
    def create_data_collection_controls(self, parent_layout):
        """
        Create the data collection control section.
        
        Args:
            parent_layout: Parent layout to add widgets to.
        """
        collection_group = QGroupBox("Data Collection")
        collection_layout = QVBoxLayout(collection_group)
        collection_layout.setSpacing(15)
        
        # Config and target distance controls - form layout for neat alignment
        params_layout = QFormLayout()
        params_layout.setVerticalSpacing(10)
        params_layout.setHorizontalSpacing(15)
        
        # Configuration name input
        self.config_entry = QLineEdit("default_config")
        self.config_entry.setMinimumHeight(30)
        self.config_entry.setPlaceholderText("Enter configuration name")
        params_layout.addRow("Config:", self.config_entry)
        
        # Target distance dropdown
        self.target_combo = QComboBox()
        self.target_combo.addItems([str(d) for d in range(5, 40, 5)])
        self.target_combo.setCurrentIndex(0)
        self.target_combo.setMinimumHeight(30)
        params_layout.addRow("Target:", self.target_combo)
        
        # Duration spinner
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(10, 300)
        self.duration_spin.setValue(60)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setMinimumHeight(30)
        params_layout.addRow("Duration:", self.duration_spin)
        
        collection_layout.addLayout(params_layout)
        
        # Action buttons in horizontal layout with improved styling
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)
        
        # Start button with primary styling
        self.start_button = QPushButton("Start Collection")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.on_start_collection)
        self.start_button.setMinimumHeight(35)
        self.start_button.setIcon(QIcon(":/icons/start.png"))
        buttons_layout.addWidget(self.start_button)
        
        # Stop button with danger styling
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.on_stop_collection)
        self.stop_button.setEnabled(False)
        self.stop_button.setMinimumHeight(35)
        self.stop_button.setIcon(QIcon(":/icons/stop.png"))
        buttons_layout.addWidget(self.stop_button)
        
        collection_layout.addWidget(buttons_frame)
        
        # Status indicator and progress bar
        status_layout = QVBoxLayout()
        status_layout.setSpacing(5)
        
        # Status label with modern styling
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        # Modern progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(15)
        status_layout.addWidget(self.progress_bar)
        
        collection_layout.addLayout(status_layout)
        
        parent_layout.addWidget(collection_group)
    
    def create_heatmap_controls(self, parent_layout):
        """
        Create the heatmap control section.
        
        Args:
            parent_layout: Parent layout to add widgets to.
        """
        heatmap_group = QGroupBox("Visualization Controls")
        heatmap_layout = QVBoxLayout(heatmap_group)
        heatmap_layout.setSpacing(12)
        
        # Top controls grid: reset, colormap
        top_layout = QGridLayout()
        top_layout.setVerticalSpacing(10)
        top_layout.setHorizontalSpacing(15)
        
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.on_reset_heatmap)
        self.reset_button.setMinimumHeight(30)
        self.reset_button.setIcon(QIcon(":/icons/reset.png"))
        top_layout.addWidget(self.reset_button, 0, 0)
        
        # Colormap combo
        colormap_label = QLabel("Colormap:")
        colormap_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_layout.addWidget(colormap_label, 0, 1)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["plasma", "viridis", "inferno", "magma", "jet"])
        self.colormap_combo.setCurrentIndex(0)
        self.colormap_combo.setMinimumHeight(30)
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        top_layout.addWidget(self.colormap_combo, 0, 2)
        
        heatmap_layout.addLayout(top_layout)
        
        # Modern sliders for decay, noise, and smoothing
        self.decay_slider = ModernSlider(
            label="Decay Factor:",
            min_val=0.8,
            max_val=0.999,
            default_val=0.98,
            scale=0.001,
            precision=3
        )
        self.decay_slider.valueChanged.connect(self.on_decay_changed)
        heatmap_layout.addWidget(self.decay_slider)
        
        self.noise_slider = ModernSlider(
            label="Noise Floor:",
            min_val=0.01,
            max_val=0.2,
            default_val=0.05,
            scale=0.01,
            precision=2
        )
        self.noise_slider.valueChanged.connect(self.on_noise_changed)
        heatmap_layout.addWidget(self.noise_slider)
        
        self.smooth_slider = ModernSlider(
            label="Smoothing:",
            min_val=0.5,
            max_val=5.0,
            default_val=2.0,
            scale=0.1,
            precision=1
        )
        self.smooth_slider.valueChanged.connect(self.on_smoothing_changed)
        heatmap_layout.addWidget(self.smooth_slider)
        
        # Visualization mode radio buttons in horizontal layout
        mode_frame = QFrame()
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(0, 5, 0, 5)
        
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        self.vis_mode_group = QButtonGroup(self)
        
        heat_radio = QRadioButton("Heatmap")
        heat_radio.setChecked(True)
        heat_radio.toggled.connect(lambda checked: checked and self.on_vis_mode_changed("heatmap"))
        self.vis_mode_group.addButton(heat_radio)
        mode_layout.addWidget(heat_radio)
        
        contour_radio = QRadioButton("Contour")
        contour_radio.toggled.connect(lambda checked: checked and self.on_vis_mode_changed("contour"))
        self.vis_mode_group.addButton(contour_radio)
        mode_layout.addWidget(contour_radio)
        
        combined_radio = QRadioButton("Combined")
        combined_radio.toggled.connect(lambda checked: checked and self.on_vis_mode_changed("combined"))
        self.vis_mode_group.addButton(combined_radio)
        mode_layout.addWidget(combined_radio)
        
        heatmap_layout.addWidget(mode_frame)
        
        parent_layout.addWidget(heatmap_group)
    
    def create_analysis_controls(self, parent_layout):
        """
        Create the analysis control section.
        
        Args:
            parent_layout: Parent layout to add widgets to.
        """
        analysis_group = QGroupBox("Analysis & Export")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(10)
        
        # ROI controls in a horizontal layout
        roi_frame = QFrame()
        roi_layout = QHBoxLayout(roi_frame)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        
        self.add_roi_button = QPushButton("Add ROI")
        self.add_roi_button.clicked.connect(lambda: self.add_roi.emit())
        self.add_roi_button.setMinimumHeight(30)
        self.add_roi_button.setIcon(QIcon(":/icons/add.png"))
        roi_layout.addWidget(self.add_roi_button)
        
        self.clear_rois_button = QPushButton("Clear ROIs")
        self.clear_rois_button.clicked.connect(lambda: self.clear_rois.emit())
        self.clear_rois_button.setMinimumHeight(30)
        self.clear_rois_button.setIcon(QIcon(":/icons/remove.png"))
        roi_layout.addWidget(self.clear_rois_button)
        
        analysis_layout.addWidget(roi_frame)
        
        # Export controls in a horizontal layout
        export_frame = QFrame()
        export_layout = QHBoxLayout(export_frame)
        export_layout.setContentsMargins(0, 0, 0, 0)
        
        self.save_heatmap_button = QPushButton("Save Heatmap")
        self.save_heatmap_button.clicked.connect(lambda: self.save_heatmap.emit())
        self.save_heatmap_button.setMinimumHeight(30)
        self.save_heatmap_button.setIcon(QIcon(":/icons/save.png"))
        export_layout.addWidget(self.save_heatmap_button)
        
        self.export_plot_button = QPushButton("Export Plot")
        self.export_plot_button.clicked.connect(lambda: self.export_plot.emit())
        self.export_plot_button.setMinimumHeight(30)
        self.export_plot_button.setIcon(QIcon(":/icons/export.png"))
        export_layout.addWidget(self.export_plot_button)
        
        analysis_layout.addWidget(export_frame)
        
        # Generate report button (primary action)
        self.report_button = QPushButton("Generate Report")
        self.report_button.setObjectName("primary")
        self.report_button.clicked.connect(lambda: self.generate_report.emit())
        self.report_button.setMinimumHeight(35)
        self.report_button.setIcon(QIcon(":/icons/report.png"))
        analysis_layout.addWidget(self.report_button)
        
        # Metrics display in a stylized frame
        metrics_frame = QFrame()
        metrics_frame.setObjectName("metrics-frame")
        metrics_frame.setStyleSheet(f"""
            #metrics-frame {{
                background-color: {Colors.LIGHT_BACKGROUND};
                border-radius: 5px;
                padding: 5px;
            }}
        """)
        metrics_layout = QVBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(10, 10, 10, 10)
        
        metrics_header = QLabel("Analysis Metrics")
        metrics_header.setStyleSheet(f"color: {Colors.TEXT}; font-weight: bold;")
        metrics_layout.addWidget(metrics_header)
        
        self.metrics_label = QLabel("No analysis data available")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        metrics_layout.addWidget(self.metrics_label)
        
        analysis_layout.addWidget(metrics_frame)
        
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
    
    def on_circle_radius_changed(self, index, value):
        """
        Handle circle radius slider change.
        
        Args:
            index: Index of circle being modified (0-2)
            value: New radius value
        """
        self.circle_radius_changed.emit(index, value)
    
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
        self.progress_bar.setValue(0)
        
        self.collection_start_time = QTimer.currentTime()
        self.collection_duration = duration
        self.progress_timer.start(100)  # Update every 0.1 seconds
        
        self.set_status(f"Collecting: {config_name}@{target_distance}m")
    
    def on_stop_collection(self):
        """Handle stop collection button click."""
        self.stop_collection.emit()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_timer.stop()
        self.progress_bar.setValue(0)
        self.set_status("Collection stopped")
    
    def update_progress(self):
        """Update the progress bar during data collection."""
        if self.collection_start_time is None:
            return
        
        # Calculate elapsed time in seconds
        elapsed = QTimer.currentTime().msecsTo(self.collection_start_time) / -1000.0
        progress = (elapsed / self.collection_duration) * 100
        
        if progress >= 100:
            self.progress_bar.setValue(100)
            self.progress_timer.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.set_status("Collection complete")
            return
        
        self.progress_bar.setValue(int(progress))
    
    def set_status(self, message):
        """
        Set the status label text.
        
        Args:
            message: Status message to display.
        """
        self.status_label.setText(message)
    
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
            value: New decay factor value.
        """
        self.decay_factor_changed.emit(value)
    
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
            value: New noise floor value.
        """
        self.noise_floor_changed.emit(value)
    
    def on_smoothing_changed(self, value):
        """
        Handle smoothing slider change.
        
        Args:
            value: New smoothing factor value.
        """
        self.smoothing_changed.emit(value)
    
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
        
        # Format metrics in a more readable way
        text = f"""
        <table style='width:100%'>
            <tr>
                <td><b>Max Intensity:</b></td>
                <td>{metrics.get('max_intensity', 0):.2f}</td>
                <td><b>Avg Intensity:</b></td>
                <td>{metrics.get('avg_intensity', 0):.2f}</td>
            </tr>
            <tr>
                <td><b>SNR:</b></td>
                <td>{metrics.get('snr_dB', 0):.1f} dB</td>
                <td><b>Coverage:</b></td>
                <td>{metrics.get('coverage_percentage', 0):.1f}%</td>
            </tr>
        </table>
        """
        self.metrics_label.setText(text)