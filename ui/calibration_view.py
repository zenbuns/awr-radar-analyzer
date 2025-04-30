#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration view widget for managing intrinsic and extrinsic parameters.
"""

import os
import sys # <-- Add sys import for path debugging
import yaml
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
                             QLineEdit, QPushButton, QFileDialog, QMessageBox,
                             QFormLayout, QSpacerItem, QSizePolicy, QSplitter, QSpinBox, QFrame,
                             QListWidget, QAbstractItemView, QListWidgetItem, QApplication, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, pyqtSignal # <-- Added pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time
import traceback
import collections # <-- Import collections for deque
from scipy.optimize import least_squares # For LM refinement (optional)
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import median_abs_deviation, zscore
# --- Add scikit-learn import ---
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors # Added for outlier removal
# --- End Add ---
try:
    import numba
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not found. MAD filtering will run unaccelerated.")

# Check for ROS2 availability and dependencies
ROS2_AVAILABLE = False
CV_BRIDGE_AVAILABLE = False
CvBridge = None
# --- Add YOLO Msg Import ---
YOLO_MSGS_AVAILABLE = False
DetectionArray = None
# --- End Add ---
try:
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    # --- Add YOLO Msg Import Attempt ---
    try:
        from yolo_msgs.msg import DetectionArray as DetectionArray_import
        DetectionArray = DetectionArray_import
        YOLO_MSGS_AVAILABLE = True
        print("CalibrationView: yolo_msgs imported successfully.")
    except ImportError as e:
        print(f"CalibrationView: yolo_msgs import error: {e}. YOLO overlay will be unavailable.")
        YOLO_MSGS_AVAILABLE = False
        # --- Add print traceback on error ---
        import traceback
        traceback.print_exc()
        # --- End Add ---
    # --- End Add ---
    ROS2_AVAILABLE = True
    try:
        import importlib.util
        cv_bridge_spec = importlib.util.find_spec("cv_bridge")
        if cv_bridge_spec:
            try:
                from cv_bridge import CvBridge as CvBridge_import
                CvBridge = CvBridge_import
                CV_BRIDGE_AVAILABLE = True
                print("CalibrationView: cv_bridge imported successfully.")
            except AttributeError as e:
                if "_ARRAY_API not found" in str(e):
                    print("CalibrationView: NumPy 2.0 compatibility issue detected with cv_bridge.")
                else: raise e
                CV_BRIDGE_AVAILABLE = False
            except ImportError as e:
                print(f"CalibrationView: CV_Bridge import error: {e}")
                CV_BRIDGE_AVAILABLE = False
        else:
            print("CalibrationView: cv_bridge package not found.")
            CV_BRIDGE_AVAILABLE = False
    except Exception as e:
        print(f"CalibrationView: Error checking/importing cv_bridge: {e}")
        CV_BRIDGE_AVAILABLE = False
except ImportError as e:
    print(f"CalibrationView: ROS2 core dependencies import error: {e}")

# --- Configuration Structs ---
class Param:
    """ Equivalent to C++ param struct for filter/processing config. """
    def __init__(self, name="", num=0, if_channel=False, if_reverse=False,
                 lower_limit=-np.inf, upper_limit=np.inf, eps=0.0):
        self.name = name
        self.num = num # Maybe axis index (0=x, 1=y, 2=z)?
        self.if_channel = if_channel # Unclear usage from C++
        self.if_reverse = if_reverse # For box filter (keep points inside or outside)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.eps = eps # Unclear usage from C++

class Config:
    """ Equivalent to C++ config struct holding multiple Params. """
    def __init__(self):
        self.x = Param(name="x", num=0)
        self.y = Param(name="y", num=1)
        self.z = Param(name="z", num=2)
        self.i = Param(name="intensity") # Assuming 'i' is intensity

# Example usage:
# filter_config = Config()
# filter_config.x.lower_limit = -5.0
# filter_config.x.upper_limit = 5.0
# filter_config.y.lower_limit = 0.1
# filter_config.y.upper_limit = 10.0
# filter_config.z.lower_limit = -1.0
# filter_config.z.upper_limit = 1.0
# filter_config.x.if_reverse = False # Keep points inside the box limits
# --- End Configuration Structs ---


class CalibrationView(QWidget):
    """
    Widget for handling camera intrinsic and radar-camera extrinsic calibration,
    following the method described in Cheng et al., arXiv:2307.15264.
    """
    # --- Decay config ---
    RADAR_DECAY_SECONDS = 1.5   # How long (seconds) points remain visible (reduced for better sync)
    RADAR_DECAY_TYPE = 'exp'    # 'exp' for exponential, 'linear' for linear fade

    # --- Signal for thread-safe frame update ---
    new_frame_signal = pyqtSignal(np.ndarray, float) # frame (np.ndarray), timestamp (float)

    def __init__(self, parent=None):
        """Initialize the CalibrationView widget."""
        super().__init__(parent)
        self.main_window = parent
        # FIX 4: Corrected history comment
        self.radar_projection_history = []  # Each entry: (u, v, timestamp)

        # Store intrinsic parameters - Initialize with new user-provided values
        self.camera_matrix = np.array([
            [1283.0, 0.0, 640.0],
            [0.0, 962.25, 360.0],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([
            [0.11480806073904032,
             -0.21946985653851792,
              0.0012002116999769957,
              0.008564577708855225,
              0.11274677130853494]
        ])

        # Checkerboard parameters (for intrinsic calibration ONLY)
        self.checkerboard_width = 9
        self.checkerboard_height = 8
        self.checkerboard_square_size = 0.02214 # meters (22.14 mm from paper)
        self.captured_intrinsic_data = [] # Stores pairs of (objectPoints, imagePoints) for intrinsic calib

        # Extrinsic calibration data (CR-based PnP)
        self.point_pairs = [] # List to store ((radar_CR_x,y,z), (image_click_u,v)) tuples
        self.last_projections = [] # Store last projected points (u_proj, v_proj) for visualization
        self.show_projections = True # Flag to control drawing of saved pair projections
        self.last_successful_click_info = {"pos": None, "time": 0} # For temporary click feedback
        self.extrinsic_R = np.eye(3)
        self.extrinsic_T = np.zeros((3, 1))
        self.avg_reprojection_error = 0.0

        # Camera feed related attributes
        self.latest_camera_frame = None
        self.latest_camera_timestamp = None
        self.frame_received = False
        self.camera_subscription = None
        self.cv_bridge = None
        # --- Add YOLO attributes ---
        self.yolo_subscription = None
        self.latest_yolo_detections = None # Stores the latest yolo_msgs/DetectionArray
        # --- Change for Tracking --- 
        self.latest_yolo_tracking = None # Stores the latest tracking message
        self.yolo_track_counts = {} # Stores history: {track_id: deque([count1, count2,...])}
        self.YOLO_COUNT_HISTORY_LEN = 10 # Number of frames for moving average
        self.YOLO_TRACK_TIMEOUT = 1.0 # Seconds before removing an old track ID
        self.last_yolo_update_time = {} # {track_id: timestamp}
        self.yolo_tracking_topic = "/yolo/tracking" # Switched topic
        # --- End Change ---
        # --- End Add ---
        if CV_BRIDGE_AVAILABLE and CvBridge:
            try:
                self.cv_bridge = CvBridge()
            except Exception as e:
                print(f"Error initializing CvBridge in CalibrationView: {e}")
                self.cv_bridge = None

        self.setup_ui()
        self.init_camera_subscriber()
        # --- Add YOLO subscriber init ---
        self.init_yolo_subscriber()
        # --- End Add ---
        # Ensure set_intrinsics is called with the new defaults for proper validation and display
        self.set_intrinsics(self.camera_matrix, self.dist_coeffs)
        self.update_intrinsic_display() # Update display with initial/default values
        self.update_extrinsic_display()

        # --- Manual Adjustment Mode ---
        self.manual_adjust_mode = False
        self.manual_adjust_step = 0.01  # meters for translation
        self.manual_rotate_step_deg = 0.5  # degrees for rotation
        self.manual_adjust_instr_label = None  # set in setup_ui

        # FIX 6: Connect the signal to the slot
        self.new_frame_signal.connect(self._update_latest_frame)

    def setup_ui(self):
        """Set up the UI components for the calibration tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)

        # --- Intrinsic Calibration Group ---
        intrinsic_group = QGroupBox("1. Intrinsic Camera Calibration (using Checkerboard)")
        intrinsic_layout = QVBoxLayout()

        # Intrinsic Calibration Controls
        intrinsic_capture_group = QGroupBox("Capture Controls & Status") # GroupBox for capture section
        intrinsic_capture_layout = QHBoxLayout() # Layout for this specific section
        self.capture_intrinsic_button = QPushButton("Capture Checkerboard Image")
        self.capture_intrinsic_button.clicked.connect(self.capture_checkerboard_image)
        self.calibrate_intrinsic_button = QPushButton("Run Intrinsic Calibration")
        self.calibrate_intrinsic_button.clicked.connect(self.run_intrinsic_calibration)
        self.captured_images_list = QListWidget()
        self.captured_images_list.setMaximumHeight(100)
        self.clear_intrinsic_captures_button = QPushButton("Clear Intrinsic Captures")
        self.clear_intrinsic_captures_button.clicked.connect(self.clear_intrinsic_captures)
        capture_controls_vbox = QVBoxLayout()
        capture_controls_vbox.addWidget(self.capture_intrinsic_button)
        capture_controls_vbox.addWidget(self.calibrate_intrinsic_button)
        capture_controls_vbox.addWidget(self.clear_intrinsic_captures_button)
        intrinsic_capture_layout.addLayout(capture_controls_vbox)
        intrinsic_capture_layout.addWidget(self.captured_images_list, 1)
        intrinsic_capture_group.setLayout(intrinsic_capture_layout) # Set layout for the capture group box
        intrinsic_layout.addWidget(intrinsic_capture_group) # Add the group box widget here

        # Checkerboard Configuration
        checkerboard_layout = QFormLayout()
        self.cb_width_input = QSpinBox()
        self.cb_width_input.setRange(2, 50)
        self.cb_width_input.setValue(self.checkerboard_width)
        self.cb_width_input.valueChanged.connect(lambda val: setattr(self, 'checkerboard_width', val))
        self.cb_height_input = QSpinBox()
        self.cb_height_input.setRange(2, 50)
        self.cb_height_input.setValue(self.checkerboard_height)
        self.cb_height_input.valueChanged.connect(lambda val: setattr(self, 'checkerboard_height', val))
        self.cb_square_size_input = QLineEdit(str(self.checkerboard_square_size))
        self.cb_square_size_input.textChanged.connect(self._update_square_size)
        checkerboard_layout.addRow("Checkerboard Width (corners):", self.cb_width_input)
        checkerboard_layout.addRow("Checkerboard Height (corners):", self.cb_height_input)
        checkerboard_layout.addRow("Square Size (meters):", self.cb_square_size_input)
        intrinsic_layout.addLayout(checkerboard_layout)

        # Separator
        separator_int = QFrame()
        separator_int.setFrameShape(QFrame.HLine)
        separator_int.setFrameShadow(QFrame.Sunken)
        intrinsic_layout.addWidget(separator_int)

        # Display Matrix K
        k_matrix_layout = QFormLayout()
        self.fx_label = QLabel("...")
        self.fy_label = QLabel("...")
        self.cx_label = QLabel("...")
        self.cy_label = QLabel("...")
        k_matrix_layout.addRow("Focal Length (fx):", self.fx_label)
        k_matrix_layout.addRow("Focal Length (fy):", self.fy_label)
        k_matrix_layout.addRow("Principal Point (cx):", self.cx_label)
        k_matrix_layout.addRow("Principal Point (cy):", self.cy_label)
        intrinsic_layout.addLayout(k_matrix_layout)

        # Display Distortion D
        dist_layout = QFormLayout()
        self.d1_label = QLabel("...") # k1
        self.d2_label = QLabel("...") # k2
        self.d3_label = QLabel("...") # p1
        self.d4_label = QLabel("...") # p2
        self.d5_label = QLabel("...") # k3
        dist_layout.addRow("Distortion (k1):", self.d1_label)
        dist_layout.addRow("Distortion (k2):", self.d2_label)
        dist_layout.addRow("Distortion (p1):", self.d3_label)
        dist_layout.addRow("Distortion (p2):", self.d4_label)
        dist_layout.addRow("Distortion (k3):", self.d5_label)
        intrinsic_layout.addLayout(dist_layout)

        # Buttons for Load/Save Intrinsics
        intrinsic_buttons_layout = QHBoxLayout()
        load_intrinsics_button = QPushButton("Load Intrinsics")
        load_intrinsics_button.clicked.connect(self.load_intrinsics)
        save_intrinsics_button = QPushButton("Save Intrinsics")
        save_intrinsics_button.clicked.connect(self.save_intrinsics)
        intrinsic_buttons_layout.addWidget(load_intrinsics_button)
        intrinsic_buttons_layout.addWidget(save_intrinsics_button)
        intrinsic_layout.addLayout(intrinsic_buttons_layout)

        intrinsic_group.setLayout(intrinsic_layout)
        # --- End Intrinsic Group ---

        # --- Extrinsic Calibration Group ---
        extrinsic_group = QGroupBox("2. Extrinsic Radar-Camera Calibration (using Corner Reflector)")
        extrinsic_main_layout = QVBoxLayout()

        # Splitter for Camera Feed and Extrinsic Controls
        extrinsic_splitter = QSplitter(Qt.Horizontal)

        # Camera Feed & Clicking Area (Left side)
        camera_feed_group = QGroupBox("Data Collection: Click on Corner Reflector in Image")
        camera_feed_layout = QVBoxLayout()
        self.camera_feed_label = QLabel()
        self.camera_feed_label.setAlignment(Qt.AlignCenter)
        self.camera_feed_label.setStyleSheet("background-color: #222; border: 1px solid #888;")
        self.camera_feed_label.setMinimumSize(640, 480)
        self.camera_feed_label.mousePressEvent = self.camera_feed_mouse_press
        camera_feed_layout.addWidget(self.camera_feed_label)
        # Add a label for radar CR data quality feedback
        self.cr_quality_label = QLabel("")
        self.cr_quality_label.setStyleSheet("color: #0af; font-size: 12px;")
        camera_feed_layout.addWidget(self.cr_quality_label)
        camera_feed_group.setLayout(camera_feed_layout)
        extrinsic_splitter.addWidget(camera_feed_group)

        # Extrinsic Controls and Results Area (Right side)
        extrinsic_controls_group = QGroupBox("Calculation & Results")
        extrinsic_controls_layout = QVBoxLayout()

        # Point Pair Management
        extrinsic_controls_layout.addWidget(QLabel("Collected Point Pairs (Radar CR 3D <-> Image Click 2D):"))
        self.point_pair_list = QListWidget()
        self.point_pair_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.point_pair_list.setMaximumHeight(150)
        extrinsic_controls_layout.addWidget(self.point_pair_list)
        pair_buttons_layout = QHBoxLayout()
        remove_pair_button = QPushButton("Remove Selected Pair")
        remove_pair_button.clicked.connect(self.remove_selected_pair)
        clear_pairs_button = QPushButton("Clear All Pairs")
        clear_pairs_button.clicked.connect(self.clear_all_pairs)
        pair_buttons_layout.addWidget(remove_pair_button)
        pair_buttons_layout.addWidget(clear_pairs_button)
        extrinsic_controls_layout.addLayout(pair_buttons_layout)

        # Button to Calculate Extrinsics (PnP + Refinement)
        calculate_extrinsics_button = QPushButton("Calculate Extrinsics (RANSAC + Refine)")
        calculate_extrinsics_button.setToolTip("Requires at least 4 point pairs. Runs RANSAC (Iterative & SQPnP) + LM Refinement.")
        calculate_extrinsics_button.clicked.connect(self.calculate_extrinsics)
        extrinsic_controls_layout.addWidget(calculate_extrinsics_button)

        # Separator
        separator_ext = QFrame()
        separator_ext.setFrameShape(QFrame.HLine)
        separator_ext.setFrameShadow(QFrame.Sunken)
        extrinsic_controls_layout.addWidget(separator_ext)

        # Display Extrinsic Results (R, T, Error)
        extrinsic_results_layout = QFormLayout()
        self.r_label = QLabel("...")
        self.t_label = QLabel("...")
        self.reprojection_error_label = QLabel("...")
        extrinsic_results_layout.addRow("Rotation (R):", self.r_label)
        extrinsic_results_layout.addRow("Translation (T):", self.t_label)
        extrinsic_results_layout.addRow("Avg Reproj Err (px):", self.reprojection_error_label)
        extrinsic_controls_layout.addLayout(extrinsic_results_layout)

        # --- Add Checkbox for Live Projection ---
        self.show_live_projection_checkbox = QCheckBox("Show Live Radar Projection")
        self.show_live_projection_checkbox.setChecked(True) # Default to on
        self.show_live_projection_checkbox.setToolTip("Toggle the display of real-time radar points projected onto the camera feed.")
        extrinsic_controls_layout.addWidget(self.show_live_projection_checkbox)
        # --- End Checkbox ---

        # --- Add Checkbox for YOLO Overlay ---
        self.show_yolo_checkbox = QCheckBox("Show YOLO Detections & Counts")
        self.show_yolo_checkbox.setChecked(False) # Default to off
        self.show_yolo_checkbox.setToolTip("Toggle the display of YOLO bounding boxes and radar point counts within them.")
        # Disable if YOLO msgs are not available
        if not YOLO_MSGS_AVAILABLE:
            self.show_yolo_checkbox.setEnabled(False)
            self.show_yolo_checkbox.setToolTip("YOLO overlay unavailable (yolo_msgs not found or import error).")
        extrinsic_controls_layout.addWidget(self.show_yolo_checkbox)
        # --- End Add ---

        # --- Manual Adjustment Controls (Sliders) ---
        from PyQt5.QtWidgets import QSlider, QGridLayout
        self.slider_group = QGroupBox("Manual Adjustment (Sliders)")
        slider_layout = QGridLayout()
        # Translation sliders
        self.trans_sliders = []
        self.trans_labels = []
        trans_names = ['X (m)', 'Y (m)', 'Z (m)']
        trans_ranges = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
        for i, (name, (minv, maxv)) in enumerate(zip(trans_names, trans_ranges)):
            label = QLabel(f"{name}: 0.00")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(400)
            slider.setValue(200)
            # FIX 5: Capture minv/maxv by value in lambda
            slider.valueChanged.connect(lambda val, idx=i, minv=minv, maxv=maxv: self._slider_update_translation(idx, val, minv, maxv))
            slider_layout.addWidget(label, i, 0)
            slider_layout.addWidget(slider, i, 1)
            self.trans_sliders.append(slider)
            self.trans_labels.append(label)
        # Rotation sliders
        self.rot_sliders = []
        self.rot_labels = []
        rot_names = ['Yaw (deg)', 'Pitch (deg)', 'Roll (deg)']
        rot_ranges = [(-30, 30), (-30, 30), (-30, 30)]
        for i, (name, (minv, maxv)) in enumerate(zip(rot_names, rot_ranges)):
            label = QLabel(f"{name}: 0.0")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(600)
            slider.setValue(300)
            # FIX 5: Capture minv/maxv by value in lambda
            slider.valueChanged.connect(lambda val, idx=i, minv=minv, maxv=maxv: self._slider_update_rotation(idx, val, minv, maxv))
            slider_layout.addWidget(label, i+3, 0)
            slider_layout.addWidget(slider, i+3, 1)
            self.rot_sliders.append(slider)
            self.rot_labels.append(label)
        self.slider_group.setLayout(slider_layout)
        extrinsic_controls_layout.addWidget(self.slider_group)

        # Buttons for Load/Save Extrinsics
        extrinsic_io_buttons_layout = QHBoxLayout()
        load_extrinsics_button = QPushButton("Load Extrinsics")
        load_extrinsics_button.clicked.connect(self.load_extrinsics)
        save_extrinsics_button = QPushButton("Save Extrinsics")
        save_extrinsics_button.clicked.connect(self.save_extrinsics)
        extrinsic_io_buttons_layout.addWidget(load_extrinsics_button)
        extrinsic_io_buttons_layout.addWidget(save_extrinsics_button)
        extrinsic_controls_layout.addLayout(extrinsic_io_buttons_layout)

        extrinsic_controls_group.setLayout(extrinsic_controls_layout)
        extrinsic_splitter.addWidget(extrinsic_controls_group)

        # Set initial splitter sizes
        extrinsic_splitter.setSizes([600, 350]) # Give slightly more space to controls

        extrinsic_main_layout.addWidget(extrinsic_splitter)
        extrinsic_group.setLayout(extrinsic_main_layout)
        # --- End Extrinsic Group ---

        # Add Intrinsic and Extrinsic groups to main layout
        main_layout.addWidget(intrinsic_group)
        main_layout.addWidget(extrinsic_group)

        # Add spacer to push content up
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Start camera update timer
        self.camera_update_timer = QTimer(self)
        self.camera_update_timer.timeout.connect(self.update_camera_display)
        self.camera_update_timer.start(33) # ~30 FPS

    def update_intrinsic_display(self):
        """Update the UI labels with current intrinsic parameters."""
        if isinstance(self.camera_matrix, np.ndarray) and self.camera_matrix.shape == (3, 3):
             self.fx_label.setText(f"{self.camera_matrix[0, 0]:.4f}")
             self.fy_label.setText(f"{self.camera_matrix[1, 1]:.4f}")
             self.cx_label.setText(f"{self.camera_matrix[0, 2]:.4f}")
             self.cy_label.setText(f"{self.camera_matrix[1, 2]:.4f}")
        else:
             self.fx_label.setText("...")
             self.fy_label.setText("...")
             self.cx_label.setText("...")
             self.cy_label.setText("...")

        if isinstance(self.dist_coeffs, np.ndarray):
             d_coeffs = self.dist_coeffs.flatten()
             labels = [self.d1_label, self.d2_label, self.d3_label, self.d4_label, self.d5_label]
             for i, label in enumerate(labels):
                 if i < len(d_coeffs):
                     label.setText(f"{d_coeffs[i]:.6f}")
                 else:
                     label.setText("...")
        else:
             for label in [self.d1_label, self.d2_label, self.d3_label, self.d4_label, self.d5_label]:
                 label.setText("...")

    def set_intrinsics(self, K, D):
         """Set the intrinsic parameters, update display, and clear dependent data."""
         valid_K = isinstance(K, np.ndarray) and K.shape == (3, 3)
         valid_D = isinstance(D, np.ndarray) and (D.ndim == 1 or D.shape[0] == 1)

         if not valid_K:
             QMessageBox.warning(self, "Error", "Invalid Camera Matrix (K) format provided.")
             return
         if not valid_D:
             QMessageBox.warning(self, "Error", "Invalid Distortion Coefficients (D) format provided.")
             return

         self.camera_matrix = K
         print(f"[set_intrinsics] Assigned self.camera_matrix:\n{self.camera_matrix}")
         # Ensure distortion coeffs have exactly 5 elements, pad/truncate if necessary
         d_flat = D.flatten()
         padded_D = np.zeros(5)
         elements_to_copy = min(len(d_flat), 5)
         padded_D[:elements_to_copy] = d_flat[:elements_to_copy]
         self.dist_coeffs = padded_D.reshape(1, 5)
         print(f"[set_intrinsics] Assigned self.dist_coeffs: {self.dist_coeffs}")

         self.update_intrinsic_display()
         print("Intrinsic parameters set and display updated.")

         # Crucially, extrinsic calibration depends on intrinsics.
         # Clear existing extrinsic pairs and results as they are now invalid.
         print("Clearing extrinsic point pairs and results due to new intrinsics.")
         self.clear_all_pairs() # Clears self.point_pairs, self.last_projections, UI list
         self.clear_extrinsic_results() # Clears R, T, error, UI display

    @pyqtSlot()
    def load_intrinsics(self):
        """Load intrinsic parameters from a YAML file."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Intrinsic Camera Parameters", "",
                                                   "YAML Files (*.yaml *.yml);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                if 'camera_matrix' not in data or 'dist_coeffs' not in data:
                     raise ValueError("YAML file missing 'camera_matrix' or 'dist_coeffs'.")

                K = np.array(data['camera_matrix'])
                D = np.array(data['dist_coeffs'])
                self.set_intrinsics(K, D) # Use the setter for validation and updates
                print(f"[load_intrinsics] Value of self.camera_matrix after set_intrinsics:\n{self.camera_matrix}")
                QMessageBox.information(self, "Success", f"Loaded intrinsics from {os.path.basename(file_path)}")

            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load intrinsics: {str(e)}")

    @pyqtSlot()
    def save_intrinsics(self):
        """Save current intrinsic parameters to a YAML file."""
        # Check if intrinsics are valid before saving
        if np.allclose(self.camera_matrix, np.eye(3)) or np.all(self.dist_coeffs == 0):
            reply = QMessageBox.question(self, "Save Default Intrinsics?",
                                         "Camera matrix is identity or distortion is zero. Save these defaults?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Intrinsic Camera Parameters", "intrinsics.yaml",
                                                   "YAML Files (*.yaml *.yml);;All Files (*)", options=options)
        if file_path:
            if not file_path.lower().endswith(('.yaml', '.yml')):
                file_path += '.yaml'
            try:
                data = {
                    'camera_matrix': self.camera_matrix.tolist(),
                    'dist_coeffs': self.dist_coeffs.tolist()
                }
                with open(file_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=None, sort_keys=False, indent=2)
                QMessageBox.information(self, "Success", f"Saved intrinsics to {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save intrinsics: {str(e)}")

    def _update_square_size(self, text):
        """Validate and update the checkerboard square size attribute."""
        try:
            size = float(text)
            if size > 1e-6: # Must be positive
                self.checkerboard_square_size = size
            else:
                 # Handle invalid input visually? (e.g., red border)
                 pass
        except ValueError:
            # Handle invalid input visually?
            pass

    def update_extrinsic_display(self):
         """Update the UI labels with current extrinsic parameters."""
         if isinstance(self.extrinsic_R, np.ndarray) and self.extrinsic_R.shape == (3, 3):
             r_str = np.array2string(self.extrinsic_R, precision=4, separator=', ', suppress_small=True)
             self.r_label.setText(r_str)
             self.r_label.setToolTip(r_str)
         else:
             self.r_label.setText("...")
             self.r_label.setToolTip("")
         if isinstance(self.extrinsic_T, np.ndarray) and self.extrinsic_T.size == 3:
             t_str = np.array2string(self.extrinsic_T.flatten(), precision=4, separator=', ', suppress_small=True)
             self.t_label.setText(t_str)
             self.t_label.setToolTip(t_str)
         else:
             self.t_label.setText("...")
             self.t_label.setToolTip("")
         if self.avg_reprojection_error > 0:
             self.reprojection_error_label.setText(f"{self.avg_reprojection_error:.4f}")
             self.reprojection_error_label.setToolTip(f"{self.avg_reprojection_error:.6f} pixels")
         else:
             self.reprojection_error_label.setText("...")
             self.reprojection_error_label.setToolTip("")

    @pyqtSlot()
    def load_extrinsics(self):
        """Load extrinsic parameters from a YAML file."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Extrinsic Parameters", "",
                                                   "YAML Files (*.yaml *.yml);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                if 'rotation_matrix' not in data or 'translation_vector' not in data:
                     raise ValueError("YAML file missing 'rotation_matrix' or 'translation_vector'.")

                R = np.array(data['rotation_matrix'])
                T = np.array(data['translation_vector'])
                err = data.get('avg_reprojection_error_px', 0.0)

                if R.shape == (3, 3) and (T.shape == (3, 1) or T.shape == (3,)): # Validate shapes
                     self.extrinsic_R = R
                     self.extrinsic_T = T.reshape(3, 1)
                     self.avg_reprojection_error = float(err)
                     self.update_extrinsic_display()
                     QMessageBox.information(self, "Success", f"Loaded extrinsics from {os.path.basename(file_path)}")
                     # Re-project points for visualization if pairs exist
                     if self.point_pairs:
                         self.project_current_pairs_for_display()
                     self.update_point_pair_list_display()
                else:
                    raise ValueError("Invalid R or T shape in YAML file.")

            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load extrinsics: {str(e)}")
                self.clear_extrinsic_results()

    @pyqtSlot()
    def save_extrinsics(self):
        """Save current extrinsic parameters to a YAML file."""
        if np.allclose(self.extrinsic_T, 0) and np.allclose(self.extrinsic_R, np.eye(3)):
             QMessageBox.warning(self, "No Data", "Extrinsic parameters appear to be default or non-calculated. Saving may not be meaningful.")
             # Allow saving anyway? Or return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Extrinsic Parameters", "extrinsics_cr_pnp.yaml",
                                                   "YAML Files (*.yaml *.yml);;All Files (*)", options=options)
        if file_path:
            if not file_path.lower().endswith(('.yaml', '.yml')):
                file_path += '.yaml'
            try:
                data = {
                    'method': 'CR_PnP_RANSAC_Refined', # Method based on paper
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'num_point_pairs': len(self.point_pairs),
                    'rotation_matrix': self.extrinsic_R.tolist(),
                    'translation_vector': self.extrinsic_T.tolist(),
                    'avg_reprojection_error_px': float(self.avg_reprojection_error)
                }
                with open(file_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=None, sort_keys=False, indent=2)
                QMessageBox.information(self, "Success", f"Saved extrinsics to {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save extrinsics: {str(e)}")

    def clear_extrinsic_results(self):
         """Reset extrinsic parameters (R, T, error) and their display."""
         self.extrinsic_R = np.eye(3)
         self.extrinsic_T = np.zeros((3, 1))
         self.avg_reprojection_error = 0.0
         # self.last_projections are cleared/updated within project_current_pairs
         self.update_extrinsic_display()
         self.update_point_pair_list_display() # Update list display (removes projection info)
         print("Cleared extrinsic results (R, T, error).")
         # Trigger display update to potentially clear projections if needed
         self.project_current_pairs_for_display() # Recalculate projections (will be None)
         self.update_camera_display() # Update camera view

    def init_camera_subscriber(self):
        """Initialize ROS2 subscription for camera feed."""
        if ROS2_AVAILABLE and self.main_window and hasattr(self.main_window, 'analyzer'):
            analyzer_node = self.main_window.analyzer
            if analyzer_node is not None:
                try:
                    # TODO: Make topic configurable
                    topic = getattr(analyzer_node, 'camera_topic', '/out') 
                    self.camera_subscription = analyzer_node.create_subscription(
                        Image, topic, self._image_callback, 10)
                    print(f"CalibrationView: Subscribed to ROS2 topic {topic}")
                except Exception as e:
                    print(f"CalibrationView: Error creating camera subscription: {e}")
                    self._create_test_pattern()
            else:
                 print("CalibrationView: Analyzer node found, but not initialized?")
                 self._create_test_pattern()
        else:
            print("CalibrationView: ROS2 or Analyzer node not available. Displaying test pattern.")
            self._create_test_pattern()

    def _image_callback(self, msg):
        """Callback function for ROS2 image messages. EMITS signal."""
        try:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            cv_image = None
            if self.cv_bridge:
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif msg.encoding == 'bgr8':
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            elif msg.encoding == 'rgb8':
                rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                cv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                gray_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                cv_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            else:
                # Maybe log unsupported encoding?
                return

            if cv_image is not None:
                # Emit signal instead of assigning directly
                self.new_frame_signal.emit(cv_image, timestamp)

        except Exception as e:
            print(f"CalibrationView: Error processing image message: {e}")
            # Optionally print traceback:
            # import traceback
            # traceback.print_exc()

    @pyqtSlot(np.ndarray, float)
    def _update_latest_frame(self, frame, timestamp):
        """Slot to receive and store the latest frame and timestamp."""
        self.latest_camera_frame = frame
        self.latest_camera_timestamp = timestamp
        self.frame_received = True
        # Note: We might not need to trigger update_camera_display here,
        # as the timer already does that periodically. If updates seem laggy,
        # uncomment the line below, but be mindful of performance.
        # self.update_camera_display()

    def update_camera_display(self):
        """Update the camera feed label with the latest frame and overlays."""
        # ADD Projection update here, BEFORE drawing
        self.project_current_pairs_for_display()

        if self.frame_received and self.latest_camera_frame is not None:
            frame = self.latest_camera_frame.copy()

            # --- Draw SAVED projection points if available and enabled ---
            if self.show_projections: # This controls the SAVED pairs
                self.draw_projection_points(frame)

            # --- Draw LIVE radar projections if enabled ---
            if self.show_live_projection_checkbox.isChecked(): # <-- CHECK THE NEW CHECKBOX
                self.draw_live_radar_projections(frame)

            # --- Draw YOLO Overlay if enabled ---
            if self.show_yolo_checkbox.isChecked():
                self.draw_yolo_overlay(frame)
            # --- End Add ---

            # Convert frame to QPixmap
            h, w = frame.shape[:2]
            bytes_per_line = 3 * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            self.display_pixmap = pixmap
            scaled_pixmap = pixmap.scaled(self.camera_feed_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_feed_label.setPixmap(scaled_pixmap)
        else:
             if not hasattr(self, 'display_pixmap') or self.display_pixmap is None:
                 self._create_test_pattern()

    def detect_and_mark_checkerboard(self, frame):
        """(Optional) Detect checkerboard in live view and add markers for user aid."""
        if frame is None: return
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pattern_size = (self.checkerboard_width, self.checkerboard_height)
            # Use FAST_CHECK for performance
            ret, corners = cv2.findChessboardCorners(gray_frame, pattern_size, cv2.CALIB_CB_FAST_CHECK)
            if ret:
                # Draw faint corners if found (useful during intrinsic calib setup)
                cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
                # Add small text indicator - MAKE MORE PROMINENT
                text = "CHECKERBOARD DETECTED"
                font_scale = 0.8
                thickness = 2
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = 20
                text_y = frame.shape[0] - 20 # Position near bottom-left
                # Optional: Add background rectangle for better visibility
                # cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                #                 (text_x + text_size[0], text_y + 5), (0,0,0), -1)
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness) # Bright Yellow
        except Exception as e:
            # Ignore errors here, it's just a visual aid
            # print(f"Checkerboard detection error: {e}")
            pass

    def _create_test_pattern(self):
        """Create a test pattern image when no real camera feed is available."""
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera Feed / ROS2 Error", (width // 6, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        self.display_pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = self.display_pixmap.scaled(self.camera_feed_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_feed_label.setPixmap(scaled_pixmap)
        self.latest_camera_frame = frame
        self.latest_camera_timestamp = time.time()
        self.frame_received = True

    def camera_feed_mouse_press(self, event):
        """Handle mouse clicks on the camera feed label to collect a point pair."""
        print(f"[Mouse Press] Checking intrinsics. Current self.camera_matrix:\n{self.camera_matrix}")

        if not self.frame_received or self.latest_camera_frame is None or not hasattr(self, 'display_pixmap'):
            QMessageBox.warning(self, "Error", "Cannot capture point pair: No valid camera frame.")
            return
        # Use a more robust check for valid intrinsics
        if np.allclose(self.camera_matrix, np.eye(3)):
             QMessageBox.warning(self, "Intrinsics Missing", "Cannot capture point pair: Valid camera intrinsics (K) must be loaded or calculated first.")
             return

        if event.button() == Qt.LeftButton:
            # Map click coordinates
            label_pos = event.pos()
            pixmap_size = self.display_pixmap.size()
            label_size = self.camera_feed_label.size()
            scale_w = label_size.width() / pixmap_size.width() if pixmap_size.width() > 0 else 0
            scale_h = label_size.height() / pixmap_size.height() if pixmap_size.height() > 0 else 0
            scale = min(scale_w, scale_h)
            if scale == 0: return # Avoid error
            scaled_pixmap_w = int(pixmap_size.width() * scale)
            scaled_pixmap_h = int(pixmap_size.height() * scale)
            offset_x = max(0, (label_size.width() - scaled_pixmap_w) // 2)
            offset_y = max(0, (label_size.height() - scaled_pixmap_h) // 2)
            img_x = int((label_pos.x() - offset_x) / scale)
            img_y = int((label_pos.y() - offset_y) / scale)
            if not (0 <= img_x < pixmap_size.width() and 0 <= img_y < pixmap_size.height()):
                print("Click outside image area.")
                return

            clicked_point_2d = (float(img_x), float(img_y))
            click_timestamp = self.latest_camera_timestamp
            print(f"Image Click: ({img_x:.1f}, {img_y:.1f}) at Timestamp: {click_timestamp:.3f}")

            # Get Corresponding 3D Radar Point (via external processing)
            print("Requesting filtered 3D radar point for CR...")
            # --- Show radar CR data quality metrics ---
            radar_metrics = getattr(self, 'last_radar_cr_metrics', {})
            dt_val = radar_metrics.get('dt', None)
            raw_val = radar_metrics.get('raw', 'n/a')
            inliers_val = radar_metrics.get('inliers', 'n/a')
            std_val = radar_metrics.get('std', None) # Expect float or tuple

            dt_str = f"{dt_val*1000:.1f}ms" if dt_val is not None else "n/a"
            std_str = "n/a"
            std_avg = None
            if isinstance(std_val, (tuple, list)) and len(std_val) == 3:
                std_str = f"({std_val[0]:.3f}, {std_val[1]:.3f}, {std_val[2]:.3f})m"
                std_avg = (std_val[0] + std_val[1] + std_val[2]) / 3.0
            elif isinstance(std_val, float):
                std_str = f"{std_val:.3f}m"
                std_avg = std_val

            msg = f"Quality - Δt: {dt_str} | Raw: {raw_val} | Inliers: {inliers_val} | StdDev: {std_str}"

            # --- Color coding based on quality --- 
            label_style = "color: #0af; font-size: 12px;" # Default: blueish
            dt_thresh_warn = 0.005 # 5 ms
            std_thresh_warn = 0.05  # 5 cm
            dt_thresh_bad = 0.010  # 10 ms
            std_thresh_bad = 0.10   # 10 cm
            is_bad = False
            is_warning = False

            if dt_val is not None and dt_val > dt_thresh_bad:
                is_bad = True
            elif std_avg is not None and std_avg > std_thresh_bad:
                is_bad = True
            elif dt_val is not None and dt_val > dt_thresh_warn:
                is_warning = True
            elif std_avg is not None and std_avg > std_thresh_warn:
                is_warning = True

            if is_bad:
                label_style = "color: #f55; font-size: 12px;" # Red for bad
            elif is_warning:
                label_style = "color: #fa0; font-size: 12px;" # Orange for warning

            self.cr_quality_label.setText(msg)
            self.cr_quality_label.setStyleSheet(label_style)

            # Get corresponding 3D point from radar data
            radar_point_3d = self.get_radar_cr_point_at_timestamp(click_timestamp)

            if radar_point_3d is not None:
                if isinstance(radar_point_3d, (list, tuple)) and len(radar_point_3d) == 3:
                    radar_point_3d = tuple(float(x) for x in radar_point_3d)
                    self.point_pairs.append((radar_point_3d, clicked_point_2d))
                    self.update_point_pair_list_display()
                    # Store info for temporary visual feedback
                    self.last_successful_click_info["pos"] = (int(round(clicked_point_2d[0])), int(round(clicked_point_2d[1])))
                    self.last_successful_click_info["time"] = time.time()
                    print(f"Success: Added Pair {len(self.point_pairs)}. Radar:{radar_point_3d} <=> Image:{clicked_point_2d}")
                    self.project_current_pairs_for_display() # Update projections
                    self.update_camera_display() # Update visualization
                else:
                     QMessageBox.warning(self, "Radar Data Error", f"Received invalid radar point format: {radar_point_3d}")
            else:
                QMessageBox.warning(self, "Radar Data Missing", f"Could not retrieve a valid radar CR point near timestamp {click_timestamp:.3f}. Ensure CR is detected and filter logic is working.")

    def get_radar_cr_point_at_timestamp(self, timestamp):
        """
        Find the synchronized radar corner reflector position at the given timestamp,
        applying velocity & range gating and robust MAD-based outlier filtering.
        Returns the mean (x, y, z) of inlier radar points, or None if insufficient data.
        """
        analyzer = None
        # Initialize metrics with default values
        self.last_radar_cr_metrics = {
            'dt': None, 'raw': 0, 'inliers': 0, 'std': None
        }

        if self.main_window and hasattr(self.main_window, 'analyzer'):
            analyzer = self.main_window.analyzer

        # Update: Check for potentially richer interface first
        sync_method_name = 'get_sync_cr_position_with_details' # Preferred
        if not (analyzer and hasattr(analyzer, sync_method_name)):
            sync_method_name = 'find_sync_corner_reflector_position' # Fallback

        if analyzer and hasattr(analyzer, sync_method_name):
            try:
                # --- Call Analyzer --- 
                # Expects: (points_array, dt, raw_count, inliers_count, std_dev) or just points_array
                sync_result = getattr(analyzer, sync_method_name)(timestamp)

                # --- Parse Analyzer Result --- 
                radar_points = None
                if isinstance(sync_result, tuple) and len(sync_result) == 5: # Rich result
                    radar_points, dt, raw, inliers_count, std = sync_result
                    self.last_radar_cr_metrics.update({'dt': dt, 'raw': raw, 'inliers': inliers_count, 'std': std})
                    print(f"[Radar Sync] Rich data @ {timestamp:.6f}: dt={dt:.4f}s, raw={raw}, inliers={inliers_count}, std={std}")
                elif isinstance(sync_result, (np.ndarray, list, tuple)): # Basic result (fallback)
                    radar_points = sync_result
                    # Metrics will need calculation here if not provided by analyzer
                    print(f"[Radar Sync] Basic data @ {timestamp:.6f}: Got {len(radar_points) if radar_points is not None else 0} points.")
                else: # No valid result
                    print(f"[Radar Sync] Unexpected result type from {sync_method_name}: {type(sync_result)}")

                # --- Basic Validation & Metric Update --- 
                if radar_points is None or len(radar_points) == 0:
                    print("[Radar Sync] No radar points found for this timestamp.")
                    return None

                # --- Normalize output: always Nx3 array ---
                # If a single tuple (3,) is returned, wrap it as a list
                if isinstance(radar_points, tuple) and len(radar_points) == 3 and not hasattr(radar_points[0], '__len__'):
                    radar_points = [radar_points]
                radar_points = np.array(radar_points, dtype=np.float32)
                if self.last_radar_cr_metrics['raw'] == 0: # Update raw count if not provided
                    self.last_radar_cr_metrics['raw'] = radar_points.shape[0]

                if radar_points.ndim == 1 and radar_points.shape[0] == 3:
                    radar_points = radar_points.reshape(1, 3)
                if radar_points.shape[1] != 3:
                    print(f"[Radar Sync] Radar points shape unexpected: {radar_points.shape}")
                    return None
                if radar_points.shape[0] < 1:
                    print("[Radar Sync] No radar points after normalization.")
                    return None
                if radar_points.shape[0] == 1:
                    print(f"[Radar Sync] Only one radar point found: {radar_points[0]}")
                    self.last_radar_cr_metrics.update({'raw': 1, 'inliers': 1, 'std': (0.0, 0.0, 0.0)})
                    return tuple(radar_points[0])
                x, y, z = radar_points[:, 0], radar_points[:, 1], radar_points[:, 2]
                n_points = radar_points.shape[0]
                mad_thresh = 3.5
                # --- Robust MAD-based outlier rejection (Numba-accelerated if available) ---
                # if NUMBA_AVAILABLE and mad_mask_numba is not None:
                #     med_x, mad_x = np.median(x), median_abs_deviation(x, scale='normal')
                #     med_y, mad_y = np.median(y), median_abs_deviation(y, scale='normal')
                #     med_z, mad_z = np.median(z), median_abs_deviation(z, scale='normal')
                #     mask_x = mad_mask_numba(x, med_x, mad_x, mad_thresh)
                #     mask_y = mad_mask_numba(y, med_y, mad_y, mad_thresh)
                #     mask_z = mad_mask_numba(z, med_z, mad_z, mad_thresh)
                #     inliers = mask_x & mask_y & mask_z
                # else:
                #     mask_x = mad_outlier_mask(x, mad_thresh)
                #     mask_y = mad_outlier_mask(y, mad_thresh)
                #     mask_z = mad_outlier_mask(z, mad_thresh)
                #     inliers = mask_x & mask_y & mask_z
                 
                # --- Z-Score based outlier rejection (as per paper) ---
                z_thresh = 3.0 # Standard Z-Score threshold
                mask_x = z_score_outlier_mask(x, thresh=z_thresh)
                mask_y = z_score_outlier_mask(y, thresh=z_thresh)
                mask_z = z_score_outlier_mask(z, thresh=z_thresh)
                inliers = mask_x & mask_y & mask_z
                print(f"[Radar Sync] Applying Z-Score filtering (thresh={z_thresh:.1f})...")
                 
                x, y, z = x[inliers], y[inliers], z[inliers]
                min_inliers = 3 # Keep minimum inlier requirement
                if len(x) < min_inliers:
                    print(f"[Radar Sync] Too few radar points after Z-Score filtering. Inliers: {len(x)} (required: {min_inliers})")
                    self.last_radar_cr_metrics.update({'inliers': len(x)}) # Update inliers count even if returning None
                    return None
 
                # --- Calculate MEAN of inliers (as per paper) --- 
                mean_x = float(np.mean(x))
                mean_y = float(np.mean(y))
                mean_z = float(np.mean(z))
                std_x, std_y, std_z = float(np.std(x)), float(np.std(y)), float(np.std(z))
                print(f"[Radar Sync] Returning MEAN CR position: ({mean_x:.3f}, {mean_y:.3f}, {mean_z:.3f}) | StdDev: ({std_x:.4f}, {std_y:.4f}, {std_z:.4f}) | Inliers: {len(x)} after Z-Score filtering")
                if self.last_radar_cr_metrics['inliers'] == 0:
                    self.last_radar_cr_metrics['inliers'] = len(x)
                if self.last_radar_cr_metrics['std'] is None:
                    self.last_radar_cr_metrics['std'] = (std_x, std_y, std_z)
                return (mean_x, mean_y, mean_z)
            except Exception as e:
                print(f"[Radar Sync] Error during radar CR point retrieval/filtering: {e}")
                import traceback
                traceback.print_exc()
                self.last_radar_cr_metrics.update({'dt': None, 'raw': 0, 'inliers': 0, 'std': None})
                return None
        else:
            print("[Radar Sync] Analyzer node or 'find_sync_corner_reflector_position' method not found.")
            return None

    @pyqtSlot()
    def remove_selected_pair(self):
        """Remove the selected point pair from the list."""
        selected_items = self.point_pair_list.selectedItems()
        if not selected_items: return
        selected_item = selected_items[0]
        row_index = self.point_pair_list.row(selected_item)
        if 0 <= row_index < len(self.point_pairs):
            del self.point_pairs[row_index]
            self.point_pair_list.takeItem(row_index)
            self.update_point_pair_list_display() # Renumber items
            print(f"Removed point pair at index {row_index}")
            self.project_current_pairs_for_display()
            self.update_camera_display()
        else:
            print(f"Error: Selected row index {row_index} out of bounds.")

    @pyqtSlot()
    def clear_all_pairs(self):
        """Clear all collected point pairs and their projections."""
        if not self.point_pairs: return # Nothing to clear
        self.point_pairs = []
        self.last_projections = []
        self.point_pair_list.clear()
        print("Cleared all point pairs.")
        self.update_camera_display() # Update display to remove points

    def update_point_pair_list_display(self):
         """Update the QListWidget for point pairs with current data and indices."""
         self.point_pair_list.clear()
         for i, (radar_pt, img_pt) in enumerate(self.point_pairs):
             radar_str = f"R:[{radar_pt[0]:.3f}, {radar_pt[1]:.3f}, {radar_pt[2]:.3f}]"
             img_str = f"I:[{img_pt[0]:.1f}, {img_pt[1]:.1f}]"
             item_text = f"Pair {i + 1}: {radar_str} <=> {img_str}"
             proj_str = ""
             if i < len(self.last_projections) and self.last_projections[i] is not None:
                 proj_pt = self.last_projections[i]
                 error = np.linalg.norm(np.array(img_pt) - proj_pt)
                 proj_str = f" -> Proj:[{proj_pt[0]:.1f}, {proj_pt[1]:.1f}] (Err: {error:.2f}px)"
             item_text += proj_str
             list_item = QListWidgetItem(item_text)
             list_item.setToolTip(item_text)
             self.point_pair_list.addItem(list_item)
         self.point_pair_list.scrollToBottom()

    @pyqtSlot()
    def calculate_extrinsics(self):
        """
        Calculate extrinsic parameters (R, T) using collected point pairs.
        Follows Algorithm 1 from Cheng et al., arXiv:2307.15264:
        1. RANSAC PnP (Iterative LM vs SQPnP) -> Choose best based on inlier error.
        2. LM Refinement using *all* points with the best RANSAC result as guess.
        """
        # --- Pre-checks ---
        if np.allclose(self.camera_matrix, np.eye(3)):
            QMessageBox.warning(self, "Intrinsics Missing", "Valid camera intrinsics (K) must be loaded or calculated first.")
            return
        min_points = 4 # Min points for PnP
        if len(self.point_pairs) < min_points:
             QMessageBox.warning(self, "Insufficient Data", f"Need at least {min_points} point pairs for PnP calculation. Currently have {len(self.point_pairs)}.")
             return

        print(f"\n--- Starting Extrinsic Calculation ({len(self.point_pairs)} pairs) ---")
        object_points = np.array([pair[0] for pair in self.point_pairs], dtype=np.float64)
        image_points = np.array([pair[1] for pair in self.point_pairs], dtype=np.float64)
        K = self.camera_matrix.astype(np.float64)
        D = self.dist_coeffs.astype(np.float64).flatten() # FLATTEN HERE

        # --- Add Check: Point Distribution ---
        points_ok, dist_msg = self._check_point_distribution(object_points, image_points)
        if not points_ok:
            QMessageBox.warning(self, "Poor Point Distribution", dist_msg)
            return
        # --- End Check ---

        # --- Step 1: RANSAC PnP (Try both ITERATIVE and SQPnP) ---
        best_rvec_ransac, best_tvec_ransac = None, None
        min_ransac_inlier_error = float('inf')
        best_ransac_inliers = None
        # best_method = None # We'll follow a fixed sequence now
        # FIX 3: Initialize variables before try block
        success_it = False
        inliers_it = None
        rvec_it, tvec_it = None, None # Also init these for potential use in fallback

        reprojection_error_threshold = 8.0 # pixels
        confidence = 0.99
        max_iterations = 1000

        # RANSAC Method 1: ITERATIVE
        print("Running RANSAC PnP (cv2.SOLVEPNP_ITERATIVE)...")
        try:
            success_it, rvec_it, tvec_it, inliers_it = cv2.solvePnPRansac(
                object_points, image_points, K, D,
                iterationsCount=max_iterations,
                reprojectionError=reprojection_error_threshold,
                confidence=confidence,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success_it and inliers_it is not None and len(inliers_it) >= min_points:
                print(f"RANSAC (ITERATIVE) succeeded with {len(inliers_it)} inliers.")
                inlier_indices = inliers_it.flatten()
                object_points_inliers = object_points[inlier_indices]
                image_points_inliers = image_points[inlier_indices]

                # --- Step 2: LM Optimization (using RANSAC inliers) --- 
                print("\nPerforming LM optimization using RANSAC inliers...")
                try:
                    # Use solvePnP with SOLVEPNP_ITERATIVE on inliers
                    success_lm, rvec_lm, tvec_lm = cv2.solvePnP(
                        object_points_inliers, image_points_inliers, K, D,
                        # rvec=rvec_it, # Optionally use RANSAC result as guess? Paper doesn't specify.
                        # tvec=tvec_it,
                        # useExtrinsicGuess=False, # Start LM fresh on inliers
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if not success_lm:
                         print("Warning: LM optimization step failed. Using initial RANSAC result for SQPnP guess.")
                         # Use the raw RANSAC result if LM fails
                         rvec_lm, tvec_lm = rvec_it, tvec_it 
                    else:
                         print("LM optimization step successful.")
                         # Project inliers with LM result for error check
                         projected_lm, _ = cv2.projectPoints(object_points_inliers, rvec_lm, tvec_lm, K, D)
                         if projected_lm is not None:
                             errors_lm = np.linalg.norm(image_points_inliers - projected_lm.reshape(-1, 2), axis=1)
                             avg_error_lm = np.mean(errors_lm)
                             print(f"  LM Inlier Avg Reproj Error: {avg_error_lm:.4f} px")
                             
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error during LM optimization: {e}. Using initial RANSAC result for SQPnP guess.")
                    # Fallback to the RANSAC result if LM crashes
                    rvec_lm, tvec_lm = rvec_it, tvec_it

                # --- Step 3: SQPnP (using LM result as guess on inliers) --- 
                print("\nPerforming SQPnP using RANSAC inliers and LM result as guess...")
                try:
                    sqpnp_available = hasattr(cv2, 'SOLVEPNP_SQPNP')
                    if sqpnp_available:
                        success_sq, rvec_final, tvec_final = cv2.solvePnP(
                            object_points_inliers, image_points_inliers, K, D,
                            rvec=rvec_lm.copy(), # Use LM result as guess
                            tvec=tvec_lm.copy(),
                            useExtrinsicGuess=True, 
                            flags=cv2.SOLVEPNP_SQPNP
                        )
                        if not success_sq:
                            print("Warning: SQPnP step failed. Falling back to LM result.")
                            rvec_final, tvec_final = rvec_lm, tvec_lm # Fallback to LM
                        else:
                            print("SQPnP step successful.")
                            # Project inliers with SQPnP result for error check
                            projected_sq, _ = cv2.projectPoints(object_points_inliers, rvec_final, tvec_final, K, D)
                            if projected_sq is not None:
                                errors_sq = np.linalg.norm(image_points_inliers - projected_sq.reshape(-1, 2), axis=1)
                                avg_error_sq = np.mean(errors_sq)
                                print(f"  SQPnP Inlier Avg Reproj Error: {avg_error_sq:.4f} px")
                    else:
                        print("cv2.SOLVEPNP_SQPNP not available. Using LM result as final.")
                        rvec_final, tvec_final = rvec_lm, tvec_lm # Fallback to LM if SQPnP not present
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error during SQPnP step: {e}. Falling back to LM result.")
                    rvec_final, tvec_final = rvec_lm, tvec_lm # Fallback to LM on error

            # --- Store Final Results (with re-orthogonalization) --- 
            # (Keep the re-orthogonalization as it ensures valid rotation)
            R_mat_final, _ = cv2.Rodrigues(rvec_final)
            # Re-orthogonalize R to ensure it's a valid rotation matrix
            try:
                u, _, vh = np.linalg.svd(R_mat_final)
                R_check = u @ vh
                # --- Fix: Ensure proper rotation (det=+1) ---
                if np.linalg.det(R_check) < 0:
                     # Flip the sign of the last column of u before reconstructing.
                     u[:, -1] *= -1
                     R_check = u @ vh
                     print("Applied SVD re-orthogonalization and correction to final rotation matrix (det=+1).")
                else:
                     print("Applied SVD re-orthogonalization to final rotation matrix.")
                # --- End Fix ---
                R_mat_final = R_check
            except np.linalg.LinAlgError:
                print("Warning: SVD for re-orthogonalization failed. Using original R matrix.")
            self.extrinsic_R = R_mat_final
            self.extrinsic_T = tvec_final

            # --- Calculate Final Reprojection Error (using ALL points with the FINAL R, T) ---
            print("\nCalculating final reprojection error using ALL points...")
            # FIX 7: Ensure R is contiguous before Rodrigues
            rvec_final_reorth, _ = cv2.Rodrigues(np.ascontiguousarray(self.extrinsic_R)) # Get rvec from potentially re-orthogonalized R
            # IMPORTANT: Use the original full object_points and image_points for final error calculation
            projected_final, _ = cv2.projectPoints(object_points, rvec_final_reorth, self.extrinsic_T, K, D)
            if projected_final is not None:
                projected_final = projected_final.reshape(-1, 2)
                errors_final = np.linalg.norm(image_points - projected_final, axis=1)
                self.avg_reprojection_error = np.mean(errors_final)
                # --- AED: Average Euclidean Distance (pixels) ---
                AED = self.avg_reprojection_error
                # --- CDSD: Corrected Distance Standard Deviation (meters, in radar frame) ---
                radar_points_cam = (self.extrinsic_R @ object_points.T + self.extrinsic_T).T
                dists = np.linalg.norm(radar_points_cam, axis=1)
                CDSD = np.std(dists)
                acc_thresh = 5.0
                Acc = np.mean(errors_final < acc_thresh)
                print(f"--- Extrinsic Calculation Complete --- Final Avg Reproj Error (AED): {AED:.4f} px")
                print(f"CDSD (std of corrected distances): {CDSD:.4f} m")
                print(f"Acc (fraction within {acc_thresh:.1f} px): {Acc*100:.2f}%")
            else:
                print("Warning: Could not re-project points with final R, T to calculate error.")
                self.avg_reprojection_error = -1.0
                AED = -1.0
                CDSD = -1.0
                Acc = -1.0

            # Update UI
            self.update_extrinsic_display()
            self.project_current_pairs_for_display() # Update projections
            self.update_point_pair_list_display() # Update list with final errors
            QMessageBox.information(
                self, "Calibration Successful",
                f"Extrinsic calibration complete (RANSAC + Refine).\n"
                f"Final Avg Reprojection Error (AED): {AED:.4f} px\n"
                f"Corrected Distance StdDev (CDSD): {CDSD:.4f} m\n"
                f"Accuracy (Acc, <5px): {Acc*100:.2f}%"
            )
        except cv2.error as e:
            # Catching specific OpenCV errors is good.
            print(f"  OpenCV Error during ITERATIVE RANSAC: {e}")
        except Exception as e: # <--- Cursor was here
            # This is a very broad exception catch.
            # Add traceback printing here for better debugging:
            import traceback
            traceback.print_exc()
            print(f"  Unexpected Error during ITERATIVE RANSAC: {e}")

        # --- Fallback: Direct Iterative PnP (no RANSAC) ---
        fallback_used = False
        # FIX 3: Check initialized variables
        if not success_it or inliers_it is None or len(inliers_it) < min_points:
            print(f"RANSAC (ITERATIVE) failed or found too few inliers ({len(inliers_it) if inliers_it is not None else 0}). Trying direct iterative PnP fallback...")
            # --- Fallback: Direct Iterative PnP (no RANSAC) --- 
            fallback_success = False
            try:
                # Uses cv2.solvePnP directly.
                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, K, D,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                # ... (Checks success, calculates error - ok) ...
                if success:
                    print("  Direct iterative PnP succeeded.")
                    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
                    if projected is not None:
                        errors = np.linalg.norm(image_points - projected.reshape(-1, 2), axis=1)
                        avg_error = np.mean(errors)
                        print(f"  Direct iterative PnP Avg Reproj Error: {avg_error:.4f} px")
                    best_rvec_ransac = rvec
                    best_tvec_ransac = tvec
                    fallback_used = True
                    # best_method = "ITERATIVE_FALLBACK" # ADDED: Label the fallback method
                else:
                    print("  Direct iterative PnP failed.")
            except Exception as e:
                # Broad exception catch again. Add traceback?
                import traceback
                traceback.print_exc()
                print(f"  Error in direct iterative PnP: {e}")

        if not fallback_used: # Check if fallback also failed
             # --- Failure Condition ---
             # ... (Failure logic) ...
             return

        # ... (Prints selected method - ok) ...

        # --- Step 2: LM Refinement (using ALL points) ---
        print("Performing LM refinement using all points with best RANSAC guess...")
        try:
            # Uses solvePnP again, but with an initial guess.
            success_refine, rvec_final, tvec_final = cv2.solvePnP(
                object_points, image_points, K, D,
                rvec=best_rvec_ransac.copy(),
                tvec=best_tvec_ransac.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            # ... (Handles refinement failure by using RANSAC result - ok) ...
            if not success_refine:
                 print("Warning: LM refinement step failed to converge. Using RANSAC result.")
                 # Fallback to the best RANSAC result if refinement fails
                 rvec_final, tvec_final = best_rvec_ransac, best_tvec_ransac
            else:
                 print("LM refinement step successful.")
        except Exception as e:
             # Broad exception catch again. Add traceback?
             import traceback
             traceback.print_exc()
             print(f"Error during LM refinement: {e}. Using RANSAC result.")
             # Uses RANSAC result if refinement crashes - ok.
             # Assign RANSAC results directly if refinement crashes
             rvec_final, tvec_final = best_rvec_ransac, best_tvec_ransac # Ensure assignment

        # --- Store Final Results (with re-orthogonalization) ---
        R_mat_final, _ = cv2.Rodrigues(rvec_final)
        # Re-orthogonalize R to ensure it's a valid rotation matrix
        try:
            u, _, vh = np.linalg.svd(R_mat_final)
            R_check = u @ vh
            # --- Fix: Ensure proper rotation (det=+1) ---
            if np.linalg.det(R_check) < 0:
                 # Flip the sign of the last column of u before reconstructing.
                 u[:, -1] *= -1
                 R_check = u @ vh
                 print("Applied SVD re-orthogonalization and correction to final rotation matrix (det=+1).")
            else:
                 print("Applied SVD re-orthogonalization to final rotation matrix.")
            # --- End Fix ---
            R_mat_final = R_check
        except np.linalg.LinAlgError:
            print("Warning: SVD for re-orthogonalization failed. Using original R matrix.")
        self.extrinsic_R = R_mat_final
        self.extrinsic_T = tvec_final

        # --- Calculate Final Reprojection Error (using ALL points) ---
        rvec_final_reorth, _ = cv2.Rodrigues(np.ascontiguousarray(self.extrinsic_R)) # Get rvec from potentially re-orthogonalized R
        projected_final, _ = cv2.projectPoints(object_points, rvec_final_reorth, self.extrinsic_T, K, D)
        if projected_final is not None:
            projected_final = projected_final.reshape(-1, 2)
            errors_final = np.linalg.norm(image_points - projected_final, axis=1)
            self.avg_reprojection_error = np.mean(errors_final)
            # --- AED: Average Euclidean Distance (pixels) ---
            AED = self.avg_reprojection_error
            # --- CDSD: Corrected Distance Standard Deviation (meters, in radar frame) ---
            radar_points_cam = (self.extrinsic_R @ object_points.T + self.extrinsic_T).T
            dists = np.linalg.norm(radar_points_cam, axis=1)
            CDSD = np.std(dists)
            acc_thresh = 5.0
            Acc = np.mean(errors_final < acc_thresh)
            print(f"--- Extrinsic Calculation Complete --- Final Avg Reproj Error (AED): {AED:.4f} px")
            print(f"CDSD (std of corrected distances): {CDSD:.4f} m")
            print(f"Acc (fraction within {acc_thresh:.1f} px): {Acc*100:.2f}%")
        else:
            print("Warning: Could not re-project points with final R, T to calculate error.")
            self.avg_reprojection_error = -1.0
            AED = -1.0
            CDSD = -1.0
            Acc = -1.0

        # Update UI
        self.update_extrinsic_display()
        self.project_current_pairs_for_display() # Update projections
        self.update_point_pair_list_display() # Update list with final errors
        QMessageBox.information(
            self, "Calibration Successful",
            f"Extrinsic calibration complete (RANSAC + Refine).\n"
            f"Final Avg Reprojection Error (AED): {AED:.4f} px\n"
            f"Corrected Distance StdDev (CDSD): {CDSD:.4f} m\n"
            f"Accuracy (Acc, <5px): {Acc*100:.2f}%"
        )

    def project_current_pairs_for_display(self):
        """Re-project all stored 3D radar CR points using current extrinsics."""
        if not self.point_pairs or np.allclose(self.extrinsic_R, np.eye(3)):
            # Avoid projection with identity matrix unless T is also non-zero
            if np.allclose(self.extrinsic_T, 0):
                self.last_projections = [None] * len(self.point_pairs)
                return

        object_points_3d = np.array([pair[0] for pair in self.point_pairs], dtype=np.float64)
        if object_points_3d.size == 0:
             self.last_projections = []
             return

        try:
            projected_points, valid_mask = project_radar_to_image(
                object_points_3d, self.camera_matrix, self.dist_coeffs,
                self.extrinsic_R, self.extrinsic_T
            )
            self.last_projections = []
            if projected_points is not None and valid_mask is not None and len(projected_points) == len(valid_mask):
                for i, proj in enumerate(projected_points):
                    self.last_projections.append(proj if valid_mask[i] else None)
            else:
                 self.last_projections = [None] * len(self.point_pairs)
        except Exception as e:
             print(f"Error during projection for display: {e}")
             self.last_projections = [None] * len(self.point_pairs)

    def draw_projection_points(self, frame):
        """Draw captured points (clicks) and their projections onto the frame."""
        if frame is None: return
        num_pairs = len(self.point_pairs)
        if num_pairs == 0: return
        if len(self.last_projections) != num_pairs:
            self.last_projections = (self.last_projections + [None] * num_pairs)[:num_pairs]

        for i in range(num_pairs):
            _, clicked_point_2d = self.point_pairs[i]
            projected_point_2d = self.last_projections[i]
            u_click, v_click = int(round(clicked_point_2d[0])), int(round(clicked_point_2d[1]))
            cv2.circle(frame, (u_click, v_click), 5, (0, 0, 255), -1) # Red click
            cv2.putText(frame, str(i+1), (int(u_click + 7), int(v_click - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if projected_point_2d is not None:
                u_proj, v_proj = int(round(projected_point_2d[0])), int(round(projected_point_2d[1]))
                cv2.line(frame, (u_proj - 5, v_proj), (u_proj + 5, v_proj), (255, 100, 0), 2) # Blue H cross
                cv2.line(frame, (u_proj, v_proj - 5), (u_proj, v_proj + 5), (255, 100, 0), 2) # Blue V cross
                cv2.putText(frame, str(i+1), (u_proj + 7, v_proj + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
                cv2.line(frame, (u_click, v_click), (u_proj, v_proj), (0, 255, 255), 1) # Yellow line

        # Draw temporary highlight for the last successful click
        feedback_duration = 0.5 # seconds
        if self.last_successful_click_info["pos"] is not None and \
           (time.time() - self.last_successful_click_info["time"]) < feedback_duration:
            click_pos = self.last_successful_click_info["pos"]
            cv2.circle(frame, (int(click_pos[0]), int(click_pos[1])), 10, (0, 255, 0), 2) # Bright green circle
        elif self.last_successful_click_info["pos"] is not None: # Reset after duration
            self.last_successful_click_info["pos"] = None

    def draw_live_radar_projections(self, frame):
        """Projects and draws the *latest* radar point cloud onto the frame, applying DBSCAN and drawing bounding boxes."""
        # print("Attempting to draw live radar projections...") # Reduce noise

        if frame is None: return

        # Check if analyzer is available
        analyzer = None
        if self.main_window and hasattr(self.main_window, 'analyzer'):
            analyzer = self.main_window.analyzer
        if not analyzer or not hasattr(analyzer, 'get_latest_radar_points'):
            return # Analyzer not ready

        # Check if calibration is valid
        intrinsics_ok = not np.allclose(self.camera_matrix, np.eye(3))
        extrinsics_ok = not (np.allclose(self.extrinsic_R, np.eye(3)) and np.allclose(self.extrinsic_T, 0))
        if not extrinsics_ok or not intrinsics_ok:
             return # Calibration not ready

        # Get latest points from analyzer
        latest_points_tuple = analyzer.get_latest_radar_points()
        if latest_points_tuple is None:
            return # No points

        x_rad, y_rad, z_rad = latest_points_tuple
        if x_rad.size == 0:
            return # Empty point cloud

        # --- Prepare 3D points ---
        radar_points_3d = np.vstack((x_rad, y_rad, z_rad)).T

        # --- Subsampling for performance ---
        point_count = radar_points_3d.shape[0]
        max_points_for_render = 1500
        if point_count > max_points_for_render:
            step = int(np.ceil(point_count / max_points_for_render))
            radar_points_3d = radar_points_3d[::step, :]
            # print(f"  - Performance: Subsampled to {radar_points_3d.shape[0]} points")


        # Project points
        image_points_2d, valid_mask = None, None
        try:
            image_points_2d, valid_mask = project_radar_to_image(
                radar_points_3d, self.camera_matrix, self.dist_coeffs,
                self.extrinsic_R, self.extrinsic_T
            )
        except Exception as e:
            print(f"Error during live radar projection: {e}")
            return

        if image_points_2d is None or valid_mask is None:
            return # Projection failed

        # --- Update History & Prune ---
        now = time.time()
        h, w = frame.shape[:2]
        u_coords = image_points_2d[:, 0]
        v_coords = image_points_2d[:, 1]
        # Combine validity mask (in front of cam) with in-bounds mask
        in_bounds_mask = (0 <= u_coords) & (u_coords < w) & (0 <= v_coords) & (v_coords < h) & valid_mask
        valid_indices_this_frame = np.where(in_bounds_mask)[0]

        # Add *newly projected valid* points to history
        for idx in valid_indices_this_frame:
            u, v = u_coords[idx], v_coords[idx]
            self.radar_projection_history.append((int(round(u)), int(round(v)), now)) # Store (u, v, timestamp)

        # Prune old points from history
        decay_window = getattr(self, 'RADAR_DECAY_SECONDS', 1.5)
        self.radar_projection_history = [(u, v, t) for u, v, t in self.radar_projection_history if now - t < decay_window]

        # --- DBSCAN Clustering on History---
        labels = None
        unique_labels = []
        cluster_colors = {}
        points_in_history = len(self.radar_projection_history)
        history_points_array = None

        if points_in_history > 10: # Only cluster if enough points in history
            history_points_array = np.array([(item[0], item[1]) for item in self.radar_projection_history])
            try:
                # DBSCAN parameters (eps: max distance between samples for one to be considered as in the neighborhood of the other)
                # (min_samples: number of samples in a neighborhood for a point to be considered as a core point)
                # --- Increased eps and min_samples for potentially larger human clusters ---
                db = DBSCAN(eps=15, min_samples=10).fit(history_points_array) # Tunable parameters
                # --- End change ---
                labels = db.labels_
                unique_labels = sorted(list(set(labels)))

                # Generate colors for clusters
                n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
                if n_clusters_ > 0:
                    cmap = cm.get_cmap('viridis', n_clusters_)
                    color_idx = 0
                    for label in unique_labels:
                        if label != -1:
                            # BGR format for OpenCV
                            cluster_colors[label] = tuple(int(c * 255) for c in cmap(color_idx)[:3][::-1])
                            color_idx += 1
                # Noise points color (Gray)
                cluster_colors[-1] = (100, 100, 100)

            except Exception as e:
                print(f"Error during DBSCAN clustering: {e}")
                labels = None # Ensure labels is None if clustering fails

        # --- Draw Faded Points and Bounding Boxes ---
        valid_points_drawn = 0
        cluster_bboxes = {} # Store points per cluster: {label: [(u,v), (u,v), ...]}

        for idx, (u, v, t) in enumerate(self.radar_projection_history):
            age = now - t
            fade = max(0.0, 1.0 - age / decay_window) # Linear fade default
            if self.RADAR_DECAY_TYPE == 'exp':
                fade = np.exp(-age / (decay_window * 0.3)) # Optional exponential

            # Determine color and collect points for bbox
            point_color = (0, 255, 0) # Default green
            label = -1 # Default to noise if no labels
            if labels is not None and idx < len(labels):
                label = labels[idx]
                point_color = cluster_colors.get(label, (100, 100, 100)) # Use noise color if label missing

                # Collect points for bounding box calculation (ignore noise)
                if label != -1:
                    if label not in cluster_bboxes:
                        cluster_bboxes[label] = []
                    cluster_bboxes[label].append((u, v))

            # Draw the point
            faded_color = tuple(int(c * fade) for c in point_color)
            cv2.circle(frame, (u, v), 3, faded_color, -1)
            valid_points_drawn += 1

        # Draw bounding boxes for actual clusters
        for label, points in cluster_bboxes.items():
            if len(points) > 0: # Should always be true if label exists here
                points_arr = np.array(points)
                min_u, min_v = points_arr.min(axis=0)
                max_u, max_v = points_arr.max(axis=0)
                box_color = cluster_colors.get(label, (255, 255, 255)) # Fallback white
                # Add padding to the box (optional)
                padding = 2
                min_u, min_v = max(0, min_u - padding), max(0, min_v - padding)
                max_u, max_v = min(w - 1, max_u + padding), min(h - 1, max_v + padding)
                cv2.rectangle(frame, (min_u, min_v), (max_u, max_v), box_color, 1) # Draw thin rectangle

    @pyqtSlot()
    def capture_checkerboard_image(self):
        """Capture the current camera frame for INTRINSIC calibration."""
        if self.latest_camera_frame is None: return QMessageBox.warning(self, "Warning", "No camera frame.")
        frame = self.latest_camera_frame.copy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pattern_size = (self.checkerboard_width, self.checkerboard_height)
        ret, corners = cv2.findChessboardCorners(gray_frame, pattern_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), criteria)
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
            objp = objp * self.checkerboard_square_size
            self.captured_intrinsic_data.append((objp, corners_refined))
            self.captured_images_list.addItem(f"Capture {len(self.captured_intrinsic_data)}: OK")
            self.captured_images_list.scrollToBottom()
            print(f"Stored intrinsic capture {len(self.captured_intrinsic_data)}.")
            frame_with_corners = cv2.drawChessboardCorners(frame, pattern_size, corners_refined, ret)
            self._display_temporary_feedback(frame_with_corners)
        else:
            self.captured_images_list.addItem(f"Capture {self.captured_images_list.count() + 1}: FAILED")
            self.captured_images_list.scrollToBottom()
            QMessageBox.warning(self, "Checkerboard Not Found", "Could not find checkerboard.")

    @pyqtSlot()
    def clear_intrinsic_captures(self):
        """Clear all captured checkerboard images and data."""
        self.captured_intrinsic_data = []
        self.captured_images_list.clear()
        print("Cleared intrinsic calibration captures.")

    @pyqtSlot()
    def run_intrinsic_calibration(self):
        """Perform camera intrinsic calibration using captured data."""
        min_captures = 5
        if len(self.captured_intrinsic_data) < min_captures:
             return QMessageBox.warning(self, "Insufficient Data", f"Need at least {min_captures} captures.")
        if self.latest_camera_frame is None:
             return QMessageBox.critical(self, "Error", "No frame size info.")

        print(f"Running intrinsic calibration ({len(self.captured_intrinsic_data)} captures)...")
        obj_points = [item[0] for item in self.captured_intrinsic_data]
        img_points = [item[1] for item in self.captured_intrinsic_data]
        frame_size = (self.latest_camera_frame.shape[1], self.latest_camera_frame.shape[0])
        try:
            # ret is RMS reprojection error
            rms_error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, frame_size, None, None) 
            
            # Always accept the result, but print the error
            print(f"cv2.calibrateCamera completed. RMS Reprojection Error: {rms_error:.4f} px")
            
            # Check if error is reasonable (optional threshold)
            error_threshold = 1.0 # Example threshold
            if rms_error > error_threshold:
                print(f"Warning: Intrinsic calibration RMS error ({rms_error:.4f}) exceeds threshold ({error_threshold:.1f} px).")

            # Calculate average reprojection error manually for comparison/logging if needed
            mean_error = 0
            for i in range(len(obj_points)):
                imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            avg_error_manual = mean_error / len(obj_points)
            print(f"Manually calculated Avg Reprojection Error: {avg_error_manual:.4f} px")

            self.set_intrinsics(mtx, dist) # Set results and clear extrinsic data
            QMessageBox.information(self, "Intrinsic Calib Success", f"RMS Reprojection Error: {rms_error:.4f} pixels")

        except Exception as e:
            print(f"Error during intrinsic calibration:")
            traceback.print_exc() # Print full traceback
            QMessageBox.critical(self, "Calibration Error", f"Error: {str(e)}")

    def _display_temporary_feedback(self, frame):
        """Display a temporary visual feedback image."""
        try:
            h, w = frame.shape[:2]
            if frame.ndim == 2: q_image = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
            elif frame.shape[2] == 3: q_image = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)
            elif frame.shape[2] == 4: q_image = QImage(frame.data, w, h, 4 * w, QImage.Format_BGRA8888)
            else: return
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.camera_feed_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_feed_label.setPixmap(scaled_pixmap)
            QApplication.processEvents()
            QTimer.singleShot(1000, self.update_camera_display)
        except Exception as e:
            print(f"Error during temporary feedback display: {e}")
            self.update_camera_display()

    # --- Slider-based adjustment handlers ---
    def _slider_update_translation(self, idx, val, minv, maxv):
        # idx: 0=X, 1=Y, 2=Z
        t = minv + (maxv - minv) * (val / 400.0)
        self.extrinsic_T[idx, 0] = t
        self.trans_labels[idx].setText(f"{'XYZ'[idx]} (m): {t:.2f}")
        self.update_extrinsic_display()
        self.project_current_pairs_for_display()
        self.update_camera_display()

    def _slider_update_rotation(self, idx, val, minv, maxv):
        # idx: 0=Yaw(Z), 1=Pitch(Y), 2=Roll(X)
        deg = minv + (maxv - minv) * (val / 600.0)
        self.rot_labels[idx].setText(f"{['Yaw','Pitch','Roll'][idx]} (deg): {deg:.1f}")
        # Compose rotation from sliders (ZYX order)
        yaw = minv + (maxv - minv) * (self.rot_sliders[0].value() / 600.0)
        pitch = minv + (maxv - minv) * (self.rot_sliders[1].value() / 600.0)
        roll = minv + (maxv - minv) * (self.rot_sliders[2].value() / 600.0)
        Rz = np.array([
            [np.cos(np.deg2rad(yaw)), -np.sin(np.deg2rad(yaw)), 0],
            [np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw)), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(np.deg2rad(pitch)), 0, np.sin(np.deg2rad(pitch))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(pitch)), 0, np.cos(np.deg2rad(pitch))]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(roll)), -np.sin(np.deg2rad(roll))],
            [0, np.sin(np.deg2rad(roll)), np.cos(np.deg2rad(roll))]
        ])
        self.extrinsic_R = Rz @ Ry @ Rx
        # Re-orthogonalize R after manual adjustment
        try:
            u, _, vh = np.linalg.svd(self.extrinsic_R)
            R_check = u @ vh
            # --- Fix: Ensure proper rotation (det=+1) ---
            if np.linalg.det(R_check) < 0:
                 # Flip the sign of the last column of u before reconstructing.
                 u[:, -1] *= -1
                 R_check = u @ vh
                 print("Corrected SVD result in slider update (det=+1).")
            # --- End Fix ---
            self.extrinsic_R = R_check
        except np.linalg.LinAlgError:
            print("Warning: SVD for re-orthogonalization failed during manual adjustment.")
        self.update_extrinsic_display()
        self.project_current_pairs_for_display()
        self.update_camera_display()

    def closeEvent(self, event):
        """Clean up resources when the widget is closed."""
        print("Stopping camera update timer...")
        self.camera_update_timer.stop()
        # ROS subscription cleanup might be handled by the main node managing the analyzer
        super().closeEvent(event)

    def _check_point_distribution(self, object_points, image_points, min_points_for_check=4, min_spread_3d=0.1, min_spread_2d=50.0):
        """Checks if the collected points have sufficient spatial spread."""
        if len(object_points) < min_points_for_check:
            return True, "" # Not enough points to check spread meaningfully

        try:
            # Check 3D spread (using range)
            min_3d = np.min(object_points, axis=0)
            max_3d = np.max(object_points, axis=0)
            range_3d = max_3d - min_3d
            # Use norm of the range vector, or check individual axes
            spread_3d_metric = np.linalg.norm(range_3d)
            # Alternative: check if range on at least two axes is sufficient
            # axes_spread_ok = np.sum(range_3d > min_spread_3d / np.sqrt(3)) >= 2
            is_spread_3d = spread_3d_metric > min_spread_3d

            # Check 2D spread (using range)
            min_2d = np.min(image_points, axis=0)
            max_2d = np.max(image_points, axis=0)
            range_2d = max_2d - min_2d
            spread_2d_metric = np.linalg.norm(range_2d)
            is_spread_2d = spread_2d_metric > min_spread_2d

            if not is_spread_3d:
                msg = (f"3D points are too clustered (Spread: {spread_3d_metric:.3f}m, Need > {min_spread_3d:.2f}m). "
                       f"Try collecting points further apart in the physical scene.")
                print(f"[Point Distribution Check] FAIL: {msg}")
                return False, msg
            if not is_spread_2d:
                msg = (f"2D image points are too clustered (Spread: {spread_2d_metric:.1f}px, Need > {min_spread_2d:.1f}px). "
                       f"Try clicking on points further apart in the camera view.")
                print(f"[Point Distribution Check] FAIL: {msg}")
                return False, msg

            print(f"[Point Distribution Check] OK (3D Spread: {spread_3d_metric:.3f}m, 2D Spread: {spread_2d_metric:.1f}px)")
            return True, ""
        except Exception as e:
            print(f"Error during point distribution check: {e}")
            return True, "" # Don't block calibration if check fails

    def init_yolo_subscriber(self):
        """Initialize ROS2 subscription for YOLO tracking.""" # <-- Docstring update
        if ROS2_AVAILABLE and YOLO_MSGS_AVAILABLE and self.main_window and hasattr(self.main_window, 'analyzer'):
            analyzer_node = self.main_window.analyzer
            if analyzer_node is not None:
                try:
                    # --- Use Tracking Topic --- 
                    self.yolo_subscription = analyzer_node.create_subscription(
                        DetectionArray, self.yolo_tracking_topic, self._yolo_tracking_callback, 10) # <-- Use tracking topic and callback
                    print(f"CalibrationView: Subscribed to YOLO topic {self.yolo_tracking_topic}")
                    # --- End Use Tracking Topic ---
                except Exception as e:
                    print(f"CalibrationView: Error creating YOLO subscription: {e}")
            else:
                 print("CalibrationView: Analyzer node found, but not initialized? Cannot subscribe to YOLO.")
        else:
            status = []
            if not ROS2_AVAILABLE: status.append("ROS2 unavailable")
            if not YOLO_MSGS_AVAILABLE: status.append("yolo_msgs unavailable")
            if not (self.main_window and hasattr(self.main_window, 'analyzer')): status.append("Analyzer node unavailable")
            print(f"CalibrationView: Cannot subscribe to YOLO ({', '.join(status)}).")

    def _yolo_detections_callback(self, msg):
        """Callback function for ROS2 YOLO DetectionArray messages."""
        try:
            # Just store the latest message
            self.latest_yolo_detections = msg
        except Exception as e:
            print(f"CalibrationView: Error processing YOLO message: {e}")
            self.latest_yolo_detections = None

    # --- New Callback for Tracking --- 
    def _yolo_tracking_callback(self, msg):
        """Callback function for ROS2 YOLO Tracking messages (DetectionArray format assumed)."""
        # --- Add Print --- 
        print("Received YOLO tracking message.") # <-- Print 1: Confirm callback execution
        # --- End Add ---
        try:
            self.latest_yolo_tracking = msg
            now = time.time() # Get current time for track timeout management
            
            # --- Update Count History --- 
            current_track_ids = set()
            history_copy = list(self.radar_projection_history) # Points to check against boxes
            
            if self.latest_yolo_tracking is None or not hasattr(self.latest_yolo_tracking, 'detections'):
                return # No valid tracking data
                
            for det in self.latest_yolo_tracking.detections:
                # Check if track_id attribute exists and is valid (assuming > 0 is valid)
                # --- Modify Check: Use 'id' and convert to int --- 
                if hasattr(det, 'id') and det.id:
                    try:
                        # Convert string 'id' to integer track_id
                        track_id = int(det.id)
                        if track_id <= 0:
                            # print(f"  Skipping det with non-positive track_id: {track_id}")
                            continue # Skip if ID is not positive after conversion
                    except (ValueError, TypeError):
                        # print(f"  Skipping det with non-integer track_id: {det.id}")
                        continue # Skip if ID cannot be converted to int
                    # --- End Modify Check --- 
                    
                    # --- Proceed with valid integer track_id --- 
                    current_track_ids.add(track_id)
                    self.last_yolo_update_time[track_id] = now # Update last seen time
                    
                    # --- Add Print --- 
                    print(f"  Processing det with valid track_id: {track_id}") # <-- Print 2: Confirm valid track ID
                    # --- End Add ---
                    
                    # Calculate instantaneous count for this detection's bbox
                    bbox = det.bbox
                    cx = int(bbox.center.position.x)
                    cy = int(bbox.center.position.y)
                    w = int(bbox.size.x)
                    h = int(bbox.size.y)
                    x1, y1 = cx - w // 2, cy - h // 2
                    x2, y2 = cx + w // 2, cy + h // 2
                    
                    inst_count = 0
                    for u_hist, v_hist, _ in history_copy:
                        if x1 <= u_hist < x2 and y1 <= v_hist < y2:
                            inst_count += 1
                            
                    # Initialize deque if new track ID
                    if track_id not in self.yolo_track_counts:
                        self.yolo_track_counts[track_id] = collections.deque(maxlen=self.YOLO_COUNT_HISTORY_LEN)
                        
                    # Append count to history
                    self.yolo_track_counts[track_id].append(inst_count)
            
            # --- Cleanup Old Tracks --- 
            ids_to_remove = []
            for track_id in self.yolo_track_counts.keys():
                if track_id not in current_track_ids:
                    # Check if track hasn't been seen for too long
                    if now - self.last_yolo_update_time.get(track_id, now) > self.YOLO_TRACK_TIMEOUT:
                        ids_to_remove.append(track_id)
                        
            for track_id in ids_to_remove:
                del self.yolo_track_counts[track_id]
                del self.last_yolo_update_time[track_id]
                # print(f"Removed old YOLO track ID: {track_id}") # Optional: for debugging
                
        except Exception as e:
            print(f"CalibrationView: Error processing YOLO tracking message: {e}")
            # Optional: print traceback
            import traceback
            traceback.print_exc()
            self.latest_yolo_tracking = None # Clear on error? Or keep stale data?
    # --- End New Callback --- 

    def draw_yolo_overlay(self, frame):
        """Draws YOLO bounding boxes and counts contained radar points."""
        # --- Modify for Tracking Data --- 
        if frame is None or self.latest_yolo_tracking is None:
            # Use self.latest_yolo_tracking now
            return

        # Get historical radar points currently displayed (needed for count calc in callback now)
        # history_copy = list(self.radar_projection_history)
        # if not history_copy:
        #     return # No radar points to count
        
        # Detections are now from the tracking message
        if not hasattr(self.latest_yolo_tracking, 'detections'): return
        detections = self.latest_yolo_tracking.detections
        # --- End Modify --- 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        box_color = (255, 0, 255) # Magenta
        text_color = (255, 0, 255)
        thickness = 1

        for det in detections:
            # --- Modify for Tracking Data & Avg Count --- 
            # Check if track_id attribute exists and is valid
            # --- Modify Check: Use 'id' and convert to int --- 
            track_id = None
            if hasattr(det, 'id') and det.id:
                 try:
                    track_id = int(det.id)
                    if track_id <= 0:
                        track_id = None # Treat non-positive ID as invalid for lookup
                 except (ValueError, TypeError):
                    track_id = None # Treat non-integer ID as invalid
            
            if track_id is None:
                continue # Skip detections without a valid positive integer track ID
            # --- End Modify Check ---
            
            # track_id = det.track_id # Original line removed
            
            # Extract bounding box info (center_x, center_y, width, height)
            bbox = det.bbox
            cx = int(bbox.center.position.x)
            cy = int(bbox.center.position.y)
            w = int(bbox.size.x)
            h = int(bbox.size.y)
            
            # Calculate top-left (x1, y1) and bottom-right (x2, y2) corners
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2

            # Get average count from history
            avg_count = 0.0
            if track_id in self.yolo_track_counts:
                counts_history = self.yolo_track_counts[track_id]
                if counts_history: # Ensure deque is not empty
                    avg_count = np.mean(counts_history)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

            # Draw the track ID and average point count near the top-left corner
            count_text = f"ID:{track_id} AvgPts:{avg_count:.1f}"
            # --- End Modify ---
            cv2.putText(frame, count_text, (x1, y1 - 5), font, font_scale, text_color, thickness)

            # Optional: Draw class ID and score near the bottom-left corner
            # class_text = f"{det.class_id}: {det.score:.2f}"

    # --- End New Methods ---

        # --- Fallback: Direct Iterative PnP (no RANSAC) ---
        fallback_used = False
        # FIX 3: Check initialized variables

def project_radar_to_image(pts_radar, K, D, R, T, *, eps_z=1e-3):
    """
    Vector-project 3-D radar points into the camera image.

    Parameters
    ----------
    pts_radar : (N, 3) array-like
        Radar points in the radar (sensor) coordinate frame.
    K         : (3, 3) array-like
        Camera intrinsic matrix.
    D         : None or (n, 1)/(1, n)/(n,) array-like
        Radial / tangential distortion coefficients (0–8 elements).
    R         : (3, 3) array-like
        Rotation from radar frame to camera frame.
    T         : (3, 1) or (3,) array-like
        Translation from radar frame to camera frame (metres).
    eps_z     : float, optional
        Small positive depth used to decide "in-front-of-camera".

    Returns
    -------
    img_pts : (N, 2) ndarray(float64)
        Pixel coordinates (sub-pixel precision).
    valid   : (N,)  ndarray(bool)
        True for points whose camera-frame Z is greater than `eps_z`.
    """
    # ── 1. Fast exit ────────────────────────────────────────────────────────────
    if pts_radar is None:
        try:
            if len(pts_radar) == 0: # Check length only if not None
                return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)
        except TypeError: # Handle cases where len() isn't applicable (e.g., scalar)
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)
            
    # Ensure pts_radar is array-like before checking length if it wasn't None initially
    try:
        pts_radar_array = np.asarray(pts_radar)
        if pts_radar_array.shape[0] == 0:
             return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)
    except Exception: # Catch potential errors during conversion/shape access
        print("Warning: Could not process pts_radar input in project_radar_to_image.")
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)


    # ── 2. Contiguity & dtype harmonisation (float64 is the native OpenCV path) ─
    P = np.ascontiguousarray(pts_radar_array, dtype=np.float64).reshape(-1, 3) # Use converted array
    K = np.ascontiguousarray(K,         dtype=np.float64).reshape(3, 3)
    R = np.ascontiguousarray(R,         dtype=np.float64).reshape(3, 3)
    T = np.ascontiguousarray(T,         dtype=np.float64).reshape(3, 1)
    D = None if D is None else np.ascontiguousarray(D, dtype=np.float64).reshape(-1, 1)

    # ── 3. One call to cv2.Rodrigues (deterministic 27-flop SVD) ────────────────
    try:
        rvec, _ = cv2.Rodrigues(R)                                           # ㊉ Rodrigues
        img_pts, _ = cv2.projectPoints(P, rvec, T, K, D)                     # ㊉ projectPoints
        if img_pts is None: # projectPoints can return None on error
             print("Warning: cv2.projectPoints returned None.")
             return np.empty((P.shape[0], 2), dtype=np.float64), np.zeros((P.shape[0],), dtype=bool)
        img_pts = img_pts.reshape(-1, 2)
    except cv2.error as e:
        print(f"Error in cv2.Rodrigues or cv2.projectPoints: {e}")
        # Return arrays matching input shape but indicate invalidity
        return np.full((P.shape[0], 2), np.nan, dtype=np.float64), np.zeros((P.shape[0],), dtype=bool)
    except Exception as e: # Catch other potential errors
        print(f"Unexpected error during projection calculation: {e}")
        return np.full((P.shape[0], 2), np.nan, dtype=np.float64), np.zeros((P.shape[0],), dtype=bool)


    # ── 4. Fast positive-depth mask  z_cam = R[2] @ Pᵀ + Tz  ──────────────────
    try:
        # Use np.dot for clarity and potentially better performance in some cases
        z_cam = np.dot(P, R[2,:]) + T[2,0] # P is (N,3), R[2,:] is (3,) -> (N,)
        # Original: z_cam = (R[2] @ P.T + T[2]).ravel() # R[2] is (1,3), P.T is (3,N) -> (1,N)
        valid_mask = z_cam > eps_z
        return img_pts, valid_mask
    except Exception as e:
         print(f"Error calculating depth mask: {e}")
         # If depth calculation fails, mark all as invalid but return points
         return img_pts, np.zeros((P.shape[0],), dtype=bool)

def z_score_outlier_mask(x, thresh=3.5):
    """
    Returns a boolean mask for inliers using the Z-Score method.
    """
    med = np.median(x)
    mad = median_abs_deviation(x, scale='normal')
    if mad < 1e-8:
        # All values are (almost) identical, keep all
        return np.ones_like(x, dtype=bool)
    z = 0.6745 * (x - med) / mad
    return np.abs(z) < thresh

if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def mad_mask_numba(x, med, mad, thresh):
        mask = np.empty(x.shape, dtype=np.bool_)
        for i in range(x.shape[0]):
            z = 0.6745 * (x[i] - med) / mad if mad > 1e-8 else 0.0
            mask[i] = np.abs(z) < thresh
        return mask
else:
    mad_mask_numba = None

if __name__ == '__main__':
    import sys
    app = QApplication.instance() or QApplication(sys.argv)

    # Example K and D (replace with actual intrinsic results)
    K_test = np.array([[800.0, 0., 320.0], [0., 800.0, 240.0], [0., 0., 1.]])
    D_test = np.zeros((1, 5))

    class MockAnalyzer:
        def find_sync_corner_reflector_position(self, timestamp):
            print(f"MockAnalyzer: Requesting CR point near {timestamp:.3f}")
            # Simulate finding a point based on time (simple oscillation)
            offset = np.sin(timestamp * np.pi) * 0.1
            pt = (1.5 + offset, 0.3 - offset, 0.1)
            print(f"  -> Returning mock point: {pt}")
            return pt
        # camera_topic = '/my_camera/image_raw' # Example of setting topic

    class MockMainWindow:
        def __init__(self):
            self.analyzer = MockAnalyzer()

    mock_main = MockMainWindow()
    window = CalibrationView(parent=mock_main)
    window.set_intrinsics(K_test, D_test) # Set dummy intrinsics for testing
    window.setWindowTitle("Radar-Camera Extrinsic Calibration (CR Method - Test)")
    window.setGeometry(100, 100, 950, 700)
    window.show()
    sys.exit(app.exec_()) 