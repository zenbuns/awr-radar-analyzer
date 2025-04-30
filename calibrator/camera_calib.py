#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from cv2 import aruco
import numpy as np
import numpy.typing as npt
import pickle
from typing import Optional, List, Tuple, Any
import os

class CameraCalibrator(Node):
    """
    ROS 2 Node for intrinsic camera calibration using a Charuco board.

    Uses ROS 2 parameters for configuration and OpenCV for detection and calibration.
    Allows interactive frame saving and triggers calibration.
    """

    def __init__(self) -> None:
        """Initializes the node, declares parameters, and sets up resources."""
        super().__init__('camera_calibrator')

        # --- Declare Parameters ---
        self.declare_parameter('image_topic', '/out')
        self.declare_parameter('charuco_rows', 7)
        self.declare_parameter('charuco_cols', 5)
        self.declare_parameter('square_length_meters', 0.04)
        self.declare_parameter('marker_length_meters', 0.02)
        # See cv2.aruco.PredefinedDictionaryType enum for options
        self.declare_parameter('aruco_dictionary_id', int(aruco.DICT_5X5_1000))
        self.declare_parameter('calibration_file', 'calibration.pckl')
        self.declare_parameter('min_frames_for_calibration', 10)
        self.declare_parameter('min_charuco_corners_detected', 5) # Min corners needed per frame

        # Register callback for parameter changes (optional but good practice)
        self.add_on_set_parameters_callback(self.parameters_callback)

        # --- Initialize Variables from Parameters ---
        self._load_parameters()

        # --- Setup OpenCV / ROS Resources ---
        self.bridge = CvBridge()
        self._setup_aruco()

        # Data storage for calibration
        self.corners_all: List[npt.NDArray[np.float32]] = []
        self.ids_all: List[npt.NDArray[np.int32]] = []
        self.image_size: Optional[Tuple[int, int]] = None # (width, height)
        self.frame_count: int = 0
        self.is_calibrated: bool = False

        # --- Setup Subscription ---
        # Use reliable QoS for potentially lossy image streams if needed, default is usually fine
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10  # QoS profile depth
        )

        self.get_logger().info(f"Camera Calibrator Node initialized.")
        self.get_logger().info(f"Subscribed to: {self.image_topic}")
        self.get_logger().info(f"Using Charuco board: {self.charuco_cols}x{self.charuco_rows}")
        self.get_logger().info(f"Sq: {self.square_length_m}m, Mk: {self.marker_length_m}m")
        self.get_logger().info(f"Dict: {self.aruco_dict_id}, Output: {self.calibration_file}")
        self.get_logger().info(f"Need {self.min_frames} frames, min {self.min_corners_per_frame} corners/frame.")
        self.get_logger().info("Press 's' in OpenCV window to save frame.")
        self.get_logger().info("Press 'q' to attempt calibration and quit.")
        self.get_logger().info("Close window ('X') to quit without calibrating.")


    def _load_parameters(self) -> None:
        """Loads parameter values into class attributes."""
        self.image_topic = self.get_parameter('image_topic').value
        self.charuco_rows = self.get_parameter('charuco_rows').value
        self.charuco_cols = self.get_parameter('charuco_cols').value
        self.square_length_m = self.get_parameter('square_length_meters').value
        self.marker_length_m = self.get_parameter('marker_length_meters').value
        self.aruco_dict_id = self.get_parameter('aruco_dictionary_id').value
        self.calibration_file = self.get_parameter('calibration_file').value
        self.min_frames = self.get_parameter('min_frames_for_calibration').value
        self.min_corners_per_frame = self.get_parameter('min_charuco_corners_detected').value


    def parameters_callback(self, params: List[Parameter]) -> SetParametersResult:
        """Handles dynamic parameter changes."""
        successful = True
        for param in params:
            if param.name == 'aruco_dictionary_id':
                try:
                    # Re-setup aruco if dictionary changes
                    self.aruco_dict_id = param.value
                    self._setup_aruco()
                    self.get_logger().info(f"Updated ArUco dictionary to ID: {self.aruco_dict_id}")
                except Exception as e:
                    self.get_logger().error(f"Failed to update ArUco dictionary: {e}")
                    successful = False
            # Add handling for other parameters if needed
            # Be cautious about changing board dimensions/sizes dynamically
            # if calibration data has already been collected.
            else:
                # Allow other parameters to be set directly if no special handling needed
                self.get_logger().info(f"Parameter '{param.name}' changed to {param.value}")
                setattr(self, param.name, param.value) # Basic update

        return SetParametersResult(successful=successful)


    def _setup_aruco(self) -> None:
        """Sets up the ArUco dictionary, detector, and Charuco board."""
        try:
            self.aruco_dict = aruco.getPredefinedDictionary(self.aruco_dict_id)

            # --- Adjust Detector Parameters --- 
            self.aruco_params = aruco.DetectorParameters() 
            # Explicitly set corner refinement (often default, but good to be sure)
            self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            # Slightly adjust adaptive threshold window size (might help in some lighting)
            self.aruco_params.adaptiveThreshWinSizeMin = 5 # Default is 3
            self.aruco_params.adaptiveThreshWinSizeMax = 25 # Default is 23
            # self.aruco_params.minMarkerPerimeterRate = 0.02 # Default 0.03, adjust if markers are very small relative to image
            # self.aruco_params.maxMarkerPerimeterRate = 4.0 # Default 4.0
            # --- End Adjustments --- 

            self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.charuco_board = aruco.CharucoBoard(
                size=(self.charuco_cols, self.charuco_rows),
                squareLength=self.square_length_m,
                markerLength=self.marker_length_m,
                dictionary=self.aruco_dict
            )
            self.get_logger().info("ArUco detector and Charuco board configured.")
        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize ArUco/Charuco: {e}. Shutting down.")
            # Destroy node if critical setup fails
            self.destroy_node()
            rclpy.shutdown()
            raise RuntimeError("ArUco setup failed") from e


    def image_callback(self, msg: Image) -> None:
        """Processes incoming image messages, detects markers, handles user input."""
        if self.is_calibrated:
            return # Stop processing if already calibrated and saved

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        except Exception as e:
             self.get_logger().error(f"Error converting image: {e}")
             return

        if frame is None or frame.size == 0:
            self.get_logger().warn("Received empty frame.")
            return

        # Store image size once
        if self.image_size is None:
            self.image_size = (frame.shape[1], frame.shape[0]) # (width, height)
            self.get_logger().info(f"Image size detected: {self.image_size}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy() # Draw on a copy

        # --- Detect Markers --- 
        corners, ids, _ = self.detector.detectMarkers(gray)

        # --- Log Detected Marker Count --- 
        num_detected_markers = len(ids) if ids is not None else 0
        self.get_logger().debug(f"Detected {num_detected_markers} ArUco markers.") 
        # --- End Log --- 

        charuco_corners = None
        charuco_ids = None
        num_detected_charuco = 0

        if ids is not None and len(ids) > 0:
            # Draw ALL detected markers
            aruco.drawDetectedMarkers(display_frame, corners, ids)
            try:
                retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=self.charuco_board
                )
                if retval and charuco_corners is not None and len(charuco_corners) >= self.min_corners_per_frame:
                    num_detected_charuco = len(charuco_corners)
                    aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids, (255, 0, 0))
                else:
                    # Not enough corners, reset for safety
                    charuco_corners = None
                    charuco_ids = None
            except cv2.error as e:
                self.get_logger().warn(f"OpenCV error during Charuco interpolation: {e}")
                charuco_corners = None
                charuco_ids = None


        # Add text overlay
        cv2.putText(display_frame, f"Frames Saved: {self.frame_count}/{self.min_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if charuco_corners is not None:
             cv2.putText(display_frame, f"Corners Found: {num_detected_charuco}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        # --- GUI Interaction ---
        cv2.imshow("Camera Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Handle closing window
        try:
            if cv2.getWindowProperty("Camera Calibration", cv2.WND_PROP_VISIBLE) < 1:
                self.get_logger().info("Window closed by user. Shutting down.")
                self.shutdown_node()
                return
        except cv2.error:
            # Can happen if window is already destroyed during shutdown sequence
            self.get_logger().warn("Error checking window property, likely during shutdown.")
            self.shutdown_node() # Ensure shutdown happens
            return


        # Handle key presses
        if key == ord('s'):
            if charuco_corners is not None and charuco_ids is not None:
                self.get_logger().info(f"Saving frame {self.frame_count} with {num_detected_charuco} corners.")
                # Make copies to prevent modification issues if frame processing is slow
                self.corners_all.append(charuco_corners.copy())
                self.ids_all.append(charuco_ids.copy())
                self.frame_count += 1
            else:
                self.get_logger().warn("Cannot save frame: Not enough Charuco corners detected.")

        elif key == ord('q'):
            self.get_logger().info("'q' pressed. Attempting calibration and quitting.")
            self.calibrate()
            self.shutdown_node()


    def calibrate(self) -> bool:
        """
        Performs camera calibration using collected data.

        Returns:
            True if calibration was successful and saved, False otherwise.
        """
        if self.frame_count < self.min_frames:
            self.get_logger().warn(f"Calibration requires at least {self.min_frames} frames, only have {self.frame_count}. Aborting.")
            return False

        if self.image_size is None:
            self.get_logger().error("Image size not determined. Cannot calibrate.")
            return False

        self.get_logger().info(f"Running calibration with {self.frame_count} frames...")

        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                charucoCorners=self.corners_all,
                charucoIds=self.ids_all,
                board=self.charuco_board,
                imageSize=self.image_size,
                cameraMatrix=None,  # Calibrate from scratch
                distCoeffs=None,
                flags=cv2.CALIB_FIX_ASPECT_RATIO # Example flag, adjust as needed
            )

            if ret:
                self.get_logger().info(f"Calibration successful! RMS reprojection error: {ret:.4f}")
                self.get_logger().info("Camera Matrix:")
                self.get_logger().info(f"\n{np.array2string(camera_matrix, precision=4)}")
                self.get_logger().info("Distortion Coefficients:")
                self.get_logger().info(f"\n{np.array2string(dist_coeffs, precision=4)}")

                # Save calibration results
                save_path = os.path.abspath(self.calibration_file)
                os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure dir exists
                try:
                    with open(save_path, "wb") as f:
                        pickle.dump(
                            {
                                'camera_matrix': camera_matrix,
                                'dist_coeffs': dist_coeffs,
                                'image_width': self.image_size[0],
                                'image_height': self.image_size[1],
                                'rms_error': ret
                             }, f)
                    self.get_logger().info(f"Saved calibration data to: {save_path}")
                    self.is_calibrated = True
                    return True
                except IOError as e:
                    self.get_logger().error(f"Failed to save calibration data to {save_path}: {e}")
                    return False
            else:
                self.get_logger().error("Calibration failed or did not converge properly (ret=0).")
                return False

        except cv2.error as e:
            self.get_logger().error(f"OpenCV Error during calibration: {e}")
            return False
        except Exception as e:
            self.get_logger().error(f"An unexpected error occurred during calibration: {e}")
            return False


    def shutdown_node(self) -> None:
        """Cleans up resources and initiates node shutdown."""
        self.get_logger().info("Shutting down node...")
        # No need to destroy subscription explicitly, destroy_node handles it
        cv2.destroyAllWindows()
        # Stop ROS 2 processing
        if rclpy.ok():
            self.destroy_node() # This should be called by the main loop's finally block too
            # Request shutdown, but don't force if already shutting down
            try:
                 if rclpy.ok():
                      rclpy.try_shutdown()
            except Exception as e:
                 self.get_logger().warn(f"Error during explicit shutdown request: {e}")


def main(args: Optional[List[str]] = None) -> None:
    """Main function to initialize and run the CameraCalibrator node."""
    rclpy.init(args=args)
    node = None
    try:
        node = CameraCalibrator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down.")
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Unhandled exception in main loop: {e}", exc_info=True)
        else:
            print(f"Unhandled exception during node initialization: {e}")
    finally:
        # Ensure resources are released even on error
        if node:
            node.get_logger().info("Cleaning up node resources...")
            # Ensure GUI windows are closed
            cv2.destroyAllWindows()
            if not node.is_destroyed():
                 node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Node shutdown complete.")

if __name__ == '__main__':
    main()
