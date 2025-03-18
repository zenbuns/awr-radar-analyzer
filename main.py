#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the Radar Point Cloud Analyzer application.

This module provides the primary entry point for launching the radar
point cloud analyzer, including command-line argument handling.
"""

import sys
import argparse
import os
import yaml
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QDir
from PyQt5.QtGui import QFontDatabase, QFont

# Import for style application
from ui.styles import DARK_STYLESHEET, apply_mpl_style


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Radar Point Cloud Analyzer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--max-range',
        type=float,
        default=35.0,
        help='Maximum radar range in meters'
    )
    
    parser.add_argument(
        '--circle-interval',
        type=float,
        default=10.0,
        help='Interval between range circles in meters'
    )
    
    parser.add_argument(
        '--heatmap-resolution',
        type=float,
        default=0.5,
        help='Resolution of the heatmap grid in meters per cell'
    )
    
    parser.add_argument(
        '--viz-only',
        action='store_true',
        help='Run in visualization-only mode (no ROS)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--light-theme',
        action='store_true',
        help='Use light theme instead of default dark theme'
    )
    
    # Add ROS-specific arguments
    ros_args = parser.add_argument_group('ROS arguments')
    ros_args.add_argument(
        '--ros-args',
        nargs='*',
        help='Arguments to pass to ROS'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary of configuration values.
    """
    # Default configuration
    config = {
        'experiment': {
            'max_range': 35.0,
            'circle_interval': 10.0,
            'collection_duration': 60,
            'current_config': 'default_config',
            'target_distance': 5.0,
            'circle_distance': 5.0,      # Add this explicitly
            'circle_radius': 0.5,        # Add this explicitly
            'heatmap_resolution': 0.5,
            'num_trials': 1
        },
        'visualization': {
            'colormap': 'plasma',
            'decay_factor': 0.98,
            'mode': 'heatmap',
            'noise_floor': 0.05,
            'smoothing_sigma': 2.0,
            'contour_levels': 6
        },
        'ros': {
            'node_name': 'radar_point_cloud_analyzer',
            'pcl_topic': '/ti_mmwave/radar_scan_pcl',
            'track_topic': '/ti_mmwave/radar_track_marker_array',
            'occupancy_topic': '/ti_mmwave/radar_occupancy',
            'qos_profile': 10
        },
        'paths': {
            'data_dir': '~/radar_experiment_data',
            'heatmap_dir': '~/radar_experiment_data/heatmaps',
            'reports_dir': '~/radar_experiment_data/reports'
        },
        'ui': {
            'dark_theme': True,
            'font_family': 'Segoe UI',
            'font_size': 10,
            'high_dpi': True
        }
    }
    
    # Try to load configuration from file
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                
            # Update default configuration with file values
            for section, values in file_config.items():
                if section in config:
                    config[section].update(values)
                else:
                    config[section] = values
                    
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
    
    return config


def setup_application_style(app, use_dark_theme=True, config=None):
    """
    Set up application styling and fonts.
    
    Args:
        app: QApplication instance
        use_dark_theme: Whether to use dark theme
        config: Configuration dictionary with UI preferences
    """
    # Apply high DPI settings
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Load preferred font if specified in config
    if config and 'ui' in config and 'font_family' in config['ui']:
        font_family = config['ui']['font_family']
        font_size = config['ui'].get('font_size', 10)
        
        # Try to load the font and set as default
        font = QFont(font_family, font_size)
        if font.exactMatch():
            app.setFont(font)
    
    # Apply dark theme stylesheet
    if use_dark_theme:
        app.setStyleSheet(DARK_STYLESHEET)
    else:
        # Could add a light theme stylesheet here if needed
        app.setStyle('Fusion')  # Use default Fusion style for light theme
    
    # Apply matplotlib style to match
    apply_mpl_style()


def main():
    """
    Main entry point for the application.
    
    Returns:
        Application exit code.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.max_range:
        config['experiment']['max_range'] = args.max_range
    if args.circle_interval:
        config['experiment']['circle_interval'] = args.circle_interval
    if args.heatmap_resolution:
        config['experiment']['heatmap_resolution'] = args.heatmap_resolution
    
    # Prepare directories
    for path_key, path_value in config['paths'].items():
        expanded_path = os.path.expanduser(path_value)
        os.makedirs(expanded_path, exist_ok=True)
    
    # Create PyQt application
    app = QApplication(sys.argv)
    
    # Set up application style with user preferences
    use_dark_theme = not args.light_theme  # Use light theme if specified
    if 'ui' in config:
        config['ui']['dark_theme'] = use_dark_theme
    setup_application_style(app, use_dark_theme, config)
    
    # Import GUI and analyzer here to avoid circular imports
    if args.viz_only:
        # Visualization-only mode
        from ui.main_window import MainWindow
        
        # Create main window without analyzer
        main_window = MainWindow(analyzer=None)
        main_window.show()
        
        # Run application
        return app.exec_()
    else:
        # Full mode with ROS integration
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            
            # Prepare ROS arguments
            ros_args = ['radar_point_cloud_analyzer']
            if args.ros_args:
                ros_args.extend(args.ros_args)
            
            # Initialize ROS
            rclpy.init(args=ros_args)
            
            # Import radar analyzer
            from radar_analyzer.core import RadarPointCloudAnalyzer
            from radar_gui import RadarAnalyzerGUI
            
            # Create analyzer node
            analyzer_node = RadarPointCloudAnalyzer()
            
            # Apply configuration to analyzer
            analyzer_node.params.max_range = config['experiment']['max_range']
            analyzer_node.params.circle_interval = config['experiment']['circle_interval']
            analyzer_node.params.heatmap_resolution = config['experiment']['heatmap_resolution']
            analyzer_node.params.collection_duration = config['experiment']['collection_duration']
            analyzer_node.params.target_distance = config['experiment']['target_distance']
            analyzer_node.params.circle_distance = config['experiment']['circle_distance']
            analyzer_node.params.circle_radius = config['experiment']['circle_radius']
            
            # Apply visualization settings
            analyzer_node.live_heatmap_decay_factor = config['visualization']['decay_factor']
            
            # Make sure to synchronize parameters
            if hasattr(analyzer_node.params, '__post_init__'):
                analyzer_node.params.__post_init__()
            
            # Create executor for ROS
            executor = MultiThreadedExecutor()
            executor.add_node(analyzer_node)
            
            # Start ROS spin in separate thread
            import threading
            ros_thread = threading.Thread(target=executor.spin, daemon=True)
            ros_thread.start()
            
            # Create GUI with analyzer
            from ui.main_window import MainWindow
            main_window = MainWindow(analyzer=analyzer_node)
            main_window.show()
            
            # Run application
            exit_code = app.exec_()
            
            # Cleanup ROS
            try:
                executor.shutdown()
                analyzer_node.destroy_node()
                rclpy.shutdown()
            except Exception as e:
                print(f"Error during ROS shutdown: {str(e)}")
            
            return exit_code
        
        except ImportError:
            print("ROS 2 (rclpy) not available. Running in visualization-only mode.")
            from ui.main_window import MainWindow
            
            # Create main window without analyzer
            main_window = MainWindow(analyzer=None)
            main_window.show()
            
            # Run application
            return app.exec_()
        
        except Exception as e:
            print(f"Error initializing ROS: {str(e)}")
            if 'rclpy' in str(e).lower():
                print("This appears to be a ROS 2 error. Make sure ROS 2 is properly installed and sourced.")
            
            # Fallback to visualization-only mode
            from ui.main_window import MainWindow
            
            # Create main window without analyzer
            main_window = MainWindow(analyzer=None)
            main_window.show()
            
            # Run application
            return app.exec_()


if __name__ == "__main__":
    sys.exit(main())