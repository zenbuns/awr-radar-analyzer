# Radar Point Cloud Analyzer

A comprehensive tool for analyzing radar point clouds from an AWR1843 mmWave radar using ROS 2. This application provides real-time visualization, data collection, experiment management, and scientific analysis capabilities.

## Features

- Real-time visualization of radar point clouds and heatmaps
- Configurable data collection for radar experiments
- Comparative analysis between different radar configurations
- Scientific visualization with comprehensive metrics
- Exportable reports and visualizations

## Requirements

- Python 3.8 or higher
- PyQt5 for the user interface
- NumPy, Pandas, Matplotlib, and SciPy for data processing and visualization
- ROS 2 (optional, for live radar data)

## Installation

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/zenbuns/awr-radar-analyzer.git
cd awr-radar-analyzer

# Install the package and dependencies
pip install -e .
```

### With ROS 2 Integration

```bash
# Make sure ROS 2 is sourced
source /opt/ros/foxy/setup.bash  # Or other ROS 2 distribution

# Install with ROS 2 dependencies
pip install -e ".[ros]"
```

## Usage

### Basic Usage

```bash
# Run the application
python main.py
```

### Command Line Options

```bash
# Run with custom parameters
python main.py --max-range 50.0 --circle-interval 5.0 --heatmap-resolution 0.25

# Run in visualization-only mode (no ROS)
python main.py --viz-only
```

## Data Collection

1. Configure the experiment parameters in the control panel
2. Set the target distance and collection duration
3. Click "Start" to begin data collection
4. Data will be automatically saved to `~/radar_experiment_data/<config_name>/`

## Analysis

- Use the visualization controls to adjust the display
- Add Regions of Interest (ROIs) to analyze specific areas
- Export scientific plots for publication
- Generate comparative reports between different configurations

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
pylint radar_point_cloud_analyzer

# Run tests
pytest
```

## License

[MIT License](LICENSE)

## Acknowledgements

- Texas Instruments for the AWR1843 mmWave radar
- ROS 2 community for robotics middleware
- PyQt team for the UI framework 