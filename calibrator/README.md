# AWR1843BOOST Calibrator

A comprehensive calibration tool for the Texas Instruments AWR1843BOOST mmWave radar.

## Overview

This calibrator automates the process of calculating the range bias and RX channel phase calibration values for the AWR1843BOOST. These calibration values are essential for accurate radar measurements, especially for precise distance measurements and angle-of-arrival calculations.

## Requirements

- Python 3.6 or higher
- pyserial library (`pip install pyserial`)
- AWR1843BOOST connected via USB to the computer
- Corner reflector placed at a known distance

## Quick Start

The easiest way to use the calibrator is with the provided helper script:

```bash
./calibrate.sh --scan                # Find available serial ports
./calibrate.sh --cli /dev/ttyACM0 --data /dev/ttyACM1  # Run calibration
```

## Features

- **Automatic port detection** on Linux systems
- **Multiple configuration templates** for different use cases
- **Calibration quality assessment** to verify results
- **Retry mechanism** for improved robustness
- **Progress visualization** during calibration
- **Validation** of calibration results
- **JSON export** for easier integration with other tools
- **Load and save** existing calibrations

## Using the Helper Script

The `calibrate.sh` script simplifies using the calibrator:

```bash
# Show help and scan for ports
./calibrate.sh

# Basic calibration
./calibrate.sh --cli /dev/ttyACM0 --data /dev/ttyACM1 --distance 1.5

# Using extended range template
./calibrate.sh --cli /dev/ttyACM0 --data /dev/ttyACM1 --template extended_range

# Load existing calibration
./calibrate.sh --cli /dev/ttyACM0 --load my_calibration.json

# Verbose output
./calibrate.sh --cli /dev/ttyACM0 --data /dev/ttyACM1 --verbose
```

## Direct Python Usage

For more control, you can use the Python script directly:

```bash
python3 cali.py --cli /dev/ttyACM0 --data /dev/ttyACM1 --distance 1.5 --template standard
```

## Configuration Templates

The calibrator supports multiple configuration templates:

- **standard**: Default configuration suitable for most use cases
- **extended_range**: Optimized for longer-range measurements

## Calibration Quality Assessment

The calibrator automatically assesses the quality of calibration results:

- **Range Bias**: Should ideally be close to zero
- **Phase Variance**: Measures consistency of phase calibration
- **Overall Quality**: Combined assessment (good, acceptable, poor)

## Troubleshooting

### No Serial Ports Found

- Make sure the radar is connected via USB
- Try unplugging and plugging back in
- On Linux, you might need to add your user to the `dialout` group:
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  (Logout and login again for this to take effect)

### Calibration Fails

- Ensure the corner reflector is positioned exactly at the specified distance
- Make sure there are no other large reflective objects in front of the radar
- Check that you're using the correct COM ports
- Try increasing the timeout with `--timeout 120`

### Poor Calibration Quality

- Reposition the corner reflector to be directly in front of the radar
- Ensure the radar and reflector are both level
- Try different distances (1.5m to 2.5m works best)
- Make sure the environment is free from interference

## Output Files

The calibrator produces two files:

1. **CFG file** (e.g., `awr1843_calibrated_20220101_120000.cfg`): Configuration file for mmWave Studio
2. **JSON file** (e.g., `awr1843_calibrated_20220101_120000.json`): Machine-readable calibration data

## Advanced Usage

### Loading Existing Calibration

```bash
python3 cali.py --cli /dev/ttyACM0 --load my_calibration.cfg
```

### Custom Output File

```bash
python3 cali.py --cli /dev/ttyACM0 --data /dev/ttyACM1 --output my_custom_calibration.cfg
```

## Integration with Radar Analyzer

The calibration values can be loaded directly into the AWR Radar Analyzer application through the settings panel under "Calibration". 