#!/usr/bin/env python3
"""
AWR1843BOOST Calibration Tool

This script automates the calibration process for the AWR1843BOOST mmWave radar.
It connects to the radar via serial ports, performs calibration measurements using
the provided configuration, and generates the compRangeBiasAndRxChanPhase values.

Requirements:
- Python 3.6+
- pyserial library (pip install pyserial)
- AWR1843BOOST connected via USB to the computer

Usage:
1. Connect the AWR1843BOOST to your computer via USB
2. Find the COM ports for CLI and Data (Windows Device Manager or ls /dev/tty* on Linux/Mac)
3. Place a corner reflector at 1.5 meters from the radar
4. Run this script and follow the prompts
"""

import serial
import time
import re
import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('AWR1843Calibrator')

class AWR1843Calibrator:
    def __init__(self, cli_port, data_port, target_distance=1.5, timeout=60):
        """
        Initialize the calibrator with COM ports and target distance
        
        Args:
            cli_port: COM port for the CLI interface
            data_port: COM port for the data interface
            target_distance: Distance to the target in meters (default 1.5)
            timeout: Max time to wait for calibration results in seconds
        """
        self.cli_port = cli_port
        self.data_port = data_port
        self.target_distance = float(target_distance)
        self.timeout = timeout
        self.cli_serial = None
        self.data_serial = None
        self.calibration_result = None
        self.retry_count = 3  # Number of retries for failed commands
        self.is_connected = False
        self.last_error = None
        self.calibration_quality = None  # Store calibration quality assessment

    def connect(self):
        """
        Connect to both serial ports with retry mechanism
        
        Returns:
            bool: True if successfully connected, False otherwise
        """
        for attempt in range(self.retry_count):
            try:
                logger.info(f"Connecting to CLI port {self.cli_port}... (Attempt {attempt+1}/{self.retry_count})")
                self.cli_serial = serial.Serial(self.cli_port, 115200, timeout=0.1)
                
                logger.info(f"Connecting to Data port {self.data_port}... (Attempt {attempt+1}/{self.retry_count})")
                self.data_serial = serial.Serial(self.data_port, 921600, timeout=0.1)
                
                logger.info("Successfully connected to AWR1843BOOST")
                self.is_connected = True
                return True
            except serial.SerialException as e:
                self.last_error = str(e)
                logger.error(f"Error connecting to serial ports: {e}")
                if attempt < self.retry_count - 1:
                    logger.info(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    logger.error(f"Failed to connect after {self.retry_count} attempts")
        return False

    def send_command(self, command, max_retries=2):
        """
        Send command to the CLI port and read response with retry mechanism
        
        Args:
            command: Command string to send
            max_retries: Maximum number of retries for failed commands
            
        Returns:
            str: Response from the radar, or None if failed
        """
        if self.cli_serial is None:
            logger.error("Error: Not connected to CLI port")
            return None
        
        # Add newline if not present
        if not command.endswith('\n'):
            command += '\n'
        
        for attempt in range(max_retries + 1):
            try:
                # Send command
                self.cli_serial.write(command.encode('utf-8'))
                time.sleep(0.1)  # Allow time for command processing
                
                # Read response
                response = b''
                start_time = time.time()
                while time.time() - start_time < 1.0:  # Wait up to 1 second for response
                    if self.cli_serial.in_waiting:
                        response += self.cli_serial.read(self.cli_serial.in_waiting)
                    time.sleep(0.1)
                
                response_text = response.decode('utf-8', errors='ignore')
                
                # Check for errors in response
                if "Error" in response_text:
                    logger.warning(f"Command returned error: {response_text.strip()}")
                    if attempt < max_retries:
                        logger.info(f"Retrying command: {command.strip()}...")
                        time.sleep(0.5)
                        continue
                
                return response_text
            except Exception as e:
                logger.error(f"Error sending command '{command.strip()}': {str(e)}")
                if attempt < max_retries:
                    logger.info("Retrying...")
                    time.sleep(0.5)
                else:
                    logger.error(f"Failed to send command after {max_retries + 1} attempts")
                    return None
        
        return None

    def generate_calibration_config(self, template="standard"):
        """
        Generate calibration configuration based on provided settings and template
        
        Args:
            template: Configuration template to use (standard, extended, etc.)
            
        Returns:
            list: Configuration commands
        """
        # Standard calibration configuration
        if template == "standard":
            config = [
                "sensorStop",
                "flushCfg",
                "dfeDataOutputMode 1",
                "channelCfg 15 7 0",
                "adcCfg 2 1",
                "adcbufCfg -1 0 1 1 1",
                "lowPower 0 0",
                "profileCfg 0 77 7 3 39 0 0 100 1 256 7200 0 0 30",
                "chirpCfg 0 0 0 0 0 0 0 1",
                "chirpCfg 1 1 0 0 0 0 0 4",
                "chirpCfg 2 2 0 0 0 0 0 2",
                "frameCfg 0 2 32 0 500 1 0",
                "guiMonitor -1 1 1 1 1 0 1",
                "cfarCfg -1 0 2 8 4 3 0 15.0 0",
                "cfarCfg -1 1 0 4 2 3 1 15.0 0",
                "multiObjBeamForming -1 1 0.5",
                "calibDcRangeSig -1 0 -5 8 256",
                "clutterRemoval -1 0",
                "compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
                f"measureRangeBiasAndRxChanPhase 1 {self.target_distance} 0.2",
                "aoaFovCfg -1 -90 90 -90 90",
                "cfarFovCfg -1 0 0.25 8.64",
                "cfarFovCfg -1 1 -7.06 7.06",
                "extendedMaxVelocity -1 0",
                "CQRxSatMonitor 0 3 4 127 0",
                "CQSigImgMonitor 0 111 4",
                "analogMonitor 0 0",
                "lvdsStreamCfg -1 0 0 0",
                "calibData 0 0 0",
                "sensorStart"
            ]
        # Extended range configuration for longer distances
        elif template == "extended_range":
            config = [
                "sensorStop",
                "flushCfg",
                "dfeDataOutputMode 1",
                "channelCfg 15 7 0",
                "adcCfg 2 1",
                "adcbufCfg -1 0 1 1 1",
                "lowPower 0 0",
                "profileCfg 0 77 20 3 40 0 0 100 1 512 10000 0 0 30",  # Modified for extended range
                "chirpCfg 0 0 0 0 0 0 0 1",
                "chirpCfg 1 1 0 0 0 0 0 4",
                "chirpCfg 2 2 0 0 0 0 0 2",
                "frameCfg 0 2 32 0 500 1 0",
                "guiMonitor -1 1 1 1 1 0 1",
                "cfarCfg -1 0 2 8 4 3 0 15.0 0",
                "cfarCfg -1 1 0 4 2 3 1 15.0 0",
                "multiObjBeamForming -1 1 0.5",
                "calibDcRangeSig -1 0 -5 8 256",
                "clutterRemoval -1 0",
                "compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
                f"measureRangeBiasAndRxChanPhase 1 {self.target_distance} 0.2",
                "aoaFovCfg -1 -90 90 -90 90",
                "cfarFovCfg -1 0 0.25 16.0",  # Modified for extended range
                "cfarFovCfg -1 1 -7.06 7.06",
                "extendedMaxVelocity -1 0",
                "CQRxSatMonitor 0 3 4 127 0",
                "CQSigImgMonitor 0 111 4",
                "analogMonitor 0 0",
                "lvdsStreamCfg -1 0 0 0",
                "calibData 0 0 0",
                "sensorStart"
            ]
        else:
            logger.error(f"Unknown template: {template}")
            config = []
        
        return config

    def run_calibration(self, template="standard"):
        """
        Run the calibration process
        
        Args:
            template: Configuration template to use
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        if not self.is_connected and not self.connect():
            logger.error("Failed to connect to AWR1843BOOST.")
            return False
        
        logger.info("\n=== Starting Calibration Process ===")
        logger.info(f"Target distance: {self.target_distance} meters")
        logger.info(f"Using template: {template}")
        logger.info("Generating and sending calibration configuration...")
        
        # Get calibration configuration
        config = self.generate_calibration_config(template)
        
        if not config:
            logger.error("Failed to generate configuration")
            return False
        
        # Send configuration commands
        logger.info("Sending configuration to radar...")
        print("\nSending commands to radar: ", end="")
        success_count = 0
        
        for i, cmd in enumerate(config):
            # Show progress
            if i % 5 == 0:
                print(".", end="", flush=True)
                
            response = self.send_command(cmd)
            if response is None or "Error" in response:
                logger.error(f"Command failed: {cmd}")
                continue
            success_count += 1
        
        if success_count < len(config):
            logger.warning(f"{len(config) - success_count} commands failed out of {len(config)}")
        
        print(" Done!")
        logger.info("\nCalibration in progress. Please do not move the target...")
        
        # Wait for calibration result
        calibration_found = False
        start_time = time.time()
        pattern = r'compRangeBiasAndRxChanPhase\s+([0-9.-]+(?:\s+[0-9.-]+)*)'
        
        # Create a progress bar
        print("\nWaiting for calibration results...")
        progress_chars = ["|", "/", "-", "\\"]
        progress_idx = 0
        
        # Read from both CLI and data ports to look for calibration results
        while time.time() - start_time < self.timeout:
            if self.cli_serial.in_waiting:
                line = self.cli_serial.readline().decode('utf-8', errors='ignore')
                if 'compRangeBiasAndRxChanPhase' in line:
                    match = re.search(pattern, line)
                    if match:
                        self.calibration_result = "compRangeBiasAndRxChanPhase " + match.group(1)
                        calibration_found = True
                        break
            
            if self.data_serial.in_waiting:
                self.data_serial.read(self.data_serial.in_waiting)  # Clear the buffer
            
            # Show progress indicator
            elapsed = int(time.time() - start_time)
            remaining = max(0, self.timeout - elapsed)
            
            # Update progress every second
            if elapsed % 1 == 0:
                progress_idx = (progress_idx + 1) % len(progress_chars)
                progress_char = progress_chars[progress_idx]
                progress_percent = min(100, int((elapsed / self.timeout) * 100))
                print(f"\r{progress_char} Calibrating... {progress_percent}% ({elapsed}s elapsed, {remaining}s remaining)", end="")
            
            time.sleep(0.1)
        
        print("\n")  # Move to next line after progress display
        
        # Stop the sensor
        logger.info("Stopping sensor...")
        self.send_command("sensorStop")
        
        if calibration_found:
            logger.info("Calibration successful!")
            logger.info("Calibration result:")
            logger.info(f"\n{self.calibration_result}\n")
            
            # Validate the calibration results
            self.validate_calibration()
            
            return True
        else:
            logger.error("Calibration failed: No calibration result received within timeout period.")
            logger.error("Possible issues:")
            logger.error("1. Target not positioned correctly")
            logger.error("2. Target not reflective enough")
            logger.error("3. Incorrect COM ports")
            logger.error("4. Other communication issues")
            return False

    def validate_calibration(self):
        """
        Validate the quality of calibration results
        """
        if not self.calibration_result:
            logger.error("No calibration result to validate")
            return False
        
        # Extract the bias and phase values
        values = re.findall(r'[-+]?\d*\.\d+|\d+', self.calibration_result)
        if not values:
            logger.error("Could not extract calibration values for validation")
            return False
        
        # Convert values to floats
        try:
            values = [float(v) for v in values]
        except ValueError:
            logger.error("Invalid numeric values in calibration result")
            return False
        
        # Analyze range bias (first value)
        range_bias = values[0]
        
        # Simple quality checks
        self.calibration_quality = {
            "range_bias": range_bias,
            "bias_quality": "unknown",
            "phase_variance": 0.0,
            "phase_quality": "unknown",
            "overall_quality": "unknown"
        }
        
        # Range bias should typically be small (usually between -0.2 to 0.2 meters)
        if abs(range_bias) < 0.1:
            self.calibration_quality["bias_quality"] = "excellent"
        elif abs(range_bias) < 0.2:
            self.calibration_quality["bias_quality"] = "good"
        elif abs(range_bias) < 0.5:
            self.calibration_quality["bias_quality"] = "acceptable"
        else:
            self.calibration_quality["bias_quality"] = "poor"
            
        # Calculate phase variance (excluding the first value which is range bias)
        phase_values = values[1:]
        if len(phase_values) > 1:
            import numpy as np
            phase_variance = np.var(phase_values) if 'numpy' in sys.modules else sum((x - sum(phase_values)/len(phase_values))**2 for x in phase_values) / len(phase_values)
            self.calibration_quality["phase_variance"] = float(phase_variance)
            
            # Phase values should be consistent
            if phase_variance < 0.1:
                self.calibration_quality["phase_quality"] = "excellent"
            elif phase_variance < 0.3:
                self.calibration_quality["phase_quality"] = "good"
            elif phase_variance < 0.6:
                self.calibration_quality["phase_quality"] = "acceptable"
            else:
                self.calibration_quality["phase_quality"] = "poor"
        
        # Overall quality
        if self.calibration_quality["bias_quality"] in ["excellent", "good"] and \
           self.calibration_quality["phase_quality"] in ["excellent", "good"]:
            self.calibration_quality["overall_quality"] = "good"
        elif self.calibration_quality["bias_quality"] == "poor" or \
             self.calibration_quality["phase_quality"] == "poor":
            self.calibration_quality["overall_quality"] = "poor"
        else:
            self.calibration_quality["overall_quality"] = "acceptable"
            
        # Log the quality assessment
        logger.info("Calibration Quality Assessment:")
        logger.info(f"  Range Bias: {range_bias:.6f} m ({self.calibration_quality['bias_quality']})")
        logger.info(f"  Phase Variance: {self.calibration_quality['phase_variance']:.6f} ({self.calibration_quality['phase_quality']})")
        logger.info(f"  Overall Quality: {self.calibration_quality['overall_quality']}")
        
        if self.calibration_quality["overall_quality"] != "good":
            logger.warning("Calibration quality is not optimal. Consider recalibrating.")
            if abs(range_bias) > 0.5:
                logger.warning("Range bias is unusually large. Check target positioning.")
        
        return True

    def save_calibration_config(self, output_file=None):
        """
        Save calibration result to a new configuration file
        
        Args:
            output_file: Path to output file (optional)
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.calibration_result:
            logger.error("Error: No calibration result available")
            return False
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"awr1843_calibrated_{timestamp}.cfg"
        
        logger.info(f"Saving calibration to {output_file}...")
        
        # Create a configuration with the calibration results
        config_lines = [
            "% ***************************************************************",
            "% AWR1843 Calibrated Configuration",
            f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"% Target Distance: {self.target_distance} meters",
            "% ***************************************************************",
            "% Carrier frequency     GHz                           77",
            "% Ramp Slope    MHz/us                                100",
            "% Num ADC Samples                                     256",
            "% ADC Sampling Rate Msps                              7.2",
            "% ADC Collection Time   us                            35.56",
            "% Extra ramp time required (start time) us            3",
            "% Chirp time (end time - start time)    us            36",
            "% Chirp duration (end time) us                        39",
            "% Sweep BW (useful) MHz                               3555.56",
            "% Total BW  MHz                                       3900",
            "% Max beat freq (80% of ADC sampling rate)  MHz       5.76",
            "% Max distance (80%)    m                             8.64",
            "% Range resolution  m                                 0.042",
            "% Range resolution (meter per 1D-FFT bin)   m/bin     0.042",
            "%                                                     ",
            "% Inter-chirp duration  us                            7",
            "% Number of chirp intervals in frame    -             96",
            "% Number of TX (TDM MIMO)                             3",
            "% Number of Tx elevation antennas                     0",
            "% Number of RX channels -                             4",
            "% Max umambiguous relative velocity kmph              25.41",
            "%   mileph                                            15.88",
            "% Max extended relative velocity    kmph              76.23",
            "%   mileph                                            47.64",
            "% Frame time (total)    ms                            4.416",
            "% Frame time (active)   ms                            3.744",
            "% Range FFT size    -                                 256",
            "% Doppler FFT size  -                                 32",
            "% Radar data memory required    KB                    400",
            "% Velocity resolution   m/s                           0.44",
            "% Velocity resolution (m/s per 2D-FFT bin)  m/s/bin   0.44",
            "% Velocity Maximum  m/s                               7.06",
            "% Extended Maximum Velocity m/s                       21.17",
            "% Maximum sweep accorss range bins  range bin         0.74",
            "% ",
            "sensorStop",
            "flushCfg",
            "dfeDataOutputMode 1",
            "channelCfg 15 7 0",
            "adcCfg 2 1",
            "adcbufCfg -1 0 1 1 1",
            "lowPower 0 0",
            "profileCfg 0 77 7 3 39 0 0 100 1 256 7200 0 0 30",
            "chirpCfg 0 0 0 0 0 0 0 1",
            "chirpCfg 1 1 0 0 0 0 0 4",
            "chirpCfg 2 2 0 0 0 0 0 2",
            "frameCfg 0 2 32 0 500 1 0",
            "guiMonitor -1 1 1 1 1 0 1",
            "cfarCfg -1 0 2 8 4 3 0 15.0 0",
            "cfarCfg -1 1 0 4 2 3 1 15.0 0",
            "multiObjBeamForming -1 1 0.5",
            "calibDcRangeSig -1 0 -5 8 256",
            "clutterRemoval -1 0",
            f"{self.calibration_result}",  # Use the calibration result
            "% measureRangeBiasAndRxChanPhase 1 1.5 0.2  % Commented out after calibration",
            "aoaFovCfg -1 -90 90 -90 90",
            "cfarFovCfg -1 0 0.25 8.64",
            "cfarFovCfg -1 1 -7.06 7.06",
            "extendedMaxVelocity -1 0",
            "CQRxSatMonitor 0 3 4 127 0",
            "CQSigImgMonitor 0 111 4",
            "analogMonitor 0 0",
            "lvdsStreamCfg -1 0 0 0",
            "calibData 0 0 0",
            "sensorStart"
        ]
        
        # Add calibration quality assessment as comments
        if self.calibration_quality:
            config_lines.insert(5, f"% Quality Assessment: {self.calibration_quality.get('overall_quality', 'unknown').upper()}")
            config_lines.insert(6, f"% Range Bias: {self.calibration_quality.get('range_bias', 0):.6f} m ({self.calibration_quality.get('bias_quality', 'unknown')})")
            config_lines.insert(7, f"% Phase Consistency: {self.calibration_quality.get('phase_quality', 'unknown')}")
        
        # Write the configuration
        try:
            with open(output_file, 'w') as f:
                for line in config_lines:
                    f.write(line + '\n')
            
            logger.info(f"Calibration configuration saved to {output_file}")
            
            # Also save calibration data in JSON format for easier parsing
            json_file = output_file.replace('.cfg', '.json')
            calib_data = {
                'timestamp': datetime.now().isoformat(),
                'target_distance': self.target_distance,
                'calibration_result': self.calibration_result,
                'quality': self.calibration_quality
            }
            
            with open(json_file, 'w') as f:
                json.dump(calib_data, f, indent=2)
                
            logger.info(f"Calibration data also saved to {json_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving calibration: {str(e)}")
            return False

    def load_calibration(self, file_path):
        """
        Load existing calibration from file
        
        Args:
            file_path: Path to calibration file (.cfg or .json)
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.calibration_result = data.get('calibration_result')
                    self.calibration_quality = data.get('quality')
                    logger.info(f"Loaded calibration from {file_path}")
            else:  # Assume .cfg
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith('compRangeBiasAndRxChanPhase'):
                            self.calibration_result = line.strip()
                            logger.info(f"Loaded calibration from {file_path}")
                            break
            
            if self.calibration_result:
                logger.info(f"Loaded calibration: {self.calibration_result}")
                return True
            else:
                logger.error(f"No calibration data found in {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading calibration: {str(e)}")
            return False

    def disconnect(self):
        """Close the serial connections"""
        if self.cli_serial:
            self.cli_serial.close()
        if self.data_serial:
            self.data_serial.close()
        logger.info("Disconnected from AWR1843BOOST")
        self.is_connected = False


def main():
    parser = argparse.ArgumentParser(description='AWR1843BOOST Calibration Tool')
    parser.add_argument('--cli', required=True, help='COM port for CLI interface (e.g., COM4 or /dev/ttyACM0)')
    parser.add_argument('--data', required=True, help='COM port for Data interface (e.g., COM5 or /dev/ttyACM1)')
    parser.add_argument('--distance', type=float, default=1.5, help='Distance to target in meters (default: 1.5)')
    parser.add_argument('--output', help='Output file name for calibration config (optional)')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout for calibration in seconds (default: 60)')
    parser.add_argument('--template', choices=['standard', 'extended_range'], default='standard', 
                        help='Configuration template to use (default: standard)')
    parser.add_argument('--load', help='Load existing calibration from file instead of performing calibration')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Banner
    print("=" * 70)
    print("           AWR1843BOOST Calibration Tool")
    print("=" * 70)
    print("\nThis tool performs range bias and RX channel phase calibration")
    print("for the AWR1843BOOST mmWave radar.\n")
    
    # Create calibrator
    calibrator = AWR1843Calibrator(
        args.cli, 
        args.data, 
        args.distance,
        args.timeout
    )
    
    try:
        # If loading existing calibration
        if args.load:
            logger.info(f"Loading existing calibration from {args.load}")
            if calibrator.load_calibration(args.load):
                # Save to a new file if requested
                if args.output:
                    calibrator.save_calibration_config(args.output)
                    logger.info(f"Existing calibration saved to new file: {args.output}")
                else:
                    logger.info("Loaded calibration successfully")
                    
                # Display the calibration data
                print("\nLoaded Calibration Data:")
                print(f"  {calibrator.calibration_result}")
                if calibrator.calibration_quality:
                    print("\nCalibration Quality Assessment:")
                    print(f"  Range Bias: {calibrator.calibration_quality.get('range_bias', 0):.6f} m "
                          f"({calibrator.calibration_quality.get('bias_quality', 'unknown')})")
                    print(f"  Overall Quality: {calibrator.calibration_quality.get('overall_quality', 'unknown').upper()}")
                
                return 0
            else:
                logger.error("Failed to load calibration file")
                return 1
        
        # Otherwise perform new calibration
        
        # Instructions
        print("IMPORTANT: Before proceeding, please ensure that:")
        print("1. The radar is firmly mounted and level")
        print("2. A corner reflector is placed at EXACTLY", args.distance, "meters from the radar")
        print("3. The reflector is positioned directly in front of the radar (0° azimuth)")
        print("4. There are no other significant reflectors in the field of view")
        print(f"5. Using configuration template: {args.template}\n")
        
        input("Press Enter when ready to proceed with calibration...")
        
        # Run calibration with the specified template
        if calibrator.run_calibration(args.template):
            # Save calibration
            calibrator.save_calibration_config(args.output)
            
            print("\nCalibration completed successfully!")
            print("Results:")
            print(f"  {calibrator.calibration_result}")
            
            if calibrator.calibration_quality:
                print("\nCalibration Quality Assessment:")
                print(f"  Range Bias: {calibrator.calibration_quality['range_bias']:.6f} m "
                     f"({calibrator.calibration_quality['bias_quality']})")
                print(f"  Phase Variance: {calibrator.calibration_quality['phase_variance']:.6f} "
                     f"({calibrator.calibration_quality['phase_quality']})")
                print(f"  Overall Quality: {calibrator.calibration_quality['overall_quality'].upper()}")
                
                if calibrator.calibration_quality['overall_quality'] != 'good':
                    print("\nWARNING: Calibration quality is not optimal. Consider recalibrating.")
                    if abs(calibrator.calibration_quality['range_bias']) > 0.5:
                        print("         Range bias is unusually large. Check target positioning.")
            
            print("\nYou can now use the generated configuration file with mmWave Demo Visualizer")
            return 0
        else:
            print("\nCalibration process failed. Please check setup and try again.")
            return 1
    
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1
    finally:
        calibrator.disconnect()

if __name__ == "__main__":
    sys.exit(main())
