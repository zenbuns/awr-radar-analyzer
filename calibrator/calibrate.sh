#!/bin/bash
# AWR1843 Calibrator Helper Script
# This script simplifies finding the correct ports and running the calibration.

echo "=== AWR1843BOOST Calibrator Helper ==="
echo 

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Check for pyserial
if ! python3 -c "import serial" &> /dev/null; then
    echo "Warning: pyserial module not found. Attempting to install..."
    pip3 install pyserial || { echo "Failed to install pyserial. Please install manually: pip3 install pyserial"; exit 1; }
fi

# Function to find available serial ports
find_ports() {
    echo "Scanning for available serial ports..."
    
    # Different commands for different operating systems
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Found ports:"
        ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "  No known serial ports found."
        
        # Auto-detection logic for Linux
        DATA_PORT=""
        CLI_PORT=""
        
        # Try to auto-detect the ports
        PORTS=($(ls /dev/ttyACM* 2>/dev/null))
        if [ ${#PORTS[@]} -ge 2 ]; then
            # Usually the lower-numbered port is the CLI
            CLI_PORT="${PORTS[0]}"
            DATA_PORT="${PORTS[1]}"
            echo
            echo "Auto-detected ports:"
            echo "  CLI port: $CLI_PORT"
            echo "  Data port: $DATA_PORT"
        fi
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Found ports:"
        ls -l /dev/tty.usbmodem* 2>/dev/null || echo "  No known serial ports found."
        
    elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows with MSYS or similar
        echo "On Windows, please use Device Manager to identify the COM ports."
        echo "Look for 'XDS110 Class Application/User UART' devices."
    else
        echo "Unsupported operating system. Please check available serial ports manually."
    fi
    
    echo
    echo "Typical connection pattern:"
    echo "  CLI port: Lower number (e.g., /dev/ttyACM0 on Linux, COM4 on Windows)"
    echo "  Data port: Higher number (e.g., /dev/ttyACM1 on Linux, COM5 on Windows)"
    echo
}

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --cli PORT     Specify CLI port (e.g., /dev/ttyACM0 or COM4)"
    echo "  --data PORT    Specify Data port (e.g., /dev/ttyACM1 or COM5)"
    echo "  --distance VAL Target distance in meters (default: 1.5)"
    echo "  --output FILE  Output file for calibration"
    echo "  --template TPL Configuration template (standard or extended_range)"
    echo "  --load FILE    Load existing calibration file"
    echo "  --scan         Scan for available serial ports"
    echo "  --help         Show this help"
    echo
    echo "Example:"
    echo "  $0 --cli /dev/ttyACM0 --data /dev/ttyACM1 --distance 1.5"
    echo
}

# Parse command line arguments
CLI_PORT_ARG=""
DATA_PORT_ARG=""
DISTANCE_ARG="1.5"
OUTPUT_ARG=""
TEMPLATE_ARG="standard"
LOAD_ARG=""
VERBOSE_ARG=""

# If no arguments, show help
if [ $# -eq 0 ]; then
    show_help
    find_ports
    exit 0
fi

# Parse arguments
while [ "$1" != "" ]; do
    case $1 in
        --cli)
            shift
            CLI_PORT_ARG=$1
            ;;
        --data)
            shift
            DATA_PORT_ARG=$1
            ;;
        --distance)
            shift
            DISTANCE_ARG=$1
            ;;
        --output)
            shift
            OUTPUT_ARG=$1
            ;;
        --template)
            shift
            TEMPLATE_ARG=$1
            ;;
        --load)
            shift
            LOAD_ARG=$1
            ;;
        --scan)
            find_ports
            exit 0
            ;;
        --verbose|-v)
            VERBOSE_ARG="--verbose"
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# Check if we have auto-detected ports and no user-specified ports
if [ -z "$CLI_PORT_ARG" ] && [ -n "$CLI_PORT" ]; then
    echo "Using auto-detected CLI port: $CLI_PORT"
    CLI_PORT_ARG=$CLI_PORT
fi

if [ -z "$DATA_PORT_ARG" ] && [ -n "$DATA_PORT" ]; then
    echo "Using auto-detected Data port: $DATA_PORT"
    DATA_PORT_ARG=$DATA_PORT
fi

# Validate required arguments
if [ -z "$CLI_PORT_ARG" ] || [ -z "$DATA_PORT_ARG" ]; then
    if [ -z "$LOAD_ARG" ]; then
        echo "Error: CLI and Data ports are required for calibration."
        echo "Use --scan to find available ports or specify with --cli and --data."
        exit 1
    elif [ -z "$CLI_PORT_ARG" ]; then
        echo "Error: CLI port is required even when loading calibration."
        exit 1
    fi
fi

# Build the command
CMD="python3 $(dirname "$0")/cali.py --cli \"$CLI_PORT_ARG\""

if [ -n "$DATA_PORT_ARG" ]; then
    CMD="$CMD --data \"$DATA_PORT_ARG\""
fi

CMD="$CMD --distance $DISTANCE_ARG --template $TEMPLATE_ARG"

if [ -n "$OUTPUT_ARG" ]; then
    CMD="$CMD --output \"$OUTPUT_ARG\""
fi

if [ -n "$LOAD_ARG" ]; then
    CMD="$CMD --load \"$LOAD_ARG\""
fi

if [ -n "$VERBOSE_ARG" ]; then
    CMD="$CMD $VERBOSE_ARG"
fi

echo "Running calibration with command:"
echo "  $CMD"
echo

# Execute the command
eval $CMD

# Display outcome
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo
    echo "Calibration completed successfully!"
else
    echo
    echo "Calibration failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE 