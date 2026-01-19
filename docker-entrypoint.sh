#!/bin/bash
set -e

# Function to handle shutdown
cleanup() {
    echo "Shutting down..."
    if [ ! -z "$POWERMONITOR_PID" ]; then
        kill $POWERMONITOR_PID 2>/dev/null || true
    fi
    if [ ! -z "$WEBAPP_PID" ]; then
        kill $WEBAPP_PID 2>/dev/null || true
    fi
    wait $POWERMONITOR_PID $WEBAPP_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start powerMonitor.py in background
echo "Starting powerMonitor.py..."
python3 powerMonitor.py &
POWERMONITOR_PID=$!
echo "powerMonitor.py started with PID: $POWERMONITOR_PID"

# Wait a bit for powerMonitor to initialize
sleep 3

# Start web application
echo "Starting web application on port 8888..."
python3 nonsense_power_analyzer.py &
WEBAPP_PID=$!
echo "Web application started with PID: $WEBAPP_PID"

# Wait for both processes
wait $POWERMONITOR_PID $WEBAPP_PID
