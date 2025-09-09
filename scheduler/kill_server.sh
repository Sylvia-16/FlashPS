#!/bin/bash

# Script: Kill all Python processes
# Usage: ./kill_python.sh

echo "Searching for all Python processes..."

# Find all Python processes
python_pids=$(ps aux | grep python | grep -v grep | awk '{print $2}')

if [ -z "$python_pids" ]; then
    echo "No Python processes found"
    exit 0
fi

echo "Found the following Python processes:"
ps aux | grep python | grep -v grep

echo ""
echo "Preparing to kill these processes..."

# Kill processes one by one
for pid in $python_pids; do
    echo "Killing process $pid..."
    kill -9 $pid
    if [ $? -eq 0 ]; then
        echo "Process $pid successfully terminated"
    else
        echo "Failed to terminate process $pid"
    fi
done

echo ""
echo "Checking for remaining Python processes..."
remaining=$(ps aux | grep python | grep -v grep)
if [ -z "$remaining" ]; then
    echo "All Python processes have been successfully terminated"
else
    echo "The following Python processes are still running:"
    echo "$remaining"
fi

echo "Identifying processes running on GPUs..."

# Get all processes running on GPUs using nvidia-smi
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

if [ -z "$GPU_PROCESSES" ]; then
    echo "No processes found running on GPUs."
    exit 0
fi

echo "Found the following PIDs running on GPUs: $GPU_PROCESSES"
echo "Killing processes..."

# Kill each process
for PID in $GPU_PROCESSES; do
    echo "Killing process $PID"
    kill -9 $PID
done

echo "All GPU processes have been terminated."
