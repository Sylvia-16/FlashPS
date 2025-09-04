
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
