timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p 8gpu_server_log
echo "Server log will be saved to: $(pwd)/8gpu_server_log/log_ootd_batch_size_8_${timestamp}.log"
python -u server.py --config dist_config.yml --worker-max-batch-size 8 --pipeline-name OOTD_HD --scheduling-baseline flops_balance > 8gpu_server_log/log_ootd_batch_size_8_${timestamp}.log 2>&1 &
