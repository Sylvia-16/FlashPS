timestamp=$(date +"%Y%m%d_%H%M%S")
echo "Server log will be saved to: $(pwd)/server_ootd_no_cb/8_batch_size_log_no_cb_${timestamp}.log"
python -u server.py --config dist_config.yml --scheduling-baseline no_cb --worker-max-batch-size 8 --pipeline-name OOTD_HD > server_ootd_no_cb/8_batch_size_log_no_cb_${timestamp}.log 2>&1 &
