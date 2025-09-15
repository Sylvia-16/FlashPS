timestamp=$(date +"%Y%m%d_%H%M%S")
echo "Server log will be saved to: $(pwd)/cb_sd2_server_log/log_sd2_${timestamp}.log"
mkdir -p cb_sd2_server_log
python -u server.py --config dist_config.yml --worker-max-batch-size 4 --scheduling-baseline "seq_length_balance" --pipeline-name SD2 > cb_sd2_server_log/log_sd2_${timestamp}.log 2>&1 &

sleep 90
echo "Server should start successfully"
tail -n 20 cb_sd2_server_log/log_sd2_${timestamp}.log