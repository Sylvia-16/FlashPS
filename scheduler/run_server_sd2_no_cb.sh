timestamp=$(date +"%Y%m%d_%H%M%S")
echo "Server log will be saved to: $(pwd)/no_cb_sd2_server_log/log_sd2_${timestamp}.log"
mkdir -p no_cb_sd2_server_log
python -u server.py --config dist_config.yml --worker-max-batch-size 4 --scheduling-baseline "no_cb" --pipeline-name SD2 > no_cb_sd2_server_log/log_sd2_${timestamp}.log 2>&1 &

sleep 90
echo "Server should start successfully"
tail -n 20 no_cb_sd2_server_log/log_sd2_${timestamp}.log