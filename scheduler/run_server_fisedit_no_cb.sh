timestamp=$(date +"%Y%m%d_%H%M%S")
echo "Server log will be saved to: $(pwd)/server_log/log_no_cb${timestamp}.log"
python -u server_no_cb_fisedit.py --config dist_config.yml > server_log/log_no_cb${timestamp}.log 2>&1 &