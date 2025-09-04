import numpy as np
import argparse
import os
import re
import glob
import json
import pandas as pd
from collections import defaultdict

def parse_log_for_one_log_file(log_file):
    """
    Parse a single log file and extract latency values.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Dictionary containing lists of latency values and p99/p90 metrics
    """
    try:
        print("log_file: ", log_file)
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Find all latency values using regex
        avg_latencies = [float(match) for match in re.findall(r'Average_latency:\s*([\d.]+)\s*seconds', content)]
        print("avg_latencies: ", avg_latencies)
        
        # Find all request latency arrays
        request_latency_patterns = re.findall(r'Request Latencies: \[([\d\., ]+)\]', content)
        inference_latency_patterns = re.findall(r'Inference latencies: \[([\d\., ]+)\]', content)
        
        request_latencies = []
        for pattern in request_latency_patterns:
            latency_values = [float(val.strip()) for val in pattern.split(',')]
            request_latencies.extend(latency_values)
            
        inference_latencies = []
        for pattern in inference_latency_patterns:
            latency_values = [float(val.strip()) for val in pattern.split(',')]
            inference_latencies.extend(latency_values)
            
        # Calculate p99 and p90 if request_latencies is not empty
        p99_latency = np.percentile(request_latencies, 99) if request_latencies else None
        p95_latency = np.percentile(request_latencies, 95) if request_latencies else None
        
        # Calculate p99 and p90 for inference latencies
        p99_inference_latency = np.percentile(inference_latencies, 99) if inference_latencies else None
        p95_inference_latency = np.percentile(inference_latencies, 95) if inference_latencies else None
        avg_inference_latency = np.mean(inference_latencies) if inference_latencies else None
        print(f"Found {len(request_latencies)} request latencies and {len(inference_latencies)} inference latencies")
        
        return {
            'avg_latency': avg_latencies[0] if avg_latencies else None,
            'p99_latency': p99_latency,
            'p95_latency': p95_latency,
            'p99_inference_latency': p99_inference_latency,
            'p95_inference_latency': p95_inference_latency,
            'avg_inference_latency':avg_inference_latency
        }
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return {
            'avg_latency': None,
            'p99_latency': None,
            'p95_latency': None,
            'p99_inference_latency': None,
            'p95_inference_latency': None
        }

def parse_log_for_one_folder(log_folder):
    """
    Parse all log files in the given folder structure: folder/seqlen_list_xxxx_datetime/xxx.log
    
    Args:
        log_folder: The root folder containing seqlen_list folders
    
    Returns:
        A dictionary with seqlen_list as keys and latency metrics as values
    """
    results = {}
    
    # Find all seqlen_list folders
    seqlen_folders = glob.glob(os.path.join(log_folder, "seqlen_list_*"))
    
    for folder in seqlen_folders:
        seqlen_name = os.path.basename(folder)
        # Find all log files in this folder
        log_files = glob.glob(os.path.join(folder, "*.log"))
        
        if log_files:
            # Only take the first log file since we only need one result per folder
            latencies = parse_log_for_one_log_file(log_files[0])
            results[seqlen_name] = latencies
    
    return results

def get_first_subdirectory(directory_path):
    # 获取目录下的所有内容
    contents = os.listdir(directory_path)
    
    # 遍历内容，找到第一个文件夹
    for item in contents:
        full_path = os.path.join(directory_path, item)
        # if item ends with backup
        if item.endswith('backup'):
            print("item",item)
            continue
        if os.path.isdir(full_path):
            return full_path
    
    return None  # 如果没有找到子文件夹，返回None

if __name__ == "__main__":
    import json
    args = argparse.ArgumentParser()
    args.add_argument("--root_folder", type=str, default="/app/image-inpainting/scheduler/test_ootd_e2e",required=False, help="Root folder containing RPS directories (e.g., ootd)")
    args.add_argument("--output_csv", type=str, default="/app/image-inpainting/scheduler/end2end_results/result.csv", help="Output CSV file path")
    args = args.parse_args()

    all_results = []
    
    # Find all RPS directories
    # ootd_client_teacache
    # ootd_client_no_cb
    # ootd_clinet_flashps
    log_dirs = glob.glob(os.path.join(args.root_folder, "ootd_*"))
    
 
        # Extract RPS value from directory name
        
        # Process both cb and no_cb directories
    for cb_type in ['teacache','flashps','no_cb']:
        for rps in ['1.0','3.25','4.0']:
            pattern = os.path.join(args.root_folder,f'ootd_client_{cb_type}_{rps}*')
            print("pattern",pattern)
            matching_dir = glob.glob(pattern)
            print("matching_dir",matching_dir)
            if len(matching_dir) > 0:
                # Get the first subdirectory which contains the logs
                log_folder = matching_dir[0]
                # log_folder = cb_dir
                if log_folder:
                    result = parse_log_for_one_folder(log_folder)
                    
                    # Process each seqlen result
                    for seqlen_name, metrics in result.items():
                        row = {
                            'name': cb_type,
                            'rps': rps,
                            'avg_latency': metrics['avg_latency'],
                            'p99_latency': metrics['p99_latency'],
                            'p95_latency': metrics['p95_latency'],
                            'p99_inference_latency': metrics['p99_inference_latency'],
                            'p95_inference_latency': metrics['p95_inference_latency'],
                            'avg_inference_latency': metrics['avg_inference_latency']
                        }
                        all_results.append(row)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")




















