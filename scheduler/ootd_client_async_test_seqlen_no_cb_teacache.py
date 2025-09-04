import argparse
import base64
import io
import asyncio
import aiohttp
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, List
import traceback
import time
from datetime import datetime
import yaml
import logging
import os.path

import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.edit_config import EditConfig
from generate_seqlen import load_all_seqlen_files_experiment
# Setup logging
def setup_logging(log_file, name=None):
    print(f"Logging to {log_file}")
    # Configure logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

# Setup main log directory
log_dir = "/app/image-inpainting/scheduler/test_ootd_e2e"
os.makedirs(log_dir, exist_ok=True)
log_dir = log_dir + f"/ootd_client_teacache_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"ootd_client.log")
logger = setup_logging(log_file)

# Dictionary to store loggers for different seqlens
seqlen_list_loggers = {}

def get_logger_for_seqlen_list(seqlen_list_id):
    """Get or create a logger for a specific sequence length list"""
    if seqlen_list_id not in seqlen_list_loggers:
        seqlen_list_log_dir = os.path.join(log_dir, f"seqlen_list_{seqlen_list_id}")
        os.makedirs(seqlen_list_log_dir, exist_ok=True)
        seqlen_list_log_file = os.path.join(seqlen_list_log_dir, f"seqlen_list_{seqlen_list_id}.log")
        seqlen_list_loggers[seqlen_list_id] = setup_logging(seqlen_list_log_file, name=f"seqlen_list_{seqlen_list_id}")
    return seqlen_list_loggers[seqlen_list_id]

def calculate_mask_seq_length(mask_path: str) -> int:
    """
    Calculate the sequence length based on the mask image proportion.
    
    Args:
        mask_path: Path to the mask image
        
    Returns:
        The calculated sequence length
    """
    try:
        if not mask_path or not os.path.exists(mask_path):
            logger.warning(f"No valid mask path or file doesn't exist: {mask_path}, using default seq length 4096")
            return 12288
        
        # Load mask image and calculate proportion of black pixels
        mask_img = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_img)
        # Count non-zero (white) pixels in the mask
        white_pixels = np.count_nonzero(mask_array)
        total_pixels = mask_array.size
        
        # Calculate proportion of black pixels (mask area)
        black_proportion = 1.0 - (white_pixels / total_pixels)
        
        # Calculate sequence length based on proportion (max 4096)
        mask_seq_length = min(int(black_proportion * 12288), 12288)
        
        # Ensure minimum sequence length

        
        logger.info(f"Calculated mask_seq_length: {mask_seq_length} from mask image {mask_path}, " 
              f"black proportion: {black_proportion:.4f}")
        
        return mask_seq_length
        
    except Exception as e:
        logger.error(f"Error calculating mask sequence length: {e}")
        traceback.print_exc()
        # Default value in case of error
        return 4096

async def run_inference_async(
    service_id: str, 
    inputs: Dict[str, Any], 
    server_url: str = "http://localhost:8005",
    session: aiohttp.ClientSession = None,
    custom_logger: Optional[logging.Logger] = None,
    sequence_id: int = 0  # 添加序列ID作为参数
) -> Dict[str, Any]:
    """Run inference on a diffusers pipeline asynchronously"""
    log = custom_logger or logger
    try:
        start_time = time.time()
        log.info(f"Sending request to {service_id} at {server_url}, sequence_id: {sequence_id}")
        
        # 添加序列ID到请求中，用于服务器端排序
        if "metadata" not in inputs:
            inputs["metadata"] = {}
        inputs["metadata"]["sequence_id"] = sequence_id
        
        async with session.post(
            f"{server_url}/api/workflow/{service_id}/inference", 
            json={"inputs": inputs}
        ) as response:
            result = await response.json()
            end_time = time.time()
            latency = end_time - start_time
            log.info(f"Request completed with latency: {latency:.2f}s, sequence_id: {sequence_id}")
            return {"response_json": result, "latency": latency}
    except Exception as e:
        log.error(f"Error in run_inference_async: {e}")
        traceback.print_exc()
        raise e

async def send_requests_async(
    service_id: str,
    inputs: Dict[str, Any],
    rps: int,
    interval: int,
    server_url: str = "http://localhost:8005",
    custom_logger: Optional[logging.Logger] = None,
    request_timestamps: List[float] = None,
):
    """Send requests asynchronously following a pre-generated Poisson trace"""
    log = custom_logger or logger
    
    number_of_requests = len(request_timestamps)
    log.info(f"Generated Poisson trace with {number_of_requests} requests over {interval} seconds")
    
    # 准备请求数据
    ordered_requests = []
    for req_idx in range(number_of_requests):
        ordered_requests.append((req_idx, inputs[req_idx]))
    
    # 发送请求
    all_tasks = []
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        for req_idx, (timestamp, req_inputs) in enumerate(zip(request_timestamps, ordered_requests)):
            # 等待直到预定的时间
            current_time = time.time() - start_time
            wait_time = max(0, timestamp - current_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            log.info(f"Sending request {req_idx+1} of {number_of_requests}, "
                    f"time since start: {time.time() - start_time:.2f}s, sequence_id: {req_idx}")
            
            # 创建异步任务发送请求
            task = asyncio.create_task(
                run_inference_async(service_id, req_inputs[1], server_url, session, 
                                 custom_logger=log, sequence_id=req_idx)
            )
            all_tasks.append(task)
            
        # 等待所有请求完成并收集响应
        log.info(f"Waiting for {len(all_tasks)} requests to complete...")
        responses = await asyncio.gather(*all_tasks)
    
    # Print latency statistics
    latencies = [r["latency"] for r in responses]
    log.info(f"Request Latencies: {latencies}")
    inference_latencies = [r["response_json"]["results"]["inference_latency"] for r in responses]
    log.info(f"Inference latencies: {inference_latencies}")
    avg_latency = sum(latencies) / len(latencies)
    
    stats_summary = "\n========== Latency Statistics:"
    stats_summary += f"\nAverage_latency: {avg_latency:.2f} seconds"
    stats_summary += f"\nMin_latency: {min(latencies):.2f} seconds"
    stats_summary += f"\nMax_latency: {max(latencies):.2f} seconds"
    stats_summary += f"\n=========="
    
    log.info(stats_summary)
    
    return responses

def set_edit_config(seqlen,
    log_folder,
    edit_config_path,
    schecule_type="flops",
    idx=0,):

 
    os.makedirs(log_folder, exist_ok=True)
    with open(edit_config_path, "r") as file:
        config = yaml.safe_load(file)
    
    config["generated_seqlen"] = seqlen

    # 为此配置创建唯一的输出目录
    
    # 临时YML文件名称
    temp_yaml_path = log_folder + f"/temp_config_{idx}.yml"
    with open(temp_yaml_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
    return temp_yaml_path

def run_inference_async_with_seqlen(
        seqlen_list,
    edit_config,
    args,
    idx,
    request_timestamps,
):
    """Run inference with a list of sequence lengths"""
    # Generate a unique ID for this seqlen_list based on the first few values
    list_id = f"{idx}"
    
    # Get logger specific to this seqlen_list
    seqlen_list_logger = get_logger_for_seqlen_list(list_id)
    seqlen_list_logger.info(f"Starting inference with seqlen_list of length {len(seqlen_list)}")
    
    # 保存当前运行的seqlen_list到文件，方便后续分析
    seqlen_list_file = os.path.join(log_dir, f"seqlen_list_{list_id}", "seqlen_list.json")
    os.makedirs(os.path.dirname(seqlen_list_file), exist_ok=True)
    with open(seqlen_list_file, 'w') as f:
        import json
        json.dump(seqlen_list, f)
    
    # generate edit_config_path
    inputs = {
        "edit_config_path": edit_config_path,
        "num_inference_steps": edit_config.num_inference_steps,
        "mask_seq_length": 4096,  # Default value, will be overridden
    }
    inputs_list = []
    interval = int(len(seqlen_list) // args.rps)
    
    # Create output directory for this seqlen_list
    output_dir = os.path.join(log_dir, f"seqlen_list_{list_id}", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(len(request_timestamps)):
        new_inputs = inputs.copy()
        new_inputs["edit_config_path"] = set_edit_config(
            seqlen_list[i],
            os.path.join(log_dir, f"seqlen_list_{list_id}"),
            edit_config_path,
            idx=i,
        )
        new_inputs['mask_seq_length'] = seqlen_list[i]
        inputs_list.append(new_inputs)
    
    seqlen_list_start_time = time.time()
    
    response_list = asyncio.run(send_requests_async(
        "FluxInpaintPipeline",
        inputs_list,
        args.rps,
        interval,
        custom_logger=seqlen_list_logger,
        request_timestamps=request_timestamps
    ))
    
    seqlen_list_end_time = time.time()
    seqlen_list_total_time = seqlen_list_end_time - seqlen_list_start_time
    seqlen_list_logger.info(f"Time taken to send {len(request_timestamps)} requests with seqlen_list: {seqlen_list_total_time:.2f} seconds")

    assert len(response_list) == len(request_timestamps)
    
    # Save output images to sequence-specific directory
    output_dir = os.path.join(log_dir, f"seqlen_list_{list_id}", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, response in enumerate(response_list):
        img_str_list = response["response_json"]["results"]["img_str_list"]
        assert isinstance(img_str_list, list), f"img_str_list is not a list: {type(img_str_list)}"
        for img_idx, img_str in enumerate(img_str_list):
            img_data = base64.b64decode(img_str)    
            img = Image.open(io.BytesIO(img_data))
            output_path = os.path.join(output_dir, f"output_request_{idx}_img_{img_idx}.png")
            img.save(output_path)
            seqlen_list_logger.info(f"Saved image to {output_path}")
        if idx > 3:
            seqlen_list_logger.info(f"Stopped after processing {idx+1} responses")
            break
            
    seqlen_list_logger.info(f"Flux client completed successfully for seqlen_list {list_id}")
    return response_list
    
if __name__ == "__main__":
    logger.info("Starting flux client")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-id", type=str, default=None)
    parser.add_argument("--action", type=str, choices=["inference"], default="inference")
    parser.add_argument("--rps", type=float, default=1)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--trace_path", type=str, default=None)
    parser.add_argument("--edit_config_path", type=str, default=None)
    args = parser.parse_args()
    request_timestamps = np.load(args.trace_path)
    assert args.action == "inference"
    logger.info(f"Parsed arguments: service-id={args.service_id}, action={args.action}, rps={args.rps}, interval={args.interval}")

    # Run the async function
    if args.edit_config_path is None:
        args.edit_config_path = "/app/image-inpainting/configs/ootd_teacache_4.yml"
    edit_config_path = args.edit_config_path
    with open(edit_config_path, 'r') as f:
        config = yaml.safe_load(f)
    edit_config = EditConfig(config)
    logger.info(f"Loaded edit config from {edit_config_path}")

    start_time = time.time()
    mask_ratio_trace_path = "/app/image-edit-data/mask_ratio_trace.npy"
    mask_ratio_list = np.load(mask_ratio_trace_path)
    seqlen_lists = []
    standard_seqlen= 12288
    seqlen_list = []
    mask_ratio_list = mask_ratio_list[:len(request_timestamps)]
    for mask_ratio in mask_ratio_list:
        seqlen_list.append(int(standard_seqlen * mask_ratio))
    seqlen_lists.append(seqlen_list)
    # 确保每次加载的seqlen_lists顺序一致
    seqlen_lists_stable = []
    for seqlen_list in seqlen_lists:
        # 创建一个新的列表副本而不是引用
        seqlen_lists_stable.append(list(seqlen_list))
    
    # 记录加载的seqlen_lists到日志
    logger.info(f"Loaded {len(seqlen_lists_stable)} seqlen lists")
    for i, seqlen_list in enumerate(seqlen_lists_stable):
        if i < 3:  # 只记录前几个示例
            logger.info(f"Seqlen list #{i+1} sample: {seqlen_list[:5]}...")
    
    seqlen_list_results = {}
    test_time = 1
    
    seqlen_lists_stable = seqlen_lists_stable[:len(request_timestamps)]
    for i, seqlen_list in enumerate(seqlen_lists_stable):
        seqlen_list = seqlen_list[:len(request_timestamps)]
        for j in range(test_time):
            logger.info(f"Running inference with seqlen_list #{i+1} of {len(seqlen_lists_stable)}")
            results = run_inference_async_with_seqlen(seqlen_list, edit_config, args, i, request_timestamps)
            seqlen_list_results[i] = results
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time taken for all seqlen_list tests: {total_time:.2f} seconds")
    logger.info("All flux client tests completed successfully")
        
