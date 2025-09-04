import argparse
import base64
import io
import asyncio
import aiohttp
from PIL import Image
import numpy as np
from typing import Dict, Any
import traceback
import time
from datetime import datetime
import yaml

import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.edit_config import EditConfig

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
            print(f"No valid mask path or file doesn't exist: {mask_path}, using default seq length 4096")
            return 4096
        
        # Load mask image and calculate proportion of black pixels
        mask_img = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_img)
        # Count non-zero (white) pixels in the mask
        white_pixels = np.count_nonzero(mask_array)
        total_pixels = mask_array.size
        
        # Calculate proportion of black pixels (mask area)
        black_proportion = 1.0 - (white_pixels / total_pixels)
        
        # Calculate sequence length based on proportion (max 4096)
        mask_seq_length = min(int(black_proportion * 4096), 4096)
        
        # Ensure minimum sequence length
        mask_seq_length = max(mask_seq_length, 256)
        
        print(f"Calculated mask_seq_length: {mask_seq_length} from mask image {mask_path}, " 
              f"black proportion: {black_proportion:.4f}")
        
        return mask_seq_length
        
    except Exception as e:
        print(f"Error calculating mask sequence length: {e}")
        traceback.print_exc()
        # Default value in case of error
        return 4096

async def run_inference_async(
    service_id: str, 
    inputs: Dict[str, Any], 
    server_url: str = "http://localhost:8005",
    session: aiohttp.ClientSession = None
) -> Dict[str, Any]:
    """Run inference on a diffusers pipeline asynchronously"""
    try:
        start_time = time.time()
        async with session.post(
            f"{server_url}/api/workflow/{service_id}/inference", 
            json={"inputs": inputs}
        ) as response:
            result = await response.json()
            end_time = time.time()
            latency = end_time - start_time
            return {"response_json": result, "latency": latency}
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise e

async def send_requests_async(
    service_id: str,
    inputs: Dict[str, Any],
    rps: int,
    interval: int,
    server_url: str = "http://localhost:8005"
):
    """Send requests asynchronously at specified RPS for given interval"""
    all_tasks = []
    number_of_requests = int(interval * rps)
    async with aiohttp.ClientSession() as session:
        for req_idx in range(number_of_requests):
            print(f"Sending request {req_idx+1} of {number_of_requests}, date {datetime.now()}")
            task = asyncio.create_task(
                run_inference_async(service_id, inputs, server_url, session)
            )
            all_tasks.append(task)
            await asyncio.sleep(1/rps)  # Space out the requests within a second
            
        # Wait for all requests to complete and collect responses
        responses = await asyncio.gather(*all_tasks)
    
    # Print latency statistics
    latencies = [r["latency"] for r in responses]
    print(f"Request Latencies: {latencies}")
    inference_latencies = [r["response_json"]["results"]["inference_latency"] for r in responses]
    print(f"Inference latencies: {inference_latencies}")
    avg_latency = sum(latencies) / len(latencies)
    print(f"\n========== Latency Statistics:")
    print(f"Average latency: {avg_latency:.2f} seconds")
    print(f"Min latency: {min(latencies):.2f} seconds")
    print(f"Max latency: {max(latencies):.2f} seconds")
    print(f"==========")
    
    return responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-id", type=str, default=None)
    parser.add_argument("--action", type=str, choices=["inference"], default="inference")
    parser.add_argument("--rps", type=float, default=1)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--edit_config_path",type=str,default=None)
    args = parser.parse_args()

    assert args.action == "inference"

    # Run the async function
    # edit_config_path = "/home/xjiangbp/image-inpainting/configs/sd2_configs/sd2_use_o_test_varlen.yml"
    # edit_config_path = "/home/xjiangbp/image-inpainting/configs/ootd_configs/ootd_config_standard.yml"
    if args.edit_config_path is None:

        args.edit_config_path = "/app/image-inpainting/configs/ootd_config_cb.yml"
    edit_config_path = args.edit_config_path
    with open(edit_config_path, 'r') as f:
        config = yaml.safe_load(f)
    edit_config = EditConfig(config)

    # Calculate mask sequence length here in the client
    # mask_seq_length = calculate_mask_seq_length(edit_config.mask_path)

    start_time = time.time()
    response_list = asyncio.run(send_requests_async(
        "OOTD_HD",
        {
            "edit_config_path": edit_config_path,
            "num_inference_steps": edit_config.num_inference_steps,
            "mask_seq_length": 400,  # Add mask_seq_length to inputs
        },
        args.rps,
        args.interval
    ))
    end_time = time.time()
    print(f"Time taken to send {args.rps * args.interval} requests: {end_time - start_time} seconds")

    assert len(response_list) == args.rps * args.interval
    for idx, response in enumerate(response_list):
        img_str_list = response["response_json"]["results"]["img_str_list"]
        assert isinstance(img_str_list, list), f"img_str_list is not a list: {type(img_str_list)}"
        for img_idx, img_str in enumerate(img_str_list):
            img_data = base64.b64decode(img_str)    
            img = Image.open(io.BytesIO(img_data))
            # Save image with request index and image index in filename
            img.save(f"output_request_{idx}_img_{img_idx}.png")
        if idx > 3:
            break
