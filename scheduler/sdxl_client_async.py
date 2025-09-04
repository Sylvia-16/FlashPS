import argparse
import base64
import io
import asyncio
import aiohttp
from PIL import Image
from typing import Dict, Any
import traceback
import time
from datetime import datetime
import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.edit_config import EditConfig
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
    parser.add_argument("--interval", type=int, default=1)
    args = parser.parse_args()

    assert args.action == "inference"

    # Run the async function
    start_time = time.time()
    response_list = asyncio.run(send_requests_async(
        "StableDiffusionXLPipeline",
        {
            "prompt": "An astronaut riding a green horse",
            "num_inference_steps": 50,
            "seed": 0,
            "height": 1024,
            "width": 1024,
            "guidance_scale": 0.0,
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