import argparse
import asyncio
import base64
import io
import json
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional
import threading

# Add the project root directory to the Python path
# Handle package import
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.multiprocessing as mp
import uvicorn
import yaml
import zmq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import DistributedConfig, NodeConfig, CacheConfig
from scheduler.schedule_methods import (
    get_seq_length_from_request,
    create_scheduler,
    Scheduler,
    cal_flops,
    FlopsBalanceScheduler
)

import uuid
import traceback

from datetime import datetime
# disable torch warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# from diffusers import StableDiffusionXLPipeline
from sdxl_pipeline_continous_batching import StableDiffusionXLPipeline
from pipeline_flux_inpaint_continuous_batching import FluxInpaintPipeline
from diffusers import StableDiffusionInpaintPipeline
sys.path.append('/app')
from ootd.ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.ootd.inference_ootd_dc import OOTDiffusionDC
class DistributedWorker:
    def __init__(
        self,
        local_rank: int,
        node_rank: int,
        node_config: NodeConfig,
        dist_config: DistributedConfig,
        cache_config: CacheConfig,
        scheduling_baseline: str = "basic",
        worker_max_batch_size: int = 2,
        pipeline_name: str = "SDXL",
    ):
        self.local_rank = local_rank
        self.node_rank = node_rank
        self.node_config = node_config
        self.dist_config = dist_config
        self.global_rank = self._calculate_global_rank()
        self.scheduling_baseline = scheduling_baseline
        self.pipeline_name = pipeline_name
        self.cache_config = cache_config
        # ZMQ setup
        self.context = zmq.Context()
        self.task_socket = self.context.socket(zmq.PULL)
        self.result_socket = self.context.socket(zmq.PUSH)

        # Worker identity
        self.worker_id = f"worker_{self.node_rank}_{self.local_rank}"

        # Setup logging
        self._setup_logging()

        # Ports for this worker
        self.task_port = self._get_task_port()
        self.result_port = self._get_result_port()

        assert torch.cuda.is_available()
        self.device = f"cuda:{self.local_rank}"
        self.max_gpu_memory_fraction = 0.95

        self.models = {}

        self.result_locations = {}

        self.running = True

        # Priority queue for requests, sorted by arrival timestamp
        self.request_queue = asyncio.PriorityQueue()
        
        # Event loop for async operations
        self.loop = None
        
        # Task for processing requests
        self.processing_task = None

        self.max_batch_size = worker_max_batch_size
        ### Suyi: initialize pipeline
        if self.pipeline_name == "SDXL":
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                torch_dtype=torch.float16, 
                use_safetensors=True, 
                variant="fp16"
            ).to(self.device)
        elif self.pipeline_name == "Flux_inpaint":
            print("self.device", self.device)
            self.pipeline = FluxInpaintPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell", 
                torch_dtype=torch.bfloat16,
                cache_dir="/project/infattllm/huggingface/hub/"
            ).to(self.device)
        elif self.pipeline_name == "SD2":
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16,
            ).to(self.device)
        elif self.pipeline_name == "OOTD_HD":
            self.pipeline = OOTDiffusionHD(gpu_id=self.local_rank)
        elif self.pipeline_name == "OOTD_DC":
            self.pipeline_name = OOTDiffusionDC(gpu_id=self.local_rank)
        ### create timesteps placeholder
        if self.pipeline_name == "SDXL":
            self.logger.info(f"SDXL pipeline, max_batch_size: {self.max_batch_size}")
            ### Suyi: create placeholders for timesteps for avoid overhead of creating new tensors
            self.timesteps_placeholder = [
                torch.tensor( [0.0]*timestep_len, dtype=torch.float32, device=self.device )
                for timestep_len in range( 2*self.max_batch_size + 1 )
            ]
        elif self.pipeline_name == "Flux_inpaint":
            self.logger.info(f"Flux_inpaint pipeline, max_batch_size: {self.max_batch_size}")
            self.timesteps_placeholder = [
                torch.tensor( [0.0]*timestep_len, dtype=torch.float32, device=self.device )
                for timestep_len in range( self.max_batch_size + 1 )
            ]
        elif self.pipeline_name == "SD2":
            self.logger.info(f"SD2 pipeline, max_batch_size: {self.max_batch_size}")
            self.timesteps_placeholder = [
                torch.tensor( [0]*timestep_len, dtype=torch.int32, device=self.device )
                for timestep_len in range( 2*self.max_batch_size + 1 )
            ]
        elif self.pipeline_name == "OOTD_HD" or self.pipeline_name == "OOTD_DC":
            self.logger.info(f"OOTD pipeline, max_batch_size: {self.max_batch_size}")
            self.timesteps_placeholder = [
                torch.tensor( [0]*timestep_len, dtype=torch.int32, device=self.device )
                for timestep_len in range( 2*self.max_batch_size + 1 )
            ]
        if self.scheduling_baseline != "no_cb":
            if self.pipeline_name == "SD2" or self.pipeline_name == "OOTD_HD" or self.pipeline_name == "OOTD_DC":
                self.load_cache_o(self.cache_config)
                # pass
            else:
                
                self.load_cache_kv(self.cache_config)
            pass
    def load_cache_kv(self, cache_config):

        def load_cache_for_one_folder(cache_config, cached_kv_folder):
            cached_kv_files = [
                item for item in os.listdir(cached_kv_folder) if item.endswith(".pt")
            ]
            self.cached_kv = {}

            print(f"Loading cached kv from {cache_config.cached_kv_folder}")
            for file in cached_kv_files:
                # tmp_key is k_{block_name}_{denoising_step}, e.g., k_sd3_0_1. The
                # filename is k_sd3_0_1.pt
                tmp_key = file.split(".")[0]
                # if starts with k, it is key; if starts with v, it is value
                # if edit_config.async_copy:
                step = tmp_key.split("_")[-1]
                # 检查特定键tmp_key是否存在，如果不存在或为None则初始化为空列表
                if (
                    tmp_key not in self.cached_kv
                    or self.cached_kv[tmp_key] is None
                ):
                    self.cached_kv[tmp_key] = []
                if step == "0" or step == "1":
                    self.cached_kv[tmp_key].append(
                        torch.load(
                            os.path.join(cache_config.cached_kv_folder, file),
                            map_location=self.device,
                        )
                    )
                    self.basic_cached_kv_shape = self.cached_kv[tmp_key][0].shape

                else:
                    self.cached_kv[tmp_key].append(
                        torch.load(
                            os.path.join(cache_config.cached_kv_folder, file),
                            map_location="cpu",
                        )
                        .contiguous()
                        .pin_memory()
                    )

        def _load_cache_kv(cache_config):
           
            # batch size 1
            if isinstance(cache_config.cached_kv_folder, list):
                for cached_kv_folder in cache_config.cached_kv_folder:
                    load_cache_for_one_folder(cache_config, cached_kv_folder)
            else:
                load_cache_for_one_folder(cache_config, cache_config.cached_kv_folder)
            print(f"Loading cached latents from {cache_config.cached_latents_folder}")
            cached_latents_files = [
                item
                for item in os.listdir(cache_config.cached_latents_folder)
                if item.endswith(".pt")
            ]
            self.cached_latents = {}
            for file in cached_latents_files:
                tmp_key = file.split(".")[0]
                self.cached_latents[tmp_key] = torch.load(
                    os.path.join(cache_config.cached_latents_folder, file),
                    map_location=self.device,
                )
            print("Cached latents loaded")

        _load_cache_kv(cache_config)
    
    def load_cache_o(self, cache_config):
        def load_cache_from_one_folder(cached_o_folder, cached_o_files):
            self.cached_o = {}
            for file in cached_o_files:
                tmp_key = file.split(".")[0]
                if tmp_key not in self.cached_o or self.cached_o[tmp_key] is None:
                    self.cached_o[tmp_key] = []
                # if async_copy, copy to cpu first
                self.cached_o[tmp_key].append(
                    torch.load(
                        os.path.join(cached_o_folder, file),
                        map_location=torch.device("cpu"),
                    ).contiguous().pin_memory()
                    # torch.load(
                    #     os.path.join(cached_o_folder, file),
                    #     map_location=self.device
                    # )
                )
            


        def _load_cache_o(cache_config):

            if isinstance(cache_config.cached_o_folder, list):
                for folder in cache_config.cached_o_folder:
                    cached_o_files = [
                        item for item in os.listdir(folder) if item.endswith(".pt")
                    ]
                    load_cache_from_one_folder(folder, cached_o_files)
            else:
                cached_o_files = [
                    item
                    for item in os.listdir(cache_config.cached_o_folder)
                    if item.endswith(".pt")
                ]
                load_cache_from_one_folder(
                    cache_config.cached_o_folder, cached_o_files
                )
        _load_cache_o(cache_config)
    def _setup_logging(self):
        """Setup logging for this worker"""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Create a unique log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/{timestamp}_{self.worker_id}.log"

        # Setup the logger
        self.logger = logging.getLogger(self.worker_id)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler that uses sys.stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Configure the root logger to use the same handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        self.logger.info(f"Initialized logging for {self.worker_id}")

    def _calculate_global_rank(self) -> int:
        """Calculate global rank based on node rank and local rank"""
        global_rank = 0
        for node in self.dist_config.nodes:
            if node.rank == self.node_rank:
                return global_rank + self.local_rank
            global_rank += node.gpu_count
        raise ValueError(f"Invalid node rank: {self.node_rank}")

    def _get_task_port(self) -> int:
        """Calculate unique task port for this worker"""
        base_task_port = self.dist_config.port + 1
        return base_task_port + self.global_rank * 2

    def _get_result_port(self) -> int:
        """Calculate unique result port for this worker"""
        return self._get_task_port() + 1

    def setup(self):
        # Initialize process group
        # self._setup_network()

        # Setup ZMQ connection
        self._setup_zmq_connection()

        ### Set GPU device for this worker
        # self.logger.info(f"Setting GPU device for worker {self.worker_id}; node_rank: {self.node_rank}, local_rank: {self.local_rank}, global_rank: {self.global_rank}")
        # self.logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        # self.logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")
        # self.logger.info(f"Current GPU device: {torch.cuda.current_device()}")

        torch.cuda.set_device(self.local_rank)

        # self.logger.info(f"After set_device, current GPU device: {torch.cuda.current_device()}")

    def _setup_zmq_connection(self):
        print(
            f"Worker {self.worker_id} => master_addr: {self.dist_config.master_addr}, Task Port: {self.task_port}, Result Port: {self.result_port}"
        )

        self.task_socket.connect(
            f"tcp://{self.dist_config.master_addr}:{self.task_port}"
        )
        self.result_socket.connect(
            f"tcp://{self.dist_config.master_addr}:{self.result_port}"
        )

        print(f"[ZMQ] Worker {self.worker_id} connected on node {self.node_rank}")

    def run(self):
        """Main worker loop"""
        self.setup()
        self.logger.info(
            f"[DistributedWorker] Initialized => Node: {self.node_rank}, "
            f"Local Rank: {self.local_rank}, Global Rank: {self.global_rank}"
        )

        # Create and set the event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Start the processing task
        self.processing_task = self.loop.create_task(self._process_queue())
        
        # Run the event loop in a separate thread
        def run_event_loop():
            self.loop.run_forever()
            
        loop_thread = threading.Thread(target=run_event_loop)
        loop_thread.daemon = True
        loop_thread.start()

        poller = zmq.Poller()
        poller.register(self.task_socket, zmq.POLLIN)

        while True:
            try:
                # Use poller with timeout instead of blocking recv
                socks = dict(poller.poll(timeout=1000))  # 1 second timeout
                if self.task_socket not in socks:
                    continue

                message = self.task_socket.recv_json()
                # self.logger.info(f"Received message keys: {message.keys()}") # dict_keys(['pipeline_name', 'inputs', 'req_id'])

                if message.get("type") == "stop":  # Poison pill
                    self.logger.info("Received stop signal, initiating shutdown...")
                    # Cancel the processing task
                    if self.processing_task:
                        self.loop.call_soon_threadsafe(self.processing_task.cancel)
                    break
                elif message.get("type") == "ping":  # Health check
                    self.result_socket.send_json({"type": "pong"})
                    continue
                elif message.get("type") == "clear_cache":  # Clear intermediate results
                    self.result_locations.clear()
                    self.result_socket.send_json({"type": "cache_cleared"})
                    continue
                elif message.get("type") == "batch":  # 处理批次消息
                    # Add the batch request to the queue with timestamp as priority
                    timestamp = time.time()
                    message["status"] = "new"
                    
                    # 使用call_soon_threadsafe将批次添加到队列
                    self.loop.call_soon_threadsafe(
                        lambda: self.request_queue.put_nowait((timestamp, message))
                    )
                    
                    batch_req_ids = [req["req_id"] for req in message.get("batch_requests", [])]
                    self.logger.info(f"Added batch with {len(batch_req_ids)} requests to queue: {batch_req_ids}")
                    continue
                # Add the request to the queue with timestamp as priority
                timestamp = time.time()
                # Use call_soon_threadsafe to safely add to the queue from a non-async context
                message["status"] = "new"
                
                # 检查消息中的inputs是否包含metadata和sequence_id
                priority = timestamp  # 默认使用时间戳
                if "inputs" in message and isinstance(message["inputs"], dict):
                    if "metadata" in message["inputs"] and "sequence_id" in message["inputs"]["metadata"]:
                        sequence_id = message["inputs"]["metadata"]["sequence_id"]
                        # 移除metadata以避免传递给模型
                        metadata = message["inputs"].pop("metadata")
                        priority = float(sequence_id)  # 使用序列ID作为优先级
                        self.logger.info(f"Using sequence_id {sequence_id} as priority for req_id {message['req_id']}")
                
                self.loop.call_soon_threadsafe(
                    lambda: self.request_queue.put_nowait((priority, message))
                )
                self.logger.info(f"Added request {message['req_id']} to queue with priority {priority}, datetime: {datetime.fromtimestamp(timestamp)}")
                
            except zmq.ZMQError as e:
                if self.running:  # Only log if not shutting down
                    self.logger.error(f"ZMQ Error in worker {self.worker_id}: {str(e)}")
            except KeyboardInterrupt:
                self.logger.info("Received KeyboardInterrupt, initiating shutdown...")
                # Cancel the processing task
                if self.processing_task:
                    self.loop.call_soon_threadsafe(self.processing_task.cancel)
                break
            except Exception as e:
                self.logger.error(f"Error in worker {self.worker_id}: {str(e)}")
                continue

        # Stop the event loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for the loop thread to finish
        if loop_thread.is_alive():
            loop_thread.join(timeout=5)
            
        self.cleanup()
        
    async def _process_queue(self):
        """Process requests from the queue using continuous batching"""
        self.logger.info(f"Starting request processing task for worker {self.worker_id}")
        
        # Queue for storing preprocessed requests
        active_batch = []
        print("pipeline_name", self.pipeline_name)
        try:
            while self.running:
                try:
                    one_step_process_queue_start = time.time()
                    # If active_batch is empty, wait for at least one request
                    if not active_batch:
                        print(f"{datetime.now()} len(active_batch) == {len(active_batch)}, add new request to active_batch")
                        timestamp, message = await self.request_queue.get()
                        
                        # 标记我们是否处理了批处理消息
                        is_batch_message = False
                        
                        # 检查是否是批处理消息
                        if message.get("type") == "batch" and "batch_requests" in message:
                            is_batch_message = True
                            print(f"Processing batch message with {len(message['batch_requests'])} requests")
                            
                            # 处理批处理中的每个请求
                            for request_info in message["batch_requests"]:
                                inputs = request_info["inputs"]
                                req_id = request_info["req_id"]
                                
                                # 创建单个请求消息
                                single_message = {
                                    "pipeline_name": message["pipeline_name"],
                                    "inputs": inputs,
                                    "req_id": req_id
                                }
                                
                                # 预处理请求
                                if self.pipeline_name == "SDXL":
                                    preprocessed = await self._preprocess_request_SDXL(single_message)
                                elif self.pipeline_name == "Flux_inpaint":
                                    preprocessed = await self._preprocess_request_Flux_inpaint(single_message)
                                elif self.pipeline_name == "SD2":
                                    preprocessed = await self._preprocess_request_SD2(single_message)
                                elif self.pipeline_name == "OOTD_HD" or self.pipeline_name == "OOTD_DC":
                                    preprocessed = await self._preprocess_request_OOTD(single_message)
                                else:
                                    raise ValueError(f"Pipeline name {self.pipeline_name} not supported")
                                
                                preprocessed["start_time"] = time.time()
                                active_batch.append(preprocessed)
                            
                            # 批处理消息在此处标记为已完成，仅标记一次
                            self.request_queue.task_done()
                        else:
                            # 正常处理单个请求
                            print("process message", message.get('req_id', 'unknown'))
                            
                            # Preprocess the request
                            if self.pipeline_name == "SDXL":
                                preprocessed = await self._preprocess_request_SDXL(message)
                            elif self.pipeline_name == "Flux_inpaint":
                                preprocessed = await self._preprocess_request_Flux_inpaint(message)
                            elif self.pipeline_name == "SD2":
                                preprocessed = await self._preprocess_request_SD2(message)
                            elif self.pipeline_name == "OOTD_HD" or self.pipeline_name == "OOTD_DC":
                                preprocessed = await self._preprocess_request_OOTD(message)
                            else:
                                raise ValueError(f"Pipeline name {self.pipeline_name} not supported")
                            
                            preprocessed["start_time"] = time.time()
                            active_batch.append(preprocessed)
                            
                            # 非批处理消息，标记为已完成
                            self.request_queue.task_done()
                            
                            # If this is a no_cb batch, wait for all requests in the batch before processing
                            if self.scheduling_baseline == "no_cb":
                                batch_size = message.get("batch_size", 1)
                                # Collect all requests in the batch
                                while len(active_batch) < batch_size and not self.request_queue.empty():
                                    # Get next request
                                    timestamp, message = self.request_queue.get_nowait()
                                    
                                    # 标记每个获取的请求为已完成
                                    self.request_queue.task_done()
                                    
                                    # Preprocess the request
                                    if self.pipeline_name == "SDXL":
                                        preprocessed = await self._preprocess_request_SDXL(message)
                                    elif self.pipeline_name == "Flux_inpaint":
                                        preprocessed = await self._preprocess_request_Flux_inpaint(message)
                                    elif self.pipeline_name == "SD2":
                                        preprocessed = await self._preprocess_request_SD2(message)
                                    elif self.pipeline_name == "OOTD_HD" or self.pipeline_name == "OOTD_DC":
                                        preprocessed = await self._preprocess_request_OOTD(message)
                                    else:
                                        raise ValueError(f"Pipeline name {self.pipeline_name} not supported")
                                    preprocessed["start_time"] = time.time()
                                    active_batch.append(preprocessed)
                                
                                self.logger.info(f"Collected {len(active_batch)} requests for no_cb batch processing")
                    
                    # Try to fill the batch up to max_batch_size
                    while self.scheduling_baseline != "no_cb" and len(active_batch) < self.max_batch_size and not self.request_queue.empty():
                        # get_nowait() is synchronous, so don't use await
                        timestamp, message = self.request_queue.get_nowait()
                        
                        # 标记从队列中获取的请求为已完成
                        self.request_queue.task_done()
                        
                        # preprocess_request is still async, so keep await
                        if self.pipeline_name == "SDXL":
                            preprocessed = await self._preprocess_request_SDXL(message)
                        elif self.pipeline_name == "Flux_inpaint":
                            preprocessed = await self._preprocess_request_Flux_inpaint(message)
                        elif self.pipeline_name == "SD2":
                            preprocessed = await self._preprocess_request_SD2(message)
                        elif self.pipeline_name == "OOTD_HD" or self.pipeline_name == "OOTD_DC":
                            preprocessed = await self._preprocess_request_OOTD(message)
                        else:
                            raise ValueError(f"Pipeline name {self.pipeline_name} not supported")
                        preprocessed["start_time"] = time.time()
                        active_batch.append(preprocessed)
                        print("add new request to active_batch")

                    # Process the batch
                    if self.pipeline_name == "SDXL":
                        await self._process_batch_SDXL(active_batch)
                    elif self.pipeline_name == "Flux_inpaint":
                        await self._process_batch_Flux_inpaint(active_batch)
                    elif self.pipeline_name == "SD2":
                        await self._process_batch_SD2(active_batch)
                    elif self.pipeline_name == "OOTD_HD" or self.pipeline_name == "OOTD_DC":
                        await self._process_batch_OOTD(active_batch)
                    else:
                        raise ValueError(f"Pipeline name {self.pipeline_name} not supported")
                    
                    # Remove completed requests from the batch
                    completed_indices = []
                    for i, req in enumerate(active_batch):
                        print("num_inference_steps",i, req['num_inference_steps'])

                        if req["scheduler_steps"] >= req["num_inference_steps"]:
                            # Process and send response for completed request
                            print("finish_req",req['req_id'])
                            await self._send_completion_response(req)
                            completed_indices.append(i)
                            # 移除这里的task_done调用，因为我们已经在获取请求时标记过了
                            # self.request_queue.task_done()
                    
                    if len(completed_indices) > 0:
                        print(f"completed_indices: {completed_indices}")
                    # Remove completed requests from active_batch (in reverse order)
                    for i in reversed(completed_indices):
                        active_batch.pop(i)
                    # await self._send_steps_update(active_batch)
                    one_step_process_queue_end = time.time()
                    one_step_process_queue_time = one_step_process_queue_end - one_step_process_queue_start
                    # print(f"One step process queue time: {one_step_process_queue_time}")

                except asyncio.CancelledError:
                    # exit(123)
                    self.logger.info(f"Request processing task for worker {self.worker_id} cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Error in processing task: {str(e)}")
                    traceback.print_exc()
                    await asyncio.sleep(1)  # Avoid tight loop on error
                    
        except asyncio.CancelledError:
            self.logger.info(f"Request processing task for worker {self.worker_id} cancelled")
        finally:
            self.logger.info(f"Request processing task for worker {self.worker_id} stopped")

    async def _preprocess_request_SDXL(self, message):
        """Preprocess a request using prepare_for_inference"""
        preprocess_start = time.time()
        pipeline_name = message["pipeline_name"]
        inputs = message["inputs"]
        
        # Run preprocessing in a thread pool
        loop = asyncio.get_event_loop()
        preprocessed = await loop.run_in_executor(
            None,
            lambda: self.pipeline.prepare_for_inference(
                prompt=inputs["prompt"],
                num_inference_steps=inputs["num_inference_steps"],
                guidance_scale=inputs["guidance_scale"],
                generator=torch.manual_seed(inputs["seed"])
            )
        )
        
        # Add additional information needed for tracking
        preprocessed.update({
            "req_id": message["req_id"],
            "scheduler": deepcopy(self.pipeline.scheduler),
            "denoising_progress": 0,
            "scheduler_steps": 0
        })
        preprocess_time = time.time() - preprocess_start
        # print(f"Preprocess time: {preprocess_time}")
        return preprocessed

    async def _preprocess_request_Flux_inpaint(self, message):
        """Preprocess a request using prepare_for_inference"""
        preprocess_start = time.time()
        pipeline_name = message["pipeline_name"]
        inputs = message["inputs"]
        print("inputs",inputs)
        print("inputs['image_path']",inputs['image_path'])
        # Run preprocessing in a thread pool
        loop = asyncio.get_event_loop()
        preprocessed = await loop.run_in_executor(
            None,
            lambda: self.pipeline.prepare_for_inference(
                prompt=inputs["prompt"],
                image_path=inputs["image_path"],
                mask_image_path=inputs["mask_image_path"],
                strength=inputs["strength"],
                generator=torch.manual_seed(inputs["seed"]),
                edit_config_path=inputs["edit_config_path"],
            )
        )
        
        # Add additional information needed for tracking
        preprocessed.update({
            "req_id": message["req_id"],
            "scheduler": deepcopy(self.pipeline.scheduler),
            "denoising_progress": 0,
            "scheduler_steps": 0
        })
        preprocess_time = time.time() - preprocess_start
        print(f"Worker {self.worker_id}: Preprocess time: {preprocess_time}")
        return preprocessed
    
    async def _process_batch_SDXL(self, batch):
        """Process a batch of requests using denoising_step"""
        if not batch:
            return
        
        running_batch_size = len(batch)
        # print(f"=====Processing batch of size {running_batch_size}=====")
        
        # Prepare inputs for denoising_step
        latents_list = [item["latents"] for item in batch]
        prompt_embeds_list = [item["prompt_embeds"] for item in batch]
        add_text_embeds_list = [item["add_text_embeds"] for item in batch]
        add_time_ids_list = [item["add_time_ids"] for item in batch]
        timestep_list = [item["timesteps"][item["denoising_progress"]] for item in batch]
        scheduler_list = [item["scheduler"] for item in batch]
        
        if self.pipeline.do_classifier_free_guidance:
            timestep_tensor = self.timesteps_placeholder[running_batch_size*2]
            step_size = 2
        else:
            timestep_tensor = self.timesteps_placeholder[running_batch_size]
            step_size = 1

        batch_idx = 0
        while batch_idx < running_batch_size * step_size:
            timestep_tensor[batch_idx:batch_idx+step_size] = timestep_list[batch_idx//step_size]
            batch_idx += step_size
        # print(f"cfg: {self.pipeline.do_classifier_free_guidance}, timestep_tensor: {timestep_tensor}")

        
        # Run denoising step in a thread pool
        denoising_start_time = time.time()
        loop = asyncio.get_event_loop()
        denoising_output = await loop.run_in_executor(
            None,
            lambda: self.pipeline.denoising_step_op(
                latents=latents_list,
                prompt_embeds=prompt_embeds_list,
                add_text_embeds=add_text_embeds_list,
                add_time_ids=add_time_ids_list,
                # timestep_list=timestep_tensor,
                timestep=timestep_tensor,
                scheduler_list=scheduler_list,
            )
        )
        denoising_time = time.time() - denoising_start_time
        # print(f"One step denoising time: {denoising_time}")
        print(f"denoising_output['scheduler_steps']: {denoising_output['scheduler_steps']}")
        print(f"{datetime.now()} req_ids: {[item['req_id'] for item in batch]}")

        # Update batch items with new state
        for i in range(len(batch)):
            batch[i]["latents"] = denoising_output["latents"][i]
            batch[i]["scheduler_steps"] = denoising_output["scheduler_steps"][i]
            batch[i]["denoising_progress"] += 1
            
        # Send current steps info to coordinator
       

    async def _process_batch_Flux_inpaint(self, batch):
        """Process a batch of requests using denoising_step"""
        if not batch:
            return
        
        running_batch_size = len(batch)
        # print(f"=====Processing batch of size {running_batch_size}=====")
        
        # Prepare inputs for denoising_step
        cur_step_list = [ item["denoising_progress"] for item in batch ]
        timesteps_list = [ item["timesteps"] for item in batch ]
        scheduler_list = [ item["scheduler"] for item in batch ]
        latents_list = [ item["latents"] for item in batch ]
        noise_list = [ item["noise"] for item in batch ]
        image_latents_list = [ item["image_latents"] for item in batch ]
        mask_list = [ item["mask"] for item in batch ]
        mask_list = torch.cat(mask_list, dim=0).cuda(self.device)
        prompt_embeds_list = [ item["prompt_embeds"] for item in batch ]
        pooled_prompt_embeds_list = [ item["pooled_prompt_embeds"] for item in batch ]
        batch[0]["edit_config"].device_num = self.local_rank
        batch[0]["edit_config"].max_batch_size = self.max_batch_size

        if hasattr(self, "cached_kv"):
            cached_kv = self.cached_kv
        else:
            cached_kv = None
        if hasattr(self, "cached_latents"):
            cached_latents = self.cached_latents
        else:
            cached_latents = None
        # # set batch_cache_map
        
        # # batch[0]["edit_config"].cached_kv = self.cached_kv
        # # batch[0]["edit_config"].cached_latents = self.cached_latents
        timestep_tensor = self.timesteps_placeholder[running_batch_size]
        step_size = 1

        batch_idx = 0
        while batch_idx < running_batch_size * step_size:
            timestep_tensor[batch_idx:batch_idx+step_size] = batch[batch_idx]["timesteps"][ batch[batch_idx]["denoising_progress"] ]
            batch_idx += step_size
        print(f"timestep_tensor: {timestep_tensor}")


        # Run denoising step in a thread pool
        denoising_start_time = time.time()
        loop = asyncio.get_event_loop()
        denoising_output = await loop.run_in_executor(
            None,
            lambda: self.pipeline.denoising_step_op(
                cur_step_list,
                timestep_tensor,
                timesteps_list,
                scheduler_list,
                latents_list,
                noise_list,
                image_latents_list,
                batch[0]["latent_image_ids"],
                mask_list,
                prompt_embeds_list,
                pooled_prompt_embeds_list,
                batch[0]["text_ids"],
                batch[0]["do_true_cfg"],
                batch[0]["true_cfg_scale"],
                batch[0]["edit_config"],
                cached_kv,
                cached_latents,
            )
        )
        denoising_time = time.time() - denoising_start_time
        print(f"One step denoising time: {denoising_time}")
        print(f"denoising_output['scheduler_steps']: {denoising_output['scheduler_steps']}")
        print(f"{datetime.now()} req_ids: {[item['req_id'] for item in batch]}")

        # Update batch items with new state
        for i in range(len(batch)):
            batch[i]["latents"] = denoising_output["latents"][i]
            batch[i]["scheduler_steps"] = denoising_output["scheduler_steps"][i]
            batch[i]["denoising_progress"] += 1
            
        # Send current steps info to coordinator


    async def _process_batch_SD2(self, batch):
        """Process a batch of requests using denoising_step"""
        if not batch:
            return
        
        running_batch_size = len(batch)
        
        # Prepare inputs for denoising_step
        cur_step_list = [item["denoising_progress"] for item in batch]
        scheduler_list = [item["scheduler"] for item in batch]
        latents_list = [item["latents"] for item in batch]
        mask_list = [item["mask"] for item in batch]
        mask_list = torch.cat(mask_list, dim=0).cuda(self.device)
        prompt_embeds_list = [item["prompt_embeds"] for item in batch]
        masked_image_latents_list = [item["masked_image_latents"] for item in batch]
        masked_image_latents_list = torch.cat(masked_image_latents_list, dim=0).cuda(self.device)
        
        timestep_tensor = self.timesteps_placeholder[running_batch_size]
        step_size = 1

        batch_idx = 0
        while batch_idx < running_batch_size * step_size:
            timestep_tensor[batch_idx:batch_idx+step_size] = batch[batch_idx]["timesteps"][batch[batch_idx]["denoising_progress"]]
            batch_idx += step_size
        print(f"timestep_tensor: {timestep_tensor}")
        
        if hasattr(self, "cached_o"):
            cached_o = self.cached_o
        else:
            cached_o = None
            
        batch[0]["edit_config"].device_num = self.local_rank
        batch[0]["edit_config"].max_batch_size = self.max_batch_size

        # Run denoising step in a thread pool
        denoising_start_time = time.time()
        loop = asyncio.get_event_loop()
        denoising_output = await loop.run_in_executor(
            None,
            lambda: self.pipeline.denoising_step_op(
                cur_denoising_step_list=cur_step_list,
                timestep=timestep_tensor,
                scheduler_list=scheduler_list,
                latents_list=latents_list,
                mask_list=mask_list,
                masked_image_latents_list=masked_image_latents_list,
                prompt_embeds_list=prompt_embeds_list,
                edit_config=batch[0]["edit_config"],
                cached_o=cached_o,
            )
        )
        denoising_time = time.time() - denoising_start_time
        print(f"One step denoising time: {denoising_time}")
        print(f"denoising_output['scheduler_steps']: {denoising_output['scheduler_steps']}")
        print(f"{datetime.now()} req_ids: {[item['req_id'] for item in batch]}")

        # Update batch items with new state
        for i in range(len(batch)):
            batch[i]["latents"] = denoising_output["latents"][i]
            batch[i]["scheduler_steps"] = denoising_output["scheduler_steps"][i]
            batch[i]["denoising_progress"] += 1
            
        # Send current steps info to coordinator

    async def _preprocess_request_SD2(self, message):
        """Preprocess a request using prepare_for_inference"""
        preprocess_start = time.time()
        pipeline_name = message["pipeline_name"]
        inputs = message["inputs"]
        print("inputs", inputs)
        
        # Run preprocessing in a thread pool
        loop = asyncio.get_event_loop()
        preprocessed = await loop.run_in_executor(
            None,
            lambda: self.pipeline.prepare_for_inference(
                prompt=inputs["prompt"],
                image_path=inputs["image_path"],
                mask_image_path=inputs["mask_image_path"],
                generator=torch.manual_seed(inputs["seed"]),
                edit_config_path=inputs["edit_config_path"],
            )
        )
        
        # Add additional information needed for tracking
        preprocessed.update({
            "req_id": message["req_id"],
            "scheduler": deepcopy(self.pipeline.scheduler),
            "denoising_progress": 0,
            "scheduler_steps": 0
        })
        preprocess_time = time.time() - preprocess_start
        print(f"Worker {self.worker_id}: Preprocess time: {preprocess_time}")
        return preprocessed

    async def _preprocess_request_OOTD(self, message):
        """Preprocess a request for OOTD (Outfit-On-The-Go Diffusion)"""
        preprocess_start = time.time()
        pipeline_name = message["pipeline_name"]
        inputs = message["inputs"]
        print("inputs", inputs)
        
        # Initialize models if needed
       
        # Set up category dictionaries
        
        # Run preprocessing in a thread pool
        loop = asyncio.get_event_loop()
        
        # Helper function to preprocess OOTD request
       
        preprocessed = await loop.run_in_executor(
            None,
            lambda: self.pipeline.prepare_for_pipeline(
                edit_config_path=inputs["edit_config_path"],
            )
        )
                
        
        # Add additional information needed for tracking
        preprocessed.update({
            "req_id": message["req_id"],
            "scheduler": deepcopy(self.pipeline.pipe.scheduler),
            "denoising_progress": 0,
            "scheduler_steps": 0
        })
        
        preprocess_time = time.time() - preprocess_start
        print(f"Worker {self.worker_id}: OOTD Preprocess time: {preprocess_time}")
        return preprocessed
    
    
    
    async def _process_batch_OOTD(self, batch):
        """Process a batch of OOTD requests using denoising_step"""
        if not batch:
            return
        
        running_batch_size = len(batch)
        print("running_batch_size",running_batch_size)
        # Extract current timestep for each request
        cur_timestep = torch.tensor([0.0]*running_batch_size, dtype=torch.float32, device=self.device)
        for i, item in enumerate(batch):
            print("item",item['timesteps'].shape)
            cur_timestep[i] = item["timesteps"][item["denoising_progress"]]
        print(f"Current timesteps: {cur_timestep}")
        timesteps_list = [item["timesteps"] for item in batch]
        
        # Prepare inputs for the denoising step
        cur_step_list = [item["denoising_progress"] for item in batch]
        latents_list = [item["latents"] for item in batch]
        vton_latents_list = [item["vton_latents"] for item in batch]
        prompt_embeds_list = [item["prompt_embeds"] for item in batch]
        spatial_attn_inputs_list = [item["spatial_attn_inputs"] for item in batch]
        image_ori_latents_list = [item["image_ori_latents"] for item in batch]
        mask_list = [item["mask_latents"] for item in batch]
        scheduler_list = [item["scheduler"] for item in batch]
        noise_list = [item["noise"] for item in batch]
        
        # Set device number and batch size in edit_config
        if self.pipeline_name == "OOTD_HD":
            model_type ="hd"
        else:
            model_type = "dc"
        batch[0]["edit_config"].device_num = self.local_rank
        batch[0]["edit_config"].max_batch_size = self.max_batch_size
        
        # Get cached outputs if available
        if hasattr(self, "cached_o"):
            cached_o = self.cached_o
        else:
            cached_o = None
        
        # Run denoising step in a thread pool
        denoising_start_time = time.time()
        loop = asyncio.get_event_loop()
        denoising_output = await loop.run_in_executor(
            None,
            lambda: self.pipeline.pipe.denoising_step_op(
                cur_denoising_step_list=cur_step_list,
                latents_list=latents_list,
                vton_latents_list=vton_latents_list,
                prompt_embeds_list=prompt_embeds_list,
                spatial_attn_inputs_list=spatial_attn_inputs_list,
                image_ori_latents_list=image_ori_latents_list,
                scheduler_list=scheduler_list,
                noise_list=noise_list,
                cur_timestep=cur_timestep,
                timestep_list=timesteps_list,
                model_type=model_type,
                mask_list=mask_list,
                edit_config=batch[0]["edit_config"],
                cached_o=cached_o,
            )
        )
        denoising_time = time.time() - denoising_start_time
        print(f"One step denoising time: {denoising_time}")
        print(f"{datetime.now()} req_ids: {[item['req_id'] for item in batch]}")
        
        # Update batch items with new state
        for i in range(len(batch)):
            batch[i]["latents"] = denoising_output["latents"][i]
            batch[i]["denoising_progress"] += 1
            batch[i]['scheduler_steps'] = denoising_output['scheduler_steps'][i]
            print("scheduler_steps",i, batch[i]['scheduler_steps'])
        # Send current steps info to coordinator
       

    async def _send_steps_update(self, batch):
        """Send current scheduler steps to coordinator"""
        try:
            # Collect step information for all requests in the batch
            steps_info = {}
            for req in batch:
                req_id = req.get("req_id")
                steps_info[req_id] = {
                    "scheduler_steps": req.get("scheduler_steps", 0),
                    "num_inference_steps": req.get("num_inference_steps", 0),
                }
            
            # Create update message
            update_message = {
                "type": "steps_update",
                "worker_id": self.worker_id,
                "steps_info": steps_info,
                "timestamp": time.time()
            }
            
            # Send the update through the result socket
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.result_socket.send_json(update_message)
            )
            
            self.logger.info(f"Sent steps update for {len(steps_info)} requests")
            
        except Exception as e:
            self.logger.error(f"Error sending steps update: {str(e)}")

    async def _send_completion_response(self, req):
        """Send completed request latents to the Coordinator for post-processing"""
        try:
            # Serialize the latents tensor
            serialization_start = time.time()
            # Convert tensor to bytes using torch.save
            latents_bytes = io.BytesIO()
            torch.save(req["latents"].cpu(), latents_bytes)
            latents_bytes = latents_bytes.getvalue()
            # Convert to base64 for JSON serialization
            latents_base64 = base64.b64encode(latents_bytes).decode('utf-8')
            serialization_time = time.time() - serialization_start

            send_response_start = time.time()
            # Send response with serialized latents
            response = {
                "worker_id": self.worker_id,
                "req_id": req["req_id"],
                "status": "completed",
                "latents_base64": latents_base64,
                "inference_latency": time.time() - req.get("start_time", time.time()),
            }
            self.result_socket.send_json(response)
            send_response_time = time.time() - send_response_start
            print(f"{datetime.now()} DistributedWorker for {req['req_id']}: Latents serial: {1000*serialization_time:.2f}ms, Send resp: {1000*send_response_time:.2f}ms")
            
            
        except Exception as e:
            raise e
            self.logger.error(f"Error sending completion for request {req['req_id']}: {str(e)}")
            traceback.print_exc()
            error_response = {
                "worker_id": self.worker_id,
                "req_id": req["req_id"],
                "status": "error",
                "error": str(e)
            }
            self.result_socket.send_json(error_response)

    def _unload_model(self, model_id: str):
        """Unload a model and free its memory"""
        if model_id not in self.models:
            return

        print(f"Unloading model: {model_id}")

        del self.models[model_id]
        torch.cuda.empty_cache()

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.running:  # Add check to prevent double cleanup
                self.running = False
                self.logger.info(f"Worker {self.worker_id} is stopping")
                
                # Cancel the processing task if it exists
                if self.processing_task and not self.processing_task.done():
                    self.processing_task.cancel()
                
                # Stop the event loop if it exists
                if self.loop and self.loop.is_running():
                    self.loop.call_soon_threadsafe(self.loop.stop)
                
                """Unload all models and free their memory"""
                for model_id in self.models:
                    self._unload_model(model_id)

                # Cleanup ZMQ resources
                self.task_socket.setsockopt(zmq.LINGER, 1000)  # 1 second timeout
                self.result_socket.setsockopt(zmq.LINGER, 1000)

                self.task_socket.close()
                self.result_socket.close()
                self.context.term()

                self.logger.info("Cleanup completed successfully")
                # Close all handlers
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)
        except Exception as e:
            self.logger.error(f"Error during worker cleanup: {e}")


def run_worker(
    local_rank: int,
    node_rank: int,
    node_config: NodeConfig,
    dist_config: DistributedConfig,
    cache_config: CacheConfig,
    scheduling_baseline: str = "basic",
    worker_max_batch_size: int = 2,
    pipeline_name: str = "SDXL",
):
    """Function to run in each worker process"""
    worker = DistributedWorker(
        local_rank, 
        node_rank, 
        node_config, 
        dist_config, 
        cache_config,
        scheduling_baseline,
        worker_max_batch_size,
        pipeline_name,
    )
    worker.run()


class Coordinator:
    def __init__(
        self,
        dist_config: DistributedConfig,
        node_rank: int,
        scheduling_baseline: str = "basic",
        worker_max_batch_size: int = 2,
        pipeline_name: str = "SDXL",
        cache_config: CacheConfig = None,
    ):
        self.dist_config = dist_config
        self.node_rank = node_rank
        self.node_config = dist_config.get_node_by_rank(node_rank)
        self.processes: List[mp.Process] = []
        self.scheduling_baseline = scheduling_baseline
        self.worker_max_batch_size = worker_max_batch_size
        self.pipeline_name = pipeline_name
        self.cache_config = cache_config
        print(f"Coordinator: Scheduling baseline: {self.scheduling_baseline}")
        
        # Initialize the scheduler based on scheduling_baseline
        # 根据调度基准选择调度器类型
        self.scheduler_type = "seq_length_balance"  # 默认使用序列长度平衡调度器
        if scheduling_baseline == "batch_balance":
            self.scheduler_type = "batch_size_balance"
        elif scheduling_baseline == "flops_balance":
            self.scheduler_type = "flops_balance"
        elif scheduling_baseline == "new_flops_balance":
            self.scheduler_type = "new_flops_balance"
        elif scheduling_baseline == "flops_batch_balance":
            self.scheduler_type = "flops_batch_balance"
        elif scheduling_baseline == "no_cb":
            self.scheduler_type = "batch_size_balance"
        elif scheduling_baseline == "step_flops_batch_balance":
            self.scheduler_type = "step_flops_batch_balance"
        try:
            self.scheduler = create_scheduler(self.scheduler_type)
            print(f"Using scheduler: {self.scheduler_type}")
        except ValueError:
            print(f"Warning: Unknown scheduler '{self.scheduler_type}', using seq_length_balance")
            self.scheduler_type = "seq_length_balance"
            self.scheduler = create_scheduler("seq_length_balance")

        # ZMQ setup
        if self.node_rank == 0:
            self.context = zmq.Context()
            self.task_sockets: Dict[str, zmq.Socket] = {}  # For sending tasks
            self.result_sockets: Dict[str, zmq.Socket] = {}  # For receiving results
        
        # Initialize local workers
        self._initialize_local_workers()

        # Coordinator setup
        if self.node_rank == 0:
            self.all_workers_info = self._gather_all_workers_info()
            self._setup_coordinator_sockets()

            self.device = torch.device("cuda:0")
            # Initialize pipeline for post-processing
            if self.pipeline_name == "SDXL":
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                ).to(self.device)  # Use the first GPU for post-processing
                self.pipeline.upcast_vae() # specific to the SDXL pipeline

            elif self.pipeline_name == "Flux_inpaint":
                self.pipeline = FluxInpaintPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell", 
                    torch_dtype=torch.bfloat16,
                    cache_dir="/project/infattllm/huggingface/hub/"
                ).to(self.device)
            elif self.pipeline_name == "SD2":
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    torch_dtype=torch.float16,
                ).to(self.device)
            elif self.pipeline_name == "OOTD_HD":
                self.pipeline = OOTDiffusionHD(gpu_id=0)
            elif self.pipeline_name == "OOTD_DC":
                self.pipeline = OOTDiffusionDC(gpu_id=0)
            else:
                raise ValueError(f"Pipeline name {self.pipeline_name} not supported")

        if self.node_rank == 0:
            # Maps node_name -> worker_id for intermediate results
            self.result_locations = {}

        self.active_tasks = {}  # Maps req_id to worker_id
        self.is_running = True

        # keep worker status 
        self.worker_status = {}
        
        # Add worker_batch_info to track sequence lengths of batches on each worker
        self.worker_batch_info = {}
        
        for worker_info in self.all_workers_info:
            worker_id = worker_info["worker_id"]
            self.worker_status[worker_id] = {
                "pipeline_name": None,
                "running_batch_size": 0,
                "status": "idle"
            }
            # Initialize empty batch info for each worker
            self.worker_batch_info[worker_id] = []

        # Request queue for handling concurrent requests
        # self.request_queue = asyncio.Queue()
        self.request_queue = asyncio.PriorityQueue()
        self.request_futures = {}  # Maps req_id to future
        self.scheduler_task = None
        
        # Task for centralized result gathering
        self.result_gatherer_task = None
    
    def change_scheduler(self, scheduler_type: str) -> str:
        """Change the scheduler type dynamically"""
        try:
            # Create the new scheduler
            new_scheduler = create_scheduler(scheduler_type)
            
            # Update scheduler type and instance
            self.scheduler_type = scheduler_type
            self.scheduler = new_scheduler
            
            print(f"Changed scheduler to: {self.scheduler_type}")
            return f"Successfully changed scheduler to {self.scheduler_type}"
        except ValueError as e:
            print(f"Error changing scheduler: {e}")
            return f"Error changing scheduler: {e}"

    def _initialize_local_workers(self):
        """Initialize workers for this node only"""
        worker_pids = []
        for local_rank in range(self.node_config.gpu_count):
            if local_rank == 0:
                # coordinator
                continue
            p = mp.Process(
                target=run_worker,
                args=(
                    local_rank,
                    self.node_rank,
                    self.node_config,
                    self.dist_config,
                    self.cache_config,
                    self.scheduling_baseline,
                    self.worker_max_batch_size,
                    self.pipeline_name,
                ),
            )
            p.start()
            self.processes.append(p)

            worker_pids.append(p.pid)

        with open("worker.pid", "w") as f:
            f.write("\n".join(map(str, worker_pids)))

    def _setup_coordinator_sockets(self):
        """Setup ZMQ sockets for the coordinator"""
        if self.node_rank != 0:
            return

        for worker_info in self.all_workers_info:
            worker_id = worker_info["worker_id"]

            task_socket = self.context.socket(zmq.PUSH)
            result_socket = self.context.socket(zmq.PULL)

            # Bind to the ports
            task_socket.bind(f"tcp://*:{worker_info['task_port']}")
            result_socket.bind(f"tcp://*:{worker_info['result_port']}")

            self.task_sockets[worker_id] = task_socket
            self.result_sockets[worker_id] = result_socket

    def _gather_all_workers_info(self) -> List[Dict]:
        """Gather information about all workers across all nodes"""
        workers_info = []
        for node in self.dist_config.nodes:
            for local_rank in range(node.gpu_count):
                worker_id = f"worker_{node.rank}_{local_rank}"
                global_rank = (
                    sum(
                        n.gpu_count
                        for n in self.dist_config.nodes
                        if n.rank < node.rank
                    )
                    + local_rank
                )
                if global_rank == 0:
                    # coordinator
                    continue
                task_port = self.dist_config.port + 1 + global_rank * 2
                result_port = task_port + 1

                print(
                    f"Worker: {worker_id}, node_rank: {node.rank}, local_rank: {local_rank}, "
                    f"task_port: {task_port}, result_port: {result_port}"
                )
                workers_info.append(
                    {
                        "worker_id": worker_id,
                        "node_rank": node.rank,
                        "local_rank": local_rank,
                        "global_rank": global_rank,
                        "task_port": task_port,
                        "result_port": result_port,
                    }
                )
        return workers_info

    async def start_scheduler(self):
        """Start the scheduler task that processes the request queue"""
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # Start the centralized result gatherer task
        if self.node_rank == 0:
            self.result_gatherer_task = asyncio.create_task(self._gather_all_results())

    async def process_no_cb_mode(self, idle_workers):
        """Process requests in no_cb mode by sending multiple requests as a single batch to a completely idle worker"""
        if not idle_workers:
            return False
            
        # Select an idle worker
        worker_id = idle_workers[0]
        
        # Collect up to max_batch_size requests
        batch_requests = []
        batch_count = 0
        
        # Check if there are requests in the queue
        if self.request_queue.empty():
            await asyncio.sleep(0.1)
            return False
            
        # Collect requests up to max_batch_size
        while batch_count < self.worker_max_batch_size and not self.request_queue.empty():
            try:
                # Get next request
                queue_item = await self.request_queue.get()
                queue_item_ts = queue_item[0]
                pipeline_name, inputs, req_id = queue_item[1]
                print("get next request", req_id)
                # Add to batch
                batch_requests.append((pipeline_name, inputs, req_id))
                batch_count += 1
            except Exception as e:
                print(f"{datetime.now()} Error collecting request for batch: {e}")
                break
        
        print("batch_count", batch_count)
        if not batch_requests:
            return False
            
        # Log batch processing
        req_ids = [req[2] for req in batch_requests]
        print(f"{datetime.now()} No-CB mode: Sending batch of {len(batch_requests)} requests to worker {worker_id}: {req_ids}")
        
        # 提取所有请求的pipeline_name，确保它们相同
        pipeline_name = batch_requests[0][0]
        
        # 创建一个包含所有请求的单一批次消息
        batch_task_message = {
            "pipeline_name": pipeline_name,
            "type": "batch",
            "batch_size": batch_count,
            "batch_requests": []
        }
        # 将每个请求添加到批次消息中
        for _, inputs, req_id in batch_requests:
            request_seq_length = get_seq_length_from_request(inputs)
            print("inputs", inputs)
            
            # 使用调度器更新工作节点信息，用于负载均衡指标
            self.worker_batch_info = self.scheduler.update_worker_info(
                worker_id, 
                req_id, 
                inputs, 
                self.worker_batch_info
            )
            
            # 将请求添加到批次消息
            batch_task_message["batch_requests"].append({
                "inputs": inputs,
                "req_id": req_id
            })
            
            # 更新活动任务映射
            self.active_tasks[req_id] = worker_id
        
        # 更新工作节点状态
        self.worker_status[worker_id]["pipeline_name"] = pipeline_name
        self.worker_status[worker_id]["running_batch_size"] += batch_count
        self.worker_status[worker_id]["status"] = "busy"
        print("set worker status", worker_id, "busy")
        
        # 发送批次消息给工作节点
        asyncio.create_task(self._send_task_to_worker(worker_id, batch_task_message))
        
        # Successfully processed batch
        return True

    async def _scheduler_loop(self):
        """Main scheduler loop that processes requests from the queue"""
        """Suyi: the scheduler logic is different from the _scheduler_loop_request"""
        """Suyi: We first get all idle workers, and then schedule requests to them"""
        """Suyi: Each idle worker will take the first request in the queue that can be scheduled to it"""
        """Suyi: Therefore, the execution sequence of the requests is the same as the requests' arrival order"""

        while self.is_running:
            try:
                # Get all idle workers
                if self.scheduling_baseline == "no_cb":
                    idle_workers = [worker_id for worker_id, status in self.worker_status.items() if status["status"] == "idle"]
                else:
                    idle_workers = [worker_id for worker_id, status in self.worker_status.items() if status["status"] == "idle" and status["running_batch_size"] < self.worker_max_batch_size]
                # exclude the coordinator node
                idle_workers = [worker_id for worker_id in idle_workers if worker_id != "worker_0_0"]
                
                if len(idle_workers) == 0:
                    await asyncio.sleep(0.1)
                    continue
                
                # Handle no_cb mode - process multiple requests at once for a single worker
                # In no_cb mode, we send up to max_batch_size requests to an idle worker at once,
                # rather than using continuous batching which assigns one request at a time.
                if self.scheduling_baseline == "no_cb":
                   
                    # Call the process_no_cb_mode function
                    processed = await self.process_no_cb_mode(idle_workers)
                    # if processed is True:
                    #     worker_id = idle_workers[0]
                    #     self.worker_status[worker_id]["status"] = "busy"
                
                    continue
                
                # Regular continuous batching mode (default)
                # Get the next request from queue
                queue_item = await self.request_queue.get()
                queue_item_ts = queue_item[0]
                if self.scheduling_baseline == "no_cb":
                    idle_workers = [worker_id for worker_id, status in self.worker_status.items() if status["status"] == "idle"]
                else:
                    idle_workers = [worker_id for worker_id, status in self.worker_status.items() if status["status"] == "idle" and status["running_batch_size"] < self.worker_max_batch_size]
                # exclude the coordinator node
                idle_workers = [worker_id for worker_id in idle_workers if worker_id != "worker_0_0"]
                if len(idle_workers) == 0:
                    await asyncio.sleep(0.1)
                    continue
                
                pipeline_name, inputs, req_id = queue_item[1]
                
                # 输出当前各worker的负载状态信息
                batch_info = {}
                for worker_id, batches in self.worker_batch_info.items():
                    batch_count = len(batches)
                    # if scheduler is seq_length_balance, then we need to sum the seq_length_total
                    flops_total = sum(cal_flops(batch.get('mask_seq_length', 0), self.pipeline_name) for batch in batches)
                    seq_length_total = sum(batch.get('mask_seq_length', 0) for batch in batches)
                    batch_info[worker_id] = {
                        "batch_count": batch_count,
                        "seq_length_total": seq_length_total,
                        "flops_total": flops_total
                    }

                    
                print(f"Current worker load before assignment - {batch_info}")
                print("self.worker_batch_info", self.worker_batch_info)
                # Use the scheduler to select a worker based on the scheduling strategy
                worker_id = self.scheduler.select_worker(
                    idle_workers, 
                    self.worker_batch_info, 
                    inputs, 
                    pipeline_name,
                    req_id
                )
                
                # Get sequence length for logging
                request_seq_length = get_seq_length_from_request(inputs)
                
                # # 让scheduler生成日志信息，而不是在coordinator中计算FLOPS
                scheduler_log_info = self.scheduler.get_request_log_info(request_seq_length, self.pipeline_name)
                
                print(f"{datetime.now()} Scheduling req_id {req_id} to worker {worker_id} " +
                     f"with seq_length {request_seq_length}{scheduler_log_info}, scheduler={self.scheduler.__class__.__name__}")
                
                # print the batch_info
             
                task_message = {
                    "pipeline_name": pipeline_name,
                    "inputs": inputs,
                    "req_id": req_id,
                }

                # Update worker status
                self.worker_status[worker_id]["pipeline_name"] = task_message["pipeline_name"]
                self.worker_status[worker_id]["running_batch_size"] += 1
                self.active_tasks[task_message["req_id"]] = worker_id
                
                # Update worker batch info using the scheduler
                self.worker_batch_info = self.scheduler.update_worker_info(
                    worker_id, 
                    req_id, 
                    inputs, 

                    self.worker_batch_info
                )

                # Schedule task to worker in a non-blocking way
                asyncio.create_task(self._send_task_to_worker(worker_id, task_message))

            except Exception as e:
                print(f"{datetime.now()} Error in scheduler loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)  # Prevent tight loop on error

    async def _send_task_to_worker(self, worker_id: str, task_message: Dict[str, Any]):
        """Send task to worker asynchronously"""
        try:
            # Send task in a non-blocking way
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.task_sockets[worker_id].send_json(task_message)
            )
            
            # No longer need to start a separate result gathering task
            # The centralized result gatherer will handle this

        except Exception as e:
            print(f"{datetime.now()} Error sending task to worker {worker_id}: {e}")
            traceback.print_exc()
            raise e

    async def _gather_all_results(self):
        """Centralized result gathering task that polls all result sockets"""
        print(f"{datetime.now()} Starting centralized result gatherer")
        
        # Create a poller for all result sockets
        poller = zmq.Poller()
        for worker_id, socket in self.result_sockets.items():
            poller.register(socket, zmq.POLLIN)
        
        while self.is_running:
            try:
                # Poll all sockets with a timeout
                socks = dict(await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: poller.poll(timeout=1000)  # 1 second timeout
                ))
                
                # Process any ready sockets
                for socket in socks:
                    # Find which worker this socket belongs to
                    worker_id = None
                    for wid, sock in self.result_sockets.items():
                        if sock == socket:
                            worker_id = wid
                            break
                    
                    if worker_id is None:
                        continue
                    
                    # Receive response in a non-blocking way
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: socket.recv_json()
                    )
                    
                    # Handle ping/pong messages
                    if response.get("type") == "pong":
                        continue
                    # Handle steps update messages
                    # if response.get("type") == "steps_update":
                    #     self._handle_steps_update(worker_id, response)
                    #     continue
                    
                    # Get the request ID from the response
                    received_req_id = response.get("req_id")
                    if not received_req_id:
                        raise ValueError(f"{datetime.now()} Received response without req_id from worker {worker_id}")
                    
                    print(f"{datetime.now()} Received response for req_id {received_req_id}: {response.get('status', 'unknown')}")
                    
                    # Update worker status
                    self.worker_status[worker_id]["pipeline_name"] = None
                    print("before worker_status[worker_id]['running_batch_size']", self.worker_status[worker_id]["running_batch_size"])

                    self.worker_status[worker_id]["running_batch_size"] -= 1
                    # Set status back to idle if no more requests
                    print("after worker_status[worker_id]['running_batch_size']", self.worker_status[worker_id]["running_batch_size"])
                    if self.worker_status[worker_id]["running_batch_size"] == 0:
                        self.worker_status[worker_id]["status"] = "idle"
                    
                    # Update worker_batch_info to remove the completed request
                    self.worker_batch_info = self.scheduler.remove_completed_request(
                        worker_id, received_req_id, self.worker_batch_info
                    )
                    
                    # Log the updated worker load statistics after task completion
                    # self.scheduler.log_statistics(self.worker_batch_info)
                    
                    # Check if this request is in our active tasks
                    print("received_req_id",received_req_id)
                    if received_req_id not in self.active_tasks:
                        # raise ValueError(f"{datetime.now()} Received response for unknown req_id {received_req_id}")
                        print((f"{datetime.now()} Received response for unknown req_id {received_req_id}"))
                    # Remove from active tasks
                    else:
                        del self.active_tasks[received_req_id]
                    # print("delete received_req_id", received_req_id)
                    # Process the latents if the status is completed
                    if response.get("status") == "completed" and "latents_base64" in response:
                        # Process latents and get final result
                        final_result = await self._process_latents(
                            response["latents_base64"], 
                            received_req_id
                        )
                        
                        # Add inference latency from worker
                        if "inference_latency" in response:
                            final_result["inference_latency"] = response["inference_latency"]
                        
                        # Set result in future
                        if received_req_id in self.request_futures:
                            self.request_futures[received_req_id].set_result(final_result)
                            del self.request_futures[received_req_id]
                        print(f"Processed request {received_req_id}.")
                    else:
                        # For error responses or other types, just set the result as is
                        if received_req_id in self.request_futures:
                            self.request_futures[received_req_id].set_result(response)
                            del self.request_futures[received_req_id]
                        print(f"{datetime.now()} Received response for req_id {received_req_id}: {response.get('status', 'unknown')}")
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"{datetime.now()} Error in result gatherer: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)  # Prevent tight loop on error
        
        print(f"{datetime.now()} Centralized result gatherer stopped")
        
    def _handle_steps_update(self, worker_id, update_message):
        """Handle step update messages from workers"""

        try:
            print("receive!!!!!!!!!!!!!!!!!!!!!!!")
            steps_info = update_message.get("steps_info", {})
            
            # Log the update
            req_ids = list(steps_info.keys())
            if req_ids:
                # Log a sample of the first request's progress
                sample_req = steps_info[req_ids[0]]
                print(f"{datetime.now()} Step update from {worker_id}: {len(req_ids)} requests, "
                      f"scheduler_steps ({sample_req.get('scheduler_steps', 0)}/{sample_req.get('num_inference_steps', 0)})")
                
            # Store the step information for possible API queries
            # This could be exposed through an endpoint if needed
            if not hasattr(self, "request_steps_info"):
                self.request_steps_info = {}
                
            # Update the stored steps info
            for req_id, step_info in steps_info.items():
                self.request_steps_info[req_id] = {
                    "worker_id": worker_id,
                    "scheduler_steps": step_info.get("scheduler_steps", 0),
                    "num_inference_steps": step_info.get("num_inference_steps", 0),
                    "last_update_time": time.time()
                }
                # Update worker_batch_info with the latest scheduler_steps
                if worker_id in self.worker_batch_info:
                    for batch_item in self.worker_batch_info[worker_id]:
                        if batch_item.get("request_id") == req_id:
                            # Update the scheduler_steps in worker_batch_info
                            batch_item["scheduler_steps"] = step_info.get("scheduler_steps", 0)
                            
                            # Calculate remaining steps for this request
                            total_steps = step_info.get("num_inference_steps", 0)
                            current_step = step_info.get("scheduler_steps", 0)
                            remaining_steps = max(0, total_steps - current_step + 1)


                            batch_item["remaining_steps"] = remaining_steps
                            
                            # Log detailed update for debugging
                            print(f"Updated worker_batch_info for req_id {req_id} on {worker_id}: "
                                  f"steps={current_step}/{total_steps}, "
                                  f"remaining={remaining_steps}")
                          
                            break
                
            # After updating worker_batch_info, tell the scheduler to recalculate load metrics
            # if hasattr(self.scheduler, "recalculate_worker_load") and callable(self.scheduler.recalculate_worker_load):
            #     self.scheduler.recalculate_worker_load(self.worker_batch_info)
                
        except Exception as e:
            print(f"{datetime.now()} Error handling steps update from {worker_id}: {e}")
            traceback.print_exc()

    def execute_workflow(
        self, pipeline_name: str, inputs: Dict[str, Any], req_id: str
    ) -> Dict[str, Any]:
        """Execute workflow by adding request to queue and waiting for result"""
        # Create future for this request
        future = asyncio.Future()
        self.request_futures[req_id] = future

        try:
            # 从元数据中获取序列ID，如果存在的话使用它作为优先级，否则使用时间戳
            priority = time.time()  # 默认使用时间戳
            
            # 检查是否有序列ID
            if "metadata" in inputs and "sequence_id" in inputs["metadata"]:
                # 使用序列ID作为优先级，小的序列ID有更高的优先级
                sequence_id = inputs["metadata"]["sequence_id"]
                # 移除metadata以避免传递给模型
                metadata = inputs.pop("metadata")
                priority = float(sequence_id)  # 转换为float以兼容优先队列
                print(f"{datetime.now()} Using sequence_id {sequence_id} as priority for req_id {req_id}")
            
            # 添加到优先队列，使用序列ID或时间戳作为优先级
            asyncio.create_task(self.request_queue.put((priority, (pipeline_name, inputs, req_id))))
            
            print(f"{datetime.now()} Added request {req_id} to queue with priority {priority}")

            # Wait for result
            return future

        except Exception as e:
            if req_id in self.request_futures:
                del self.request_futures[req_id]
            raise e

    def cleanup(self):
        """Cleanup coordinator resources"""
        self.is_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
        if self.result_gatherer_task:
            self.result_gatherer_task.cancel()
        if not self.is_running:  # Already cleaned up
            return

        if self.node_rank == 0:
            # Send stop signal to all workers
            for worker_id, socket in list(self.task_sockets.items()):
                try:
                    socket.close(linger=1000)  # 1 second linger
                    del self.task_sockets[worker_id]
                except Exception as e:
                    print(f"Error closing task socket for {worker_id}: {e}")

            for worker_id, socket in list(self.result_sockets.items()):
                try:
                    socket.close(linger=1000)
                    del self.result_sockets[worker_id]
                except Exception as e:
                    print(f"Error closing result socket for {worker_id}: {e}")

        # Clean up processes with timeout
        for p in self.processes:
            try:
                p.terminate()
                p.join(timeout=2)  # Wait up to 2 seconds
                if p.is_alive():
                    print(f"Process {p.pid} still alive after terminate, killing...")
                    p.kill()
                    p.join(timeout=1)
            except Exception as e:
                print(f"Error cleaning up process {p.pid}: {e}")

    async def _process_latents(self, latents_base64: str, req_id: str) -> Dict[str, Any]:
        """Process latents from worker and convert to images"""
        try:
            # Deserialize the latents tensor
            deserialization_start = time.time()
            latents_bytes = base64.b64decode(latents_base64)
            latents_buffer = io.BytesIO(latents_bytes)
            latents = torch.load(latents_buffer)
            if self.pipeline_name == "SDXL":
                latents = latents.to(dtype=torch.float16).cuda(self.device)
            elif self.pipeline_name == "Flux_inpaint":
                latents = latents.to(dtype=torch.bfloat16).cuda(self.device)
            elif self.pipeline_name == "SD2":
                latents = latents.to(dtype=torch.float16).cuda(self.device)
            elif self.pipeline_name == "OOTD_HD" or self.pipeline_name == "OOTD_DC":
                latents = latents.to(dtype=torch.float16).cuda(self.device)
            else:
                raise ValueError(f"Pipeline name {self.pipeline_name} not supported")            
            deserialization_time = time.time() - deserialization_start

            # Run post-processing in a thread pool
            post_process_start = time.time()
            loop = asyncio.get_event_loop()
            
            # 使用生成器调用post_process_inference
            if self.pipeline_name == "SD2":
                generator = torch.Generator(device=self.device).manual_seed(1000)
                result = await loop.run_in_executor(
                    None,
                    lambda: self.pipeline.post_process_inference(latents, generator=generator)
                )
            elif self.pipeline_name == "OOTD_DC" or self.pipeline_name == "OOTD_HD":
                result = await loop.run_in_executor(
                    None,
                    lambda: self.pipeline.pipe.post_process_inference(latents)
                )
            else:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.pipeline.post_process_inference(latents)
                )
            post_process_time = time.time() - post_process_start

            # Convert images to base64
            image_to_base64_start = time.time()
            img_str_list = []
            for image in result.images:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_str_list.append(img_str)
            image_to_base64_time = time.time() - image_to_base64_start
            print(f"Coordinator: for {req_id}, deserialization: {1000*deserialization_time:.2f}ms, Post process: {1000*post_process_time:.2f}ms, Image to base64: {1000*image_to_base64_time:.2f}ms")

            return {
                "req_id": req_id,
                "status": "completed",
                "img_str_list": img_str_list,
                "post_processing_latency": post_process_time + image_to_base64_time
            }
        except Exception as e:
            print(f"{datetime.now()} Error processing latents for request {req_id}: {e}")
            print(f"latents dtype, device: {latents.dtype}, {latents.device}")
            traceback.print_exc()
            return {
                "req_id": req_id,
                "status": "error",
                "error": str(e)
            }
class WorkflowService:
    def __init__(
        self,
        dist_config: DistributedConfig,
        node_rank: int = 0,
        scheduling_baseline: str = "basic",
        worker_max_batch_size: int = 2,
        pipeline_name: str = "SDXL",
        cache_config: CacheConfig = None,
    ):
        self.dist_config = dist_config
        self.node_rank = node_rank
        self.scheduling_baseline = scheduling_baseline
        self.worker_max_batch_size = worker_max_batch_size
        self.coordinator = None  # Will be initialized during startup
        self.pipeline_name = pipeline_name
        self.cache_config = cache_config
        self.setup_signal_handlers()

    async def startup(self):
        """Initialize the distributed system on service startup"""
        print(f"Starting workflow service on node {self.node_rank}")

        self.coordinator = Coordinator(
            self.dist_config, 
            self.node_rank, 
            scheduling_baseline=self.scheduling_baseline,
            worker_max_batch_size=self.worker_max_batch_size,
            pipeline_name=self.pipeline_name,
            cache_config=self.cache_config,
        )

        # Wait for all workers to be ready before accepting requests
        await self._wait_for_workers_ready()
        print(f"self._wait_for_workers_ready() completed")

        # Start the scheduler
        await self.coordinator.start_scheduler()
        print(f"self.coordinator.start_scheduler() completed")

        print(f"Workflow service ready on node {self.node_rank}")
    
    async def change_scheduler(self, scheduler_type: str) -> Dict[str, str]:
        """Change the scheduler type"""
        if self.node_rank != 0:
            return {"status": "error", "message": "Scheduler can only be changed on master node"}
        
        if not self.coordinator:
            return {"status": "error", "message": "Coordinator not initialized"}
        
        result = self.coordinator.change_scheduler(scheduler_type)
        return {"status": "success", "message": result}

    async def _wait_for_workers_ready(self, timeout_seconds: int = 60):
        """Wait for all workers to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if self._check_workers_ready():
                return
            await asyncio.sleep(1)
        raise TimeoutError("Workers failed to initialize within timeout period")

    def _check_workers_ready(self) -> bool:
        """Check if all workers are ready"""
        if self.node_rank == 0:
            try:
                # Send ping to all workers
                print("self.coordinator.task_sockets.items()", self.coordinator.task_sockets.items())
                for worker_id, socket in self.coordinator.task_sockets.items():
                    socket.send_json({"type": "ping"})

                # Wait for responses
                for socket in self.coordinator.result_sockets.values():
                    response = socket.recv_json()
                    if response.get("type") != "pong":
                        return False
                return True
            except zmq.ZMQError:
                return False

        # For worker nodes: check local workers
        return all(p.is_alive() for p in self.coordinator.processes)

    async def run_inference(
        self, service_id: str, inputs: Dict[str, Any], req_id: str
    ) -> Dict[str, Any]:
        
        pipeline_name = service_id
        print(f"{datetime.now()} Running inference for workflow: {pipeline_name}, request_id: {req_id}")

        # Make sure mask_seq_length is set for load balancing
        if 'mask_seq_length' not in inputs and pipeline_name == "Flux_inpaint":
            # If mask_image_path is provided, calculate mask_seq_length from it
            if 'mask_image_path' in inputs and os.path.exists(inputs['mask_image_path']):
                from scheduler.flux_client_async import calculate_mask_seq_length
                inputs['mask_seq_length'] = calculate_mask_seq_length(inputs['mask_image_path'])
                print(f"Calculated mask_seq_length: {inputs['mask_seq_length']} for req_id: {req_id}")
            else:
                # Default to a reasonable value if mask not provided
                inputs['mask_seq_length'] = 4096
                print(f"Using default mask_seq_length: {inputs['mask_seq_length']} for req_id: {req_id}")

        # Execute workflow and wait for result
        future = self.coordinator.execute_workflow(pipeline_name, inputs, req_id)
        return await future    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}. Starting graceful shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def shutdown(self):
        """Cleanup on service shutdown"""
        print("Shutting down workflow service...")
        if self.coordinator:
            try:
                # Set timeout for cleanup operations
                shutdown_timeout = 10  # seconds

                # Stop all workers first with timeout
                if self.node_rank == 0:
                    for worker_id, socket in self.coordinator.task_sockets.items():
                        try:
                            # Use non-blocking send with retry
                            for _ in range(3):
                                try:
                                    socket.send_json({"type": "stop"}, zmq.NOBLOCK)
                                    break
                                except zmq.Again:
                                    await asyncio.sleep(0.1)
                        except Exception as e:
                            print(f"Error sending stop signal to {worker_id}: {e}")

                # Give workers time to process stop signal
                await asyncio.sleep(2)

                try:
                    # Cleanup coordinator with timeout
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self.coordinator.cleanup
                        ),
                        timeout=shutdown_timeout,
                    )
                except asyncio.TimeoutError:
                    print("Coordinator cleanup timed out, forcing cleanup...")
                    # Force cleanup of remaining processes
                    if self.coordinator.processes:
                        for p in self.coordinator.processes:
                            try:
                                p.terminate()
                                await asyncio.sleep(0.1)
                                if p.is_alive():
                                    p.kill()
                            except Exception as e:
                                print(f"Error forcing process cleanup: {e}")
            except Exception as e:
                print(f"Error during coordinator cleanup: {e}")
            finally:
                # Ensure ZMQ context is terminated
                if hasattr(self.coordinator, "context"):
                    try:
                        self.coordinator.context.term()
                    except Exception as e:
                        print(f"Error terminating ZMQ context: {e}")

        print("Workflow service shutdown complete")


class InferenceRequest(BaseModel):
    inputs: Dict[str, Any]


class SchedulerChangeRequest(BaseModel):
    scheduler_type: str


class ProgressRequest(BaseModel):
    req_id: Optional[str] = None  # Optional, if not provided, return all active requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle"""
    # Initialize service
    await workflow_service.startup()

    yield

    # Cleanup on shutdown
    await workflow_service.shutdown()


app = FastAPI(lifespan=lifespan)



# service_id is the pipeline class name in diffusers
@app.post("/api/workflow/{service_id}/inference")
async def run_inference(service_id: str, request: InferenceRequest):
    try:
        req_id = str(uuid.uuid4())[:8]
        print(f"{datetime.now()} Handle request_id: {req_id}")
        
        # Run inference asynchronously
        results = await workflow_service.run_inference(service_id, request.inputs, req_id)

        assert results["status"] == "completed", f"Unknown status: {results['status']}"
        return {"status": "success", "results": results}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/change_scheduler")
async def change_scheduler(request: SchedulerChangeRequest):
    """Endpoint to change the scheduler type dynamically"""
    try:
        result = await workflow_service.change_scheduler(request.scheduler_type)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/progress")
async def get_request_progress(request: ProgressRequest):
    """Get progress information for active requests"""
    try:
        # Check if the workflow service is initialized
        if not workflow_service or not workflow_service.coordinator:
            raise HTTPException(status_code=500, detail="Workflow service not initialized")
            
        # Create request_steps_info if it doesn't exist
        if not hasattr(workflow_service.coordinator, "request_steps_info"):
            workflow_service.coordinator.request_steps_info = {}
            
        # If req_id is provided, get progress for that specific request
        if request.req_id:
            if request.req_id in workflow_service.coordinator.request_steps_info:
                # Return progress for the specific request
                progress_info = workflow_service.coordinator.request_steps_info[request.req_id]
                return {
                    "status": "success",
                    "progress": {
                        request.req_id: progress_info
                    }
                }
            else:
                # Check if it's in active tasks but no progress update yet
                if request.req_id in workflow_service.coordinator.active_tasks:
                    worker_id = workflow_service.coordinator.active_tasks[request.req_id]
                    return {
                        "status": "success",
                        "progress": {
                            request.req_id: {
                                "worker_id": worker_id,
                                "scheduler_steps": 0,
                                "num_inference_steps": "unknown",
                                "progress_percentage": 0,
                                "status": "pending"
                            }
                        }
                    }
                else:
                    # Request not found
                    return {
                        "status": "error",
                        "message": f"Request {request.req_id} not found or completed"
                    }
        else:
            # Return progress for all active requests
            all_progress = {}
            
            # Add known progress info
            for req_id, progress_info in workflow_service.coordinator.request_steps_info.items():
                # Only include active requests (ones in the active_tasks dict)
                if req_id in workflow_service.coordinator.active_tasks:
                    all_progress[req_id] = progress_info
            
            # Add active requests without progress updates yet
            for req_id, worker_id in workflow_service.coordinator.active_tasks.items():
                if req_id not in all_progress:
                    all_progress[req_id] = {
                        "worker_id": worker_id,
                        "scheduler_steps": 0, 
                        "num_inference_steps": "unknown",
                        "progress_percentage": 0,
                        "status": "pending"
                    }
            
            return {
                "status": "success",
                "progress": all_progress
            }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def get_node_config():
    import socket
    import subprocess

    # 获取主机名
    hostname = socket.gethostname()
    print(f"主机名: {hostname}")

    # 获取GPU数量
    try:
        nvidia_smi_output = subprocess.check_output("nvidia-smi -L", shell=True, text=True)
        gpu_count = nvidia_smi_output.count("GPU ")
        # gpu_count = 2
        print(f"GPU数量: {gpu_count}")
    except subprocess.CalledProcessError:
        print("无法获取GPU信息, 可能没有安装NVIDIA驱动或nvidia-smi工具")
    rank = 0
    return {'rank':rank, "address": hostname, "gpu_count": gpu_count}
if __name__ == "__main__":
    mp.set_start_method("spawn")

    with open("server.pid", "w") as f:
        f.write(str(os.getpid()))

    # Set up signal handlers
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        try:
            os.remove("server.pid")
        except Exception as e:
            print(f"Error removing server.pid: {e}")

        if workflow_service:
            asyncio.run(workflow_service.shutdown())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--config", type=str, default="local_config.yml")
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--scheduling-baseline", type=str, 
                       choices=["batch_balance", "seq_length_balance", "flops_balance", "new_flops_balance","flops_batch_balance", "no_cb", "step_flops_batch_balance"], 
                       default="flops_balance",
                       help="Scheduling strategy: batch_balance (balance batch counts), seq_length_balance (balance sequence lengths), flops_balance (balance computational load), or no_cb (disable continuous batching)")
    parser.add_argument("--worker-max-batch-size", type=int, default=8)
    parser.add_argument("--pipeline-name", type=str, default="SDXL", choices=["SDXL", "Flux_inpaint", "SD2", "OOTD_HD", "OOTD_DC"])
    
    args = parser.parse_args()
    if args.pipeline_name == "SD2":
        args.cache_config = "/home/xjiangbp/image-inpainting/scheduler/cache_configs/sd2_cache_config.yml"
    elif args.pipeline_name == "OOTD_HD" or args.pipeline_name == "OOTD_DC":
        args.cache_config = "/app/image-inpainting/scheduler/cache_configs/ootd_cache_config.yml"
    else:
        args.cache_config = "/app/image-inpainting/scheduler/cache_configs/ootd_cache_config.yml"
    print("schedule_baseline", args.scheduling_baseline)
    # Load DistributedConfig from YAML
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
    with open(args.cache_config, "r") as f:
        cache_config_data = yaml.safe_load(f)
    node_dict = get_node_config()
    # Build NodeConfig objects from the `nodes` list
    # node_configs = [NodeConfig(**node_dict) for node_dict in config_data["nodes"]]
    node_configs = [NodeConfig(**node_dict)]
    dist_config = DistributedConfig(
        nodes=node_configs, port=config_data.get("port", 29500)
    )
    cache_config = CacheConfig(**cache_config_data)
    # Initialize the service
    workflow_service = WorkflowService(
        dist_config=dist_config, 
        node_rank=args.node_rank, 
        scheduling_baseline=args.scheduling_baseline, 
        worker_max_batch_size=args.worker_max_batch_size,
        pipeline_name=args.pipeline_name,
        cache_config=cache_config,
    )

    # Run the FastAPI server
    try:
        if args.node_rank == 0:
            config = uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                loop="asyncio",
                timeout_keep_alive=30,
                timeout_graceful_shutdown=30,
            )
            server = uvicorn.Server(config)
            print("server.run()")
            server.run()
            print("server.run() done")
        else:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(workflow_service.startup())
            try:
                loop.run_forever()
            except KeyboardInterrupt:
                loop.run_until_complete(workflow_service.shutdown())
            finally:
                loop.close()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        asyncio.run(workflow_service.shutdown())
    except Exception as e:
        print(f"Error during server execution: {e}")
        asyncio.run(workflow_service.shutdown())





