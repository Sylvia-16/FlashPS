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

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import uvicorn
import yaml
import zmq
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from config import DistributedConfig, NodeConfig

from collections import defaultdict
import uuid
import traceback

from datetime import datetime
# disable torch warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from diffusers import StableDiffusionXLPipeline


class DistributedWorker:
    def __init__(
        self,
        local_rank: int,
        node_rank: int,
        node_config: NodeConfig,
        dist_config: DistributedConfig,
        scheduling_baseline: str = "basic",
    ):
        self.local_rank = local_rank
        self.node_rank = node_rank
        self.node_config = node_config
        self.dist_config = dist_config
        self.global_rank = self._calculate_global_rank()
        self.scheduling_baseline = scheduling_baseline

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

        self.intermediate_results = {}
        self.result_locations = {}

        self.running = True

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16"
        ).to(self.device)

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

        poller = zmq.Poller()
        poller.register(self.task_socket, zmq.POLLIN)

        while True:
            try:
                # Use poller with timeout instead of blocking recv
                socks = dict(poller.poll(timeout=1000))  # 1 second timeout
                if self.task_socket not in socks:
                    continue

                message = self.task_socket.recv_json()
                # self.logger.info(f"Received message: {message}")

                if message.get("type") == "stop":  # Poison pill
                    self.logger.info("Received stop signal, initiating shutdown...")
                    break
                elif message.get("type") == "ping":  # Health check
                    self.result_socket.send_json({"type": "pong"})
                    continue
                elif message.get("type") == "clear_cache":  # Clear intermediate results
                    self.intermediate_results.clear()
                    self.result_locations.clear()
                    self.result_socket.send_json({"type": "cache_cleared"})
                    continue
                
                #### Execute the workflow ####
                ### Find the pipeline and load it
                pipeline_name = message["pipeline_name"]

                pipeline = self.pipeline

                ### Pasre the inputs
                inputs = message["inputs"]
                prompt = inputs["prompt"]
                num_inference_steps = inputs["num_inference_steps"]
                guidance_scale = inputs["guidance_scale"]
                sd_generator = torch.manual_seed(inputs["seed"])
                
                ### Run the pipeline
                inference_start_time = time.time()
                self.logger.info(f"Running pipeline {pipeline_name} with for req_id: {message['req_id']}")
                images = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=sd_generator,
                ).images
                assert isinstance(images, list) and all(isinstance(img, Image.Image) for img in images), f"image is not an instance of Image.Image"
                # self.logger.info(f"Completed running pipeline {pipeline_name} with for req_id: {message['req_id']}")
                #### Execute the workflow ####

                #### Serialize the images ####
                img_str_list = []
                for image in images:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str_list.append(img_str)
                inference_end_time = time.time()
                inference_latency = inference_end_time - inference_start_time

                #### Serialize the images ####

                response = {
                    "worker_id": self.worker_id,
                    "req_id": message["req_id"],
                    "status": "completed",
                    "img_str_list": img_str_list,
                    "inference_latency": inference_latency,
                }

                self.result_socket.send_json(response)
                
            except zmq.ZMQError as e:
                if self.running:  # Only log if not shutting down
                    self.logger.error(f"ZMQ Error in worker {self.worker_id}: {str(e)}")
            except KeyboardInterrupt:
                self.logger.info("Received KeyboardInterrupt, initiating shutdown...")
                break
            except Exception as e:
                self.logger.error(f"Error in worker {self.worker_id}: {str(e)}")
                continue

        self.cleanup()

    def _get_gpu_memory_info(self, device: str) -> Dict[str, Any]:
        """Get GPU memory information in bytes"""
        if not torch.cuda.is_available():
            return {"free": 0, "total": 0, "used": 0}

        device_idx = int(device.split(":")[-1])
        torch.cuda.synchronize(device_idx)

        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        reserved_memory = torch.cuda.memory_reserved(device_idx)
        allocated_memory = torch.cuda.memory_allocated(device_idx)
        free_memory = total_memory - reserved_memory

        return {"free": free_memory, "total": total_memory, "used": allocated_memory}

    def _unload_model(self, model_id: str):
        """Unload a model and free its memory"""
        if model_id not in self.models:
            return

        print(f"Unloading model: {model_id}")

        del self.models[model_id]
        torch.cuda.empty_cache()

    def cleanup(self):
        try:
            if self.running:  # Add check to prevent double cleanup
                self.running = False
                self.logger.info(f"Worker {self.worker_id} is stopping")

                """Unload all models and free their memory"""
                for model_id in self.models:
                    self._unload_model(model_id)

                # Cleanup ZMQ resources
                self.task_socket.setsockopt(zmq.LINGER, 1000)  # 1 second timeout
                self.result_socket.setsockopt(zmq.LINGER, 1000)

                self.task_socket.close()
                self.result_socket.close()
                self.context.term()

                # Cleanup NCCL process group explicitly
                if dist.is_initialized():
                    dist.destroy_process_group()

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
    scheduling_baseline: str = "basic",
):
    """Function to run in each worker process"""
    worker = DistributedWorker(local_rank, node_rank, node_config, dist_config, scheduling_baseline)
    worker.run()


class Coordinator:
    def __init__(self, dist_config: DistributedConfig, node_rank: int, scheduling_baseline: str = "basic"):
        self.dist_config = dist_config
        self.node_rank = node_rank
        self.node_config = dist_config.get_node_by_rank(node_rank)
        self.processes: List[mp.Process] = []
        self.scheduling_baseline = scheduling_baseline
        print(f"Coordinator: Scheduling baseline: {self.scheduling_baseline}")

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

        if self.node_rank == 0:
            # Maps node_name -> worker_id for intermediate results
            self.result_locations = {}

        self.active_tasks = {}  # Maps req_id to worker_id
        self.is_running = True

        # keep worker status 
        self.worker_status = {}
        for worker_info in self.all_workers_info:
            self.worker_status[worker_info["worker_id"]] = {
                "status": "idle",
                "pipeline_name": None,
            }

        # Request queue for handling concurrent requests
        # self.request_queue = asyncio.Queue()
        self.request_queue = asyncio.PriorityQueue()
        self.request_futures = {}  # Maps req_id to future
        self.scheduler_task = None

    def _initialize_local_workers(self):
        """Initialize workers for this node only"""
        worker_pids = []
        for local_rank in range(self.node_config.gpu_count):
            p = mp.Process(
                target=run_worker,
                args=(
                    local_rank,
                    self.node_rank,
                    self.node_config,
                    self.dist_config,
                    self.scheduling_baseline,
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

    async def _scheduler_loop(self):
        """Main scheduler loop that processes requests from the queue"""
        """Suyi: the scheduler logic is different from the _scheduler_loop_request"""
        """Suyi: We first get all idle workers, and then schedule requests to them"""
        """Suyi: Each idle worker will take the first request in the queue that can be scheduled to it"""
        """Suyi: Therefore, the execution sequence of the requests is the same as the requests' arrival order"""

        while self.is_running:
            try:
                # Get all idle workerss
                idle_workers = [worker_id for worker_id, status in self.worker_status.items() if status["status"] == "idle"]
                if len(idle_workers) == 0:
                    # print(f"{datetime.now()} No idle workers")
                    await asyncio.sleep(0.1)
                    continue
                
                ##### scheduling logic #####
                # For each idle worker, get the first request that can be scheduled to it
                worker_id = None
                for idle_worker_id in idle_workers:
                    ### FIFO queue
                    # pipeline_name, inputs, req_id = await self.request_queue.get()
                    ### Priority queue
                    queue_item = await self.request_queue.get()
                    ### (timestamp, (pipeline_name, inputs, req_id))
                    queue_item_ts = queue_item[0]
                    # print(f"queue_item_ts: {queue_item_ts}")
                    pipeline_name, inputs, req_id = queue_item[1]

                    
                    for worker_info in self.all_workers_info:
                        if worker_info["worker_id"] == idle_worker_id:
                            worker_id = idle_worker_id
                            break
                    
                    ### Suyi: when we find a worker_id, we break the loop;
                    if worker_id is not None:
                        print(f"Find a worker_id: {worker_id}")
                        break
                    ### Suyi: if no worker_id found, put the request back in the queue
                    else:
                        print(f"No worker_id found for req_id {req_id}")
                        # put the request back in the queue
                        await self.request_queue.put((queue_item_ts, (pipeline_name, inputs, req_id)))
                        await asyncio.sleep(0.1)
                        continue
                
                print(f"{datetime.now()} Scheduling req_id {req_id} to worker {worker_id}")
                ##### scheduling logic #####


                task_message = {
                    "pipeline_name": pipeline_name,
                    "inputs": inputs,
                    "req_id": req_id,
                }

                # Update worker status
                self.worker_status[worker_id]["status"] = "busy"
                self.worker_status[worker_id]["pipeline_name"] = task_message["pipeline_name"]
                self.active_tasks[task_message["req_id"]] = worker_id

                # Schedule task to worker in a non-blocking way
                asyncio.create_task(self._send_task_to_worker(worker_id, task_message))

            except Exception as e:
                logging.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error

    async def _send_task_to_worker(self, worker_id: str, task_message: Dict[str, Any]):
        """Send task to worker asynchronously"""
        try:
            # Send task in a non-blocking way
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.task_sockets[worker_id].send_json(task_message)
            )

            # Start result gathering task
            asyncio.create_task(self._gather_result_for_request(task_message["req_id"], worker_id))

        except Exception as e:
            logging.error(f"Error sending task to worker {worker_id}: {e}")
            traceback.print_exc()
            raise e

    async def _gather_result_for_request(self, req_id: str, worker_id: str):
        """Gather result for a specific request"""
        start_time = time.time()
        max_wait_time = 300  # 5 minutes maximum wait time
        
        try:
            result_socket = self.result_sockets[worker_id]
            poller = zmq.Poller()
            poller.register(result_socket, zmq.POLLIN)

            # # Store for mismatched responses
            # pending_responses = {}

            while True:
                # Check if we've exceeded max wait time
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(f"Request {req_id} timed out after {max_wait_time} seconds")

                # Poll for result in a non-blocking way
                socks = dict(await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: poller.poll(timeout=12000)
                ))
                
                if result_socket in socks:
                    # Receive response in a non-blocking way
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: result_socket.recv_json()
                    )
                    
                    received_req_id = response.get("req_id")
                    print(f"{datetime.now()} Received response for req_id {received_req_id}: {response.get('status', 'unknown')}")
                    
                    if response.get("type") == "pong":
                        continue

                    if received_req_id == req_id:
                        # This is the response we're waiting for
                        self.worker_status[worker_id]["status"] = "idle"
                        self.worker_status[worker_id]["pipeline_name"] = None
                        del self.active_tasks[req_id]

                        # Set result in future
                        if req_id in self.request_futures:
                            self.request_futures[req_id].set_result(response)
                            del self.request_futures[req_id]
                        return
                    else:
                        ### Suyi: if received_req_id is not equal to req_id, raise an error
                        ### reason: there may be consistency issue in send_json of the scheduler_loop
                        raise ValueError(f"Received response for req_id {received_req_id} (waiting for {req_id})")
                
                # No response yet, sleep briefly before next poll
                await asyncio.sleep(0.1)

        except Exception as e:
            logging.error(f"Error gathering result for request {req_id}: {e}")
            if req_id in self.request_futures:
                self.request_futures[req_id].set_exception(e)
                del self.request_futures[req_id]
            # Reset worker status on error
            self.worker_status[worker_id]["status"] = "idle"
            self.worker_status[worker_id]["pipeline_name"] = None
            if req_id in self.active_tasks:
                del self.active_tasks[req_id]

    def execute_workflow(
        self, pipeline_name: str, inputs: Dict[str, Any], req_id: str
    ) -> Dict[str, Any]:
        """Execute workflow by adding request to queue and waiting for result"""
        # Create future for this request
        future = asyncio.Future()
        self.request_futures[req_id] = future

        try:
            # Add request to queue
            # asyncio.create_task(self.request_queue.put((pipeline_name, inputs, req_id)))
            ### Priority queue
            asyncio.create_task(self.request_queue.put((time.time(), (pipeline_name, inputs, req_id))))

            print(f"{datetime.now()} Added request {req_id} to queue")

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

        # Cleanup NCCL process group
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f"Error destroying process group: {e}")


class WorkflowService:
    def __init__(
        self,
        dist_config: DistributedConfig,
        node_rank: int = 0,
        scheduling_baseline: str = "basic",
    ):
        self.dist_config = dist_config
        self.node_rank = node_rank
        self.scheduling_baseline = scheduling_baseline
        self.coordinator = None  # Will be initialized during startup
        self.setup_signal_handlers()

    async def startup(self):
        """Initialize the distributed system on service startup"""
        print(f"Starting workflow service on node {self.node_rank}")

        self.coordinator = Coordinator(self.dist_config, self.node_rank)

        # Wait for all workers to be ready before accepting requests
        await self._wait_for_workers_ready()
        print(f"self._wait_for_workers_ready() completed")

        # Start the scheduler
        await self.coordinator.start_scheduler()
        print(f"self.coordinator.start_scheduler() completed")

        print(f"Workflow service ready on node {self.node_rank}")

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
        req_id = str(uuid.uuid4())
        print(f"{datetime.now()} Handle request_id: {req_id}")
        
        # Run inference asynchronously
        results = await workflow_service.run_inference(service_id, request.inputs, req_id)

        assert results["status"] == "completed", f"Unknown status: {results['status']}"
        return {"status": "success", "results": results}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


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
    parser.add_argument("--config", type=str, default="dist_config.yml")
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--scheduling-baseline", type=str, choices=["basic"], default="basic")
    args = parser.parse_args()

    # Load DistributedConfig from YAML
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)

    # Build NodeConfig objects from the `nodes` list
    node_configs = [NodeConfig(**node_dict) for node_dict in config_data["nodes"]]

    dist_config = DistributedConfig(
        nodes=node_configs, port=config_data.get("port", 29500)
    )

    # Initialize the service
    workflow_service = WorkflowService(
        dist_config=dist_config, node_rank=args.node_rank, scheduling_baseline=args.scheduling_baseline
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
            server.run()
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