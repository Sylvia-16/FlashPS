# Artifact Evaluation for Eurosys 2026
This guide provides step-by-step instructions to reproduce the experiments and results presented in our Flashps paper. Follow these steps to validate our claims regarding **performance improvements** and **image quality preservation**.

To simplify reproducibility, we provide an off-the-shelf Docker image, `jiangxiaoxiao/flashps` which includes all the dependencies and configurations required to run the experiments. This eliminates the need for complex environment setup. You can pull the image from [Docker Hub](https://hub.docker.com/r/jiangxiaoxiao/flashps) and use it as follows.
## Run Flashps with Docker
We have pulled the image on the provided machine, as its size is nearly 100 GiB.
```bash
 # We have pulled the image on the provided machine. You can skip this. On your machine, you can pull the prebuilt Docker image with the following command.
docker pull jiangxiaoxiao/flashps:latest

docker run -it --rm --name flashps-ae --runtime=nvidia --gpus all --shm-size=16g \
-e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
jiangxiaoxiao/flashps bash

conda activate flashps
```
## Image Quality Assessment
Our image quality evaluation ensures that **performance optimizations** do not compromise **output quality**. 

Evaluate Image Quality:
```
bash scripts/test_quality.sh 
```
## End-to-end Performance of OOTD
```bash
cd scheduler/
# run server to test teacache and diffusers baseline
bash run_server_ootd_no_cb.sh 
bash /app/image-inpainting/scheduler/test_ootd_teacache.sh
bash /app/image-inpainting/scheduler/test_ootd_diffusers.sh
# kill the server
bash kill_server.sh

# run server to test flashps baseline
bash run_server_ootd.sh
bash /app/image-inpainting/scheduler/test_ootd_flashps.sh
# analyze the result 
python scripts/parse_end2end.py 
```
When the server starts successfully, it will output information like "INFO: Uvicorn running on http://0.0.0.0:8005 (Press CTRL+C to quit)" in the log. Please wait to see this message before starting the client.
## End-to-end Performance of SD2
Because SD2's baseline FISEdit is not compatible with advanced GPUs, we have provided a machine with a pre-configured environment to facilitate execution for review purposes.
```bash

source activate pytorch
cd image-inpainting/scheduler

# run flashps server
bash run_server_sd2_cb.sh
bash scripts/test_cb_sd2.sh
# kill the server
bash scripts/kill_gpu_processes.sh

# run server for testing diffusers
bash run_server_sd2_no_cb.sh
# test diffusers
bash scripts/test_no_cb_sd2.sh
bash scripts/kill_gpu_processes.sh

# activate fisedit environment
conda activate fisedit
source ~/Hetu/hetu.exp

# run server for testing fisedit
bash run_server_fisedit_no_cb.sh
bash scripts/test_fisedit_e2e.sh
# analyze the result
python scripts/parse_end2end.py 

```
