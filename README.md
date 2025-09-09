# Artifact Evaluation for Eurosys 2026
This guide provides step-by-step instructions to reproduce the experiments and results presented in our Flashps paper. Follow these steps to validate our claims regarding **performance improvements** and **image quality preservation**.

To simplify reproducibility, we provide an off-the-shelf Docker image, `jiangxiaoxiao/flashps` which includes all the dependencies and configurations required to run the experiments. This eliminates the need for complex environment setup. We also provided AWS EC2 instances for evaluation. 
We provide two types of EC2 instances, one with **A100 GPUs** and the other with **A10 GPUs**.

## On the machine with A100 GPUs
Please comment on us in the HotCRP with your public key to get the IP address and access the machine.

### Run FlashPS with Docker
We have pulled the image on the provided machine, as its size is nearly 100 GiB.
```bash
# We will add your public key to the machine. You can log in to the machine with your private key by replacing the <IP_ADDRESS> with the actual IP.
ssh ubuntu@<IP_ADDRESS>

# We have pulled the image on the provided machine. You can skip this. On your machine, you can pull the prebuilt Docker image with the following command.
docker pull jiangxiaoxiao/flashps:latest

# Run the following command to spin up the container. This may take a few minutes.
docker run -d --name flashps-ae --runtime=nvidia --gpus all --shm-size=16g \
-e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
-e CONDA_DEFAULT_ENV="" \
-e CONDA_AUTO_ACTIVATE_BASE=false \
jiangxiaoxiao/flashps sleep infinity

docker exec -it flashps-ae zsh

# Activate the environment
conda activate flashps

```

### End-to-end Performance of OOTD
```bash
cd /app/image-inpainting/scheduler/

# Ensure the repo up-to-date
git pull

# Spin up the server to test TeaCache and diffusers baseline. It may take two minutes to start the server.
# When the server successfully starts, it will print
# "INFO:     Uvicorn running on http://0.0.0.0:8005 (Press CTRL+C to quit)"
# on the console.

bash run_server_ootd_no_cb.sh

# Send requests to the server. Note that the first 10 requests are for warm-up purposes.
# For each baseline, we send requests with different RPS. 

# Send requests to evaluate the baseline TeaCache.
bash /app/image-inpainting/scheduler/test_ootd_teacache.sh

# Send requests to evaluate the baseline diffusers.
bash /app/image-inpainting/scheduler/test_ootd_diffusers.sh

# Remember to kill the server.
bash kill_server.sh

# Spin up the server to test FlashPS baseline. It may take two minutes to start the server.
# When the server successfully starts, it will print
# "INFO:     Uvicorn running on http://0.0.0.0:8005 (Press CTRL+C to quit)"
# on the console.

bash run_server_ootd.sh

# Send requests to evaluate FlashPS.
bash /app/image-inpainting/scheduler/test_ootd_flashps.sh

# Remember to kill the server.
bash kill_server.sh

# Analyze and plot the results. The script will print out the path to the figure.
python scripts/parse_end2end.py

```
You may compare the figure with Figure 12 in the paper.

### Image Quality Assessment
Our image quality evaluation ensures that **performance optimizations** do not compromise **output quality**. 
As generating all images takes hours, we have cached them for evaluation.

Evaluate Image Quality. This may take minutes.
```
cd /app/image-inpainting/
bash scripts/test_quality.sh 
```
You may compare the printed results with those in Table 2 in the paper.

## On the machine with A10 GPUs
Please comment on us in the HotCRP with your public key to get the IP address and access the machine.

### End-to-end Performance of SD2
Because SD2's baseline FISEdit is not compatible with advanced GPUs, we have provided a machine with a pre-configured environment to facilitate execution for review purposes.
We will add your public key to the machine. You can log in to the machine with your private key by replacing the <SD2_IP_ADDRESS> with the actual IP.
Please comment us on the HotCRP for the IP address.

```bash
ssh ubuntu@<SD2_IP_ADDRESS>
```


```bash
# Initialize the environment
source activate pytorch

# Go to the project directory
cd /home/ubuntu/image-inpainting/scheduler

# Spin up the server to evaluate FlashPS. It may take two minutes to start the server.
# When the server successfully starts, it will print
# "INFO:     Uvicorn running on http://0.0.0.0:8005 (Press CTRL+C to quit)"
# on the console.
bash run_server_sd2_cb.sh

# Send requests to evaluate FlashPS.
bash scripts/test_cb_sd2.sh

# kill the server
bash scripts/kill_gpu_processes.sh


# Spin up the server to evaluate Diffusers. It may take two minutes to start the server.
# When the server successfully starts, it will print
# "INFO:     Uvicorn running on http://0.0.0.0:8005 (Press CTRL+C to quit)"
# on the console.
bash run_server_sd2_no_cb.sh

# Send requests to evaluate Diffusers.
bash scripts/test_no_cb_sd2.sh

# kill the server
bash scripts/kill_gpu_processes.sh

# activate fisedit environment
conda activate fisedit
source ~/Hetu/hetu.exp

# Spin up the server to evaluate FisEdit. It may take two minutes to start the server.
# When the server successfully starts, it will print
# "INFO:     Uvicorn running on http://0.0.0.0:8005 (Press CTRL+C to quit)"
# on the console.
bash run_server_fisedit_no_cb.sh

# Send requests to evaluate FisEdit.
bash scripts/test_fisedit_e2e.sh

# kill the server
bash scripts/kill_gpu_processes.sh

# analyze and plot the result. The script will print out the path to the figure.
python scripts/parse_end2end.py 

```
You may compare the figure with Figure 12 in the paper.

### Distribution of Mask Ratios

In the same machine, which is used to evaluate SD2. 
```bash
# Go to the trace directory
cd /home/ubuntu/plot_mask_ratio

# Plot the distribution
python3 plot_mask_ratio_2traces.py
```
You may compare the plot /home/ubuntu/plot_mask_ratio/mask_ratio_2traces.pdf with the Figure 3 in the paper.
