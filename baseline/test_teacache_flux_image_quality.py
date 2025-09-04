import json
import os
import re
import subprocess
import sys
import time
import glob

import torch
import yaml
from diffusers.utils import load_image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.edit_config import EditConfig
from teacache_flux import get_pipeline_use_teacache
def parse_data_folder(folder_path):
    """
    Parse a folder containing subfolders with the structure:
    folder/subfolder/data_image_{i}.png, folder/subfolder/data_mask_{i}.png, folder/subfolder/data_prompt_{i}.pt
    
    Returns:
    List of tuples containing (image_path, mask_path, prompt_path) for each valid set
    """
    print(f"Parsing data folder: {folder_path}")
    
    data_sets = []
    
    # Get all immediate subdirectories
    subdirs = [os.path.join(folder_path, d) for d in os.listdir(folder_path) 
               if os.path.isdir(os.path.join(folder_path, d))]
    
    if not subdirs:
        print(f"Warning: No subdirectories found in {folder_path}")
    
    # Process each subfolder
    for subdir in subdirs:
        print(f"Processing subfolder: {subdir}")
        
        # Find all image files in this subfolder
        image_files = glob.glob(os.path.join(subdir, "data_image_*.png"))
        
        for image_file in image_files:
            # Extract the index number from image filename
            match = re.search(r'data_image_(\d+)\.png', os.path.basename(image_file))
            if not match:
                continue
                
            idx = match.group(1)
            mask_file = os.path.join(subdir, f"data_mask_{idx}.png")
            prompt_file = os.path.join(subdir, f"data_prompt_{idx}.pt")
            
            # Check if corresponding mask and prompt files exist
            if os.path.exists(mask_file) and os.path.exists(prompt_file):
                data_sets.append((subdir, image_file, mask_file, prompt_file))
                print(f"Found complete data set for index {idx} in {subdir}")
            else:
                print(f"Warning: Incomplete data set for index {idx} in {subdir}")
    
    print(f"Found {len(data_sets)} complete data sets across all subdirectories")
    return data_sets

def test_quality(
    mask_path,
    image_path,
    prompt_path,
    save_image_path,
    pipeline,
):
    current_datetime = time.strftime("%m%d_%H%M%S")
    source = load_image(image_path)
    height = source.height
    width = source.width
    prompt = torch.load(prompt_path)
    mask = load_image(mask_path)
    generator = torch.Generator("cpu").manual_seed(42)
    image = pipeline(
        prompt=prompt,
        image=source,
        mask_image=mask,
        height=height,
        width=width,
        strength=1.0,
        generator=generator,
        num_inference_steps=20,
        edit_config=None,
    ).images[0]
    image.save(save_image_path)
    

def main():
    # assert len(sys.argv) >= 2, "Usage: python test_quality.py <config_path> <data_folder>"
    project_path = os.getcwd()
    print("Project Path", project_path)

 
    data_folder = '/project/infattllm/xjiangbp/test_image/'
    teacache_flux_folder = '/project/infattllm/xjiangbp/flux_inpainting/teacache_flux_6' #0.6
    if not os.path.exists(teacache_flux_folder):
        # if parent folder not exists, create it,递归
        os.makedirs(teacache_flux_folder, exist_ok=True)

    # create log folder

    env_variables = os.environ.copy()
    
    # Parse data folder to get all image, mask, prompt sets
    data_sets = parse_data_folder(data_folder)
    
    if not data_sets:
        print(f"No valid data sets found in {data_folder}")
        return
    
    # put all result log in one folder
    current_datetime = time.strftime("%m%d_%H%M%S")
    
    test_times = 1
    idx = 0
    
    # Group data sets by subfolder
    subfolder_data = {}
    for sub_folder, image_path, mask_path, prompt_path in data_sets:
        subfolder_name = os.path.basename(sub_folder)
        if subfolder_name not in subfolder_data:
            subfolder_data[subfolder_name] = []
        subfolder_data[subfolder_name].append((image_path, mask_path, prompt_path))
    # Process each subfolder's data sets
    pipeline = get_pipeline_use_teacache()
    for subfolder_name, subfolder_datasets in subfolder_data.items():
        # Create a specific log folder for this subfolder
     
        # os.makedirs(subfolder_log_folder, exist_ok=True)
        print(f"Created log folder for subfolder {subfolder_name}: {teacache_flux_folder}")
        
        subfolder_idx = 0
        for image_path, mask_path, prompt_path in subfolder_datasets:
            print(f"Processing dataset from {subfolder_name}: Image={image_path}, Mask={mask_path}, Prompt={prompt_path}")
            for _ in range(test_times):
                # 创建标准图片文件夹
                standard_img_path = os.path.join(teacache_flux_folder, subfolder_name)
                os.makedirs(standard_img_path, exist_ok=True)
                save_image_path = os.path.join(standard_img_path, os.path.basename(image_path))
                print("save_image_path",save_image_path)
                # if not os.path.exists(save_image_path):
                test_quality(
                    mask_path,
                    image_path,
                    prompt_path,
                    save_image_path,
                    pipeline
                )
            

if __name__ == "__main__":
    main()
