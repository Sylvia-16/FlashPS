import os
import glob
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from scipy import linalg
# import tensorflow as tf
import torch

from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

def preprocess_image(img_path):
    """Preprocess the image for inception model"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2.0  # Normalize to [-1, 1]
    return img

# def calculate_fid_single(img1, img2):
#     fid_metric = pyiqa.create_metric('fid')
#     fid_score = fid_metric(img1, img2)
#     return fid_score

def cal_ssim(image1_path, image2_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    height = image1.height
    width = image1.width
    # Convert images to numpy arrays
    array1 = torch.from_numpy(np.array(image1))
    array2 = torch.from_numpy(np.array(image2))
    print(array1.shape, array2.shape)
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 255.0))

    # convert to BxCxHxW
    # gold_images = torch.stack(gold_images, dim=0) * 1.0
    # pred_images = torch.stack(pred_images, dim=0) * 1.0
    array1 = array1.view(1, 3, height, width)
    # add one dimension for numpy array

    array2 = array2.view(1, 3, height, width)
    ssim_score = ssim(array1, array2)

    return ssim_score
def calculate_fid(real_images, fake_images):
   
    # clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    # fid score
    fid_metric = FrechetInceptionDistance(feature=64)
    
    for i in range(len(real_images)):
        # Load and process images
        image1 = Image.open(real_images[i])
        image2 = Image.open(fake_images[i])
        array1 = torch.from_numpy(np.array(image1))
        array2 = torch.from_numpy(np.array(image2))
        
        # Convert to BxCxHxW format
        height, width = array1.shape[:2]
        array1 = array1.view(1, 3, height, width)
        array2 = array2.view(1, 3, height, width)
        
        # Update FID metric
        fid_metric.update(array1, real=True)
        fid_metric.update(array2, real=False)
    
    # Compute final FID score
    final_fid = fid_metric.compute()
    print(f"Overall FID Score: {final_fid}")
def cal_clip(fake_images, prompt_folder):
    """Calculate CLIP score between images and text prompts derived from their paths."""
    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    
    clip_scores = []
    for i in tqdm(range(len(fake_images)), desc="Calculating CLIP scores"):
        # Load image
        fake_image = Image.open(fake_images[i])
        
        # Convert fake image to tensor (CLIP score expects tensor for image)
        fake_tensor = torch.from_numpy(np.array(fake_image)).permute(2, 0, 1)
        
        # Parse path to extract subfolder and filename
        path_parts = fake_images[i].split('/')
        # Extract filename without extension
        filename = path_parts[-1].split('.')[0]
        
        # Extract numeric value from filename (assuming format like "data_image_43")
        import re
        numbers = re.findall(r'\d+', filename)
        image_number = numbers[-1] if numbers else ""  # Take the last number if multiple exist
        
        # Extract immediate subfolder (assuming structure like .../subfolder/filename.png)
        subfolder = path_parts[-2] if len(path_parts) > 1 else ""
        
        print(f"Path: {fake_images[i]}")
        print(f"Subfolder: {subfolder}")
        print(f"Filename: {filename}")
        print(f"Image Number: {image_number}")
        prompt_file_name = f"data_prompt_{image_number}.pt"
        # Construct the prompt path using the image number
        prompt_path = os.path.join(prompt_folder, subfolder, prompt_file_name)
        
        # Try to read the prompt from file if it exists, otherwise use subfolder and filename
        text_prompt = ""
        if os.path.exists(prompt_path):
            text_prompt = torch.load(prompt_path)
        else:
            # Use subfolder and filename as text prompt
            raise ValueError(f"Prompt file not found: {prompt_path}")
        
        print(f"Text Prompt: {text_prompt}")
        
        # Calculate CLIP score (measures how well the image matches the text)
        clip_score = clip_metric(fake_tensor, text_prompt)
        clip_scores.append(clip_score.item())
        
    avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0
    print(f"Average CLIP Score: {avg_clip_score}")
    
    return clip_scores, avg_clip_score
def cal_psnr(image1_path, image2_path):
    """Calculate Peak Signal-to-Noise Ratio between two images using torchmetrics."""
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    height = image1.height
    width = image1.width
    
    # Convert images to tensors
    array1 = torch.from_numpy(np.array(image1))
    array2 = torch.from_numpy(np.array(image2))
    
    # Initialize PSNR metric
    psnr = PeakSignalNoiseRatio(data_range=255.0)
    
    # Convert to BxCxHxW format
    array1 = array1.view(1, 3, height, width)
    array2 = array2.view(1, 3, height, width)
    
    # Calculate PSNR score
    psnr_score = psnr(array1, array2)
    
    return psnr_score
def find_matching_images(dir1, dir2):
    """Find images with the same name in corresponding subfolders of both directories"""
    matching_pairs = []
    
    # Walk through all subdirectories in dir1
    for root, _, files in os.walk(dir1):
        # Get the relative path from dir1
        rel_path = os.path.relpath(root, dir1)
        # Construct the corresponding path in dir2
        corresponding_dir = os.path.join(dir2, rel_path) if rel_path != '.' else dir2
        # Skip if corresponding directory doesn't exist in dir2
        if not os.path.exists(corresponding_dir):
            continue
            
        # Check each file in current dir1 subdirectory
        for file in files:
            # Only consider image files
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file1_path = os.path.join(root, file)
                file2_path = os.path.join(corresponding_dir, file)
                
                # Check if the corresponding file exists in dir2
                if os.path.exists(file2_path):
                    matching_pairs.append((file1_path, file2_path))
    
    return matching_pairs

def main():
    parser = argparse.ArgumentParser(description='Compare image quality using FID score')
    # parser.add_argument('--dir1', type=str, default='/project/infattllm/xjiangbp/flux_inpainting/standard_img_flux/',help='First directory containing images')
    # parser.add_argument('--dir2', type=str, default='/project/infattllm/xjiangbp/flux_inpainting/teacache_flux_6/',help='Second directory containing images')
    # parser.add_argument('--prompt_folder', type=str, default='/project/infattllm/xjiangbp/test_image/')
    parser.add_argument('--dir1', type=str, default='/project/infattllm/xjiangbp/ootd/standard_img_ootd',help='First directory containing images')
    parser.add_argument('--dir2', type=str, default='/project/infattllm/xjiangbp/ootd/use_o/',help='Second directory containing images')
    parser.add_argument('--prompt_folder', type=str, default='/project/infattllm/xjiangbp/test_image/')
    args = parser.parse_args()
    # Find matching images
    matching_pairs = find_matching_images(args.dir1, args.dir2)
    
    if not matching_pairs:
        print("No matching image files found between the two directories.")
        return
    
    print(f"Found {len(matching_pairs)} matching image files.")
    
    # Calculate FID score
    images1 = [pair[0] for pair in matching_pairs]
    images2 = [pair[1] for pair in matching_pairs]
    

    avg_ssim = 0
    avg_psnr = 0
    for i in range(len(images1)):
        ssim_score = cal_ssim(images1[i], images2[i])
        psnr_score = cal_psnr(images1[i], images2[i])
        print(f"SSIM Score: {images1[i]} {images2[i]} {ssim_score}")
        print(f"PSNR Score: {images1[i]} {images2[i]} {psnr_score}")
        avg_ssim+=ssim_score
        avg_psnr+=psnr_score
    avg_ssim = avg_ssim/len(images1)
    avg_psnr = avg_psnr/len(images1)
    print(f"Average SSIM Score: {avg_ssim}")
    print(f"Average PSNR Score: {avg_psnr}")

    # # Calculate FID for individual image pairs
    # # print("Calculating FID for individual image pairs:")
    fid_score = calculate_fid(images1, images2)
    print(f"Average fid {fid_score}")
    # # Calculate FID for the entire dataset
    # print("\nCalculating FID for the entire dataset:")
    # clip_score = cal_clip(images2, args.prompt_folder)

if __name__ == "__main__":
    main()
# FLUX
# Average SSIM Score: 0.880126416683197
# Average FID score: 0.04090195149183273
# Average CLIP score: 30.90982951337134

# TEACACHE FLUX 0.6
# CLIP:30.79946631716009
# Average SSIM Score: 0.805854320526123
# Overall FID Score: 0.05508570373058319

# OOTD
# Average SSIM Score: 0.990889847278595
# Average PSNR Score: 34.81057357788086
# Overall FID Score: 0.009806415997445583



