import os
import time
import cv2
import torch
import numpy as np
import requests
from tqdm import tqdm
import re
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

input_dir = '../../results/extract_keyframes/vins-fusion/keyframe_images'
output_dir = '../../results/ai_depthestimate'

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

checkpoint_urls = {
    'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true',
    'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true',
    'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true'
}

def check_and_download_checkpoint(encoder):
    checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
    if not os.path.exists(checkpoint_path):
        print(f'Checkpoint for {encoder} not found, downloading...')
        os.makedirs('checkpoints', exist_ok=True)
        url = checkpoint_urls[encoder]
        download_file(url, checkpoint_path)
    return checkpoint_path

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  

    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

encoder = 'vits'  
checkpoint_path = check_and_download_checkpoint(encoder)
model = DepthAnythingV2(**model_configs[encoder])
state_dict = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

cmap = matplotlib.colormaps.get_cmap('Spectral_r')

os.makedirs(output_dir, exist_ok=True)
processed_files = set()

while True:
    files = os.listdir(input_dir)
    new_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg')) and file not in processed_files]

    new_files.sort(key=natural_sort_key)

    for file in new_files:
        processed_files.add(file)
        img_path = os.path.join(input_dir, file)
        print(f'Processing: {img_path}')
        raw_image = cv2.imread(img_path)
        depth = model.infer_image(raw_image)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth_colored = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
        output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '_depth.png')
        cv2.imwrite(output_path, depth_colored)
        print(f'Depth estimation saved to: {output_path}')

    time.sleep(5)
