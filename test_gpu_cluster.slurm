#!/bin/bash
#SBATCH --job-name=test_gpu_cv2
#SBATCH --output=test_gpu_cv2.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --partition=gpuai
#SBATCH --qos=gpu-default

module purge
module load conda/23.11-py311
conda activate eyetheia

python3 - <<'EOF'
import torch
import cv2

print("=== Environment Check ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.rand((1000,1000), device='cuda')
    print("Computation result:", torch.sum(x @ x.T).item())

print("\n=== OpenCV Check ===")
print("OpenCV version:", cv2.__version__)

import numpy as np
img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("OpenCV test image shape:", gray.shape)
print("Mean pixel value:", np.mean(gray))
print("==========================")
EOF
