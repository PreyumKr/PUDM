#!/bin/bash

# Exit on any error
set -e

# Set environment variables to avoid interactive prompts
export DEBIAN_FRONTEND=noninteractive

# Install basic dependencies, tools, and Python 3.7
apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.7
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.7 get-pip.py && \
    rm get-pip.py

# Set Python 3.7 as the default python3 (optional, depending on system needs)
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Ensure pip is up-to-date for Python 3.7
python3.7 -m pip install --upgrade pip

# Install CUDA-related package (assuming base image has CUDA 11.1)
python3.7 -m pip install nvidia-cudnn-cu11

# Install PyTorch and related packages with CUDA 11.1
python3.7 -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install additional Python dependencies
python3.7 -m pip install open3d termcolor tqdm einops transforms3d==0.3.1
python3.7 -m pip install msgpack-numpy lmdb h5py hydra-core==0.11.3 pytorch-lightning==0.7.1
python3.7 -m pip install scikit-image black usort flake8 matplotlib jupyter imageio fvcore plotly opencv-python
python3.7 -m pip install markdown==3.1.0

# Install pytorch3d using pip from GitHub
python3.7 -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.1"

# Install GCC 8 for CUDA compatibility
apt-get update && apt-get install -y gcc-8 g++-8 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80

# Compile C++ extensions (assuming PUDM-main is mounted)
bash compile.sh

# Verify setup
echo "Setup complete. Verifying environment..."
gcc --version
nvcc --version
python3.7 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3.7 -c "import sys; print(sys.version)"

echo "PUDM environment ready. Run 'cd /app/PUDM-main/pointnet2 && python3.7 train.py' to start training."