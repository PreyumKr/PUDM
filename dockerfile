# Use an official NVIDIA CUDA base image with Ubuntu 20.04 (CUDA 11.1 compatible)
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-11.1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.1"

# Install basic utilities (including git)
RUN apt update && apt install -y \
    wget \
    software-properties-common \
    git \
    libgl1-mesa-glx \
    libglu1-mesa \
    libx11-6 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install CUB
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/NVIDIA/cub.git /opt/cub-1.9.10 && \
    cd /opt/cub-1.9.10 && \
    git checkout 1.9.10 && \
    rm -rf .git
ENV CUB_HOME=/opt/cub-1.9.10

# Add deadsnakes PPA to get Python 3.7 (Ubuntu 20.04 defaults to 3.8)
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y \
    python3.7 \
    python3.7-dev \
    python3-pip \
    python3.7-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.7 as the default 'python' and install pip for it
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 && \
    wget https://bootstrap.pypa.io/pip/3.7/get-pip.py -O /tmp/get-pip.py && \
    python3.7 /tmp/get-pip.py && \
    rm /tmp/get-pip.py && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

# Install CUDA-related packages
RUN python -m pip install nvidia-cudnn-cu11

# Install PyTorch and related packages with CUDA 11.1
RUN python -m pip install \
    torch==1.9.1+cu111 \
    torchvision==0.10.1+cu111 \
    torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install additional Python dependencies
RUN python -m pip install \
    open3d \
    termcolor \
    tqdm \
    einops \
    transforms3d==0.3.1 \
    msgpack-numpy \
    lmdb \
    h5py \
    hydra-core==0.11.3 \
    pytorch-lightning==0.7.1 \
    scikit-image \
    black \
    usort \
    flake8 \
    matplotlib \
    jupyter \
    imageio \
    fvcore \
    plotly \
    opencv-python \
    markdown==3.1.0 \
    chainer==7.8.1 \
    cupy-cuda111==8.1.0

# Install cuDNN for CUDA 11.1 using CuPy's tool
RUN python -m cupyx.tools.install_library --library cudnn --cuda 11.1
    
# Install GCC 8 for CUDA compatibility (default GCC in Ubuntu 20.04 is 9.3)
RUN apt update && apt install -y \
    gcc-8 \
    g++-8 \
    ninja-build \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80 \
    && rm -rf /var/lib/apt/lists/*

# Clone and build pytorch3d v0.6.1
RUN git clone https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && \
    git checkout v0.6.1 && \
    MAX_JOBS=2 python setup.py install --verbose