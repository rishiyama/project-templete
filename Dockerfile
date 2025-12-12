# Base image with CUDA 11.8 and Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set noninteractive environment to avoid timezone prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    tzdata \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Optional libs for cv2 / cairo / ffi, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1-mesa-glx \
    libcairo2 \
    libcairo2-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# uv installation
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory
WORKDIR /workspace

# Default command
CMD ["bash"]