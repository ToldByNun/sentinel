# Sentinel C++/CUDA LM framework — Cloud Agent base image.
#
# Ubuntu 24.04 + CUDA 13.0 toolkit (nvcc, cudart, cuBLAS) + GNU toolchain,
# Ninja and Python dev headers. The CUDA install mirrors the repo CI
# (.github/workflows/c-cpp.yml). GPU/driver are provided by the host at run
# time; when absent the library falls back to an OpenMP host training path.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Base build toolchain, OpenMP, Python dev headers, and fetch tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        ninja-build \
        libomp-dev \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        git \
        curl \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CUDA 13.0 toolkit: compiler, runtime dev headers, and cuBLAS dev libraries.
# Only the pieces needed to compile and link the engine (no full CUDA install).
RUN cd /tmp \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-nvcc-13-0 \
        cuda-cudart-dev-13-0 \
        libcublas-dev-13-0 \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring_1.1-1_all.deb \
    && ln -sfn /usr/local/cuda-13.0 /usr/local/cuda \
    && echo "/usr/local/cuda-13.0/lib64" > /etc/ld.so.conf.d/cuda-13.conf \
    && ldconfig

# Make the CUDA toolchain discoverable for all shells and build steps.
ENV CUDA_HOME=/usr/local/cuda-13.0 \
    CUDA_PATH=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:${PATH}

# Ubuntu 24.04 ships a default non-root `ubuntu` user (uid 1000).
USER ubuntu
