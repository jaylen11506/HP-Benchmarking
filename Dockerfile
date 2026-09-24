# 1. Base Image with fixed CUDA and OS versions
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Install pinned system dependencies & Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 3. Build llama.cpp (CPU mode for laptop dry run; set GGML_CUDA=ON on NVIDIA target)
WORKDIR /opt
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    git checkout b3500 && \
    cmake -B build -DGGML_CUDA=OFF && \
    cmake --build build --config Release -j$(nproc)

# 4. Install pinned PyTorch & Python benchmark dependencies
WORKDIR /app
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    huggingface_hub==0.23.0 \
    psutil==5.9.8 \
    pynvml==11.5.0

# 5. Copy project files into container
COPY . /app

# Entrypoint to run the benchmark harness script
ENTRYPOINT ["python", "run_benchmark.py"]
