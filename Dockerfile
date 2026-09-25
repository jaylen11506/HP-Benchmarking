# Benchmark image: llama.cpp + lightweight Python harness. No PyTorch.
# One file for every machine; only the build args change.
#
#   Nano:          docker build -f docker/bench.Dockerfile -t nano-bench:gb10 .
#   RTX 4090 box:  docker build -f docker/bench.Dockerfile --build-arg CUDA_ARCH=89 -t nano-bench:4090 .
#   Laptop (CPU):  docker build -f docker/bench.Dockerfile --build-arg GGML_CUDA=OFF -t nano-bench:cpu .

# CUDA must be new enough for Blackwell (GB10) and not newer than the Nano's driver.
# Confirm this tag exists and matches `nvidia-smi` on the Nano. Once chosen, pin by
# digest (image@sha256:...) so the base can never silently change underneath you.
ARG BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# GPU architecture to compile for. GB10 should report compute capability 12.1 -> "121".
# Verify on the Nano: nvidia-smi --query-gpu=compute_cap --format=csv
# Baselines: RTX 3090 = 86, RTX 4090 = 89, A100 = 80, H100 = 90.
ARG GGML_CUDA=ON
ARG CUDA_ARCH=121
ARG LLAMA_CPP_TAG=b11170

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        git cmake build-essential curl ca-certificates \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp at an exact release. The linker flag lets the CUDA build finish
# inside `docker build`, where no GPU driver is present (same trick llama.cpp's own
# Dockerfiles use). BUILD_INFO records how this binary was made so the harness can
# write it into the runtime_version column of every result row.
WORKDIR /opt
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=${GGML_CUDA} \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build -j"$(nproc)" && \
    echo "llama.cpp=${LLAMA_CPP_TAG} GGML_CUDA=${GGML_CUDA} CUDA_ARCH=${CUDA_ARCH}" > /opt/BUILD_INFO
ENV PATH="/opt/llama.cpp/build/bin:${PATH}"

# Python tools go in a virtual environment (required on Ubuntu 24.04).
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app
COPY requirements-bench.txt .
RUN pip install --no-cache-dir -r requirements-bench.txt

COPY . /app

# Models and results live OUTSIDE the image and are mounted at run time:
#   /models   (read-only GGUF files)
#   /app/results  (CSV output, same folder on every machine)
ENTRYPOINT ["python", "run_benchmark.py"]
