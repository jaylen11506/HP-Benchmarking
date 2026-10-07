#!/bin/bash
# Builds llama.cpp at the pinned tag (once; the binary is kept under ~/SageMaker), then
# re-runs the team's four Qwen3-8B T4 configurations so our numbers can be compared with theirs.
set -euxo pipefail
LLAMA_CPP_TAG=b11170  # must match the Dockerfile
CUDA_ARCH=75          # T4
# The notebook ships several CUDA toolkits; 13.0 is the one the Dockerfile's base image uses.
CUDA_HOME=/usr/local/cuda-13.0

cd "$WORK"
aws s3 cp --quiet "s3://$BUCKET/code/harness.tar.gz" .
mkdir -p repo && tar -xzf harness.tar.gz -C repo

export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

BIN="$WORK/llama-$LLAMA_CPP_TAG-sm$CUDA_ARCH"
if [ ! -x "$BIN/llama-server" ]; then
  nvcc --version
  rm -rf llama.cpp
  git clone --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp.git
  cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
  cmake --build llama.cpp/build -j"$(nproc)" --target llama-server
  mkdir -p "$BIN" && cp llama.cpp/build/bin/llama-server "$BIN/"
  echo "llama.cpp=$LLAMA_CPP_TAG GGML_CUDA=ON CUDA_ARCH=$CUDA_ARCH" > "$BIN/BUILD_INFO"
fi

# Only the packages run_benchmark.py imports, at the versions pinned in requirements-bench.txt.
if [ ! -x venv/bin/python ]; then
  for py in /home/ec2-user/anaconda3/envs/python3/bin/python /home/ec2-user/anaconda3/bin/python python3.12 python3.11 python3; do
    if "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      "$py" -m venv venv
      break
    fi
  done
  venv/bin/pip install -q $(grep -oE '^(psutil|nvidia-ml-py|huggingface_hub|requests|aiohttp|jsonschema)==[^ ]+' repo/requirements-bench.txt)
fi

export PATH="$BIN:$PATH" BUILD_INFO_PATH="$BIN/BUILD_INFO" MODELS_DIR="$WORK/models" RESULTS_DIR="$WORK/results"
OUT="$RESULTS_DIR/results_aws.csv"
LABEL="AWS SageMaker ml.g4dn.xlarge notebook"
cd repo
../venv/bin/python run_benchmark.py --only Qwen3-8B --contexts 1024 --batch-sizes 1,2,4 --output "$OUT" --env-label "$LABEL"
../venv/bin/python run_benchmark.py --only Qwen3-8B --contexts 4096 --batch-sizes 1 --output "$OUT" --env-label "$LABEL"
echo JOB_DONE
