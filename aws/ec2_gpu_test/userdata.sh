#!/bin/bash
# One-shot test on an EC2 GPU instance: build llama.cpp, run the four Qwen3-8B reference
# configurations, print the rows to the serial console, and terminate.
#
# The instance needs no IAM role, no S3 and no SSH. Everything it downloads is public,
# and results are read back from outside with `aws ec2 get-console-output`.
#
# Cost guard: the instance is launched with shutdown behaviour "terminate", and the very
# first thing this script does is schedule a shutdown, so it cannot outlive MAX_MINUTES
# whatever happens below. launch.sh adds a CloudWatch alarm as a second, AWS-side stop.
MAX_MINUTES=150   # launch.sh replaces this with the limit worked out from the instance's price
shutdown -h +"$MAX_MINUTES"

exec > >(tee /var/log/hpbench.log | logger -t hpbench -s 2>/dev/console) 2>&1

# launch.sh waits for this line. If it never appears, launch.sh terminates the instance.
if [ -f /run/systemd/shutdown/scheduled ]; then
  echo "HPBENCH_TIMER_ARMED minutes=$MAX_MINUTES"
else
  echo "HPBENCH_TIMER_FAILED"
  shutdown -h now
  exit 1
fi

finish() {
  code=$?
  set +e
  if [ "$code" -ne 0 ]; then
    echo "HPBENCH_STATUS=failed exit=$code"
    tail -n 40 /var/log/hpbench-build.log 2>/dev/null
  else
    echo "HPBENCH_STATUS=ok"
  fi
  echo "HPBENCH_END $(date -u +%FT%TZ) uptime_s=$(cut -d. -f1 /proc/uptime)"
  sleep 300   # leave time for the console output to be read ($0.28)
  shutdown -h now
}
trap finish EXIT
set -euo pipefail

LLAMA_CPP_TAG=b11170   # must match the Dockerfile
HARNESS_BRANCH=benchmark-harness
export HOME=/root DEBIAN_FRONTEND=noninteractive

echo "HPBENCH_START $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')

missing=""
for tool in git cmake g++; do command -v "$tool" >/dev/null || missing=1; done
python3 -c 'import ensurepip' 2>/dev/null || missing=1
if [ -n "$missing" ]; then
  apt-get -o DPkg::Lock::Timeout=300 update -qq
  apt-get -o DPkg::Lock::Timeout=300 install -y -qq git cmake build-essential python3-venv >/dev/null
fi

# Same toolkit as the T4 run if the image has it, otherwise the newest one installed.
CUDA_HOME=/usr/local/cuda-13.0
[ -x "$CUDA_HOME/bin/nvcc" ] || CUDA_HOME=$(ls -d /usr/local/cuda-1[2-9].* | sort -V | tail -1)
export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME=$CUDA_HOME CUDA_ARCH=$CUDA_ARCH vcpus=$(nproc)"

cd /opt
git clone -q --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp.git
{
  cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
  cmake --build llama.cpp/build -j"$(nproc)" --target llama-server
} > /var/log/hpbench-build.log 2>&1
echo "HPBENCH_BUILD_DONE $(date -u +%T) uptime_s=$(cut -d. -f1 /proc/uptime)"

git clone -q --depth 1 --branch "$HARNESS_BRANCH" https://github.com/jaylen11506/HP-Benchmarking.git repo
echo "harness commit $(git -C repo rev-parse --short HEAD)"
python3 -m venv venv
venv/bin/pip install -q psutil==7.2.2 nvidia-ml-py==13.610.43 huggingface_hub==1.33.0 requests==2.32.4 aiohttp==3.14.3 jsonschema==4.26.0

echo "llama.cpp=$LLAMA_CPP_TAG GGML_CUDA=ON CUDA_ARCH=$CUDA_ARCH" > /opt/BUILD_INFO
export PATH="/opt/llama.cpp/build/bin:$PATH" BUILD_INFO_PATH=/opt/BUILD_INFO MODELS_DIR=/opt/models RESULTS_DIR=/opt/results
OUT=/opt/results/results_ec2.csv
cd repo
MODE=__MODE__   # launch.sh fills in "test" or "sweep"
if [ "$MODE" = sweep ]; then
  # Level 2: every model in models.json, three context lengths, five batch sizes.
  /opt/venv/bin/python run_benchmark.py --contexts 1024,4096,16384 --batch-sizes 1,2,4,8,16 --output "$OUT" --env-label "AWS EC2 __INSTANCE_TYPE__"
else
  # The four Qwen3-8B configurations that the T4 and Colab rows also have.
  /opt/venv/bin/python run_benchmark.py --only Qwen3-8B --contexts 1024 --batch-sizes 1,2,4 --output "$OUT" --env-label "AWS EC2 __INSTANCE_TYPE__"
  /opt/venv/bin/python run_benchmark.py --only Qwen3-8B --contexts 4096 --batch-sizes 1 --output "$OUT" --env-label "AWS EC2 __INSTANCE_TYPE__"
fi

echo HPBENCH_RESULT_BEGIN
gzip -c "$OUT" | base64 -w 100
echo HPBENCH_RESULT_END
