#!/bin/bash
# Records what the notebook instance actually provides, before anything is built on it.
set -x
cat /etc/os-release | head -3
nproc; free -g | head -2
df -h / /home/ec2-user/SageMaker
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
which nvcc cmake gcc g++ git docker
nvcc --version
ls -d /usr/local/cuda* /opt/cuda* 2>/dev/null
ls /usr/local/cuda/lib64 2>/dev/null | grep -E 'libcublas|libcudart' | head
gcc --version | head -1
cmake --version | head -1
python3 --version
ls /home/ec2-user/anaconda3/envs 2>/dev/null
for p in /home/ec2-user/anaconda3/envs/*/bin/python; do echo "$p $($p --version 2>&1)"; done
docker info 2>/dev/null | grep -iE 'runtimes|root dir|server version'
curl -sI --max-time 10 https://huggingface.co | head -1
echo PROBE_DONE
