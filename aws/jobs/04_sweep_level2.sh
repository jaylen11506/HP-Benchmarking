#!/bin/bash
# Level 2 sweep: every model in models.json at three context lengths and five batch sizes.
# Configurations already in results_aws.csv for this GPU are skipped, so the job can be
# re-sent after an interruption and will carry on where it stopped.
set -euxo pipefail
cd "$WORK"
aws s3 cp --quiet "s3://$BUCKET/code/harness.tar.gz" .
tar -xzf harness.tar.gz -C repo

CUDA_HOME=/usr/local/cuda-13.0
BIN="$WORK/llama-b11170-sm75"
export PATH="$BIN:$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export BUILD_INFO_PATH="$BIN/BUILD_INFO" MODELS_DIR="$WORK/models" RESULTS_DIR="$WORK/results"

cd repo
../venv/bin/python run_benchmark.py --contexts 1024,4096,16384 --batch-sizes 1,2,4,8,16 --skip-existing \
  --output "$RESULTS_DIR/results_aws.csv" --env-label "AWS SageMaker ml.g4dn.xlarge notebook"
echo JOB_DONE
