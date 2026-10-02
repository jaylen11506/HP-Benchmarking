#!/bin/bash
# SageMaker notebook "on start" hook. Runs as root and must return within 5 minutes,
# so it only fetches the driver from S3 and leaves it running in the background.
# aws/setup.sh fills in the bucket name from .env before uploading this.
set -eu
BUCKET="__HP_BENCH_BUCKET__"
WORK=/home/ec2-user/SageMaker/hp-bench

# If the driver cannot be fetched the start fails, rather than leaving a notebook
# running with nothing to stop it.
sudo -u ec2-user -i bash <<EOF
set -e
mkdir -p "$WORK"
aws s3 cp "s3://$BUCKET/control/agent.sh" "$WORK/agent.sh"
setsid nohup bash "$WORK/agent.sh" "$BUCKET" > "$WORK/agent.log" 2>&1 < /dev/null &
EOF
