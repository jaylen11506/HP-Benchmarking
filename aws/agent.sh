#!/bin/bash
# Driver that runs on the SageMaker notebook for as long as it is up.
#
#   1. Runs s3://BUCKET/control/job.sh whenever a new version of it appears.
#   2. Copies logs, results and a heartbeat back to S3 every POLL_S seconds.
#   3. Stops the notebook when nothing has happened for IDLE_LIMIT_S, or after
#      MAX_UPTIME_S no matter what, so a forgotten notebook cannot keep billing.
#
# Everything lives under ~/SageMaker, the only folder that survives a stop/start.

BUCKET="$1"
WORK=/home/ec2-user/SageMaker/hp-bench
IDLE_LIMIT_S=$((45 * 60))
MAX_UPTIME_S=$((8 * 3600))
POLL_S=60

meta() { python3 -c "import json; d = json.load(open('/opt/ml/metadata/resource-metadata.json')); print($1)"; }
NAME=$(meta "d['ResourceName']")
REGION=$(meta "d['ResourceArn'].split(':')[3]")
export BUCKET WORK AWS_DEFAULT_REGION="$REGION"

mkdir -p "$WORK/logs"
boot=$(date +%s)
last_active=$boot
job_pid=""
job_name=""

job_running() { [ -n "$job_pid" ] && kill -0 "$job_pid" 2>/dev/null; }

# Seconds since the last Jupyter kernel or terminal activity; empty if it cannot be read.
jupyter_idle_s() {
  python3 - <<'PY' 2>/dev/null
import json, ssl, urllib.request
from datetime import datetime, timezone

ctx = ssl._create_unverified_context()
newest = None
for path in ("sessions", "terminals"):
    with urllib.request.urlopen(f"https://localhost:8443/api/{path}", context=ctx, timeout=5) as r:
        for item in json.load(r):
            kernel = item.get("kernel", item)
            if kernel.get("execution_state") == "busy":
                print(0)
                raise SystemExit
            stamp = kernel.get("last_activity")
            if stamp:
                t = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                newest = t if newest is None or t > newest else newest
if newest is not None:
    print(int((datetime.now(timezone.utc) - newest).total_seconds()))
PY
}

push() {
  local state="$1" now gpu
  now=$(date +%s)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null | head -1)
  cat > "$WORK/heartbeat.json" <<EOF
{"time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "notebook": "$NAME", "state": "$state", "uptime_s": $((now - boot)), "idle_s": $((now - last_active)), "job": "$job_name", "job_running": $(job_running && echo true || echo false), "gpu": "$gpu"}
EOF
  aws s3 cp --quiet "$WORK/heartbeat.json" "s3://$BUCKET/status/heartbeat.json"
  aws s3 cp --quiet "$WORK/agent.log" "s3://$BUCKET/logs/agent.log"
  aws s3 sync --quiet "$WORK/logs" "s3://$BUCKET/logs"
  [ -d "$WORK/results" ] && aws s3 sync --quiet "$WORK/results" "s3://$BUCKET/results"
}

stop_notebook() {
  echo "$(date -u +%FT%TZ) stopping notebook: $1"
  push "stopping: $1"
  aws sagemaker stop-notebook-instance --notebook-instance-name "$NAME"
  sleep 900  # the instance goes down underneath us
}

echo "$(date -u +%FT%TZ) agent started on $NAME ($REGION), bucket $BUCKET"
while true; do
  now=$(date +%s)

  if ! job_running; then
    etag=$(aws s3api head-object --bucket "$BUCKET" --key control/job.sh --query ETag --output text 2>/dev/null)
    if [ -n "$etag" ] && [ "$etag" != "$(cat "$WORK/.last_job_etag" 2>/dev/null)" ]; then
      aws s3 cp --quiet "s3://$BUCKET/control/job.sh" "$WORK/job.sh"
      echo "$etag" > "$WORK/.last_job_etag"
      job_name="job-$(date -u +%Y%m%dT%H%M%SZ)"
      echo "$(date -u +%FT%TZ) starting $job_name"
      ( bash "$WORK/job.sh" > "$WORK/logs/$job_name.log" 2>&1; echo $? > "$WORK/logs/$job_name.exit" ) &
      job_pid=$!
    fi
  fi

  idle=$(jupyter_idle_s)
  if job_running; then
    last_active=$now
  elif [ -n "$idle" ] && [ $((now - idle)) -gt "$last_active" ]; then
    last_active=$((now - idle))
  fi

  if [ $((now - boot)) -gt "$MAX_UPTIME_S" ]; then
    stop_notebook "up for more than $((MAX_UPTIME_S / 3600)) hours"
  elif [ $((now - last_active)) -gt "$IDLE_LIMIT_S" ]; then
    stop_notebook "idle for more than $((IDLE_LIMIT_S / 60)) minutes"
  fi

  push running
  sleep "$POLL_S"
done
