#!/bin/bash
# Launches the one-shot EC2 GPU test (see userdata.sh), watches it, and terminates it.
# Run it in an ordinary terminal and leave it open; it takes about half an hour.
#
#   bash aws/ec2_gpu_test/launch.sh [instance-type] [minutes-to-keep-trying-for-capacity] [test|sweep]
#   bash aws/ec2_gpu_test/launch.sh g7e.2xlarge 20           # four Qwen3-8B configurations
#   bash aws/ec2_gpu_test/launch.sh g7.2xlarge 20 sweep      # all of level 2: five models, 75 configurations
#
# Spending is limited by running time, because billing data arrives hours too late to act on.
# Four things stop the instance, and the first three do not depend on each other:
#   1. this script terminates it when the test ends, when the time limit passes, or when
#      this script itself is interrupted or its terminal is closed;
#   2. the instance shuts itself down (which terminates it) at the time limit. This script
#      does not leave an instance running unless it has seen that timer confirmed armed;
#   3. a CloudWatch alarm terminates it from the AWS side within 15 minutes after the limit;
#   4. the instance shuts itself down as soon as its own script finishes or fails.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
REGION=${AWS_REGION:?set AWS_REGION in .env}
TYPE=${1:-g7e.2xlarge}
WAIT_MINUTES=${2:-0}
MODE=${3:-test}
case "$MODE" in test|sweep) ;; *) echo "third argument must be 'test' or 'sweep'" >&2; exit 1 ;; esac
NAME=hp-bench-gpu-test
CAP_USD=10
mkdir -p results
DONE=results/gpu_test_done   # written when this script finishes, so a waiting session knows
rm -f "$DONE"
LIMIT_MINUTES=90     # the test should take about 30
ARM_MINUTES=${ARM_MINUTES:-12}   # how long to wait for the instance to confirm its own timer
POLL_S=${POLL_S:-30}

# On-demand Linux prices in us-west-2, from the AWS pricing API on 2026-10-02.
case "$TYPE" in
  g7e.2xlarge) PRICE=3.36312 ;;
  g7e.4xlarge) PRICE=3.99816 ;;
  g7.2xlarge)  PRICE=2.52 ;;
  g7.4xlarge)  PRICE=3.04208 ;;
  g6e.xlarge)  PRICE=1.861 ;;
  g6e.2xlarge) PRICE=2.24208 ;;
  *) echo "No price recorded for $TYPE, so no safe time limit can be set. Add it to this script first." >&2; exit 1 ;;
esac
MAX_MINUTES=$(python3 -c "print(min($LIMIT_MINUTES, int(8.40 / $PRICE * 60)))")
PERIODS=$(( (MAX_MINUTES + 4) / 5 + 2 ))   # alarm window in 5-minute datapoints
WORST=$(python3 -c "print(f'{($PERIODS * 5 + 5) / 60 * $PRICE:.2f}')")
echo "Mode: $MODE. $TYPE at \$$PRICE/hour. Time limit $MAX_MINUTES min (\$$(python3 -c "print(f'{$MAX_MINUTES / 60 * $PRICE:.2f}')"))."
echo "If every stop but the alarm failed: $((PERIODS * 5 + 5)) min, \$$WORST. Cap \$$CAP_USD."
python3 -c "import sys; sys.exit(0 if $WORST < $CAP_USD else 1)" || { echo "worst case is not under the cap; not launching" >&2; exit 1; }

# One test at a time, so one alarm and one watcher always cover the only instance.
RUNNING=$(aws ec2 describe-instances --region "$REGION" --filters Name=tag:Name,Values="$NAME" \
  Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
[ -z "$RUNNING" ] || { echo "a test instance already exists ($RUNNING); terminate it before starting another" >&2; exit 1; }

# The alarm's terminate action is carried out through this AWS-managed role; make sure it exists.
aws iam get-role --role-name AWSServiceRoleForCloudWatchEvents >/dev/null 2>&1 \
  || aws iam create-service-linked-role --aws-service-name events.amazonaws.com >/dev/null \
  || echo "note: could not create the role up front; creating the alarm normally creates it"

USERDATA=$(mktemp)
sed -e "s/^MAX_MINUTES=.*/MAX_MINUTES=$MAX_MINUTES/" -e "s/__INSTANCE_TYPE__/$TYPE/g" -e "s/__MODE__/$MODE/" aws/ec2_gpu_test/userdata.sh > "$USERDATA"
AMI=$(aws ssm get-parameter --region "$REGION" --query Parameter.Value --output text \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id)
ROOT_DEV=$(aws ec2 describe-images --region "$REGION" --image-ids "$AMI" --query 'Images[0].RootDeviceName' --output text)
ZONES=$(aws ec2 describe-instance-type-offerings --region "$REGION" --location-type availability-zone \
  --filters Name=instance-type,Values="$TYPE" --query 'InstanceTypeOfferings[].Location' --output text)
[ -n "$ZONES" ] || { echo "$TYPE is not offered in $REGION" >&2; exit 1; }

# Try each availability zone, moving on when one is out of capacity. Nothing is billed until one succeeds.
ID=""
ERR=$(mktemp)
deadline=$(( $(date +%s) + WAIT_MINUTES * 60 ))
while :; do
  for zone in $ZONES; do
    SUBNET=$(aws ec2 describe-subnets --region "$REGION" --filters Name=default-for-az,Values=true \
      Name=availability-zone,Values="$zone" --query 'Subnets[0].SubnetId' --output text)
    if [ -z "$SUBNET" ] || [ "$SUBNET" = None ]; then continue; fi
    if ID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --subnet-id "$SUBNET" \
        --associate-public-ip-address \
        --instance-initiated-shutdown-behavior terminate \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --block-device-mappings "DeviceName=$ROOT_DEV,Ebs={VolumeSize=120,VolumeType=gp3,DeleteOnTermination=true}" \
        --user-data "file://$USERDATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=hp-bench}]" \
                             "ResourceType=volume,Tags=[{Key=Project,Value=hp-bench}]" \
        --query 'Instances[0].InstanceId' --output text 2>"$ERR"); then
      break 2
    fi
    ID=""
    if ! grep -q InsufficientInstanceCapacity "$ERR"; then
      cat "$ERR" >&2   # quota or permission errors will not improve in another zone
      exit 1
    fi
  done
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "no zone in $REGION has $TYPE capacity right now; nothing was launched" >&2
    echo "launched=no reason=no-capacity type=$TYPE" > "$DONE"
    exit 1
  fi
  echo "  no $TYPE capacity in any zone at $(date +%H:%M:%S); trying again in a minute"
  sleep 60
done
T0=$(date +%s)
echo "launched $ID ($TYPE) in $zone at $(date -u +%H:%M:%SZ)"

# From here on, however this script ends, the instance is terminated and that is confirmed.
ALARM="$NAME-$ID"
LOG=results/gpu_test_console_$ID.log
cleanup() {
  trap - EXIT
  echo "terminating $ID"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$ID" >/dev/null || true
  if aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$ID"; then
    minutes=$(( ($(date +%s) - T0 + 59) / 60 ))
    echo "CONFIRMED terminated. Ran about $minutes min: roughly \$$(python3 -c "print(f'{$minutes / 60 * $PRICE:.2f}')") at \$$PRICE/hour."
    aws cloudwatch delete-alarms --region "$REGION" --alarm-names "$ALARM" || true
    echo "terminated=confirmed mode=$MODE id=$ID type=$TYPE minutes=$minutes price=$PRICE log=$LOG" > "$DONE"
  else
    echo "COULD NOT CONFIRM termination of $ID. Check it now:" >&2
    echo "  aws ec2 terminate-instances --region $REGION --instance-ids $ID" >&2
    echo "terminated=UNCONFIRMED id=$ID type=$TYPE log=$LOG" > "$DONE"
  fi
  echo "console log saved to $LOG"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

BEHAVIOUR=$(aws ec2 describe-instance-attribute --region "$REGION" --instance-id "$ID" \
  --attribute instanceInitiatedShutdownBehavior --query InstanceInitiatedShutdownBehavior.Value --output text)
[ "$BEHAVIOUR" = terminate ] || { echo "shutdown behaviour is '$BEHAVIOUR', not 'terminate'" >&2; exit 1; }

# AWS-side stop: once all but two of the last $PERIODS five-minute datapoints exist, terminate.
aws cloudwatch put-metric-alarm --region "$REGION" --alarm-name "$ALARM" \
  --alarm-description "Terminate $ID if it is still running after about $((PERIODS * 5)) minutes" \
  --namespace AWS/EC2 --metric-name CPUUtilization --dimensions Name=InstanceId,Value="$ID" \
  --statistic Maximum --period 300 --evaluation-periods "$PERIODS" --datapoints-to-alarm "$((PERIODS - 2))" \
  --threshold 0 --comparison-operator GreaterThanOrEqualToThreshold --treat-missing-data notBreaching \
  --alarm-actions "arn:aws:automate:$REGION:ec2:terminate"
echo "alarm $ALARM set. Watching; leave this window open. Closing it or pressing Ctrl-C terminates the instance."

mkdir -p results
armed=""
shown=0
while :; do
  sleep "$POLL_S"
  elapsed=$(( $(date +%s) - T0 ))
  # A failed status call (a network blip) is not a reason to kill a healthy test; the limits below still apply.
  state=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null) || state=unknown
  aws ec2 get-console-output --region "$REGION" --instance-id "$ID" --latest --output text > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
  if [ -f "$LOG" ]; then
    total=$(grep -c 'hpbench' "$LOG" || true)
    if [ "$total" -gt "$shown" ]; then grep 'hpbench' "$LOG" | tail -n "$((total - shown))" | cut -c1-200; shown=$total; fi
    if [ -z "$armed" ] && grep -q HPBENCH_TIMER_ARMED "$LOG"; then
      armed=1
      echo ">> the instance's own shutdown timer is confirmed armed ($((elapsed / 60)) min after launch)"
    fi
    if grep -q HPBENCH_END "$LOG"; then echo ">> test finished"; exit 0; fi
  fi
  case "$state" in pending|running|unknown) ;; *) echo ">> instance is $state"; exit 0 ;; esac
  if [ -z "$armed" ] && [ "$elapsed" -gt $((ARM_MINUTES * 60)) ]; then
    echo ">> no confirmation of the instance's timer after $ARM_MINUTES minutes; stopping the test" >&2
    exit 1
  fi
  if [ "$elapsed" -gt $((MAX_MINUTES * 60)) ]; then echo ">> time limit reached" >&2; exit 1; fi
done
