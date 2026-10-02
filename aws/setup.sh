#!/bin/bash
# One-time AWS setup for running the benchmark on a SageMaker GPU notebook.
#
# Reads AWS_ACCOUNT_ID, AWS_REGION and HP_BENCH_BUCKET from .env (copy .env.example), then creates:
#   - a private S3 bucket for code, logs and results
#   - an IAM role the notebook runs as (aws/iam/*.json: that bucket, its own logs, stopping itself)
#   - a lifecycle config that starts aws/agent.sh on every boot
#   - a request to raise the EC2 GPU quota to 16 vCPUs
#   - one ml.g4dn.xlarge notebook (T4), billed at $0.736/hour while it is running
#
# The notebook stops itself after 45 idle minutes or 8 hours (see aws/agent.sh).
# Safe to re-run: anything that already exists is left alone.
#
#   bash aws/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "No .env found. Copy .env.example to .env and fill in your AWS account ID and bucket name." >&2
  exit 1
fi
set -a; . ./.env; set +a
: "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID in .env}" "${AWS_REGION:?set AWS_REGION in .env}" "${HP_BENCH_BUCKET:?set HP_BENCH_BUCKET in .env}"

REGION=$AWS_REGION
BUCKET=$HP_BENCH_BUCKET
ROLE=hp-bench-notebook-role
LIFECYCLE=hp-bench-onstart
NOTEBOOK=hp-bench-t4

# The files under aws/ carry placeholders instead of account details; this fills them in.
render() {
  sed -e "s/__AWS_ACCOUNT_ID__/$AWS_ACCOUNT_ID/g" -e "s/__AWS_REGION__/$AWS_REGION/g" \
      -e "s/__HP_BENCH_BUCKET__/$HP_BENCH_BUCKET/g" "$1"
}
if [ "${1:-}" = "--render" ]; then  # print one rendered file and stop; changes nothing in AWS
  render "$2"
  exit 0
fi

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
if [ "$ACCOUNT" != "$AWS_ACCOUNT_ID" ]; then
  echo ".env says account $AWS_ACCOUNT_ID, but the AWS CLI is signed in to $ACCOUNT." >&2
  exit 1
fi

echo "1/6 bucket $BUCKET"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

echo "2/6 role $ROLE"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document "$(render aws/iam/trust.json)" \
    --tags Key=Project,Value=hp-bench >/dev/null
  NEW_ROLE=1
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name hp-bench-notebook --policy-document "$(render aws/iam/policy.json)"

echo "3/6 upload harness, driver and first job"
BUNDLE=$(mktemp -d)/harness.tar.gz
COPYFILE_DISABLE=1 tar -czf "$BUNDLE" run_benchmark.py models.json results_schema.json requirements-bench.txt
aws s3 cp --quiet "$BUNDLE" "s3://$BUCKET/code/harness.tar.gz"
aws s3 cp --quiet aws/agent.sh "s3://$BUCKET/control/agent.sh"
aws s3 cp --quiet aws/jobs/01_probe.sh "s3://$BUCKET/control/job.sh"

echo "4/6 lifecycle config $LIFECYCLE"
if ! aws sagemaker describe-notebook-instance-lifecycle-config --region "$REGION" \
    --notebook-instance-lifecycle-config-name "$LIFECYCLE" >/dev/null 2>&1; then
  aws sagemaker create-notebook-instance-lifecycle-config --region "$REGION" \
    --notebook-instance-lifecycle-config-name "$LIFECYCLE" \
    --on-start Content="$(render aws/onstart.sh | base64 | tr -d '\n')" >/dev/null
fi

echo "5/6 EC2 GPU quota request (G and VT on-demand, 16 vCPUs)"
aws service-quotas request-service-quota-increase --region "$REGION" --service-code ec2 \
  --quota-code L-DB2E81BA --desired-value 16 --query 'RequestedQuota.[Id,Status]' --output text \
  || echo "  not filed (already requested or granted, or refused); the notebook does not depend on it"

echo "6/6 notebook $NOTEBOOK"
if ! aws sagemaker describe-notebook-instance --region "$REGION" --notebook-instance-name "$NOTEBOOK" >/dev/null 2>&1; then
  [ -n "${NEW_ROLE:-}" ] && sleep 15  # a new role takes a few seconds to become usable
  aws sagemaker create-notebook-instance --region "$REGION" \
    --notebook-instance-name "$NOTEBOOK" \
    --instance-type ml.g4dn.xlarge \
    --role-arn "arn:aws:iam::$ACCOUNT:role/$ROLE" \
    --volume-size-in-gb 100 \
    --lifecycle-config-name "$LIFECYCLE" \
    --platform-identifier notebook-al2023-v1 \
    --instance-metadata-service-configuration MinimumInstanceMetadataServiceVersion=2 \
    --tags Key=Project,Value=hp-bench >/dev/null
fi
aws sagemaker describe-notebook-instance --region "$REGION" --notebook-instance-name "$NOTEBOOK" \
  --query '[NotebookInstanceName,InstanceType,NotebookInstanceStatus]' --output text
echo "Done. The notebook takes about 5 minutes to reach InService."
