import boto3

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.model_monitor.model_monitoring import DefaultModelMonitor
from sagemaker.core.model_monitor.dataset_format import DatasetFormat


PROFILE = "cristhian-dev"
REGION = "us-east-1"

ROLE_ARN = (
    "arn:aws:iam::528162482936:"
    "role/aws-health-ai-sagemaker-execution-role"
)

BUCKET = "aws-health-ai-data-3b2f0d9f09215b58379c9aabbd"

BASELINE_DATASET = (
    f"s3://{BUCKET}/"
    "monitoring/baseline/input/train_features.csv"
)

BASELINE_OUTPUT = (
    f"s3://{BUCKET}/"
    "monitoring/baseline/results/"
)


boto_session = boto3.Session(
    profile_name=PROFILE,
    region_name=REGION,
)

sagemaker_session = Session(
    boto_session=boto_session,
    default_bucket=BUCKET,
)


monitor = DefaultModelMonitor(
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=1800,
    sagemaker_session=sagemaker_session,
)


print("Starting Model Monitor baseline job...")
print("Input:", BASELINE_DATASET)
print("Output:", BASELINE_OUTPUT)


monitor.suggest_baseline(
    baseline_dataset=BASELINE_DATASET,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=BASELINE_OUTPUT,
    wait=True,
    logs=True,
)


print("Baseline completed.")
print("Results:", BASELINE_OUTPUT)