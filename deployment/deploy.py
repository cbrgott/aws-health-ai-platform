import boto3
import os
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import SourceCode
from sagemaker.serve.model_builder import ModelBuilder
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")

PROFILE = os.environ.get("AWS_PROFILE")

MODEL_PACKAGE_ARN = os.environ["MODEL_PACKAGE_ARN"]

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "aws-health-ai-heart-disease-v3",
)

ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]


# AWS session
boto_session = boto3.Session(
    profile_name=PROFILE,
    region_name=REGION,
)

PROJECT_BUCKET = os.environ["PROJECT_BUCKET"]

sagemaker_session = Session(
    boto_session=boto_session,
    default_bucket=PROJECT_BUCKET,
    default_bucket_prefix="deployment",
)

sm_client = boto_session.client("sagemaker")


# Read the registered model package
package = sm_client.describe_model_package(
    ModelPackageName=MODEL_PACKAGE_ARN
)

container = package["InferenceSpecification"]["Containers"][0]

image_uri = container["Image"]
model_data = container["ModelDataUrl"]


# Tell SageMaker which inference script to use
source_code = SourceCode(
    source_dir="deployment",
    entry_script="inference.py",
)


# Build SageMaker deployment definition
model_builder = ModelBuilder(
    image_uri=image_uri,
    s3_model_data_url=model_data,
    source_code=source_code,
    role_arn=ROLE_ARN,
    sagemaker_session=sagemaker_session,
    content_type="application/json",
    accept_type="application/json",
)


print("Building SageMaker Model...")
print("Model artifact:", model_data)
print("Inference image:", image_uri)
print("Inference script: deployment/inference.py")


model = model_builder.build(
    model_name=MODEL_NAME
)

print("SageMaker Model created successfully:")
print(MODEL_NAME)