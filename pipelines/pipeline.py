import boto3

from sagemaker.core import image_uris
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.pipeline import Pipeline

from pipelines.processing_step import build_processing_step
from pipelines.training_step import build_training_step
from pipelines.evaluation_step import build_evaluation_step
from pipelines.registry_step import build_registry_step


import os
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")

PROJECT_BUCKET = os.environ["PROJECT_BUCKET"]

ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]

INSTANCE_TYPE = "ml.m5.large"

PIPELINE_NAME = "aws-health-ai-heart-disease-pipeline"

MODEL_PACKAGE_GROUP = "aws-health-ai-heart-disease-models"

AUC_THRESHOLD = 0.80


# AWS session
boto_session = boto3.Session(
    profile_name="cristhian-dev",
    region_name=REGION,
)

# SageMaker Pipeline session
pipeline_session = PipelineSession(
    boto_session=boto_session,
    default_bucket=PROJECT_BUCKET,
    default_bucket_prefix="ml/heart-disease/pipeline",
)

# SageMaker scikit-learn image
sklearn_image = image_uris.retrieve(
    framework="sklearn",
    region=REGION,
    version="1.2-1",
    py_version="py3",
    instance_type=INSTANCE_TYPE,
    image_scope="training",
)


# 1. Preprocessing
step_process = build_processing_step(
    pipeline_session=pipeline_session,
    sklearn_image=sklearn_image,
    project_bucket=PROJECT_BUCKET,
    role_arn=ROLE_ARN,
    instance_type=INSTANCE_TYPE,
)


# 2. Training
step_train = build_training_step(
    pipeline_session=pipeline_session,
    step_process=step_process,
    project_bucket=PROJECT_BUCKET,
    role_arn=ROLE_ARN,
    region=REGION,
    instance_type=INSTANCE_TYPE,
)


# 3. Evaluation
step_evaluate, evaluation_report = build_evaluation_step(
    pipeline_session=pipeline_session,
    step_process=step_process,
    step_train=step_train,
    sklearn_image=sklearn_image,
    project_bucket=PROJECT_BUCKET,
    role_arn=ROLE_ARN,
    instance_type=INSTANCE_TYPE,
)


# 4. Quality gate + Model Registry
step_condition = build_registry_step(
    pipeline_session=pipeline_session,
    step_train=step_train,
    step_evaluate=step_evaluate,
    evaluation_report=evaluation_report,
    sklearn_image=sklearn_image,
    role_arn=ROLE_ARN,
    model_package_group=MODEL_PACKAGE_GROUP,
    auc_threshold=AUC_THRESHOLD,
)


# Assemble pipeline
pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[
        step_process,
        step_train,
        step_evaluate,
        step_condition,
    ],
    sagemaker_session=pipeline_session,
)

# Compile pipeline definition
pipeline_definition = pipeline.definition()

print("Pipeline definition created successfully")

# Create or update the pipeline in SageMaker
pipeline.upsert(
    role_arn=ROLE_ARN,
)

print(f"Pipeline '{PIPELINE_NAME}' created/updated successfully")