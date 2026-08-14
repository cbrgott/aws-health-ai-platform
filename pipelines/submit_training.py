import boto3
from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import SourceCode, Compute, InputData
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session

#source code config: train.py as entry point
source_code = SourceCode(
    source_dir="pipelines",
    entry_script="train.py",
)
#instance config
compute = Compute(
    instance_type="ml.m5.large",
    instance_count=1,
    volume_size_in_gb=10,
)
#S3 input channel
training_data = InputData(
    channel_name="train",
    data_source="s3://aws-health-ai-data-3b2f0d9f09215b58379c9aabbd/ml/heart-disease/train/",
    content_type="text/csv",
)

#SageMaker scikit-learn training image

REGION = "us-east-1"

training_image = image_uris.retrieve(
    framework="sklearn",
    region=REGION,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.m5.large",
    image_scope="training",
)

#CREATING MODEL TRAINER (SageMaker Python SDK class that configures and launches model training.)

boto_session = boto3.Session(
    profile_name="cristhian-dev",
    region_name=REGION,
)

sagemaker_session = Session(
    boto_session=boto_session,
)

ROLE_ARN = "arn:aws:iam::528162482936:role/aws-health-ai-sagemaker-execution-role"

trainer = ModelTrainer(
    role=ROLE_ARN,
    base_job_name="aws-health-ai-heart-disease",
    source_code=source_code,
    compute=compute,
    training_image=training_image,
    sagemaker_session=sagemaker_session,
)

trainer.train(
    input_data_config=[training_data],
    wait=True,
    logs=True,
)