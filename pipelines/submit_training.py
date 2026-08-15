import boto3

from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import SourceCode, Compute, InputData
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.shapes import OutputDataConfig


REGION = "us-east-1"
PROJECT_BUCKET = "aws-health-ai-data-3b2f0d9f09215b58379c9aabbd"
ROLE_ARN = "arn:aws:iam::528162482936:role/aws-health-ai-sagemaker-execution-role"
INSTANCE_TYPE = "ml.m5.large"


def main():

    # Source code configuration
    source_code = SourceCode(
        source_dir="pipelines",
        entry_script="train.py",
    )

    # Training compute
    compute = Compute(
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        volume_size_in_gb=10,
    )

    # S3 training input
    training_data = InputData(
        channel_name="train",
        data_source=(
            f"s3://{PROJECT_BUCKET}/"
            "ml/heart-disease/train/"
        ),
        content_type="text/csv",
    )

    # AWS-managed SageMaker scikit-learn training image
    training_image = image_uris.retrieve(
        framework="sklearn",
        region=REGION,
        version="1.2-1",
        py_version="py3",
        instance_type=INSTANCE_TYPE,
        image_scope="training",
    )

    # Explicit AWS profile/session
    boto_session = boto3.Session(
        profile_name="cristhian-dev",
        region_name=REGION,
    )

    sagemaker_session = Session(
        boto_session=boto_session,
        default_bucket=PROJECT_BUCKET,
        default_bucket_prefix="ml/heart-disease",
    )

    # Model artifact output
    output_config = OutputDataConfig(
        s3_output_path=(
            f"s3://{PROJECT_BUCKET}/"
            "ml/heart-disease"
        )
    )

    # Configure SageMaker Training Job
    trainer = ModelTrainer(
        role=ROLE_ARN,
        base_job_name="aws-health-ai-heart-disease",
        source_code=source_code,
        compute=compute,
        training_image=training_image,
        output_data_config=output_config,
        sagemaker_session=sagemaker_session,
    )

    # Launch training
    trainer.train(
        input_data_config=[training_data],
        wait=True,
        logs=False,
    )


if __name__ == "__main__":
    main()