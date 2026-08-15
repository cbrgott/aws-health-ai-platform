from sagemaker.train import ModelTrainer
from sagemaker.core import image_uris
from sagemaker.core.training.configs import SourceCode, Compute, InputData
from sagemaker.core.shapes import OutputDataConfig
from sagemaker.mlops.workflow.steps import TrainingStep


def build_training_step(
    pipeline_session,
    step_process,
    project_bucket,
    role_arn,
    region,
    instance_type,
):
    # Source code executed by the SageMaker Training Job
    source_code = SourceCode(
        source_dir="pipelines",
        entry_script="train.py",
    )

    # Training compute configuration
    compute = Compute(
        instance_type=instance_type,
        instance_count=1,
        volume_size_in_gb=10,
    )

    # Training data produced by the ProcessingStep
    training_data = InputData(
        channel_name="train",
        data_source=step_process.properties.ProcessingOutputConfig.Outputs[
            "processed_data"
        ].S3Output.S3Uri,
        content_type="text/csv",
    )

    # SageMaker managed scikit-learn training image
    training_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type=instance_type,
        image_scope="training",
    )

    # Location for model artifacts
    output_config = OutputDataConfig(
        s3_output_path=(
            f"s3://{project_bucket}/"
            "ml/heart-disease/models"
        )
    )
    # Configure the SageMaker Training Job
    trainer = ModelTrainer(
        role=role_arn,
        base_job_name="aws-health-ai-heart-disease",
        source_code=source_code,
        compute=compute,
        training_image=training_image,
        output_data_config=output_config,
        sagemaker_session=pipeline_session,
    )

    # Capture training arguments for the pipeline
    training_args = trainer.train(
        input_data_config=[training_data],
        wait=False,
        logs=False,
    )

    # Create the TrainingStep
    step_train = TrainingStep(
        name="TrainHeartDiseaseModel",
        step_args=training_args,
    )

    return step_train