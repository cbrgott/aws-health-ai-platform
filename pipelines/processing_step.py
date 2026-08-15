from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingS3Input,
    ProcessingOutput,
    ProcessingS3Output,
)
from sagemaker.mlops.workflow.steps import ProcessingStep


def build_processing_step(
    pipeline_session,
    sklearn_image,
    project_bucket,
    role_arn,
    instance_type,
):
    processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        role=role_arn,
        env={
            "SM_PROCESSING_INPUT_DIR": "/opt/ml/processing/input",
            "SM_PROCESSING_OUTPUT_DIR": "/opt/ml/processing/output",
        },
        sagemaker_session=pipeline_session,
    )

    processing_args = processor.run(
        code="pipelines/preprocess.py",
        inputs=[
            ProcessingInput(
                input_name="raw_data",
                s3_input=ProcessingS3Input(
                    s3_uri=(
                        f"s3://{project_bucket}/"
                        "ml/heart-disease/raw/processed.cleveland.data"
                    ),
                    local_path="/opt/ml/processing/input",
                    s3_data_type="S3Prefix",
                ),
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                s3_output=ProcessingS3Output(
                    s3_uri=(
                        f"s3://{project_bucket}/"
                        "ml/heart-disease/pipeline/processed"
                    ),
                    local_path="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob",
                ),
            )
        ],
    )

    step_process = ProcessingStep(
        name="PreprocessHeartDiseaseData",
        step_args=processing_args,
    )

    return step_process