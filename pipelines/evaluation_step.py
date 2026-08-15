from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingS3Input,
    ProcessingOutput,
    ProcessingS3Output,
)
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.mlops.workflow.steps import ProcessingStep


def build_evaluation_step(
    pipeline_session,
    step_process,
    step_train,
    sklearn_image,
    project_bucket,
    role_arn,
    instance_type,
):
    # Processor used to evaluate the trained model
    evaluation_processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        role=role_arn,
        env={
            "SM_MODEL_DIR": "/opt/ml/processing/model",
            "SM_CHANNEL_TEST": "/opt/ml/processing/test",
            "SM_OUTPUT_DATA_DIR": "/opt/ml/processing/evaluation",
        },
        sagemaker_session=pipeline_session,
    )

    # Inputs:
    # 1. Model artifact produced by TrainingStep
    # 2. Test data produced by ProcessingStep
    evaluation_args = evaluation_processor.run(
        code="pipelines/evaluate.py",
        inputs=[
            ProcessingInput(
                input_name="model",
                s3_input=ProcessingS3Input(
                    s3_uri=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    local_path="/opt/ml/processing/model",
                    s3_data_type="S3Prefix",
                ),
            ),
            ProcessingInput(
                input_name="test",
                s3_input=ProcessingS3Input(
                    s3_uri=step_process.properties.ProcessingOutputConfig.Outputs[
                        "processed_data"
                    ].S3Output.S3Uri,
                    local_path="/opt/ml/processing/test",
                    s3_data_type="S3Prefix",
                ),
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                s3_output=ProcessingS3Output(
                    s3_uri=(
                        f"s3://{project_bucket}/"
                        "ml/heart-disease/pipeline/evaluation"
                    ),
                    local_path="/opt/ml/processing/evaluation",
                    s3_upload_mode="EndOfJob",
                ),
            ),
        ],
    )

    # Make evaluation.json available to later pipeline steps
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_evaluate = ProcessingStep(
        name="EvaluateHeartDiseaseModel",
        step_args=evaluation_args,
        property_files=[evaluation_report],
    )

    return step_evaluate, evaluation_report