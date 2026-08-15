from sagemaker.core.workflow.functions import JsonGet
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops import ModelBuilder


def build_registry_step(
    pipeline_session,
    step_train,
    step_evaluate,
    evaluation_report,
    sklearn_image,
    role_arn,
    model_package_group,
    auc_threshold,
):
    # Quality gate: ROC-AUC must meet the required threshold
    auc_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="roc_auc",
        ),
        right=auc_threshold,
    )

    # Build the model from the artifact produced by TrainingStep
    model_builder = ModelBuilder(
        s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        image_uri=sklearn_image,
        role_arn=role_arn,
        sagemaker_session=pipeline_session,
    )

    # Configure Model Registry registration
    register_args = model_builder.register(
        model_package_group_name=model_package_group,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        approval_status="PendingManualApproval",
        description="Heart disease Random Forest model",
    )

    # Registration step
    step_register = ModelStep(
        name="RegisterHeartDiseaseModel",
        step_args=register_args,
    )

    # Register only if ROC-AUC passes the quality gate
    step_condition = ConditionStep(
        name="CheckModelQuality",
        conditions=[auc_condition],
        if_steps=[step_register],
        else_steps=[],
    )

    return step_condition