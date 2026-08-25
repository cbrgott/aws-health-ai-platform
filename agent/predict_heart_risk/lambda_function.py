import json
import os

import boto3


REGION = os.environ.get("AWS_REGION", "us-east-1")

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

sagemaker_runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION,
)

def predict_heart_risk(patient_features):
    endpoint_name = os.environ["SAGEMAKER_ENDPOINT_NAME"]

    payload = {
        feature: patient_features[feature]
        for feature in FEATURES
    }

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(
        response["Body"].read().decode("utf-8")
    )

    return result

def lambda_handler(event, context):
    print(json.dumps(event))

    patient_features = {}

    for feature in FEATURES:
        if feature not in event:
            raise ValueError(
                f"Missing required feature: {feature}"
            )

        patient_features[feature] = event[feature]

    result = predict_heart_risk(patient_features)

    return result