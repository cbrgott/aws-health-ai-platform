import json
import joblib
import pandas as pd
from pathlib import Path


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


def model_fn(model_dir):
    model_path = Path(model_dir) / "model.joblib"
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: {request_content_type}"
        )

    payload = json.loads(request_body)

    # Accept either one patient object or {"instances": [...]}
    if "instances" in payload:
        records = payload["instances"]
    else:
        records = [payload]

    df = pd.DataFrame(records)

    missing_columns = [
        column for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    df = df[FEATURES]

    if df.isnull().any().any():
        raise ValueError(
            "Missing feature values are not supported by this model."
        )

    return df


def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    return [
        {
            "prediction": int(pred),
            "risk_probability": float(prob),
        }
        for pred, prob in zip(predictions, probabilities)
    ]


def output_fn(prediction, accept):
    if accept not in ("application/json", "*/*"):
        raise ValueError(
            f"Unsupported accept type: {accept}"
        )

    return json.dumps(
        {"predictions": prediction}
    ), "application/json"