import argparse

import boto3
import numpy as np
import pandas as pd


PROFILE = "cristhian-dev"
REGION = "us-east-1"

BASELINE_FILE = "data/monitoring/train_features.csv"

ENDPOINT_NAME = "aws-health-ai-heart-disease-endpoint"

CLOUDWATCH_NAMESPACE = "AWSHealthAI"

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

SIGNIFICANT_DRIFT_THRESHOLD = 0.25


def calculate_psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # Create histogram bins from the baseline distribution
    breakpoints = np.percentile(
        expected,
        np.linspace(0, 100, bins + 1),
    )

    breakpoints = np.unique(breakpoints)

    expected_counts, _ = np.histogram(
        expected,
        bins=breakpoints,
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=breakpoints,
    )

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    epsilon = 1e-6

    expected_pct = np.maximum(
        expected_pct,
        epsilon,
    )

    actual_pct = np.maximum(
        actual_pct,
        epsilon,
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


def classify_drift(psi):
    if psi < 0.10:
        return "NO_DRIFT"

    if psi < SIGNIFICANT_DRIFT_THRESHOLD:
        return "MODERATE_DRIFT"

    return "SIGNIFICANT_DRIFT"


def publish_cloudwatch_metrics(
    drift_detected,
    significant_drift_features,
):
    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )

    cloudwatch = session.client("cloudwatch")

    cloudwatch.put_metric_data(
        Namespace=CLOUDWATCH_NAMESPACE,
        MetricData=[
            {
                "MetricName": "DriftDetected",
                "Dimensions": [
                    {
                        "Name": "EndpointName",
                        "Value": ENDPOINT_NAME,
                    }
                ],
                "Value": drift_detected,
                "Unit": "Count",
            },
            {
                "MetricName": "SignificantDriftFeatureCount",
                "Dimensions": [
                    {
                        "Name": "EndpointName",
                        "Value": ENDPOINT_NAME,
                    }
                ],
                "Value": len(significant_drift_features),
                "Unit": "Count",
            },
        ],
    )

    print("\nCloudWatch metrics published successfully.")
    print(f"Namespace: {CLOUDWATCH_NAMESPACE}")
    print(f"DriftDetected: {drift_detected}")
    print(
        "SignificantDriftFeatureCount:",
        len(significant_drift_features),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Detect production feature drift using PSI."
    )

    parser.add_argument(
        "--production-file",
        required=True,
        help="Production dataset to compare against the baseline.",
    )

    args = parser.parse_args()

    baseline = pd.read_csv(BASELINE_FILE)
    production = pd.read_csv(args.production_file)

    significant_drift_features = []

    print("\nAWS Health AI - Drift Report")
    print("=" * 55)

    for feature in FEATURES:
        psi = calculate_psi(
            baseline[feature],
            production[feature],
        )

        status = classify_drift(psi)

        if status == "SIGNIFICANT_DRIFT":
            significant_drift_features.append(feature)

        print(
            f"{feature:12} "
            f"PSI={psi:.4f} "
            f"{status}"
        )

    print("\n" + "=" * 55)

    if significant_drift_features:
        drift_detected = 1

        print("DRIFT DETECTED")
        print(
            "Significant features:",
            ", ".join(significant_drift_features),
        )
    else:
        drift_detected = 0

        print("NO SIGNIFICANT DRIFT")

    print(f"DriftDetected={drift_detected}")

    publish_cloudwatch_metrics(
        drift_detected,
        significant_drift_features,
    )


if __name__ == "__main__":
    main()