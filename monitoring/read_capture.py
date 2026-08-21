import argparse
import base64
import json

import boto3
import pandas as pd


PROFILE = "cristhian-dev"
REGION = "us-east-1"

BUCKET = "aws-health-ai-data-3b2f0d9f09215b58379c9aabbd"

CAPTURE_PREFIX = (
    "monitoring/data-capture/"
    "aws-health-ai-heart-disease-endpoint/"
)

OUTPUT_FILE = "monitoring/production_capture.csv"

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


def decode_input(encoded_data):
    """Decode the Base64 endpoint input captured by SageMaker."""

    decoded_bytes = base64.b64decode(encoded_data)
    decoded_text = decoded_bytes.decode("utf-8")

    return json.loads(decoded_text)


def main():
    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )

    s3 = session.client("s3")

    paginator = s3.get_paginator("list_objects_v2")

    records = []

    print("Reading SageMaker Data Capture...")

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=CAPTURE_PREFIX,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.endswith(".jsonl"):
                continue

            print(f"Reading: {key}")

            response = s3.get_object(
                Bucket=BUCKET,
                Key=key,
            )

            content = response["Body"].read().decode("utf-8")

            # Each line is one captured inference event
            for line in content.splitlines():
                if not line.strip():
                    continue

                event = json.loads(line)

                encoded_input = (
                    event["captureData"]
                    ["endpointInput"]
                    ["data"]
                )

                payload = decode_input(encoded_input)

                # Our endpoint accepts either:
                # {patient}
                #
                # or:
                # {"instances": [{patient}, ...]}

                if "instances" in payload:
                    records.extend(payload["instances"])
                else:
                    records.append(payload)

    if not records:
        print("No captured inference records found.")
        return

    df = pd.DataFrame(records)

    # Ensure exactly the same feature order used by the model
    df = df[FEATURES]

    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Removed {before - len(df)} duplicate rows.")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Capture extraction complete.")
    print("Records:", len(df))
    print("Output:", OUTPUT_FILE)

    print()
    print(df.head())


if __name__ == "__main__":
    main()