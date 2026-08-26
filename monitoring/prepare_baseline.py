import boto3
import pandas as pd
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()

PROFILE = os.environ.get("AWS_PROFILE")
REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ["PROJECT_BUCKET"]

SOURCE_KEY = (
    "ml/heart-disease/pipeline/processed/train.csv"
)

DESTINATION_KEY = (
    "monitoring/baseline/input/train_features.csv"
)


session = boto3.Session(
    region_name=REGION,
)

s3 = session.client("s3")


# Download the training dataset from S3
response = s3.get_object(
    Bucket=BUCKET,
    Key=SOURCE_KEY,
)

df = pd.read_csv(
    BytesIO(response["Body"].read())
)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())


# Remove the target because production requests
# contain only model input features
features_df = df.drop(columns=["target"])

print("Baseline feature shape:", features_df.shape)
print("Baseline columns:", features_df.columns.tolist())


# Convert back to CSV in memory
csv_buffer = BytesIO()

features_df.to_csv(
    csv_buffer,
    index=False,
)

csv_buffer.seek(0)


# Upload baseline dataset to S3
s3.put_object(
    Bucket=BUCKET,
    Key=DESTINATION_KEY,
    Body=csv_buffer.getvalue(),
    ContentType="text/csv",
)

print(
    "Baseline uploaded to:",
    f"s3://{BUCKET}/{DESTINATION_KEY}",
)