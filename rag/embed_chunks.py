import json
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")

INPUT_FILE = "rag/chunks.json"
OUTPUT_FILE = "rag/chunks_with_embeddings.json"


session = boto3.Session(
    region_name=REGION,
)
bedrock = session.client("bedrock-runtime")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

for i, chunk in enumerate(chunks, start=1):
    body = json.dumps({
        "inputText": chunk["text"]
    })

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())

    chunk["embedding"] = result["embedding"]

    print(f"Embedded {i}/{len(chunks)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Saved: {OUTPUT_FILE}")