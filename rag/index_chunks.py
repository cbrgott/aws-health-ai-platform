import json
import boto3
import requests
from requests_aws4auth import AWS4Auth


ENDPOINT = "https://x16m2ax2gmt70fgkred6.aoss.us-east-1.on.aws"
INDEX = "heart-disease-guidelines"
INPUT_FILE = "rag/chunks_with_embeddings.json"

session = boto3.Session(
    profile_name="cristhian-dev",
    region_name="us-east-1",
)

credentials = session.get_credentials().get_frozen_credentials()

auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    "us-east-1",
    "aoss",
    session_token=credentials.token,
)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

for i, chunk in enumerate(chunks, start=1):
    response = requests.put(
        f"{ENDPOINT}/{INDEX}/_doc/{chunk['chunk_id']}",
        auth=auth,
        headers={"Content-Type": "application/json"},
        json={
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
            "embedding": chunk["embedding"],
        },
    )

    response.raise_for_status()
    print(f"Indexed {i}/{len(chunks)}")

print("Indexing complete.")