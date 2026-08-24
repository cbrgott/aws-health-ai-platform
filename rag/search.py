import json
import boto3
import requests
from requests_aws4auth import AWS4Auth


ENDPOINT = "https://x16m2ax2gmt70fgkred6.aoss.us-east-1.on.aws"
INDEX = "heart-disease-guidelines"
QUESTION = "What are the main recommendations for preventing cardiovascular disease?"

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

bedrock = session.client("bedrock-runtime")

body = json.dumps({
    "inputText": QUESTION
})

response = bedrock.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    contentType="application/json",
    accept="application/json",
    body=body,
)

result = json.loads(response["body"].read())
query_embedding = result["embedding"]

search_body = {
    "size": 3,
    "query": {
        "knn": {
            "embedding": {
                "vector": query_embedding,
                "k": 3
            }
        }
    },
    "_source": [
        "text",
        "page",
        "source"
    ]
}

response = requests.post(
    f"{ENDPOINT}/{INDEX}/_search",
    auth=auth,
    headers={"Content-Type": "application/json"},
    json=search_body,
)

response.raise_for_status()
results = response.json()

print(f"\nQuestion: {QUESTION}\n")

for i, hit in enumerate(results["hits"]["hits"], start=1):
    source = hit["_source"]

    print(f"Result {i}")
    print(f"Score: {hit['_score']}")
    print(f"Page: {source['page']}")
    print(f"Source: {source['source']}")
    print(source["text"])
    print("-" * 80)