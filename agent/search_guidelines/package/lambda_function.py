import json
import os

import boto3
import requests
from requests_aws4auth import AWS4Auth


REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
INDEX = os.environ.get(
    "OPENSEARCH_INDEX",
    "heart-disease-guidelines"
)

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
)


def get_auth():
    credentials = boto3.Session().get_credentials().get_frozen_credentials()

    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        REGION,
        "aoss",
        session_token=credentials.token,
    )


def search_guidelines(question, k=3):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": question
        }),
    )

    result = json.loads(response["body"].read())
    query_embedding = result["embedding"]

    search_body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": k
                }
            }
        },
        "_source": [
            "text",
            "page",
            "source"
        ],
    }

    response = requests.post(
        f"{ENDPOINT}/{INDEX}/_search",
        auth=get_auth(),
        headers={"Content-Type": "application/json"},
        json=search_body,
        timeout=15,
    )

    response.raise_for_status()

    hits = response.json()["hits"]["hits"]

    return [
        {
            "text": hit["_source"]["text"],
            "page": hit["_source"]["page"],
            "source": hit["_source"]["source"],
            "score": hit["_score"],
        }
        for hit in hits
    ]


def lambda_handler(event, context):
    print(json.dumps(event))

    # AgentCore Gateway format
    question = event.get("question")

    # Fallback for the previous/manual test format
    if not question:
        parameters = {
            item["name"]: item["value"]
            for item in event.get("parameters", [])
        }

        question = parameters.get("question")

    if not question:
        raise ValueError("Missing required parameter: question")

    results = search_guidelines(question)

    return {
        "results": results
    }