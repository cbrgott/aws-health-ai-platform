import json
import os

import boto3
import requests
from dotenv import load_dotenv
from requests_aws4auth import AWS4Auth


load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]

INDEX = "heart-disease-guidelines"

session = boto3.Session(
    region_name=REGION,
)

credentials = session.get_credentials().get_frozen_credentials()

auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION ,
    "aoss",
    session_token=credentials.token,
)

bedrock = session.client("bedrock-runtime")


def retrieve_documents(question, k=3):
    body = json.dumps({
        "inputText": question
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

    documents = []

    for hit in results["hits"]["hits"]:
        source = hit["_source"]

        documents.append({
            "text": source["text"],
            "page": source["page"],
            "source": source["source"],
            "score": hit["_score"],
        })

    return documents


if __name__ == "__main__":
    question = "What are the main recommendations for preventing cardiovascular disease?"

    docs = retrieve_documents(question)

    print(f"\nQuestion: {question}\n")

    for i, doc in enumerate(docs, start=1):
        print(f"Result {i}")
        print(f"Score: {doc['score']}")
        print(f"Page: {doc['page']}")
        print(f"Source: {doc['source']}")
        print(doc["text"])
        print("-" * 80)