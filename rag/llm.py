import os

import boto3
from dotenv import load_dotenv

REGION = os.environ.get("AWS_REGION", "us-east-1")

session = boto3.Session(
    region_name=REGION,
)

bedrock = session.client("bedrock-runtime")


def invoke_llm(prompt):
    response = bedrock.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": 500,
            "temperature": 0.2,
        },
    )

    return response["output"]["message"]["content"][0]["text"]


if __name__ == "__main__":
    prompt = "Explain in two sentences why cardiovascular disease prevention is important."

    answer = invoke_llm(prompt)

    print("\nResponse:\n")
    print(answer)