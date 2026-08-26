import os
import uuid

import boto3


REGION = os.environ.get("AWS_REGION", "us-east-1")

HARNESS_ARN = os.environ.get(
    "AGENTCORE_HARNESS_ARN"
)

session = boto3.Session(
    profile_name="cristhian-dev",
    region_name=REGION,
)
agentcore = session.client("bedrock-agentcore")


def invoke_agent(question: str, user_id: str) -> str:
    if not HARNESS_ARN:
        raise ValueError(
            "AGENTCORE_HARNESS_ARN environment variable is not configured."
        )

    session_id = str(uuid.uuid4())

    response = agentcore.invoke_harness(
        harnessArn=HARNESS_ARN,
        runtimeSessionId=session_id,
        actorId=user_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": question
                    }
                ]
            }
        ]
    )

    chunks = []

    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})

            if "text" in delta:
                chunks.append(delta["text"])

    return "".join(chunks)