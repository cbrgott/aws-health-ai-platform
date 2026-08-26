import os
import boto3

from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")

GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID")

GUARDRAIL_VERSION = os.environ.get(
    "GUARDRAIL_VERSION",
    "DRAFT"
)

session = boto3.Session(
    region_name=REGION,
)

bedrock = session.client("bedrock-runtime")


def check_input(text: str) -> dict:
    response = bedrock.apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source="INPUT",
        content=[
            {
                "text": {
                    "text": text
                }
            }
        ]
    )

    return response

def check_output(text: str) -> dict:
    response = bedrock.apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source="OUTPUT",
        content=[
            {
                "text": {
                    "text": text
                }
            }
        ]
    )

    return response