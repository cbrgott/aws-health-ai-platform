import boto3


REGION = "us-east-1"
GUARDRAIL_ID = "t5j8b0cx9xse"
GUARDRAIL_VERSION = "DRAFT"

session = boto3.Session(
    profile_name="cristhian-dev",
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