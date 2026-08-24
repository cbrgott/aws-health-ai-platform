import boto3


session = boto3.Session(
    profile_name="cristhian-dev",
    region_name="us-east-1",
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