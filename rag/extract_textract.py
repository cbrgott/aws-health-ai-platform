import json
import os

import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
JOB_ID = os.environ["TEXTRACT_JOB_ID"]

session = boto3.Session(
    region_name=REGION,
)

textract = session.client("textract")

pages = {}
next_token = None

while True:
    args = {"JobId": JOB_ID}

    if next_token:
        args["NextToken"] = next_token

    response = textract.get_document_text_detection(**args)

    for block in response["Blocks"]:
        if block["BlockType"] == "LINE":
            page = block["Page"]
            pages.setdefault(page, []).append(block["Text"])

    next_token = response.get("NextToken")

    if not next_token:
        break

result = [
    {
        "page": page,
        "text": "\n".join(lines),
    }
    for page, lines in sorted(pages.items())
]

with open("rag/textract_pages.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"Pages extracted: {len(result)}")