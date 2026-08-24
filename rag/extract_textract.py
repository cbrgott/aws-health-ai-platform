import json
import boto3

JOB_ID = "0fa40e6c77f9499387d99b2aecd4de59311e31d1c0d40bc7cb8aa5ccb8d99ac4"

session = boto3.Session(
    profile_name="cristhian-dev",
    region_name="us-east-1",
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