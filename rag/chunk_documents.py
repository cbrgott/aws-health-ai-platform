import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_FILE = "rag/textract_pages.json"
OUTPUT_FILE = "rag/chunks.json"
SOURCE = "acc_aha_guidelines_made_simple.pdf"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    pages = json.load(f)

chunks = []

for page in pages:
    page_number = page["page"]
    text = page["text"]

    page_chunks = splitter.split_text(text)

    for i, chunk_text in enumerate(page_chunks, start=1):
        chunks.append(
            {
                "chunk_id": f"page-{page_number}-chunk-{i}",
                "source": SOURCE,
                "page": page_number,
                "text": chunk_text,
            }
        )

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Chunks created: {len(chunks)}")