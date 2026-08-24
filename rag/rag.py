from rag.search import retrieve_documents
from rag.llm import invoke_llm


def ask_rag(question):
    documents = retrieve_documents(question, k=3)

    context = "\n\n".join(
        f"Source: {doc['source']}, Page: {doc['page']}\n{doc['text']}"
        for doc in documents
    )

    prompt = f"""
You are a healthcare information assistant.

Answer the user's question using only the context provided below.

If the context does not contain enough information to answer the question,
say that the available clinical document does not contain enough information.

Do not invent facts.
Do not provide information that is not supported by the context.

Context:
{context}

Question:
{question}
"""

    answer = invoke_llm(prompt)

    return {
        "answer": answer,
        "sources": [
            {
                "source": doc["source"],
                "page": doc["page"],
                "score": doc["score"],
            }
            for doc in documents
        ],
    }


if __name__ == "__main__":
    question =  "What is the recommended treatment for kidney cancer?"

    result = ask_rag(question)

    print("\nQuestion:\n")
    print(question)

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:\n")

    for source in result["sources"]:
        print(
            f"{source['source']} "
            f"- Page {source['page']} "
            f"- Score {source['score']:.3f}"
        )