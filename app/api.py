from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.rag import ask_rag


app = FastAPI(
    title="AWS Health AI Platform",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        return ask_rag(request.question)

    except Exception as exc:
        print(f"RAG error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="The RAG service is temporarily unavailable."
        )