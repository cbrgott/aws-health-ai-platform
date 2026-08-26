from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.rag import ask_rag
from app.guardrails import check_input, check_output
from app.agent import invoke_agent

app = FastAPI(
    title="AWS Health AI Platform",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str

class AgentRequest(BaseModel):
    user_id: str
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

    guardrail_result = check_input(request.question)

    if guardrail_result["action"] == "GUARDRAIL_INTERVENED":
        raise HTTPException(
            status_code=400,
            detail="The request was blocked by the clinical safety guardrail."
        )

    try:
        result = ask_rag(request.question)

    except Exception as exc:
        print(f"RAG error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="The RAG service is temporarily unavailable."
        )

    answer = result["answer"]

    guardrail_output = check_output(answer)

    if guardrail_output["action"] == "GUARDRAIL_INTERVENED":
        raise HTTPException(
            status_code=400,
            detail="The response was blocked by the clinical safety guardrail."
        )

    return result
@app.post("/agent")
def agent(request: AgentRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    guardrail_result = check_input(request.question)

    if guardrail_result["action"] == "GUARDRAIL_INTERVENED":
        raise HTTPException(
            status_code=400,
            detail="The request was blocked by the clinical safety guardrail."
        )

    try:
        answer = invoke_agent(
                    request.question,
                    request.user_id
                )
        
    except Exception as exc:
        print(f"Agent error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="The AgentCore service is temporarily unavailable."
        )

    guardrail_output = check_output(answer)

    if guardrail_output["action"] == "GUARDRAIL_INTERVENED":
        raise HTTPException(
            status_code=400,
            detail="The response was blocked by the clinical safety guardrail."
        )

    return {
        "answer": answer
    }