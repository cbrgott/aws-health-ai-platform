import json
from pathlib import Path
from app.guardrails import check_input
from rag.rag import ask_rag
import uuid
import boto3

DATASET_PATH = Path("evaluation/evaluation_dataset.json")
HARNESS_ARN = (
    "arn:aws:bedrock-agentcore:us-east-1:528162482936:"
    "harness/aws_health_ai_clinical_agent-p2P0cxPLoQ"
)

session = boto3.Session(
    profile_name="cristhian-dev",
    region_name="us-east-1",
)

agentcore = session.client("bedrock-agentcore")

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

def evaluate_guardrail(case):
    result = check_input(case["question"])
    action = result["action"]

    if case["expected_behavior"] == "allowed":
        passed = action == "NONE"

    elif case["expected_behavior"] == "blocked":
        passed = action == "GUARDRAIL_INTERVENED"

    else:
        passed = False

    return {
        "id": case["id"],
        "capability": case["capability"],
        "expected": case["expected_behavior"],
        "actual": action,
        "passed": passed,
    }
def evaluate_rag(case):
    result = ask_rag(case["question"])

    answer = result["answer"]
    sources = result["sources"]

    if case["expected_behavior"] == "answer_with_sources":
        passed = bool(answer.strip()) and len(sources) > 0

    elif case["expected_behavior"] == "insufficient_context":
        normalized_answer = answer.lower()

        passed = (
            "not contain enough information" in normalized_answer
            or "insufficient information" in normalized_answer
            or "does not contain enough information" in normalized_answer
        )

    else:
        passed = False

    return {
        "id": case["id"],
        "capability": case["capability"],
        "expected": case["expected_behavior"],
        "answer": answer,
        "sources_count": len(sources),
        "passed": passed,
    }

def invoke_agent(question: str) -> str:
    session_id = str(uuid.uuid4())

    response = agentcore.invoke_harness(
        harnessArn=HARNESS_ARN,
        runtimeSessionId=session_id,
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

if __name__ == "__main__":
    dataset = load_dataset()

    results = []

    for case in dataset:

        if case["capability"] == "guardrail":
            result = evaluate_guardrail(case)
            results.append(result)

        elif case["capability"] == "rag":
            result = evaluate_rag(case)
            results.append(result)

    passed = sum(result["passed"] for result in results)
    total = len(results)
    pass_rate = passed / total if total else 0

    summary = {
        "total_tests": total,
        "passed_tests": passed,
        "failed_tests": total - passed,
        "pass_rate": pass_rate,
        "results": results,
    }

    with open(
        "evaluation/evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Evaluation complete: "
        f"{passed}/{total} passed "
        f"({pass_rate:.0%})"
    )

    print(
        "Saved: evaluation/evaluation_results.json"
    )