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

    # Automated evaluations
    for case in dataset:

        if case["capability"] == "guardrail":
            result = evaluate_guardrail(case)
            result["evaluation_mode"] = "automated"
            results.append(result)

        elif case["capability"] == "rag":
            result = evaluate_rag(case)
            result["evaluation_mode"] = "automated"
            results.append(result)

    # AgentCore trace-based evaluations
    trace_results = [
        {
            "id": "agent_guideline_tool_01",
            "capability": "agent",
            "expected": "use_search_guidelines",
            "evaluation_mode": "trace",
            "observed_tool": "search_guidelines",
            "trace_id": "6a8e442d75b8026848dcfd336e2e05f4",
            "passed": True,
        },
        {
            "id": "agent_prediction_tool_01",
            "capability": "agent",
            "expected": "use_predict_heart_risk",
            "evaluation_mode": "trace",
            "observed_tool": "predict_heart_risk",
            "trace_id": "6a8e4562035a6db52870931a191901e5",
            "note": (
                "Tool routing succeeded. The downstream SageMaker endpoint "
                "was intentionally deleted for cost control."
            ),
            "passed": True,
        },
        {
            "id": "agent_multitool_01",
            "capability": "agent",
            "expected": "use_prediction_and_guidelines",
            "evaluation_mode": "trace",
            "observed_tools": [
                "predict_heart_risk",
                "search_guidelines",
            ],
            "trace_id": "6a8e46345f29b776519dddd7632f0a1b",
            "note": (
                "Both tools were observed in the same AgentCore trace. "
                "The prediction backend was unavailable because the "
                "SageMaker endpoint had been intentionally deleted."
            ),
            "passed": True,
        },
    ]

    results.extend(trace_results)

    # Cross-session memory evaluation
    memory_result = {
        "id": "memory_cross_session_01",
        "capability": "memory",
        "expected": "recall_previous_patient_context",
        "evaluation_mode": "manual",
        "observed": {
            "age": 58,
            "cholesterol_mg_dl": 245,
        },
        "note": (
            "Patient values were provided in one Harness session, "
            "the session was stopped, and the values were correctly "
            "recalled in a new session."
        ),
        "passed": True,
    }

    results.append(memory_result)

    passed = sum(result["passed"] for result in results)
    total = len(results)
    pass_rate = passed / total if total else 0

    summary = {
        "total_tests": total,
        "passed_tests": passed,
        "failed_tests": total - passed,
        "pass_rate": pass_rate,
        "automated_tests": sum(
            result["evaluation_mode"] == "automated"
            for result in results
        ),
        "trace_tests": sum(
            result["evaluation_mode"] == "trace"
            for result in results
        ),
        "manual_tests": sum(
            result["evaluation_mode"] == "manual"
            for result in results
        ),
        "results": results,
    }

    with open(
        "evaluation/evaluation_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Evaluation complete: "
        f"{passed}/{total} passed "
        f"({pass_rate:.0%})"
    )

    print(
        "Modes: "
        f"{summary['automated_tests']} automated, "
        f"{summary['trace_tests']} trace-based, "
        f"{summary['manual_tests']} manual"
    )

    print(
        "Saved: evaluation/evaluation_results.json"
    )