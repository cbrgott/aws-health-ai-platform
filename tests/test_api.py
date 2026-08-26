from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={"question": "   "}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_agent_rejects_empty_question():
    response = client.post(
        "/agent",
        json={
            "user_id": "pytest-user",
            "question": "   "
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."