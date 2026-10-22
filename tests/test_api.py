
from fastapi.testclient import TestClient

from src.api import app, determine_routing

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Banking Operations AI Copilot API is running." in (
    response.json()["message"]
)
    assert response.json()["version"] == "2.1.0"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_card_arrival():
    response = client.post(
        "/predict",
        json={"message": "I am still waiting for my card"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["predicted_category"] == "card_arrival"
    assert 0 <= result["confidence"] <= 1
    assert result["routing_status"] in {
        "HUMAN_REVIEW",
        "AUTO_ROUTE_CANDIDATE",
        "ESCALATE_SENSITIVE_INTENT",
    }
    assert result["routing_reason"]
    assert len(result["top_3_predictions"]) == 3


def test_sensitive_intent_requires_escalation():
    result = determine_routing(
        category="lost_or_stolen_card",
        confidence=0.99,
    )
    assert result["routing_status"] == "ESCALATE_SENSITIVE_INTENT"


def test_low_confidence_requires_review():
    result = determine_routing(
        category="card_arrival",
        confidence=0.40,
    )
    assert result["routing_status"] == "HUMAN_REVIEW"


def test_confident_non_sensitive_intent_is_candidate():
    result = determine_routing(
        category="card_arrival",
        confidence=0.85,
    )
    assert result["routing_status"] == "AUTO_ROUTE_CANDIDATE"


def test_empty_message_is_rejected():
    response = client.post("/predict", json={"message": ""})
    assert response.status_code == 422


def test_missing_message_is_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_whitespace_message_is_rejected():
    response = client.post("/predict", json={"message": "   "})
    assert response.status_code == 422



def test_ask_returns_relevant_stolen_card_document():
    response = client.post(
        "/ask",
        json={"question": "Someone stole my card. What should I do?"},
    )

    assert response.status_code == 200
    result = response.json()

    assert result["answer_status"] == "GROUNDED_EXTRACTIVE"
    assert len(result["sources"]) > 0
    assert result["sources"][0]["topic"] == "lost_or_stolen_card"
    assert result["sources"][0]["title"] == "Lost or stolen card"


def test_ask_returns_source_metadata():
    response = client.post(
        "/ask",
        json={"question": "Why has my refund not arrived?"},
    )

    assert response.status_code == 200
    source = response.json()["sources"][0]

    assert source["id"]
    assert source["title"]
    assert source["source"]
    assert source["content"]
    assert 0 <= source["similarity"] <= 1


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_ask_preserves_official_source_metadata():
    response = client.post(
        "/ask",
        json={
            "question": (
                "RBI Integrated Ombudsman Scheme eligibility "
                "and complaint process"
            ),
            "top_k": 6,
        },
    )

    assert response.status_code == 200

    result = response.json()
    rbi_sources = [
        source
        for source in result["sources"]
        if source["id"] in {"rbi_001", "rbi_002"}
    ]

    assert rbi_sources, "Expected at least one RBI document"

    for source in rbi_sources:
        assert source.get("source_url", "").startswith("https://")
        assert source.get("source_type", "").startswith("official_")
        assert source.get("verified_on") == "2026-10-21"
