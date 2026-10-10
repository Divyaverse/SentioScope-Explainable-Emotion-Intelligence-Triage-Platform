import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.normalizer import clean_and_normalize
from app.triage import compute_urgency

client = TestClient(app)
client.__enter__()

def test_normalization_slang_and_emoji():
    raw = "tbh idk why I feel fml 😭"
    cleaned = clean_and_normalize(raw)
    assert "to be honest" in cleaned
    assert "i do not know" in cleaned
    assert "i am devastated" in cleaned
    assert "crying face" in cleaned

def test_triage_critical_detection():
    high_distress_probs = {"sadness": 0.85, "joy": 0.0, "love": 0.0, "anger": 0.1, "fear": 0.05, "surprise": 0.0}
    triage = compute_urgency(high_distress_probs)
    assert triage["tier"] == "CRITICAL"
    assert triage["score"] >= 60.0

def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert len(data["classes"]) == 6

def test_empty_string_validation():
    response = client.post("/api/v1/analyze", json={"text": "", "explain": False})
    assert response.status_code == 422

def test_successful_analysis():
    payload = {"text": "I received an unexpected gift and I am so amazed", "explain": False}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dominant_emotion"] in ["joy", "surprise"]
    assert "urgency" in data
    assert "latency_ms" in data

def test_history_lifecycle():
    client.post("/api/v1/analyze", json={"text": "feeling so lonely and sad", "explain": False})
    res = client.get("/api/v1/history")
    assert res.status_code == 200
    assert len(res.json()) >= 1
