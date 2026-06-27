"""
API tests using FastAPI's TestClient. This runs the REAL app — lifespan startup,
routing, Pydantic validation, the error boundary — but injects a fake LLM into
the agent so there's no model, no network, no cost. Proves the HTTP layer works
end to end.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.agent.agent import EvidenceAgent
from app.agent.llm import LLMResponse, ToolCall


class _FakeClient:
    def __init__(self, scripted):
        self._s, self._i = scripted, 0

    def complete(self, system, messages, tools):
        r = self._s[self._i]; self._i += 1
        return r


@pytest.fixture
def client():
    with TestClient(app) as c:
        # Replace the agent's LLM with a scripted fake after startup loaded the corpus.
        scripted = [
            LLMResponse(tool_calls=[ToolCall(id="c0", name="search_evidence",
                                             arguments={"query": "annual leave"})]),
            LLMResponse(text="A minimum of 25 days [Civil Service Annual Leave Policy #0]."),
        ]
        app.state.agent = EvidenceAgent(store=app.state.store, client=_FakeClient(scripted))
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["corpus_chunks"] > 0


def test_ask_returns_cited_answer(client):
    r = client.post("/ask", json={"question": "How much annual leave?"})
    assert r.status_code == 200
    body = r.json()
    assert "25 days" in body["answer"]
    assert len(body["evidence"]) > 0
    assert body["steps"] == 2


def test_ask_validates_empty_question(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422  # Pydantic min_length rejects it
