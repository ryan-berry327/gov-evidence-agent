"""
Tests the agent LOOP MECHANICS with no API key, no network, no cost, by injecting
a fake LLMClient that returns scripted LLMResponses. Proves the loop is correct
independently of which model backend is used.
"""
from app.semantic.loader import load_corpus
from app.agent.agent import EvidenceAgent
from app.agent.llm import LLMResponse, ToolCall


class _FakeClient:
    """Implements the LLMClient interface with a scripted sequence of responses."""

    def __init__(self, scripted: list[LLMResponse]):
        self._scripted = scripted
        self._i = 0

    def complete(self, system, messages, tools) -> LLMResponse:
        resp = self._scripted[self._i]
        self._i += 1
        return resp


def test_agent_searches_then_answers():
    store = load_corpus("corpus/sample_corpus.json")
    scripted = [
        # Turn 1: model decides to search.
        LLMResponse(tool_calls=[ToolCall(id="c0", name="search_evidence",
                                         arguments={"query": "annual leave entitlement"})]),
        # Turn 2: model returns a final cited answer.
        LLMResponse(text="Civil servants get a minimum of 25 days "
                         "[Civil Service Annual Leave Policy #0]."),
    ]
    agent = EvidenceAgent(store=store, client=_FakeClient(scripted))
    result = agent.answer("How much annual leave do civil servants get?")

    assert result.steps == 2
    assert "25 days" in result.answer
    assert len(result.evidence) > 0
    assert result.trace[0]["query"] == "annual leave entitlement"


def test_agent_respects_step_limit():
    """If the model never stops calling tools, the bound must terminate the loop."""
    store = load_corpus("corpus/sample_corpus.json")
    always_search = LLMResponse(
        tool_calls=[ToolCall(id="c", name="search_evidence", arguments={"query": "x"})]
    )

    class _Loopy:
        def complete(self, system, messages, tools):
            return always_search

    result = EvidenceAgent(store=store, client=_Loopy()).answer("loop forever?")
    assert "max agent steps" in result.answer
    assert result.steps > 0
