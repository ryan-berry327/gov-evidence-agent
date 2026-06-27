"""
The agent loop: an LLM with a single search tool, run until it produces a final
answer or hits a step limit.

Each turn we send the conversation plus the tool definition to the model. It
either asks to search (we run the search, feed the results back, loop) or returns
text (the final answer). max_agent_steps bounds the loop so it always terminates.

Every chunk retrieved across the run is collected and returned with the answer as
an audit trail, and the system prompt requires the answer to cite sources with
[Title #n] markers.

The loop is provider-agnostic (see llm.py). The only provider-specific bit is how
tool calls and results are echoed back into history, handled in _append_* below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.agent.llm import LLMClient, LLMResponse, get_llm
from app.semantic.store import VectorStore


SYSTEM_PROMPT = """You are an evidence assistant for UK government policy questions.

Rules:
- Answer ONLY using information returned by the search_evidence tool. Do not rely
  on prior knowledge.
- Before answering, search for the evidence you need. You may search more than
  once with different queries if the first results are insufficient.
- Every factual claim in your final answer MUST carry a citation marker in the
  form [Document Title #chunk] taken from the evidence you retrieved.
- If the evidence does not contain the answer, say so plainly. Do not guess.
- Be concise and precise. You are writing for senior, time-poor readers.
"""

SEARCH_TOOL = {
    "name": "search_evidence",
    "description": (
        "Search the document corpus for passages relevant to a query. Returns the "
        "most relevant chunks with citation markers. Call before answering, and "
        "again with a refined query if the first results are insufficient."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused search query capturing what you need to find.",
            }
        },
        "required": ["query"],
    },
}


@dataclass
class AgentResult:
    answer: str
    evidence: list[dict] = field(default_factory=list)
    steps: int = 0
    trace: list[dict] = field(default_factory=list)


class EvidenceAgent:
    def __init__(self, store: VectorStore, client: LLMClient | None = None):
        self.store = store
        self.client = client or get_llm()

    def _run_search(self, query: str) -> tuple[str, list[dict]]:
        hits = self.store.search(query, top_k=settings.top_k)
        evidence, lines = [], []
        for chunk, score in hits:
            evidence.append(
                {
                    "citation": chunk.citation,
                    "doc_title": chunk.doc_title,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "score": round(score, 4),
                }
            )
            lines.append(f"{chunk.citation} (relevance {score:.3f})\n{chunk.text}")
        formatted = "\n\n".join(lines) if lines else "No relevant passages found."
        return formatted, evidence

    def answer(self, question: str) -> AgentResult:
        messages = [{"role": "user", "content": question}]
        all_evidence: list[dict] = []
        trace: list[dict] = []

        for step in range(1, settings.max_agent_steps + 1):
            resp: LLMResponse = self.client.complete(
                system=SYSTEM_PROMPT, messages=messages, tools=[SEARCH_TOOL]
            )

            if not resp.tool_calls:
                return AgentResult(
                    answer=resp.text, evidence=all_evidence, steps=step, trace=trace
                )

            # Echo the assistant turn into history (provider-native form).
            messages.append(self._assistant_turn(resp))

            for tc in resp.tool_calls:
                query = tc.arguments.get("query", "")
                formatted, evidence = self._run_search(query)
                all_evidence.extend(evidence)
                trace.append({"step": step, "query": query, "n_hits": len(evidence)})
                messages.append(self._tool_result_turn(tc.id, formatted))

        return AgentResult(
            answer="(stopped: reached max agent steps without a final answer)",
            evidence=all_evidence,
            steps=settings.max_agent_steps,
            trace=trace,
        )

    # --- provider-format helpers -------------------------------------------
    def _assistant_turn(self, resp: LLMResponse):
        # Reuse the provider's raw turn so tool-call ids line up; otherwise fall
        # back to a generic assistant message.
        if resp.raw_assistant_turn is not None:
            if isinstance(resp.raw_assistant_turn, list):  # Anthropic blocks
                return {"role": "assistant", "content": resp.raw_assistant_turn}
            return resp.raw_assistant_turn  # Ollama message dict
        return {"role": "assistant", "content": resp.text}

    def _tool_result_turn(self, tool_call_id: str, content: str):
        # Anthropic carries tool results in a user turn; Ollama uses a 'tool' role.
        from app.agent.llm import AnthropicClient

        if isinstance(self.client, AnthropicClient):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                    }
                ],
            }
        return {"role": "tool", "content": content}
