"""
LLM provider abstraction.

The agent talks to one small interface (LLMClient): given a conversation and a set
of tools, return either tool calls or final text. Each backend implements it and
normalises to a common LLMResponse, so the agent loop is identical regardless of
provider.

Backends:
  - OllamaClient    : local models (llama3.1, qwen2.5, ...). Free, for dev/CI.
  - AnthropicClient : frontier API model, for production.
  - the test suite injects a fake implementing the same interface.

The detail being hidden is that providers express tool-calling differently:
Anthropic returns typed content blocks; Ollama (OpenAI-style) returns a tool_calls
array with JSON-string arguments. The adapters absorb that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # raw provider turn, so the agent can append it to history verbatim
    raw_assistant_turn: Any = None


class LLMClient(Protocol):
    def complete(
        self, system: str, messages: list[dict], tools: list[dict]
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------
# Ollama (OpenAI-compatible chat API, served locally at :11434)
# --------------------------------------------------------------------------
class OllamaClient:
    """Talks to a local Ollama server. Requires `ollama serve` running."""

    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def complete(self, system, messages, tools) -> LLMResponse:
        import httpx

        # Ollama uses OpenAI-style tool schemas: {"type":"function","function":{...}}
        ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": ollama_tools,
            "stream": False,
        }
        resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=120.0)
        resp.raise_for_status()
        msg = resp.json()["message"]

        tool_calls = []
        for i, tc in enumerate(msg.get("tool_calls", []) or []):
            fn = tc["function"]
            args = fn["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(id=f"call_{i}", name=fn["name"], arguments=args))

        return LLMResponse(
            text=msg.get("content", "") or "",
            tool_calls=tool_calls,
            raw_assistant_turn=msg,
        )


# --------------------------------------------------------------------------
# Anthropic (production model)
# --------------------------------------------------------------------------
class AnthropicClient:
    """Frontier API model. Used in production."""

    def __init__(self, model: str, api_key: str):
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=api_key)

    def complete(self, system, messages, tools) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input)
            for b in resp.content
            if b.type == "tool_use"
        ]
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(text=text, tool_calls=tool_calls, raw_assistant_turn=resp.content)


def get_llm() -> LLMClient:
    """Pick a backend from env. Defaults to local Ollama."""
    import os

    provider = os.environ.get("LLM_PROVIDER", "ollama")
    if provider == "anthropic":
        return AnthropicClient(
            model=os.environ.get("CHAT_MODEL", "claude-sonnet-4-6"),
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    return OllamaClient(model=os.environ.get("OLLAMA_MODEL", "llama3.1"))
