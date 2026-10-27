"""LLM backends for the tool-calling loop: Anthropic SDK, and an offline deterministic stub."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from ..config import DEFAULT_MODEL, env


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def blocks(self) -> list[dict]:
        out: list[dict] = [{"type": "text", "text": self.text}] if self.text else []
        out += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in self.tool_calls]
        return out


class LLM(Protocol):
    def generate(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class AnthropicLLM:
    """Thin wrapper over the Anthropic Messages API. Key comes from ANTHROPIC_API_KEY."""

    def __init__(self, model: str | None = None, max_tokens: int = 1024) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model or env("LIFEOS_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def generate(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=messages, tools=tools
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text, calls)


def default_llm() -> LLM:
    """Real model if ANTHROPIC_API_KEY is set, otherwise the offline stub."""
    return AnthropicLLM() if env("ANTHROPIC_API_KEY") else StubLLM()


def _tool_results(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in m["content"]:
                if block.get("type") == "tool_result" and not block.get("is_error"):
                    out.append(json.loads(block["content"]))
    return out


class StubLLM:
    """Deterministic scripted policy, not a language model.

    It picks the tool sequence for each agent role (marked `[agent:<role>]` in the
    system prompt) and writes an extractive answer from tool results, so the whole
    loop can be exercised offline and in tests.
    """

    def generate(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        role = (re.search(r"\[agent:(\w+)\]", system) or [None, ""])[1]
        user = messages[0]["content"]
        results = _tool_results(messages)
        step = getattr(self, f"_{role}", None)
        if step is None:
            return LLMResponse(text="(stub) unknown agent role")
        return step(user, results)

    @staticmethod
    def _call(n: int, tool: str, **args) -> LLMResponse:
        return LLMResponse(tool_calls=[ToolCall(f"stub_{n}", tool, args)])

    def _capture(self, user: str, results: list) -> LLMResponse:
        if not results:
            return self._call(0, "add_note", text=user)
        r = results[0]
        by_type: dict[str, list[str]] = {}
        for e in r["entities"]:
            by_type.setdefault(e["type"], []).append(e["name"])
        cats = "; ".join(f"{t}: {', '.join(v)}" for t, v in sorted(by_type.items())) or "none"
        status = "stored" if r["created"] else "already stored (duplicate, no change)"
        return LLMResponse(text=f"Note '{r['title']}' {status}. Categorised entities -> {cats}.")

    def _answer(self, user: str, results: list) -> LLMResponse:
        if not results:
            return self._call(0, "search_memory", query=user, k=4)
        hits = results[0]
        if not hits:
            return LLMResponse(text="I found nothing relevant in memory.")
        lines = []
        for h in hits[:3]:
            first = re.split(r"(?<=[.!?])\s+", h["text"].split("\n", 1)[-1].strip())[0]
            lines.append(f"- {h['title']}: {first} [why: {h['why'][0] if h['why'] else 'n/a'}]")
        return LLMResponse(text="From memory (extractive stub answer):\n" + "\n".join(lines))

    def _planner(self, user: str, results: list) -> LLMResponse:
        if not results:
            return self._call(0, "search_memory", query=user, k=4)
        if len(results) == 1:
            names = [e for h in results[0] for e in h["entities"] if e["type"] in ("Project", "Event")]
            if names:
                return self._call(1, "get_entity", name=names[0]["name"])
        entity = results[-1] if isinstance(results[-1], dict) else None
        tasks = [n for n in (entity or {}).get("neighbors", []) if n["type"] == "Task"]
        if not tasks:
            return LLMResponse(text="No open tasks found; nothing to plan from memory.")
        steps = [
            f"{i}. {t['name']}" + (f" (owner: {', '.join(t['assigned_to'])})" if t.get("assigned_to") else "")
            for i, t in enumerate(tasks, 1)
        ]
        return LLMResponse(text=f"Plan for '{entity['name']}' from open tasks in memory:\n" + "\n".join(steps))
