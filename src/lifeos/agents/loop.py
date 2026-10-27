"""Minimal tool-calling loop: call the model, run requested tools, feed results back."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLM
from .tools import Tool


@dataclass
class AgentRun:
    final_text: str
    trace: list[dict] = field(default_factory=list)  # [{"tool", "input", "output"|"error"}]
    steps: int = 0


def run_agent(llm: LLM, system: str, user: str, tools: list[Tool], max_steps: int = 6) -> AgentRun:
    registry = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    messages: list[dict] = [{"role": "user", "content": user}]
    run = AgentRun("")
    for step in range(1, max_steps + 1):
        resp = llm.generate(system, messages, specs)
        run.steps = step
        if not resp.tool_calls:
            run.final_text = resp.text
            return run
        messages.append({"role": "assistant", "content": resp.blocks()})
        results = []
        for call in resp.tool_calls:
            tool = registry.get(call.name)
            try:
                if tool is None:
                    raise LookupError(f"unknown tool {call.name!r}")
                output = tool.fn(**call.input)
                content = json.dumps(output)
                run.trace.append({"tool": call.name, "input": call.input, "output": output})
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": content})
            except Exception as exc:  # surfaced to the model so it can recover
                run.trace.append({"tool": call.name, "input": call.input, "error": str(exc)})
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": str(exc), "is_error": True})
        messages.append({"role": "user", "content": results})
    run.final_text = f"(stopped after {max_steps} steps without a final answer)"
    return run
