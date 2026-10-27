"""Three small agents: each is a system prompt plus a subset of the memory tools."""
from __future__ import annotations

from dataclasses import dataclass

from ..memory import Memory
from .llm import LLM, default_llm
from .loop import AgentRun, run_agent
from .tools import memory_tools


@dataclass
class Agent:
    name: str
    system: str
    tool_names: tuple[str, ...]
    memory: Memory
    llm: LLM

    def run(self, user: str, max_steps: int = 6) -> AgentRun:
        tools = memory_tools(self.memory)
        return run_agent(self.llm, self.system, user, [tools[n] for n in self.tool_names], max_steps)


def capture_agent(memory: Memory, llm: LLM | None = None) -> Agent:
    return Agent("capture", (
        "[agent:capture] You file incoming notes into a personal knowledge graph. Call add_note once with the "
        "user's text, then summarise what was stored and which entities (people, projects, topics, tasks, events) "
        "it was categorised into."), ("add_note",), memory, llm or default_llm())


def answer_agent(memory: Memory, llm: LLM | None = None) -> Agent:
    return Agent("answer", (
        "[agent:answer] Answer questions using only search_memory results. Cite the note titles and the 'why' "
        "provenance. If memory has nothing relevant, say so."), ("search_memory",), memory, llm or default_llm())


def planner_agent(memory: Memory, llm: LLM | None = None) -> Agent:
    return Agent("planner", (
        "[agent:planner] Turn a goal into an ordered plan. Use search_memory to find the relevant project or "
        "event, get_entity to read its open tasks and owners, then write a numbered plan."),
        ("search_memory", "get_entity"), memory, llm or default_llm())
