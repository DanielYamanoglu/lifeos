"""Tool definitions shared by the agents (and mirrored by the MCP server)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..memory import Memory


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., object]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def memory_tools(memory: Memory) -> dict[str, Tool]:
    def get_entity(name: str):
        found = memory.get_entity(name)
        if found is None:
            raise LookupError(f"no entity named {name!r}")
        return found

    return {
        "search_memory": Tool(
            "search_memory",
            "Hybrid (vector + graph) search over stored notes. Returns notes with a 'why' provenance list.",
            {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 5}},
             "required": ["query"]},
            lambda query, k=5: memory.search(query, k=k),
        ),
        "add_note": Tool(
            "add_note",
            "Store a note. Entities and relations are extracted and linked into the graph.",
            {"type": "object", "properties": {"text": {"type": "string"}, "title": {"type": "string"}},
             "required": ["text"]},
            lambda text, title=None: memory.add_note(text, title=title, source="agent"),
        ),
        "get_entity": Tool(
            "get_entity",
            "Look up an entity (person, project, topic, task, event): neighbours, tasks and notes mentioning it.",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            get_entity,
        ),
    }
