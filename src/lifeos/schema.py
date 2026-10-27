"""Typed graph schema shared by ingestion, storage and retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ENTITY_TYPES = ("Person", "Project", "Topic", "Task", "Event")
NOTE = "Note"
REL_TYPES = ("MENTIONS", "PART_OF", "RELATED_TO", "ASSIGNED_TO", "WORKS_ON", "ATTENDS")
ENTITY_REL_TYPES = tuple(r for r in REL_TYPES if r != "MENTIONS")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def entity_id(name: str) -> str:
    """Entities are keyed by normalised name, so 'Garden Bot' == 'garden bot'."""
    return f"e:{slugify(name)}"


@dataclass(frozen=True)
class Entity:
    name: str
    type: str


@dataclass(frozen=True)
class Relation:
    """Directed relation between two entity names."""

    src: str
    rel: str
    dst: str


@dataclass
class Node:
    id: str
    label: str  # one of ENTITY_TYPES or "Note"
    name: str
    props: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    """One hop of a graph path; `forward` tells whether the edge points frm -> to."""

    frm: str
    rel: str
    to: str
    forward: bool
