"""Storage interfaces. In-memory and server-backed implementations are interchangeable."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..schema import Node, Step


@dataclass
class VectorPoint:
    id: str
    vector: list[float]
    payload: dict = field(default_factory=dict)


@dataclass
class Hit:
    id: str
    score: float
    payload: dict


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, points: list[VectorPoint]) -> None: ...

    @abstractmethod
    def search(self, vector: list[float], k: int) -> list[Hit]: ...

    @abstractmethod
    def count(self) -> int: ...


class GraphStore(ABC):
    @abstractmethod
    def upsert_node(self, node: Node) -> None: ...

    @abstractmethod
    def upsert_edge(self, src: str, rel: str, dst: str, note: str | None = None) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> Node | None: ...

    @abstractmethod
    def entities_in_text(self, text: str) -> list[Node]:
        """Known (non-Task) entities whose name occurs in `text` (case-insensitive, whole word)."""

    @abstractmethod
    def neighbors(self, entity: str) -> list[tuple[str, bool, Node]]:
        """Entity-to-entity edges as (rel, forward, other_node)."""

    @abstractmethod
    def expand(self, seed: str, hops: int) -> dict[str, list[Step]]:
        """Shortest path (undirected, entity edges only) from seed to every entity within `hops`."""

    @abstractmethod
    def mentioning_notes(self, entity: str) -> list[Node]: ...

    @abstractmethod
    def entities_of_note(self, note: str) -> list[Node]: ...

    @abstractmethod
    def mention_count(self, entity: str) -> int: ...

    @abstractmethod
    def stats(self) -> dict[str, int]: ...

    def has_node(self, node_id: str) -> bool:
        return self.get_node(node_id) is not None

    def close(self) -> None:
        return None
