"""In-memory graph and vector stores: the whole system runs with no Docker."""
from __future__ import annotations

import re
from collections import Counter, defaultdict, deque

import numpy as np

from ..schema import NOTE, Node, Step
from .base import GraphStore, Hit, VectorPoint, VectorStore


class InMemoryVectors(VectorStore):
    def __init__(self) -> None:
        self._points: dict[str, VectorPoint] = {}

    def upsert(self, points: list[VectorPoint]) -> None:
        for p in points:
            self._points[p.id] = p

    def search(self, vector: list[float], k: int) -> list[Hit]:
        if not self._points:
            return []
        ids = list(self._points)
        matrix = np.array([self._points[i].vector for i in ids], dtype=np.float32)
        scores = matrix @ np.asarray(vector, dtype=np.float32)  # cosine: vectors are unit-norm
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(ids[i], float(scores[i]), self._points[ids[i]].payload) for i in order]

    def count(self) -> int:
        return len(self._points)


class InMemoryGraph(GraphStore):
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str, str], dict] = {}
        self._adj: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)
        self._mentions: dict[str, list[str]] = defaultdict(list)  # entity -> notes
        self._note_entities: dict[str, list[str]] = defaultdict(list)

    def upsert_node(self, node: Node) -> None:
        self._nodes[node.id] = node

    def upsert_edge(self, src: str, rel: str, dst: str, note: str | None = None) -> None:
        key = (src, rel, dst)
        if key in self._edges:
            return
        self._edges[key] = {"note": note}
        if rel == "MENTIONS":
            self._mentions[dst].append(src)
            self._note_entities[src].append(dst)
        else:
            self._adj[src].append((rel, True, dst))
            self._adj[dst].append((rel, False, src))

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def entities_in_text(self, text: str) -> list[Node]:
        out = []
        for node in self._nodes.values():
            if node.label in (NOTE, "Task") or len(node.name) < 3:
                continue
            if re.search(rf"(?<!\w){re.escape(node.name)}(?!\w)", text, re.IGNORECASE):
                out.append(node)
        return sorted(out, key=lambda n: n.id)

    def neighbors(self, entity: str) -> list[tuple[str, bool, Node]]:
        return [(rel, fwd, self._nodes[o]) for rel, fwd, o in self._adj.get(entity, [])]

    def expand(self, seed: str, hops: int) -> dict[str, list[Step]]:
        paths: dict[str, list[Step]] = {seed: []}
        queue = deque([seed])
        while queue:
            cur = queue.popleft()
            if len(paths[cur]) >= hops:
                continue
            for rel, fwd, other in self._adj.get(cur, []):
                if other not in paths:
                    paths[other] = paths[cur] + [Step(cur, rel, other, fwd)]
                    queue.append(other)
        paths.pop(seed)
        return paths

    def mentioning_notes(self, entity: str) -> list[Node]:
        return [self._nodes[n] for n in self._mentions.get(entity, [])]

    def entities_of_note(self, note: str) -> list[Node]:
        return [self._nodes[e] for e in self._note_entities.get(note, [])]

    def mention_count(self, entity: str) -> int:
        return len(self._mentions.get(entity, []))

    def stats(self) -> dict[str, int]:
        counts = Counter(n.label for n in self._nodes.values())
        return {**counts, "edges": len(self._edges)}
