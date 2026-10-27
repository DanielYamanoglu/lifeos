"""Capture pipeline: text -> chunks -> embeddings -> entities/relations -> idempotent upserts."""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..embeddings import Embedder
from ..schema import NOTE, Entity, Node, Relation, entity_id
from ..storage.base import GraphStore, VectorPoint, VectorStore
from .adapters import ChatAdapter
from .chunking import chunk_text
from .extract import Extractor, RuleBasedExtractor

_NS = uuid.UUID("6f1d3f0e-5a77-4d52-9d1e-0b8f2b7c9a11")


def content_hash(text: str) -> str:
    normalized = text.replace("\r\n", "\n").strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class IngestResult:
    note_id: str
    title: str
    created: bool
    chunks: int
    entities: list[Entity]
    relations: list[Relation]


def _title(text: str) -> str:
    m = re.match(r"\s*#\s+(.+)", text)
    if m:
        return m.group(1).strip()
    first = text.strip().splitlines()[0]
    return first if len(first) <= 50 else first[:50].rsplit(" ", 1)[0] + "..."


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)


class Ingestor:
    def __init__(self, graph: GraphStore, vectors: VectorStore, embedder: Embedder, extractor: Extractor | None = None):
        self.graph, self.vectors, self.embedder = graph, vectors, embedder
        self.extractor = extractor or RuleBasedExtractor()

    def ingest_text(self, text: str, title: str | None = None, source: str = "text") -> IngestResult:
        """Idempotent: the note id is a content hash; re-ingesting identical text changes nothing."""
        text = _strip_frontmatter(text.replace("\r\n", "\n")).strip()
        if not text:
            raise ValueError("empty note")
        note_id = content_hash(text)
        title = title or _title(text)
        extraction = self.extractor.extract(text)
        if self.graph.has_node(note_id):
            return IngestResult(note_id, title, False, 0, extraction.entities, extraction.relations)

        # Vectors first, the Note node last: the Note node is the "ingest completed" marker,
        # so a crash in between is repaired by simply re-running (all upserts are idempotent).
        chunks = chunk_text(f"{title}\n\n{text}")
        vectors = self.embedder.embed(chunks)
        self.vectors.upsert([
            VectorPoint(
                str(uuid.uuid5(_NS, f"{note_id}:{i}")),
                vec.tolist(),
                {"note_id": note_id, "chunk_idx": i, "text": chunk, "title": title},
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ])

        known = self.graph.entities_in_text(text)  # link mentions of entities seen in earlier notes
        ids: dict[str, str] = {}
        for ent in extraction.entities:
            ids[ent.name] = self._upsert_entity(ent)
        for node in known:
            ids.setdefault(node.name, node.id)

        self.graph.upsert_node(Node(note_id, NOTE, title, {"source": source, "text": text, "chunks": len(chunks)}))
        for eid in ids.values():
            self.graph.upsert_edge(note_id, "MENTIONS", eid, note_id)
        for rel in extraction.relations:
            if rel.src in ids and rel.dst in ids:
                self.graph.upsert_edge(ids[rel.src], rel.rel, ids[rel.dst], note_id)
        return IngestResult(note_id, title, True, len(chunks), extraction.entities, extraction.relations)

    def _upsert_entity(self, ent: Entity) -> str:
        eid = entity_id(ent.name)
        existing = self.graph.get_node(eid)
        if existing and (existing.label != "Topic" or ent.type == "Topic"):
            return eid  # first-seen type wins unless it was only a generic Topic
        self.graph.upsert_node(Node(eid, ent.type, existing.name if existing else ent.name))
        return eid

    def ingest_markdown(self, path: str | Path) -> IngestResult:
        path = Path(path)
        return self.ingest_text(path.read_text(encoding="utf-8"), source=path.stem)

    def ingest_chat(self, adapter: ChatAdapter) -> list[IngestResult]:
        return [
            self.ingest_text(m.text, source=f"{adapter.name}:{m.id}")
            for m in adapter.fetch()
        ]
