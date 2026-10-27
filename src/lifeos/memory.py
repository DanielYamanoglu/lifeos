"""`Memory`: one facade over graph, vectors, ingestion and retrieval.
The API, the MCP server and the agents' tools all go through it."""
from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embedder, get_embedder
from .ingest import Ingestor
from .retrieval import HybridRetriever
from .schema import entity_id
from .storage import GraphStore, VectorStore, build_graph, build_vectors


@dataclass
class Memory:
    graph: GraphStore
    vectors: VectorStore
    embedder: Embedder
    ingestor: Ingestor
    retriever: HybridRetriever

    def add_note(self, text: str, title: str | None = None, source: str = "api") -> dict:
        r = self.ingestor.ingest_text(text, title=title, source=source)
        return {
            "note_id": r.note_id, "title": r.title, "created": r.created, "chunks": r.chunks,
            "entities": [{"name": e.name, "type": e.type} for e in r.entities],
            "relations": [{"src": x.src, "rel": x.rel, "dst": x.dst} for x in r.relations],
        }

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
        out = []
        for res in self.retriever.search(query, k=k, mode=mode):
            d = res.to_dict()
            d["entities"] = [{"name": e.name, "type": e.label} for e in self.graph.entities_of_note(res.note_id)]
            out.append(d)
        return out

    def get_entity(self, name: str) -> dict | None:
        node = self.graph.get_node(entity_id(name))
        if node is None or node.label == "Note":
            return None
        neighbors = []
        for rel, fwd, other in self.graph.neighbors(node.id):
            item = {"rel": rel, "direction": "out" if fwd else "in", "name": other.name, "type": other.label}
            if other.label == "Task":
                item["assigned_to"] = [o.name for r, f, o in self.graph.neighbors(other.id) if r == "ASSIGNED_TO" and f]
            neighbors.append(item)
        notes = [{"note_id": n.id, "title": n.name} for n in self.graph.mentioning_notes(node.id)]
        return {"name": node.name, "type": node.label, "neighbors": neighbors, "notes": notes}


def build_memory(embedder: Embedder | None = None, graph: GraphStore | None = None,
                 vectors: VectorStore | None = None, hops: int = 2) -> Memory:
    embedder = embedder or get_embedder()
    graph = graph or build_graph()
    vectors = vectors or build_vectors(embedder.dim)
    return Memory(graph, vectors, embedder, Ingestor(graph, vectors, embedder),
                  HybridRetriever(graph, vectors, embedder, hops=hops))
