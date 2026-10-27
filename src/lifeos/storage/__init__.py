from __future__ import annotations

from ..config import env
from .base import GraphStore, Hit, VectorPoint, VectorStore
from .memory import InMemoryGraph, InMemoryVectors


def build_graph() -> GraphStore:
    """Neo4j when NEO4J_URI is set, otherwise in-memory."""
    uri = env("NEO4J_URI")
    if not uri:
        return InMemoryGraph()
    from .neo4j_store import Neo4jGraph

    return Neo4jGraph(uri, env("NEO4J_USER", "neo4j"), env("NEO4J_PASSWORD", ""))


def build_vectors(dim: int) -> VectorStore:
    """Qdrant when QDRANT_URL is set, otherwise in-memory."""
    url = env("QDRANT_URL")
    if not url:
        return InMemoryVectors()
    from .qdrant_store import QdrantVectors

    return QdrantVectors(url, dim)


__all__ = ["GraphStore", "VectorStore", "VectorPoint", "Hit", "InMemoryGraph", "InMemoryVectors", "build_graph", "build_vectors"]
