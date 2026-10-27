"""Qdrant vector store. Used when QDRANT_URL is set."""
from __future__ import annotations

from qdrant_client import QdrantClient, models

from .base import Hit, VectorPoint, VectorStore


class QdrantVectors(VectorStore):
    def __init__(self, url: str, dim: int, collection: str = "lifeos_chunks") -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )

    def upsert(self, points: list[VectorPoint]) -> None:
        self._client.upsert(
            self._collection,
            points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
        )

    def search(self, vector: list[float], k: int) -> list[Hit]:
        res = self._client.query_points(self._collection, query=vector, limit=k, with_payload=True)
        return [Hit(str(p.id), float(p.score), p.payload or {}) for p in res.points]

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count
