"""Hybrid retrieval: Qdrant vector search + Neo4j graph expansion, fused with RRF."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..embeddings import Embedder
from ..schema import Step
from ..storage.base import GraphStore, VectorStore
from .fusion import reciprocal_rank_fusion

MODES = ("hybrid", "vector", "graph")


@dataclass
class SearchResult:
    note_id: str
    title: str
    source: str
    text: str
    score: float
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "note_id": self.note_id, "title": self.title, "source": self.source,
            "text": self.text, "score": round(self.score, 4), "why": self.provenance,
        }


def _path(graph: GraphStore, seed: str, steps: list[Step]) -> str:
    def name(i: str) -> str:
        node = graph.get_node(i)
        return f"'{node.name}'" if node else i

    out = name(seed)
    for s in steps:
        out += f" -{s.rel}-> {name(s.to)}" if s.forward else f" <-{s.rel}- {name(s.to)}"
    return out


class HybridRetriever:
    def __init__(self, graph: GraphStore, vectors: VectorStore, embedder: Embedder, hops: int = 2,
                 vector_k: int = 20, seed_top_n: int = 3, rrf_k: int = 60, hub_ratio: float = 0.4,
                 seed_ratio: float = 0.6, graph_weight: float = 0.6) -> None:
        self.graph, self.vectors, self.embedder = graph, vectors, embedder
        self.hops, self.vector_k, self.seed_top_n, self.rrf_k = hops, vector_k, seed_top_n, rrf_k
        self.hub_ratio, self.seed_ratio, self.graph_weight = hub_ratio, seed_ratio, graph_weight

    # -- vector side -------------------------------------------------------
    def _vector_ranking(self, query: str) -> tuple[list[str], dict[str, str], dict[str, float]]:
        hits = self.vectors.search(self.embedder.embed([query])[0].tolist(), self.vector_k)
        best: dict[str, tuple[float, int]] = {}
        for h in hits:
            nid = h.payload["note_id"]
            if nid not in best:  # hits arrive best-first
                best[nid] = (h.score, h.payload["chunk_idx"])
        ranking = list(best)
        why = {nid: f"vector: best chunk #{c} cosine={s:.2f} (vector rank {i})" for i, (nid, (s, c)) in enumerate(best.items(), 1)}
        return ranking, why, {n: sc for n, (sc, _) in best.items()}

    # -- graph side --------------------------------------------------------
    def _graph_ranking(self, query: str, vector_ranking: list[str], v_scores: dict[str, float]) -> tuple[list[str], dict[str, list[str]]]:
        seeds: dict[str, tuple[float, str]] = {}  # entity id -> (weight, reason)
        # Only confident vector hits seed the graph: within seed_ratio of the best cosine.
        best = v_scores[vector_ranking[0]] if vector_ranking else 0.0
        seed_notes = [n for n in vector_ranking[: self.seed_top_n] if best > 0.05 and v_scores[n] >= self.seed_ratio * best]
        n_notes = max(1, self.graph.stats().get("Note", 1))
        for node in self.graph.entities_in_text(query):
            seeds[node.id] = (1.0, f"query mentions '{node.name}'")
        for nid in seed_notes:
            title = self.graph.get_node(nid).name
            for ent in self.graph.entities_of_note(nid):
                is_hub = self.graph.mention_count(ent.id) / n_notes > self.hub_ratio
                if ent.label != "Task" and not is_hub:  # a hub (e.g. the note-taker) links everything to everything
                    seeds.setdefault(ent.id, (0.5, f"top vector hit '{title}' mentions '{ent.name}'"))

        contributions: dict[str, list[tuple[float, str]]] = {}
        hub: dict[str, float] = {}

        def damp(entity: str) -> float:  # hub entities (mentioned everywhere) say little
            if entity not in hub:
                hub[entity] = 1.0 / (1.0 + math.log(max(1, self.graph.mention_count(entity))))
            return hub[entity]

        for seed, (weight, reason) in seeds.items():
            reached = {seed: []} | self.graph.expand(seed, self.hops)
            for ent, steps in reached.items():
                named = weight == 1.0 and not steps  # an entity the user named is not a hub, however common
                score = weight / (1 + len(steps)) * (1.0 if named else damp(ent))
                where = _path(self.graph, seed, steps) if steps else f"'{self.graph.get_node(seed).name}'"
                for note in self.graph.mentioning_notes(ent):
                    hop_txt = f"{len(steps)} hop{'s' if len(steps) != 1 else ''}"
                    contributions.setdefault(note.id, []).append(
                        (score, f"graph: {reason}; {where} (mentioned in this note, {hop_txt})")
                    )
        totals = {n: sum(s for s, _ in c) for n, c in contributions.items()}
        ranking = sorted(totals, key=lambda n: (-totals[n], self.graph.get_node(n).name))
        why = {n: [r for _, r in sorted(c, key=lambda x: -x[0])[:2]] for n, c in contributions.items()}
        return ranking, why

    # -- public ------------------------------------------------------------
    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[SearchResult]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        v_rank, v_why, v_scores = self._vector_ranking(query)
        g_why: dict[str, list[str]] = {}
        if mode == "vector":
            fused = [(n, v_scores[n], {}) for n in v_rank]
        else:
            g_rank, g_why = self._graph_ranking(query, v_rank, v_scores)
            if mode == "graph":
                fused = reciprocal_rank_fusion({"graph": g_rank}, self.rrf_k)
            else:
                fused = reciprocal_rank_fusion({"vector": v_rank, "graph": g_rank}, self.rrf_k,
                                               {"graph": self.graph_weight})
        results = []
        for note_id, score, ranks in fused[:k]:
            node = self.graph.get_node(note_id)
            why = ([v_why[note_id]] if note_id in v_why and mode != "graph" else []) + g_why.get(note_id, [])
            if mode == "hybrid":
                why.append("fusion: " + ", ".join(f"{s} rank {r}" for s, r in ranks.items()) + f" -> weighted RRF {score:.4f}")
            results.append(SearchResult(note_id, node.name, node.props.get("source", ""), node.props.get("text", ""), score, why))
        return results
