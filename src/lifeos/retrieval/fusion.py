"""Reciprocal rank fusion (Cormack et al., 2009)."""
from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], k: int = 60, weights: dict[str, float] | None = None
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked id lists. Returns (id, score, {source: 1-based rank}) best first.

    score(d) = sum over sources of weight_source / (k + rank_source(d)) (weights default to 1). Ties break on id for determinism.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for source, ids in rankings.items():
        for pos, doc in enumerate(ids, start=1):
            scores[doc] = scores.get(doc, 0.0) + (weights or {}).get(source, 1.0) / (k + pos)
            ranks.setdefault(doc, {})[source] = pos
    return [(d, scores[d], ranks[d]) for d in sorted(scores, key=lambda d: (-scores[d], d))]
