"""Embedders: sentence-transformers MiniLM, plus a deterministic hashing fallback."""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

from .config import env

log = logging.getLogger(__name__)
DIM = 384  # matches all-MiniLM-L6-v2, so the stores are interchangeable
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_STOP = frozenset(
    "a an and are as at be by for from has have i in is it its me my of on or that the "
    "this to was were what when where which who whom why with you your do does did how "
    "about all any can should would will".split()
)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def tokens(text: str) -> list[str]:
    out = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOP:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class HashingEmbedder:
    """Signed feature hashing of unigrams and bigrams. Deterministic and offline.

    It is lexical (no semantics), so it is a test fallback and a baseline,
    not a replacement for a trained model.
    """

    name = "hashing"

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int(hashlib.md5(feature.encode()).hexdigest()[:12], 16)
        return h % self.dim, 1.0 if (h >> 40) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokens(text)
            feats = [(t, 1.0) for t in toks] + [(f"{a}_{b}", 0.5) for a, b in zip(toks, toks[1:])]
            for feat, weight in feats:
                idx, sign = self._bucket(feat)
                out[row, idx] += sign * weight
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:
    name = "minilm"
    dim = DIM

    def __init__(self, model: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(kind: str | None = None) -> Embedder:
    """kind: 'hashing' | 'minilm' | 'auto' (default; MiniLM, else hashing with a warning)."""
    kind = (kind or env("LIFEOS_EMBEDDER", "auto") or "auto").lower()
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "minilm":
        return SentenceTransformerEmbedder()
    if kind != "auto":
        raise ValueError(f"unknown embedder {kind!r}")
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:  # model download or import failure
        log.warning("MiniLM unavailable (%s); using hashing embedder", exc)
        return HashingEmbedder()
