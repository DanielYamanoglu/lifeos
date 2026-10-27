"""Paragraph-aware chunking with a hard-split fallback for very long sentences."""
from __future__ import annotations

import re


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    step = max(1, max_chars - overlap)
    return [text[i : i + max_chars].strip() for i in range(0, len(text), step) if text[i : i + max_chars].strip()]


def _pieces(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        out.extend([sentence] if len(sentence) <= max_chars else _hard_split(sentence, max_chars, overlap))
    return out


def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
    """Pack paragraphs (then sentences) into chunks of at most `max_chars`."""
    chunks: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text.strip())):
        for piece in _pieces(paragraph, max_chars, overlap) if paragraph else []:
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks
