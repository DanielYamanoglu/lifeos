import os

for var in ("NEO4J_URI", "QDRANT_URL", "ANTHROPIC_API_KEY"):
    os.environ.pop(var, None)
os.environ["LIFEOS_EMBEDDER"] = "hashing"

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from lifeos.embeddings import HashingEmbedder  # noqa: E402
from lifeos.memory import build_memory  # noqa: E402
from lifeos.storage import InMemoryGraph, InMemoryVectors  # noqa: E402

NOTES = Path(__file__).resolve().parent.parent / "examples" / "sample_notes"


@pytest.fixture
def memory():
    return build_memory(HashingEmbedder(), InMemoryGraph(), InMemoryVectors())


@pytest.fixture
def loaded(memory):
    for p in sorted(NOTES.glob("*.md")):
        memory.ingestor.ingest_markdown(p)
    return memory
