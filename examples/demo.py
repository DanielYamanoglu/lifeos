"""Offline demo: ingest the fictional notes, run hybrid queries with provenance, run the agents.

    python examples/demo.py            # prints markdown to stdout
Uses the deterministic hashing embedder and the stub LLM, so it needs no network, keys or Docker.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from lifeos.agents import StubLLM, answer_agent, capture_agent, planner_agent
from lifeos.embeddings import HashingEmbedder
from lifeos.ingest import ChatMessage, FakeChatAdapter
from lifeos.memory import build_memory
from lifeos.storage import InMemoryGraph, InMemoryVectors

NOTES = Path(__file__).parent / "sample_notes"
QUERIES = [
    "What is Jonas working on?",
    "Who is helping with the trip?",
    "What did Lena recommend?",
]


def snippet(text: str, n: int = 150) -> str:
    body = " ".join(text.split("\n", 2)[-1].split())
    return body if len(body) <= n else body[: n - 3] + "..."


def main() -> None:
    mem = build_memory(HashingEmbedder(), InMemoryGraph(), InMemoryVectors())
    print("# LifeOS demo output\n")
    print("Embedder: hashing (offline). LLM: stub (offline). Stores: in-memory. All notes are fictional.\n")

    print("## 1. Ingest\n")
    results = [mem.ingestor.ingest_markdown(p) for p in sorted(NOTES.glob("*.md"))]
    results += mem.ingestor.ingest_chat(FakeChatAdapter([
        ChatMessage("101", "alex", "Remember to ask Tomas about the Lisbon Trip tram pass. #travel"),
    ]))
    print(f"Ingested {sum(r.created for r in results)} notes. Graph: {mem.graph.stats()}. Vector chunks: {mem.vectors.count()}.\n")
    again = [mem.ingestor.ingest_markdown(p) for p in sorted(NOTES.glob("*.md"))]
    print(f"Re-ingesting the same files: {sum(r.created for r in again)} created, "
          f"{sum(not r.created for r in again)} skipped as duplicates. Vector chunks still {mem.vectors.count()}.\n")

    print("## 2. Hybrid retrieval with provenance\n")
    for q in QUERIES:
        print(f"### Query: {q}\n")
        for i, r in enumerate(mem.retriever.search(q, k=4), 1):
            print(f"{i}. **{r.title}** (score {r.score:.4f})  ")
            print(f"   {snippet(r.text)}")
            for why in r.provenance:
                print(f"   - {why}")
        print()

    print("## 3. Graph lookup: `get_entity('Garden Bot')`\n")
    ent = mem.get_entity("Garden Bot")
    print(f"{ent['name']} ({ent['type']}); mentioned in {len(ent['notes'])} notes.\n")
    for n in ent["neighbors"]:
        arrow = "->" if n["direction"] == "out" else "<-"
        extra = f" assigned_to={n['assigned_to']}" if n.get("assigned_to") else ""
        print(f"- {arrow} {n['rel']} {n['name']} [{n['type']}]{extra}")
    print()

    print("## 4. Agents (stub LLM, tool-calling loop)\n")
    llm = StubLLM()
    for agent, prompt in [
        (capture_agent(mem, llm), "Jonas suggested a waterproof case for Garden Bot. Alex will order it. #electronics"),
        (answer_agent(mem, llm), "What is Jonas working on?"),
        (planner_agent(mem, llm), "Plan the Lisbon Trip"),
    ]:
        run = agent.run(prompt)
        print(f"### {agent.name} agent\n")
        print(f"Input: {prompt}\n")
        print("Tool calls: " + (", ".join(f"`{t['tool']}`" for t in run.trace) or "none") + f" ({run.steps} model turns)\n")
        print(textwrap.indent(run.final_text, "> ") + "\n")


if __name__ == "__main__":
    main()
