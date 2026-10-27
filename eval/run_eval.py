"""Retrieval evaluation on the small FICTIONAL sample set: vector-only vs graph vs hybrid.

    python eval/run_eval.py [--embedders hashing,minilm]
Writes reports/eval_results.json, reports/eval_results.md and reports/figures/retrieval_eval.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifeos.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from lifeos.memory import build_memory
from lifeos.retrieval import HybridRetriever
from lifeos.storage import InMemoryGraph, InMemoryVectors

ROOT = Path(__file__).resolve().parent.parent
KS = (3, 5)
CONFIGS = [
    ("vector only", "vector", 2),
    ("graph only", "graph", 2),
    ("hybrid (1 hop)", "hybrid", 1),
    ("hybrid (2 hops)", "hybrid", 2),
]


def metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    out = {f"recall@{k}": len(relevant & set(ranked[:k])) / len(relevant) for k in KS}
    out["mrr"] = next((1.0 / i for i, s in enumerate(ranked, 1) if s in relevant), 0.0)
    return out


def evaluate(embedder) -> dict:
    mem = build_memory(embedder, InMemoryGraph(), InMemoryVectors())
    for p in sorted((ROOT / "examples" / "sample_notes").glob("*.md")):
        mem.ingestor.ingest_markdown(p)
    queries = json.loads((ROOT / "eval" / "queries.json").read_text())
    table = {}
    for name, mode, hops in CONFIGS:
        retriever = HybridRetriever(mem.graph, mem.vectors, embedder, hops=hops)
        per_type: dict[str, list[dict]] = {"all": [], "lexical": [], "relational": []}
        for item in queries:
            ranked = [r.source for r in retriever.search(item["q"], k=10, mode=mode)]
            m = metrics(ranked, set(item["relevant"]))
            per_type["all"].append(m)
            per_type[item["type"]].append(m)
        table[name] = {
            t: {key: sum(m[key] for m in ms) / len(ms) for key in ms[0]} | {"n": len(ms)}
            for t, ms in per_type.items()
        }
    return table


def markdown(results: dict) -> str:
    lines = ["# Retrieval evaluation (small, fictional set)\n"]
    n_notes = len(list((ROOT / "examples" / "sample_notes").glob("*.md")))
    n_q = len(json.loads((ROOT / "eval" / "queries.json").read_text()))
    lines.append(f"{n_notes} fictional notes, {n_q} hand-written queries with hand-labelled relevant notes. "
                 "Metrics are macro-averaged over queries; MRR uses the top 10.\n")
    for emb, table in results.items():
        for subset in ("all", "lexical", "relational"):
            n = next(iter(table.values()))[subset]["n"]
            lines += [f"## Embedder: {emb} - {subset} queries (n={n})\n",
                      "| Retriever | Recall@3 | Recall@5 | MRR |", "|---|---|---|---|"]
            for name, t in table.items():
                m = t[subset]
                lines.append(f"| {name} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} |")
            lines.append("")
    return "\n".join(lines)


def plot(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#8b8fa3", "#c9b27c", "#4f7cac", "#1f4e79"]
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.4), squeeze=False, facecolor="white")
    for ax, (emb, table) in zip(axes[0], results.items()):
        keys = ["recall@3", "recall@5", "mrr"]
        width = 0.2
        for i, (name, t) in enumerate(table.items()):
            vals = [t["all"][k] for k in keys]
            xs = [j + (i - 1.5) * width for j in range(len(keys))]
            bars = ax.bar(xs, vals, width, label=name, color=colors[i])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(keys)), ["Recall@3", "Recall@5", "MRR"])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Embedder: {emb} (13 fictional queries)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", default="hashing,minilm")
    args = ap.parse_args()
    factories = {"hashing": HashingEmbedder, "minilm": SentenceTransformerEmbedder}
    results = {}
    for name in args.embedders.split(","):
        try:
            results[name] = evaluate(factories[name]())
        except Exception as exc:  # e.g. model not downloadable offline
            print(f"skipping embedder {name}: {exc}")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "eval_results.json").write_text(json.dumps(results, indent=2))
    md = markdown(results)
    (ROOT / "reports" / "eval_results.md").write_text(md)
    plot(results, ROOT / "reports" / "figures" / "retrieval_eval.png")
    print(md)


if __name__ == "__main__":
    main()
