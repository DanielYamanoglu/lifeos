def test_graph_expansion_surfaces_note_vector_search_misses(loaded):
    # The parts note never says "Jonas"; it is reachable only Jonas -> Garden Bot -> note.
    vector = loaded.retriever.search("What is Jonas working on?", k=10, mode="vector")
    hybrid = loaded.retriever.search("What is Jonas working on?", k=10, mode="hybrid")
    parts = "Garden Bot parts list"
    rank = lambda res: next((i for i, r in enumerate(res) if r.title == parts), None)  # noqa: E731
    assert rank(hybrid) is not None
    assert rank(vector) is None or rank(hybrid) <= rank(vector)


def test_results_have_provenance(loaded):
    results = loaded.retriever.search("What is Jonas working on?", k=3)
    assert results
    for r in results:
        assert any(w.startswith("vector:") for w in r.provenance) or any(w.startswith("graph:") for w in r.provenance)
        assert r.provenance[-1].startswith("fusion:")
    assert any("Jonas" in w for r in results for w in r.provenance if w.startswith("graph:"))


def test_vector_mode_has_no_graph_provenance(loaded):
    for r in loaded.retriever.search("sourdough starter", k=3, mode="vector"):
        assert all(w.startswith("vector:") for w in r.provenance)


def test_exact_topic_query_finds_matching_note(loaded):
    vector = loaded.retriever.search("sourdough starter feeding schedule", k=1, mode="vector")
    assert vector[0].title == "Sourdough starter notes"
    hybrid = loaded.retriever.search("sourdough starter feeding schedule", k=3)
    assert "Sourdough starter notes" in [r.title for r in hybrid]


def test_bad_mode_rejected(loaded):
    import pytest

    with pytest.raises(ValueError):
        loaded.retriever.search("x", mode="nope")
