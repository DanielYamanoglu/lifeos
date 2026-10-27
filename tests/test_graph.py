from lifeos.schema import Node
from lifeos.storage import InMemoryGraph


def build():
    g = InMemoryGraph()
    for i, label in [("e:a", "Person"), ("e:b", "Project"), ("e:c", "Topic"), ("e:d", "Event")]:
        g.upsert_node(Node(i, label, i[2:].upper()))
    g.upsert_edge("e:a", "WORKS_ON", "e:b")
    g.upsert_edge("e:b", "RELATED_TO", "e:c")
    g.upsert_edge("e:d", "RELATED_TO", "e:c")
    g.upsert_node(Node("n1", "Note", "note one", {}))
    g.upsert_edge("n1", "MENTIONS", "e:d")
    return g


def test_expand_respects_hops_and_records_paths():
    g = build()
    one = g.expand("e:a", 1)
    assert set(one) == {"e:b"}
    two = g.expand("e:a", 3)
    assert set(two) == {"e:b", "e:c", "e:d"}
    assert [s.rel for s in two["e:d"]] == ["WORKS_ON", "RELATED_TO", "RELATED_TO"]
    assert [s.forward for s in two["e:d"]] == [True, True, False]


def test_mentions_do_not_create_entity_paths():
    g = build()
    assert "n1" not in g.expand("e:d", 3)
    assert [n.id for n in g.mentioning_notes("e:d")] == ["n1"]
    assert g.mention_count("e:d") == 1


def test_edge_upsert_is_idempotent():
    g = build()
    before = g.stats()
    g.upsert_edge("e:a", "WORKS_ON", "e:b")
    assert g.stats() == before


def test_entities_in_text_is_whole_word_and_case_insensitive():
    g = build()
    g.upsert_node(Node("e:garden-bot", "Project", "Garden Bot"))
    assert [n.id for n in g.entities_in_text("the garden bot works")] == ["e:garden-bot"]
    assert g.entities_in_text("gardenbots") == []
