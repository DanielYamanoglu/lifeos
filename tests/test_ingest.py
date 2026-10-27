import json

import pytest

from lifeos.embeddings import HashingEmbedder
from lifeos.ingest import ChatMessage, FakeChatAdapter, LLMExtractor, RuleBasedExtractor


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.embed(["garden bot watering"]), e.embed(["garden bot watering"])
    assert (a == b).all()
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5
    related = float(e.embed(["watering the garden"])[0] @ a[0])
    unrelated = float(e.embed(["tomato ragu recipe"])[0] @ a[0])
    assert related > unrelated


def test_rule_extractor_types_and_relations():
    text = ("Garden Bot project notes. Jonas works on the sensor driver for the Garden Bot project. #electronics\n"
            "- [ ] Order a relay board (Alex) for the Garden Bot project")
    x = RuleBasedExtractor().extract(text)
    types = {e.name: e.type for e in x.entities}
    assert types["Jonas"] == "Person" and types["Garden Bot"] == "Project" and types["electronics"] == "Topic"
    assert types["Order a relay board for the Garden Bot project"] == "Task"
    rels = {(r.src, r.rel, r.dst) for r in x.relations}
    assert ("Jonas", "WORKS_ON", "Garden Bot") in rels
    assert ("Order a relay board for the Garden Bot project", "ASSIGNED_TO", "Alex") in rels


def test_idempotent_upsert(memory):
    text = "# Pasta night\n\nPasta Night is on Friday with Jonas and Priya. #cooking"
    first = memory.ingestor.ingest_text(text)
    stats = (memory.graph.stats(), memory.vectors.count())
    second = memory.ingestor.ingest_text(text + "\n")  # same content modulo trailing whitespace
    assert first.created and not second.created
    assert first.note_id == second.note_id
    assert (memory.graph.stats(), memory.vectors.count()) == stats


def test_entities_are_shared_across_notes(memory):
    memory.ingestor.ingest_text("Jonas works on the sensor for the Garden Bot project.")
    memory.ingestor.ingest_text("Reminder: garden bot needs a new pump, ask Jonas.")
    assert memory.graph.mention_count("e:garden-bot") == 2  # second note linked by name, not by capitalisation
    assert memory.graph.mention_count("e:jonas") == 2


def test_empty_note_rejected(memory):
    with pytest.raises(ValueError):
        memory.ingestor.ingest_text("   ")


def test_chat_adapter_ingests_messages(memory):
    adapter = FakeChatAdapter([ChatMessage("1", "alex", "Ask Tomas about the tram pass. #travel")])
    [res] = memory.ingestor.ingest_chat(adapter)
    assert res.created
    assert memory.ingestor.ingest_chat(adapter)[0].created is False


def test_llm_extractor_parses_json_and_falls_back():
    payload = json.dumps({"entities": [{"name": "Maya", "type": "Person"}, {"name": "Trip", "type": "Event"},
                                       {"name": "x", "type": "Bogus"}],
                          "relations": [{"src": "Maya", "rel": "ATTENDS", "dst": "Trip"},
                                        {"src": "Maya", "rel": "MENTIONS", "dst": "Trip"}]})
    x = LLMExtractor(lambda prompt: f"Sure! {payload}").extract("whatever")
    assert [(e.name, e.type) for e in x.entities] == [("Maya", "Person"), ("Trip", "Event")]
    assert [(r.rel) for r in x.relations] == ["ATTENDS"]
    fallback = LLMExtractor(lambda prompt: "not json").extract("Ask Tomas about it. #travel")
    assert {e.name for e in fallback.entities} == {"travel", "Tomas"}
