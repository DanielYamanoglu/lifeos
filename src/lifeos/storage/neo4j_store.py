"""Neo4j graph store. Used when NEO4J_URI is set.

Model: (:Note {id,name,text,...})-[:MENTIONS]->(:Entity:<Type> {id,name,type}),
entities linked by PART_OF / RELATED_TO / ASSIGNED_TO / WORKS_ON / ATTENDS.
Labels and relationship types are interpolated into Cypher, so they are only
ever taken from the whitelists in `schema`.
"""
from __future__ import annotations

import re

from neo4j import GraphDatabase

from ..schema import ENTITY_REL_TYPES, ENTITY_TYPES, NOTE, REL_TYPES, Node, Step
from .base import GraphStore

_ENTITY_RELS = "|".join(ENTITY_REL_TYPES)


def _node(record) -> Node:
    props = dict(record)
    label = props.pop("type", None) or NOTE
    return Node(props.pop("id"), label, props.pop("name"), props)


class Neo4jGraph(GraphStore):
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        for label in ("Entity", "Note"):
            self._run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

    def _run(self, query: str, **params):
        records, _, _ = self._driver.execute_query(query, params)
        return records

    def upsert_node(self, node: Node) -> None:
        if node.label == NOTE:
            self._run("MERGE (n:Note {id:$id}) SET n.name=$name, n += $props", id=node.id, name=node.name, props=node.props)
            return
        if node.label not in ENTITY_TYPES:
            raise ValueError(f"unknown node label {node.label!r}")
        drop = "".join(f":{t}" for t in ENTITY_TYPES if t != node.label)
        self._run(
            f"MERGE (n:Entity {{id:$id}}) SET n:{node.label}, n.name=$name, n.type=$type, n += $props REMOVE n{drop}",
            id=node.id, name=node.name, type=node.label, props=node.props,
        )

    def upsert_edge(self, src: str, rel: str, dst: str, note: str | None = None) -> None:
        if rel not in REL_TYPES:
            raise ValueError(f"unknown relation {rel!r}")
        self._run(
            f"MATCH (a:Entity|Note {{id:$src}}) MATCH (b:Entity {{id:$dst}}) "
            f"MERGE (a)-[r:{rel}]->(b) ON CREATE SET r.note=$note",
            src=src, dst=dst, note=note,
        )

    def get_node(self, node_id: str) -> Node | None:
        rows = self._run("MATCH (n:Entity|Note {id:$id}) RETURN n", id=node_id)
        return _node(rows[0]["n"]) if rows else None

    def entities_in_text(self, text: str) -> list[Node]:
        rows = self._run(
            "MATCH (e:Entity) WHERE e.type <> 'Task' AND size(e.name) >= 3 "
            "AND toLower($text) CONTAINS toLower(e.name) RETURN e ORDER BY e.id",
            text=text,
        )
        nodes = [_node(r["e"]) for r in rows]
        return [n for n in nodes if re.search(rf"(?<!\w){re.escape(n.name)}(?!\w)", text, re.IGNORECASE)]

    def neighbors(self, entity: str) -> list[tuple[str, bool, Node]]:
        rows = self._run(
            f"MATCH (a:Entity {{id:$id}})-[r:{_ENTITY_RELS}]-(b:Entity) "
            "RETURN type(r) AS rel, startNode(r).id = a.id AS fwd, b ORDER BY b.id",
            id=entity,
        )
        return [(r["rel"], r["fwd"], _node(r["b"])) for r in rows]

    def expand(self, seed: str, hops: int) -> dict[str, list[Step]]:
        hops = int(hops)
        rows = self._run(
            f"MATCH p=(s:Entity {{id:$id}})-[:{_ENTITY_RELS}*1..{hops}]-(t:Entity) WHERE t.id <> $id "
            "WITH t, p ORDER BY length(p) WITH t, collect(p)[0] AS best "
            "RETURN t.id AS target, [n IN nodes(best) | n.id] AS ids, "
            "[r IN relationships(best) | type(r)] AS rels, "
            "[r IN relationships(best) | startNode(r).id] AS starts",
            id=seed,
        )
        out = {}
        for r in rows:
            ids, rels, starts = r["ids"], r["rels"], r["starts"]
            out[r["target"]] = [Step(ids[i], rels[i], ids[i + 1], starts[i] == ids[i]) for i in range(len(rels))]
        return out

    def mentioning_notes(self, entity: str) -> list[Node]:
        rows = self._run("MATCH (n:Note)-[:MENTIONS]->(:Entity {id:$id}) RETURN n ORDER BY n.id", id=entity)
        return [_node(r["n"]) for r in rows]

    def entities_of_note(self, note: str) -> list[Node]:
        rows = self._run("MATCH (:Note {id:$id})-[:MENTIONS]->(e:Entity) RETURN e ORDER BY e.id", id=note)
        return [_node(r["e"]) for r in rows]

    def mention_count(self, entity: str) -> int:
        rows = self._run("MATCH (:Note)-[:MENTIONS]->(:Entity {id:$id}) RETURN count(*) AS c", id=entity)
        return int(rows[0]["c"])

    def stats(self) -> dict[str, int]:
        out = {r["l"]: r["c"] for r in self._run("MATCH (n:Entity|Note) RETURN coalesce(n.type,'Note') AS l, count(*) AS c")}
        out["edges"] = self._run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        return out

    def close(self) -> None:
        self._driver.close()
