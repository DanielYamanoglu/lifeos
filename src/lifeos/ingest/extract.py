"""Entity and relation extraction into the typed schema.

`RuleBasedExtractor` is a transparent offline baseline (cues, markup, capitalised
phrases). `LLMExtractor` asks any LLM for JSON and falls back to the rules on
malformed output. Both return the same `Extraction`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from ..schema import ENTITY_TYPES, REL_TYPES, Entity, Relation, slugify


@dataclass
class Extraction:
    entities: list[Entity]
    relations: list[Relation]


class Extractor(Protocol):
    def extract(self, text: str) -> Extraction: ...


_STOP = {
    "i", "we", "it", "he", "she", "they", "the", "this", "that", "these", "those", "then", "there", "next",
    "also", "after", "before", "today", "tomorrow", "yesterday", "note", "remember", "todo", "my", "our",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "a", "an", "and", "but",
    "if", "so", "for", "on", "in", "at", "to", "with", "when", "while", "all", "any", "some", "one",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "ask", "call", "email", "remind", "tell", "ping", "send", "check", "finish", "mqtt", "decided", "plan", "agenda", "ideas", "need", "needs",
}
_EVENT_WORDS = {"night", "trip", "meeting", "sync", "dinner", "workshop", "retro", "standup", "club", "festival", "call"}
_PERSON_BEFORE = re.compile(r"\b(with|from|ask|asked|call|called|email|emailed|told|tell|thanks to|met|meet|ping|and)\s+$", re.I)
_PERSON_VERBS = {
    "said", "suggested", "recommended", "owns", "will", "is", "wants", "leads", "mentioned", "loves", "prefers",
    "plans", "works", "builds", "gave", "asked", "brings", "can", "needs", "thinks", "offered", "found",
}
_WORK_CUE = re.compile(r"\b(works? on|working on|builds?|building|owns?|leads?)\b", re.I)
_ATTEND_CUE = re.compile(r"\b(with|attend\w*|join\w*|going)\b", re.I)
_CAP = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[ ][A-Z][A-Za-z0-9]*)*\b")
_TASK_LINE = re.compile(r"^\s*(?:[-*]\s*\[ \]|TODO:?)\s*(.+?)\s*$", re.M)


def _classify(sentence: str, start: int, end: int, prev: Entity | None, prev_end: int) -> str:
    before, after = sentence[:start], sentence[end:]
    if re.search(r"\bproject\s+$", before, re.I) or re.match(r"\s+project\b", after, re.I):
        return "Project"
    if sentence[start:end].split()[-1].lower() in _EVENT_WORDS:
        return "Event"
    if _PERSON_BEFORE.search(before):
        # "and"/"," continues a list only if the previous mention was a person
        if not re.search(r"\band\s+$", before, re.I) or (prev and prev.type == "Person" and sentence[prev_end:start].strip() == "and"):
            return "Person"
    nxt = after.split()[:1]
    if len(sentence[start:end].split()) == 1 and nxt and nxt[0].strip(",.:;").lower() in _PERSON_VERBS:
        return "Person"
    return "Topic"


def _relation(a: Entity, b: Entity, sentence: str) -> Relation | None:
    pair = {a.type: a, b.type: b}
    if a.type == b.type == "Topic" or a.name == b.name:
        return Relation(a.name, "RELATED_TO", b.name) if a.name != b.name else None
    if "Person" in pair and "Project" in pair:
        rel = "WORKS_ON" if _WORK_CUE.search(sentence) else "RELATED_TO"
        return Relation(pair["Person"].name, rel, pair["Project"].name)
    if "Person" in pair and "Event" in pair:
        rel = "ATTENDS" if _ATTEND_CUE.search(sentence) else "RELATED_TO"
        return Relation(pair["Person"].name, rel, pair["Event"].name)
    if "Task" in pair:
        other = b if a.type == "Task" else a
        rel = {"Person": "ASSIGNED_TO", "Project": "PART_OF", "Event": "PART_OF"}.get(other.type, "RELATED_TO")
        return Relation(pair["Task"].name, rel, other.name)
    return Relation(a.name, "RELATED_TO", b.name)


class RuleBasedExtractor:
    def extract(self, text: str) -> Extraction:
        entities: dict[str, Entity] = {}
        relations: dict[tuple, Relation] = {}

        def add(ent: Entity) -> Entity:
            existing = entities.get(slugify(ent.name))
            if existing and (existing.type != "Topic" or ent.type == "Topic"):
                return existing
            entities[slugify(ent.name)] = ent
            return ent

        for m in re.finditer(r"(?<![\w&])#([a-z][\w-]*)", text):
            add(Entity(m.group(1), "Topic"))

        # Tasks first: "- [ ] Do the thing (Assignee)"
        for m in _TASK_LINE.finditer(text):
            raw = m.group(1)
            assignee = re.search(r"\(([A-Z][a-z]+)\)\s*", raw)
            name = re.sub(r"\s*\([A-Z][a-z]+\)", "", raw).strip(" .")
            task = add(Entity(name, "Task"))
            members = [task]
            if assignee:
                person = add(Entity(assignee.group(1), "Person"))
                relations[(task.name, "ASSIGNED_TO", person.name)] = Relation(task.name, "ASSIGNED_TO", person.name)
                members.append(person)
            for ent in self._sentence_entities(name.split(" ", 1)[-1], add):  # skip the leading verb
                members.append(ent)
                rel = _relation(task, ent, name)
                if rel:
                    relations[(rel.src, rel.rel, rel.dst)] = rel

        body = _TASK_LINE.sub("", text)
        for line in body.splitlines():
            if line.lstrip().startswith("#"):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                found = self._sentence_entities(sentence, add)
                for i, a in enumerate(found):
                    for b in found[i + 1 :]:
                        if len(found) >= 3 and "Person" not in (a.type, b.type):
                            continue  # enumerations ("A, B and C") do not imply the items are related
                        rel = _relation(a, b, sentence)
                        if rel:
                            relations[(rel.src, rel.rel, rel.dst)] = rel
        return Extraction(list(entities.values()), list(relations.values()))

    def _sentence_entities(self, sentence: str, add) -> list[Entity]:
        found: list[Entity] = []
        for m in re.finditer(r"@([A-Za-z]\w*)", sentence):
            found.append(add(Entity(m.group(1), "Person")))
        sentence = re.sub(r"@\w+", " ", sentence)
        for m in re.finditer(r"\*([^*\n]{2,60})\*", sentence):  # *Italic Titles* are topics (books, papers)
            found.append(add(Entity(m.group(1).strip(), "Topic")))
        sentence = re.sub(r"\*[^*\n]+\*", " ", sentence)

        prev: Entity | None = None
        prev_end = 0
        for m in _CAP.finditer(sentence):
            words = m.group().split()
            lead = 0
            while lead < len(words) and words[lead].lower() in _STOP:
                lead += 1
            words = words[lead:]
            if not words:
                continue
            start = m.start() + len(m.group()) - len(" ".join(words))
            end = m.end()
            at_sentence_start = not sentence[:start].strip(" -*>\"'(")
            etype = _classify(sentence, start, end, prev, prev_end)
            if len(words) == 1 and at_sentence_start and etype != "Person":
                continue
            if len(words) == 1 and words[0].lower() in _STOP:
                continue
            name = " ".join(words)
            if etype == "Project":
                name = re.sub(r"\s+[Pp]roject$", "", name)
            prev = add(Entity(name, etype))
            prev_end = end
            found.append(prev)
        # de-duplicate, keep order
        seen, uniq = set(), []
        for e in found:
            if slugify(e.name) not in seen:
                seen.add(slugify(e.name))
                uniq.append(e)
        return uniq


_LLM_PROMPT = """Extract entities and relations from the note as JSON only.
Entity types: {types}. Relation types: {rels}.
Schema: {{"entities":[{{"name":str,"type":str}}],"relations":[{{"src":str,"rel":str,"dst":str}}]}}
Relations connect entity names; do not output MENTIONS.

Note:
{text}"""


class LLMExtractor:
    """Wraps any callable `complete(prompt) -> str`. Invalid output falls back to rules."""

    def __init__(self, complete, fallback: Extractor | None = None) -> None:
        self._complete = complete
        self._fallback = fallback or RuleBasedExtractor()

    def extract(self, text: str) -> Extraction:
        prompt = _LLM_PROMPT.format(types=", ".join(ENTITY_TYPES), rels=", ".join(REL_TYPES), text=text)
        try:
            raw = self._complete(prompt)
            data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            ents = [Entity(str(e["name"]).strip(), e["type"]) for e in data["entities"] if e["type"] in ENTITY_TYPES]
            names = {e.name for e in ents}
            rels = [
                Relation(r["src"], r["rel"], r["dst"])
                for r in data.get("relations", [])
                if r["rel"] in REL_TYPES and r["rel"] != "MENTIONS" and r["src"] in names and r["dst"] in names
            ]
            return Extraction(ents, rels)
        except (ValueError, KeyError, TypeError):
            return self._fallback.extract(text)
