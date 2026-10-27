"""FastAPI service: POST /notes, GET /search, GET /entities/{name}."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .memory import Memory, build_memory
from .retrieval import MODES


class NoteIn(BaseModel):
    text: str = Field(min_length=1)
    title: str | None = None


def create_app(memory: Memory | None = None) -> FastAPI:
    app = FastAPI(title="LifeOS", version="0.1.0")
    mem = memory or build_memory()

    @app.post("/notes", status_code=201)
    def post_note(note: NoteIn) -> dict:
        try:
            return mem.add_note(note.text, title=note.title, source="api")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/search")
    def search(q: str = Query(min_length=1), k: int = Query(5, ge=1, le=50), mode: str = "hybrid") -> dict:
        if mode not in MODES:
            raise HTTPException(422, f"mode must be one of {MODES}")
        return {"query": q, "mode": mode, "results": mem.search(q, k=k, mode=mode)}

    @app.get("/entities/{name}")
    def entity(name: str) -> dict:
        found = mem.get_entity(name)
        if found is None:
            raise HTTPException(404, f"no entity named {name!r}")
        return found

    return app
