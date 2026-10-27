"""MCP server (stdio) exposing the memory as tools: search_memory, add_note, get_entity.

Run:  python -m lifeos.mcp_server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .memory import Memory, build_memory


def create_server(memory: Memory | None = None) -> FastMCP:
    server = FastMCP("lifeos")
    holder: dict[str, Memory] = {}

    def mem() -> Memory:
        if memory is not None:
            return memory
        return holder.setdefault("m", build_memory())

    @server.tool()
    def search_memory(query: str, k: int = 5) -> list[dict]:
        """Hybrid (vector + graph) search over stored notes; each result carries a 'why' provenance list."""
        return mem().search(query, k=k)

    @server.tool()
    def add_note(text: str, title: str | None = None) -> dict:
        """Store a note; entities and relations are extracted and linked into the graph. Idempotent."""
        return mem().add_note(text, title=title, source="mcp")

    @server.tool()
    def get_entity(name: str) -> dict:
        """Look up an entity: its type, graph neighbours and the notes that mention it."""
        found = mem().get_entity(name)
        if found is None:
            raise ValueError(f"no entity named {name!r}")
        return found

    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
