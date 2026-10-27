import asyncio

from lifeos.mcp_server import create_server


def test_mcp_tools_listed_and_callable(memory):
    server = create_server(memory)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"search_memory", "add_note", "get_entity"}

    asyncio.run(server.call_tool("add_note", {"text": "Lisbon Trip is planned with Maya. #travel"}))
    result = asyncio.run(server.call_tool("get_entity", {"name": "Maya"}))
    assert "Lisbon Trip" in str(result)
