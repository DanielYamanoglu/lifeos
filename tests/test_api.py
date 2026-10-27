from fastapi.testclient import TestClient

from lifeos.api import create_app


def test_api_roundtrip(memory):
    client = TestClient(create_app(memory))
    r = client.post("/notes", json={"text": "# Trip\n\nLisbon Trip is planned with Maya. #travel"})
    assert r.status_code == 201 and r.json()["created"] is True
    assert client.post("/notes", json={"text": "# Trip\n\nLisbon Trip is planned with Maya. #travel"}).json()["created"] is False

    found = client.get("/search", params={"q": "Lisbon trip", "k": 3}).json()
    assert found["results"][0]["title"] == "Trip" and found["results"][0]["why"]

    ent = client.get("/entities/maya").json()
    assert ent["type"] == "Person" and ent["neighbors"][0]["name"] == "Lisbon Trip"


def test_api_errors(memory):
    client = TestClient(create_app(memory))
    assert client.get("/entities/nobody").status_code == 404
    assert client.get("/search").status_code == 422
    assert client.get("/search", params={"q": "x", "mode": "bogus"}).status_code == 422
    assert client.post("/notes", json={"text": "   "}).status_code == 422
