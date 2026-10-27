"""`python -m lifeos` starts the HTTP API."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("lifeos.api:create_app", factory=True, host="0.0.0.0", port=8000)
