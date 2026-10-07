"""Sidecar server exposing search/open/find for SDPO_ReAct, built on i-DeepSearch's ``BrowserPool``."""

import logging

from browser import BrowserPool
from fastapi import FastAPI
from pydantic import BaseModel
from retrieval_adapter import app as retrieval_adapter_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("search_sidecar")

app = FastAPI()

# The retrieval adapter (/search, /get_content) is mounted as a sub-app on a
# LOCAL-only path; BrowserPool talks to it via search_url below, over this
# same process's own event loop -- no second port, no second container.
app.mount("/_retrieval_adapter", retrieval_adapter_app)

_POOL = BrowserPool(search_url="http://127.0.0.1:8421/_retrieval_adapter", browser_backend="local")
_INITIALIZED_SESSIONS: set[str] = set()


class CallRequest(BaseModel):
    tool: str
    args: dict = {}


class CallResponse(BaseModel):
    observation: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/session/{session_id}/call", response_model=CallResponse)
async def call(session_id: str, req: CallRequest) -> CallResponse:
    if req.tool not in ("search", "open", "find"):
        return CallResponse(observation=f"error:\nunknown search tool '{req.tool}' (expected search|open|find)")

    if session_id not in _INITIALIZED_SESSIONS:
        _POOL.init_session(session_id)
        _INITIALIZED_SESSIONS.add(session_id)

    try:
        observation = await _POOL.call_tool(session_id, req.tool, req.args)
        return CallResponse(observation=observation)
    except Exception as e:  # a tool-usage error must surface as an observation, not a 500
        logger.warning("session=%s tool=%s failed: %r", session_id, req.tool, e)
        return CallResponse(observation=f"error:\n{e}")


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    """Explicit cleanup hook (not required for correctness -- see the module
    docstring -- but lets a caller that DOES track trajectory completion free
    memory proactively instead of waiting for process exit)."""
    _POOL.cleanup(session_id)
    _INITIALIZED_SESSIONS.discard(session_id)
    return {"status": "ok"}
