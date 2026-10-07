"""Public entry point for the webshop_step sidecar, routing requests to N worker processes by task_id."""

import asyncio
import hashlib
import json
import os

import httpx
from fastapi import FastAPI, Request
from starlette.responses import Response

N_WORKERS = int(os.environ.get("WEBSHOP_N_WORKERS", "16"))
_WORKER_BASE_PORT = int(os.environ.get("WEBSHOP_WORKER_BASE_PORT", "9101"))
_WORKER_URLS = [f"http://127.0.0.1:{_WORKER_BASE_PORT + i}" for i in range(N_WORKERS)]

app = FastAPI()
_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))


def _worker_index(key: str) -> int:
    """Stable hash of ``key`` (a task_id or session_id) -> worker index.

    blake2b, not the builtin hash() (PYTHONHASHSEED-salted per process, so it
    would map the same key to a different worker on every restart).
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % N_WORKERS


async def _proxy(worker_url: str, method: str, path: str, body: bytes | None) -> Response:
    try:
        resp = await _client.request(method, f"{worker_url}{path}", content=body)
    except httpx.HTTPError as e:
        return Response(
            content=json.dumps({"error": f"worker unreachable: {e}"}).encode(),
            status_code=503,
            media_type="application/json",
        )
    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


@app.get("/health")
async def health() -> Response:
    return await _proxy(_WORKER_URLS[0], "GET", "/health", None)


@app.post("/session/{session_id}/step")
async def step(session_id: str, request: Request) -> Response:
    body = await request.body()
    # task_id (when present) is the routing key -- see module docstring.
    # Falls back to session_id-hash only if the request has no task_id
    # (every real webshop_step call has one; see ../client.py).
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = {}
    task_id = payload.get("task_id")
    key = str(task_id) if task_id is not None else session_id
    worker_url = _WORKER_URLS[_worker_index(key)]
    return await _proxy(worker_url, "POST", f"/session/{session_id}/step", body)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> Response:
    # No task_id in a DELETE -- can't route by the same key /step used to
    # create the session (see module docstring), so broadcast to every
    # worker instead of guessing. All but the one owning worker no-op
    # (cleanup() on an absent session_id is a plain dict.pop(..., None)).
    responses = await asyncio.gather(
        *(_proxy(url, "DELETE", f"/session/{session_id}", None) for url in _WORKER_URLS)
    )
    return next((r for r in responses if r.status_code == 200), responses[0])
