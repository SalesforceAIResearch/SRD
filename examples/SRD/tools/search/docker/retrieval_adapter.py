"""Adapter exposing ``POST /search`` + ``POST /get_content`` backed by the wiki-18 ``/retrieve`` retriever."""

import hashlib
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

RETRIEVAL_URL = os.environ.get("SDPO_REACT_SEARCH_URL", "http://127.0.0.1:8000/retrieve")

app = FastAPI()

# pseudo-url -> (title, full body text). One process per search sidecar
# container, so this is naturally scoped to that container's lifetime -- see
# server.py's module docstring for why no persistence/cleanup is needed.
_DOC_CACHE: dict[str, tuple[str, str]] = {}


def _split_title_and_text(contents: str) -> tuple[str, str]:
    """wiki-18 rows are stored as '"Title"\\nBody text...' (see
    examples/SRD/tools/registry.py's original web_search parsing of the
    same corpus)."""
    contents = contents or ""
    if contents.startswith('"'):
        end = contents.find('"', 1)
        if end > 0:
            title = contents[1:end].strip()
            body = contents[end + 1 :].strip()
            return title or "(untitled)", body
    head, _, rest = contents.partition("\n")
    return (head.strip() or "(untitled)"), rest.strip()


class SearchRequest(BaseModel):
    query: str
    topn: int = 10
    reasoning: str | None = None


class SearchResponseItem(BaseModel):
    url: str
    title: str
    summary: str


class FetchRequest(BaseModel):
    url: str


class FetchResponse(BaseModel):
    title: str
    content: str


@app.get("/", tags=["meta"])
def root():
    return {"service": "wiki18 retrieval adapter", "endpoints": ["/search", "/get_content"]}


@app.post("/search", response_model=dict[str, list[SearchResponseItem]], tags=["search"])
async def search(request: SearchRequest):
    qhash = hashlib.sha1(request.query.encode()).hexdigest()[:12]
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(RETRIEVAL_URL, json={"queries": [request.query], "topk": request.topn, "return_scores": False})
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"retriever error {resp.status_code}: {resp.text}")
        data = resp.json()

    hits = (data.get("result") or [[]])[0]
    results = []
    for rank, doc in enumerate(hits):
        contents = doc.get("contents", "") if isinstance(doc, dict) else str(doc)
        title, body = _split_title_and_text(contents)
        url = f"wiki18://{qhash}/{rank}"
        _DOC_CACHE[url] = (title, body)
        results.append(SearchResponseItem(url=url, title=title, summary=body[:500]))
    return {"results": results}


@app.post("/get_content", response_model=FetchResponse, tags=["content"])
def get_content(request: FetchRequest):
    cached = _DOC_CACHE.get(request.url)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"URL not found: {request.url} (only urls from the MOST RECENT /search in this "
            "container's memory can be fetched -- the wiki-18 index has no stable per-doc id to "
            "re-look-up by).",
        )
    title, content = cached
    return FetchResponse(title=title, content=content)
