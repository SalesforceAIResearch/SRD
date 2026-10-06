"""Torch-GPU dense retrieval server — drop-in replacement for Search-R1's faiss-based retrieval_server.py.

Usage (inside the miles container, 1 GPU):
    CUDA_VISIBLE_DEVICES=7 python -m examples.SRD.tools.search.retrieval_server \
        --index_path /root/data/wiki18_e5/e5_Flat.index \
        --corpus_path /root/data/wiki18_e5/wiki-18.jsonl \
        --retriever_model intfloat/e5-base-v2 --topk 3 --port 8000
"""

import argparse
import json

import datasets
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI()
_STATE = {}


def _load_corpus(path):
    return datasets.load_dataset("json", data_files=path, split="train", num_proc=4)


def _load_embeddings_to_gpu(index_path, device, dtype=torch.float16):
    """Reconstruct the raw vectors from the prebuilt faiss Flat index (CPU only)
    and move them onto the GPU as one tensor. float16 halves the 64GB -> ~32GB
    and matmul is fine in fp16 for retrieval ranking."""
    import faiss  # CPU use only (reconstruct); no GPU faiss op -> no sm_90 issue

    print(f"[torch-retriever] reading faiss index {index_path} (CPU)...", flush=True)
    index = faiss.read_index(index_path)
    n, d = index.ntotal, index.d
    print(f"[torch-retriever] reconstructing {n} x {d} vectors...", flush=True)
    # reconstruct_n in chunks to bound host memory spikes
    chunk = 500_000
    parts = []
    for start in range(0, n, chunk):
        k = min(chunk, n - start)
        vecs = index.reconstruct_n(start, k)  # (k, d) float32
        parts.append(torch.from_numpy(np.ascontiguousarray(vecs)).to(dtype))
        if start % 5_000_000 == 0:
            print(f"[torch-retriever]   reconstructed {start + k}/{n}", flush=True)
    emb = torch.cat(parts, dim=0)
    del parts, index
    print(f"[torch-retriever] moving {tuple(emb.shape)} {dtype} to {device} (~{emb.numel()*2/1e9:.0f}GB)...", flush=True)
    emb = emb.to(device)
    # e5 vectors are L2-normalized already, but normalize defensively so a plain
    # dot product == cosine similarity (matches the e5/faiss inner-product ranking).
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb


class Encoder:
    def __init__(self, model_path, device, max_length=256):
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).half().eval()
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, queries):
        # e5 convention: prefix "query: " for queries
        qs = [f"query: {q}" for q in queries]
        inp = self.tok(qs, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        # mean pooling over tokens (e5)
        mask = inp["attention_mask"][..., None].bool()
        summed = (out.last_hidden_state.masked_fill(~mask, 0.0)).sum(dim=1)
        emb = summed / inp["attention_mask"].sum(dim=1)[..., None]
        return torch.nn.functional.normalize(emb, dim=-1).half()


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


@app.post("/retrieve")
def retrieve(req: QueryRequest):
    topk = req.topk or _STATE["default_topk"]
    enc = _STATE["encoder"].encode(req.queries)  # (B, d) on GPU
    emb = _STATE["emb"]  # (N, d) on GPU
    with torch.no_grad():
        scores = enc @ emb.T  # (B, N) cosine sim
        top_scores, top_idx = torch.topk(scores, topk, dim=1)
    corpus = _STATE["corpus"]
    top_idx = top_idx.cpu().tolist()
    top_scores = top_scores.float().cpu().tolist()
    resp = []
    for qi, idxs in enumerate(top_idx):
        hits = []
        for rank, di in enumerate(idxs):
            doc = corpus[int(di)]
            if req.return_scores:
                hits.append({"document": doc, "score": top_scores[qi][rank]})
            else:
                hits.append(doc)
        resp.append(hits)
    return {"result": resp}


@app.get("/health")
def health():
    return {"status": "ok", "n_vectors": _STATE["emb"].shape[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--corpus_path", required=True)
    ap.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[torch-retriever] device={device}", flush=True)
    _STATE["corpus"] = _load_corpus(args.corpus_path)
    print(f"[torch-retriever] corpus loaded: {len(_STATE['corpus'])} docs", flush=True)
    _STATE["emb"] = _load_embeddings_to_gpu(args.index_path, device)
    _STATE["encoder"] = Encoder(args.retriever_model, device)
    _STATE["default_topk"] = args.topk
    print(f"[torch-retriever] READY on :{args.port} ({_STATE['emb'].shape[0]} vectors)", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
