#!/bin/bash
# One-click HOST bring-up for the search domain's two sidecars (wiki-18 retriever + search/open/find sidecar).
#
# Usage:
#   SDPO_REACT_LOCAL_ROOT=/mnt/bigdisk/miles-run \
#   SQSH=/mnt/bigdisk/miles-run/enroot/miles-v0.1.0-cu12.sqsh \
#     bash examples/SRD/tools/search/prepare-search-host.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Secrets (HF_TOKEN for corpus/index download) from the same .env the launcher uses.
if [ -f "$REPO_ROOT/examples/SRD/.env" ]; then set -a; . "$REPO_ROOT/examples/SRD/.env"; set +a; fi

DATA_DIR="${DATA_DIR:-${SDPO_REACT_LOCAL_ROOT:?set SDPO_REACT_LOCAL_ROOT or DATA_DIR}/data/$USER}"
SQSH="${SQSH:?set SQSH to the imported training-image squashfs}"
BM25_CONTAINER="${BM25_CONTAINER:-miles-bm25}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
SEARCH_PORT="${SEARCH_PORT:-8421}"
BM25_DIR="$DATA_DIR/wiki18_bm25"
mkdir -p "$BM25_DIR"

health_bm25() { curl -sf "http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" -X POST \
    -H 'Content-Type: application/json' -d '{"queries":["health probe"],"topk":1}' >/dev/null 2>&1; }

# --- 1. retriever (dedicated container, detached, kept alive) -----------------
if health_bm25; then
    echo "[search-host] BM25 retriever already healthy on :${RETRIEVAL_PORT}"
else
    enroot list 2>/dev/null | grep -qx "$BM25_CONTAINER" || enroot create --name "$BM25_CONTAINER" "$SQSH"
    # In-container entry: stage corpus (standalone, NOT the 40GB e5 index) then
    # hand off to run_retrieval.sh's BM25 branch (index dl + jdk + pyserini +
    # serve). `tail -f` keeps the enroot session -- and thus the nohup'd server
    # -- alive after run_retrieval.sh returns.
    cat > "$BM25_DIR/_entry.sh" <<'ENTRY'
set -eux
BM25_DIR=/root/data/wiki18_bm25
if [ ! -f "$BM25_DIR/wiki-18.jsonl" ]; then
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('PeterJinGo/wiki-18-corpus','wiki-18.jsonl.gz',repo_type='dataset',local_dir='$BM25_DIR')"
    gunzip -kf "$BM25_DIR/wiki-18.jsonl.gz"
fi
RETRIEVAL_BACKEND=bm25 BM25_ALLOW_DEP_SURGERY=1 bash /root/miles/examples/SRD/tools/search/run_retrieval.sh
echo "[search-host] retriever serving; holding session open"
tail -f "$BM25_DIR/retrieval_server.log"
ENTRY
    echo "[search-host] starting BM25 retriever in $BM25_CONTAINER (detached)..."
    nohup enroot start --rw \
        --mount "$REPO_ROOT":/root/miles \
        --mount "$DATA_DIR":/root/data \
        --env HOME=/root --env HF_TOKEN="${HF_TOKEN:-}" \
        --env XDG_CACHE_HOME=/root/data/.cache \
        --env NVIDIA_VISIBLE_DEVICES=all \
        --env SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
        "$BM25_CONTAINER" bash /root/data/wiki18_bm25/_entry.sh \
        > "$BM25_DIR/prepare-retriever.log" 2>&1 &
    echo "[search-host] retriever launcher pid $! (log: $BM25_DIR/prepare-retriever.log)"
    echo "[search-host] waiting for :${RETRIEVAL_PORT} (corpus dl + deps, up to ~15min)..."
    for _ in $(seq 1 180); do health_bm25 && break; sleep 5; done
    health_bm25 || { echo "[search-host] retriever failed to come up; see $BM25_DIR/{prepare-retriever,retrieval_server}.log" >&2; exit 1; }
    echo "[search-host] BM25 retriever healthy on :${RETRIEVAL_PORT}"
fi

# --- 2. search sidecar (docker, detached) ------------------------------------
PORT="$SEARCH_PORT" RETRIEVAL_PORT="$RETRIEVAL_PORT" bash "$SCRIPT_DIR/run_search_sidecar.sh"
echo "[search-host] search stack READY (retriever :${RETRIEVAL_PORT}, sidecar :${SEARCH_PORT})."
