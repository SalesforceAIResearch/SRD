#!/bin/bash
# One-click launcher for the local dense-retrieval sidecar that backs the search/open/find tool set.
#
# Usage:
#   # 1. stage the index+corpus ONCE (~40GB download; skips if present):
#   bash examples/SRD/tools/search/run_retrieval.sh --download-only
#   # 2. start the server (idempotent -- no-op if already healthy):
#   bash examples/SRD/tools/search/run_retrieval.sh
set -euo pipefail

# this script lives at examples/SRD/tools/search/ -> repo root is 4 levels up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RETRIEVER_SRC="$REPO_ROOT/examples/search-r1/local_dense_retriever"

RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
RETRIEVAL_DIR="${RETRIEVAL_DIR:-/root/data/wiki18_e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVAL_TOPK="${RETRIEVAL_TOPK:-3}"
FAISS_GPU="${FAISS_GPU:-1}"

INDEX_PATH="$RETRIEVAL_DIR/e5_Flat.index"
CORPUS_PATH="$RETRIEVAL_DIR/wiki-18.jsonl"

mkdir -p "$RETRIEVAL_DIR"

# --------------------------------------------------------------------------- #
# BM25 backend (RETRIEVAL_BACKEND=bm25): a prebuilt Lucene index served by
# pyserini -- pure CPU, no GPU, no e5 encoder. This is the STABLE search engine
# for the multi-task run (the e5 torch-GPU retriever competed with training for
# GPU and was flaky). It is a fully self-contained short-circuit: it stages its
# own index+corpus, installs its own deps, launches, and exits -- the e5/torch/
# faiss code below is never reached when RETRIEVAL_BACKEND=bm25.
#
# Deps note: pyserini's install UPGRADES torch 2.11 -> 2.13, which is ABI-
# incompatible with the container's torchvision/torchaudio 2.11+cu129
# (torchvision::nms missing; torchaudio CUDA-version RuntimeError) and thus
# breaks `import sglang` -- i.e. it BREAKS THE TRAINING CONTAINER. pyserini's
# __init__ also eagerly imports transformers (which trips over those same ABI
# mismatches) and openai (missing-key). BM25/Lucene is pure-Java (pyjnius) and
# needs NONE of torch/torchvision/transformers, so we: bump safetensors back,
# hide torchvision/torchaudio dist-info (transformers availability probes ->
# False), and set a dummy OPENAI_API_KEY.
#
# *** CRITICAL: enroot `--rw` rootfs is PERSISTENT and SHARED across every
# `enroot start` of the same container name -- it is NOT a per-invocation
# overlay. So these torch-mutating installs must run in a DEDICATED throwaway
# container (miles-bm25), NEVER in the training container (miles-sdpo-react-
# cu12), or they silently corrupt training's torch stack. The dedicated
# retriever launcher sets BM25_ALLOW_DEP_SURGERY=1; without it, we refuse the
# destructive install to protect a shared training container. ***
# --------------------------------------------------------------------------- #
if [ "${RETRIEVAL_BACKEND:-}" = "bm25" ]; then
    if [ "${BM25_ALLOW_DEP_SURGERY:-0}" != "1" ]; then
        echo "ERROR: BM25 backend upgrades torch (2.11->2.13) and hides torchvision," >&2
        echo "       which corrupts a SHARED enroot rootfs and breaks training." >&2
        echo "       Run this in a DEDICATED container and set BM25_ALLOW_DEP_SURGERY=1." >&2
        exit 1
    fi
    BM25_DIR="${BM25_DIR:-/root/data/wiki18_bm25}"
    BM25_INDEX="$BM25_DIR/bm25"
    BM25_CORPUS="$BM25_DIR/wiki-18.jsonl"
    # install-jdk names the extracted dir after the EXACT build it fetches
    # (e.g. jdk-21.0.12+8 -> jdk-21.0.12.1+1 as upstream 21 moves), so we install
    # into a fixed PARENT and resolve the actual jdk-* dir below instead of
    # hardcoding a version that drifts and leaves JAVA_HOME/libjvm.so unfindable.
    JDK_PARENT="${JDK_PARENT:-/root/data/.jdk}"

    # Already healthy? -> no-op (idempotent).
    if curl -sf "http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" -X POST \
         -H 'Content-Type: application/json' -d '{"queries":["health probe"],"topk":1}' >/dev/null 2>&1; then
        echo "BM25 retrieval sidecar already healthy on port ${RETRIEVAL_PORT}"
        exit 0
    fi

    # Stage index (prebuilt Lucene, ~430MB) + corpus (the index does NOT store
    # raw docs, so BM25Retriever looks them up positionally: corpus[int(docid)]).
    if [ ! -d "$BM25_INDEX" ] || [ ! -f "$BM25_INDEX/segments_1" ]; then
        echo "Downloading prebuilt wiki-18 BM25 Lucene index to $BM25_DIR ..."
        python -c "from huggingface_hub import snapshot_download; snapshot_download('PeterJinGo/wiki-18-bm25-index', repo_type='dataset', local_dir='$BM25_DIR', allow_patterns=['bm25/*'])"
    fi
    [ -f "$BM25_CORPUS" ] || { echo "ERROR: BM25 corpus $BM25_CORPUS missing (extract from the e5 wiki-18 tar into $BM25_DIR/wiki-18.jsonl)" >&2; exit 1; }

    # Install the JVM once (persistent, on /fsx/data via the mounted /root/data).
    if ! ls "$JDK_PARENT"/jdk-*/bin/java >/dev/null 2>&1; then
        echo "Installing JDK 21 to $JDK_PARENT ..."
        pip install --no-cache-dir --break-system-packages install-jdk >/dev/null 2>&1
        python -c "import jdk; jdk.install('21', path='$JDK_PARENT')"
    fi
    # Resolve whatever jdk-* dir install-jdk actually produced (see comment above).
    JAVA_HOME_DIR="${JAVA_HOME_DIR:-$(ls -d "$JDK_PARENT"/jdk-*/ 2>/dev/null | head -1)}"
    JAVA_HOME_DIR="${JAVA_HOME_DIR%/}"
    if [ ! -f "$JAVA_HOME_DIR/lib/server/libjvm.so" ]; then
        echo "ERROR: libjvm.so not found under $JAVA_HOME_DIR/lib/server (JDK install layout changed?)" >&2; exit 1
    fi
    export JAVA_HOME="$JAVA_HOME_DIR"
    export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"   # pyjnius: skip its version-guessing lookup
    export PATH="$JAVA_HOME/bin:$PATH"

    # Install pyserini into THIS (dedicated, throwaway) container, then repair
    # the deps it perturbs (safetensors pin) and neutralize the imports BM25
    # doesn't need. (Guarded above: BM25_ALLOW_DEP_SURGERY=1 required.)
    python -c "import pyserini" 2>/dev/null || \
        pip install --no-cache-dir --break-system-packages --ignore-installed blinker "pyserini==0.44.0" >/dev/null 2>&1
    pip install --no-cache-dir --break-system-packages "safetensors>=0.8.0" >/dev/null 2>&1
    # pyserini also drags tokenizers DOWN (it resolves for transformers 4.x; the
    # image ships 5.x), and transformers hard-fails at import on the mismatch:
    #   ImportError: tokenizers>=0.23.1,<0.24.0 is required ... found 0.22.2
    # retrieval_server.py imports transformers at module top, so the server dies
    # instantly and the caller only sees a health-probe timeout. BM25 is pure
    # Java/Lucene and never touches tokenizers, so restoring the image's pin is
    # free.
    #
    # Read the range out of dependency_versions_table.py AS TEXT. Two traps:
    # importing anything under `transformers` runs the very check that is failing,
    # and this image's dist-info METADATA disagrees with its own source (METADATA
    # says tokenizers<=0.23.0,>=0.22.0 while the runtime check demands
    # >=0.23.1,<0.24.0), so installing what METADATA asks for changes nothing.
    # find_spec locates the package without executing its __init__.
    _TOK_REQ=$(python - <<'PY' 2>/dev/null
import importlib.util, os, re
spec = importlib.util.find_spec("transformers")
path = os.path.join(spec.submodule_search_locations[0], "dependency_versions_table.py")
m = re.search(r'"tokenizers":\s*"([^"]+)"', open(path).read())
print(m.group(1) if m else "")
PY
)
    if [ -n "$_TOK_REQ" ]; then
        python -c "import transformers" 2>/dev/null || \
            pip install --no-cache-dir --break-system-packages "$_TOK_REQ" >/dev/null 2>&1
    fi
    python -c "import transformers" 2>/dev/null || {
        echo "ERROR: transformers still un-importable after dep repair (wanted '${_TOK_REQ:-?}'):" >&2
        python -c "import transformers" 2>&1 | tail -3 >&2
        exit 1
    }
    # retrieval_server.py imports faiss unconditionally at module top (only the
    # DenseRetriever uses it); BM25 doesn't, but the import must resolve. faiss-cpu
    # is standalone (no torch dep), so it won't perturb the torch stack further.
    python -c "import faiss" 2>/dev/null || \
        pip install --no-cache-dir --break-system-packages faiss-cpu >/dev/null 2>&1
    SP=$(python -c "import site;print(site.getsitepackages()[0])")
    for pkg in torchvision torchaudio; do
        for d in "$SP"/$pkg "$SP"/$pkg-*.dist-info; do
            [ -e "$d" ] && mv "$d" "${d}.HIDDEN" 2>/dev/null || true
        done
    done
    export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy-bm25-not-used}"

    echo "Launching BM25 retrieval server on port ${RETRIEVAL_PORT} (CPU, topk=${RETRIEVAL_TOPK})..."
    nohup python "$RETRIEVER_SRC/retrieval_server.py" \
        --index_path "$BM25_INDEX" \
        --corpus_path "$BM25_CORPUS" \
        --topk "$RETRIEVAL_TOPK" \
        --retriever_name bm25 \
        > "$BM25_DIR/retrieval_server.log" 2>&1 &
    echo "BM25 retrieval server pid $! (log: $BM25_DIR/retrieval_server.log)"
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" -X POST \
             -H 'Content-Type: application/json' -d '{"queries":["health probe"],"topk":1}' >/dev/null 2>&1; then
            echo "BM25 retrieval sidecar up on port ${RETRIEVAL_PORT}"
            exit 0
        fi
        sleep 5
    done
    echo "BM25 retrieval sidecar failed to become healthy within 600s" >&2
    tail -30 "$BM25_DIR/retrieval_server.log" >&2 || true
    exit 1
fi

stage_index() {
    # The index ships as two split parts (part_aa/part_ab) that concatenate into
    # e5_Flat.index, plus a gzipped corpus. Idempotent: skip anything present.
    if [ ! -f "$INDEX_PATH" ]; then
        echo "Downloading wiki-18 e5 index + corpus to $RETRIEVAL_DIR (large, one-time)..."
        python "$RETRIEVER_SRC/download.py" --save_path "$RETRIEVAL_DIR"
        if [ -f "$RETRIEVAL_DIR/part_aa" ] && [ -f "$RETRIEVAL_DIR/part_ab" ]; then
            echo "Concatenating index parts -> e5_Flat.index"
            cat "$RETRIEVAL_DIR/part_aa" "$RETRIEVAL_DIR/part_ab" > "$INDEX_PATH"
            rm -f "$RETRIEVAL_DIR/part_aa" "$RETRIEVAL_DIR/part_ab"
        fi
    fi
    if [ ! -f "$CORPUS_PATH" ] && [ -f "$RETRIEVAL_DIR/wiki-18.jsonl.gz" ]; then
        echo "Decompressing corpus -> wiki-18.jsonl"
        gunzip -k "$RETRIEVAL_DIR/wiki-18.jsonl.gz"
    fi
    [ -f "$INDEX_PATH" ] || { echo "ERROR: $INDEX_PATH missing after staging" >&2; exit 1; }
    [ -f "$CORPUS_PATH" ] || { echo "ERROR: $CORPUS_PATH missing after staging" >&2; exit 1; }
    echo "Index + corpus staged under $RETRIEVAL_DIR"
}

if [ "${1:-}" = "--download-only" ]; then
    stage_index
    exit 0
fi

# Already healthy? -> no-op (idempotent).
if curl -sf "http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" -X POST \
     -H 'Content-Type: application/json' -d '{"queries":["health probe"],"topk":1}' >/dev/null 2>&1; then
    echo "Retrieval sidecar already healthy on port ${RETRIEVAL_PORT}"
    exit 0
fi

stage_index

# Backend: torch (default) | faiss.
#   torch  -> examples/SRD/tools/search/retrieval_server.py: reconstruct
#            the index vectors (CPU faiss) into a GPU torch tensor, search via
#            matmul+topk. Needed because faiss-GPU has NO sm_90 kernels on the
#            H200 (CUDA error 209), while torch runs there fine; CPU-faiss
#            search is ~15s/query (unusable). Needs faiss (CPU, for
#            reconstruct) -- auto-installed if absent.
#   faiss  -> the original Search-R1 server (FAISS_GPU controls CPU/GPU). Kept for
#            hosts where faiss-GPU works.
RETRIEVAL_BACKEND="${RETRIEVAL_BACKEND:-torch}"
echo "Launching retrieval server on port ${RETRIEVAL_PORT} (backend=${RETRIEVAL_BACKEND}, model=${RETRIEVER_MODEL}, topk=${RETRIEVAL_TOPK})..."
if [ "$RETRIEVAL_BACKEND" = "torch" ]; then
    python -c "import faiss" 2>/dev/null || pip install --no-cache-dir --break-system-packages faiss-gpu-cu12 >/dev/null 2>&1 || pip install --no-cache-dir --break-system-packages faiss-cpu >/dev/null 2>&1
    nohup python -m examples.SRD.tools.search.retrieval_server \
        --index_path "$INDEX_PATH" \
        --corpus_path "$CORPUS_PATH" \
        --topk "$RETRIEVAL_TOPK" \
        --retriever_model "$RETRIEVER_MODEL" \
        --port "$RETRIEVAL_PORT" \
        > "$RETRIEVAL_DIR/retrieval_server.log" 2>&1 &
else
    FAISS_FLAG=""
    [ "$FAISS_GPU" = "1" ] && FAISS_FLAG="--faiss_gpu"
    nohup python "$RETRIEVER_SRC/retrieval_server.py" \
        --index_path "$INDEX_PATH" \
        --corpus_path "$CORPUS_PATH" \
        --topk "$RETRIEVAL_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL" \
        $FAISS_FLAG \
        > "$RETRIEVAL_DIR/retrieval_server.log" 2>&1 &
fi

echo "retrieval server pid $! (log: $RETRIEVAL_DIR/retrieval_server.log)"
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" -X POST \
         -H 'Content-Type: application/json' -d '{"queries":["health probe"],"topk":1}' >/dev/null 2>&1; then
        echo "Retrieval sidecar up on port ${RETRIEVAL_PORT}"
        exit 0
    fi
    sleep 5
done
echo "Retrieval sidecar failed to become healthy within 300s" >&2
tail -30 "$RETRIEVAL_DIR/retrieval_server.log" >&2 || true
exit 1
