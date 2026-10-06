#!/bin/bash
# One-click ENROOT launcher for SDPO_ReAct — no sudo, no Docker daemon inside
# the training container. Sibling of examples/SRD/enroot-run-sdpo.sh: same
# asset/cache/enroot plumbing, pointed at the SDPO_ReAct run script.
#
# Docker note: the code_interpreter sandbox sidecar (tools/docker/) is a REAL
# Docker container, started on the HOST via tools/run_sandbox.sh BEFORE the
# enroot session starts (not from inside enroot -- enroot has no Docker-in-
# Docker story, and doesn't need one: unlike Docker, enroot containers share
# the host's network namespace by default, so 127.0.0.1:8420 inside the
# enroot session already reaches the host-side sandbox container's published
# port). This keeps the "exactly one extra port for the whole job" property:
# the sandbox is a host-level singleton, independent of how many enroot/train
# sessions come and go.
#
# The search/open/find sidecar (tools/search/docker/, port 8421) and the
# wiki-18 retriever it routes onto (tools/search/run_retrieval.sh, port 8000)
# are STARTED BY THE RUN SCRIPT ITSELF (the ablation/run-*-mathcodesearch.sh
# matrix), conditionally on $SDPO_REACT_DOMAIN -- not unconditionally here, since only
# the multitask/search domains need them and the retriever needs its own GPU
# reservation, which is a per-run decision, not a per-enroot-session one.
#
# The webshop/alfworld (SDPO_REACT_RUN_FAMILY=agentic) train/eval jsonl files
# ARE built HERE, on the host, unlike search's -- because build_webshop_data.py
# / build_alfworld_data.py MUST import gym / alfworld, which are deliberately
# NOT installed in the training image (see tools/{webshop,alfworld}/docker/
# Dockerfile), and the training session has no `docker` binary to shell out
# to the sidecar with once it's inside enroot. So this launcher starts both
# sidecars and `docker exec`s the builders into them before `enroot start`,
# writing straight onto $DATA_DIR (the host path enroot mounts as /root/data)
# -- the run script's own [-f ...] || python -m ...build_*_data then no-ops.
#
#   IMAGE   (default radixark/miles:latest-cu12)   docker image (driver 570 -> cu12)
#   SQSH    (default $ENROOT_NVME/miles-cu12.sqsh)  imported squashfs image
#   CONTAINER (default miles-sdpo-react-cu12)       enroot container name
#   ASSETS  (default /opt/dlami/nvme/miles-assets)  models + data (local nvme, ephemeral)
#   DATA_DIR (default /fsx/data/$USER)              training checkpoints (shared, durable
#                                                    network storage -- NOT /fsx/home, which
#                                                    is much smaller/quota-limited and where a
#                                                    checkpoint write once genuinely failed
#                                                    mid-save from disk pressure)
#   SDPO_REACT_MODEL (default qwen2.5)              qwen2.5 | olmo3 -- picks the HF repo,
#                                                    local asset dir name, megatron model-arg
#                                                    script, and run script (same switch
#                                                    pattern as examples/SRD/enroot-run-sdpo.sh's
#                                                    SDPO_MODEL)
#   SDPO_REACT_RUN_FAMILY (default native)          native (math/code/search, run-*-native.sh)
#                                                    | agentic (webshop/alfworld, run-*-agentic.sh)
#                                                    -- orthogonal to SDPO_REACT_MODEL (which
#                                                    picks weights/tool-grammar within EITHER
#                                                    family); each model-family case branch below
#                                                    sets *_NATIVE_RUN_SH/*_AGENTIC_RUN_SH,
#                                                    this var picks which one actually runs.
#   PREP_ONLY=1  prepare assets but do not train
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Auto-load secrets (WANDB_API_KEY, HF_TOKEN, and the optional LLM-judge gateway
# LLM_GATEWAY_KEY / OPENAI_API_URL) from examples/SRD/.env if present.
# The file is gitignored — never commit it.
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; . "$SCRIPT_DIR/.env"; set +a
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# radixark/miles:v0.1.0-cu12 -- the RELEASE image (torch 2.11.0+cu129, same
# torch as the original validated dev-cu12-202607040446 / sglang dev13799
# build, which upstream has since PRUNED). A release tag (unlike dated dev-cu12-*
# builds) is not garbage-collected, so it stays reproducible; verified on this
# host to import cleanly, pass megatron.core+TE import, carry the Qwen3.5 plugin
# (get_qwen3_5_spec), and convert Qwen3.5-4B HF->torch_dist. Do NOT use
# latest-cu12: it has drifted to torch 2.13 with a mismatched TransformerEngine
# 2.17 (undefined symbol at import) and cannot run the convert or training.
IMAGE="${IMAGE:-radixark/miles:v0.1.0-cu12}"

# SDPO_REACT_LOCAL_ROOT: one knob to relocate ALL of this launcher's local
# scratch/asset/data trees. The defaults below target this cluster's ephemeral
# NVMe (/opt/dlami/nvme) + shared /fsx; on any other host (no /fsx, no dlami
# nvme) point everything at one big local disk with e.g.
#   SDPO_REACT_LOCAL_ROOT=/mnt/bigdisk/miles-run
# Every individual path (ENROOT_NVME/ASSETS/DATA_DIR/...) still overrides this
# if set explicitly; leave LOCAL_ROOT unset to keep the original cluster paths.
if [ -n "${SDPO_REACT_LOCAL_ROOT:-}" ]; then
    ENROOT_NVME="${ENROOT_NVME:-$SDPO_REACT_LOCAL_ROOT/enroot}"
    ASSETS="${ASSETS:-$SDPO_REACT_LOCAL_ROOT/assets}"
    DATA_DIR="${DATA_DIR:-$SDPO_REACT_LOCAL_ROOT/data/$USER}"
fi

ENROOT_NVME="${ENROOT_NVME:-/opt/dlami/nvme/miles-enroot}"
SQSH="${SQSH:-$ENROOT_NVME/miles-v0.1.0-cu12.sqsh}"

export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$ENROOT_NVME/cache}"
CONTAINER="${CONTAINER:-miles-sdpo-react-cu12}"
ASSETS="${ASSETS:-/opt/dlami/nvme/miles-assets}"
CACHES="${CACHES:-$ASSETS/caches}"
DATA_DIR="${DATA_DIR:-/fsx/data/$USER}"

# JIT compile caches (Triton/inductor/torch-ext/nvrtc) MUST live on a
# POSIX-coherent LOCAL filesystem, NOT on $CACHES if that is a networked/shared
# mount (NFS, Lustre, virtiofs, overlay, ...). SGLang runs one rollout engine per
# GPU, and all of them JIT-compile the SAME kernels (e.g. Qwen3.5's
# chunk_gated_delta_rule / _fused_sigmoid_mul) concurrently into this ONE dir;
# on a shared FS the write-then-rename the cache relies on is not atomic across
# processes, so a peer opens a .cubin that is not there yet and the engine dies
# with `FileNotFoundError: .../<kernel>.cubin` (observed on a virtiofs $CACHES).
# Default to tmpfs (/dev/shm) -- always local, atomic, cleared on reboot (the
# recompile is cheap). Override with SDPO_REACT_JIT_CACHE_DIR if /dev/shm is
# absent/small.
JIT_CACHE_DIR="${SDPO_REACT_JIT_CACHE_DIR:-/dev/shm/miles-jit}"

mkdir -p "$ENROOT_NVME" "$ENROOT_CACHE_PATH" "$ASSETS" "$ASSETS/hf_cache" \
    "$CACHES/triton" "$CACHES/inductor" "$CACHES/torch_extensions" "$CACHES/nv" \
    "$JIT_CACHE_DIR/triton" "$JIT_CACHE_DIR/inductor" "$JIT_CACHE_DIR/torch_extensions" "$JIT_CACHE_DIR/nv" \
    "$DATA_DIR/sdpo_ckpts"

# --- 0. sandbox sidecar on the HOST (before entering enroot) -----------------
# $CONTAINER/$IMAGE here mean the ENROOT container and image (see the header), but
# run_sandbox.sh reads CONTAINER/IMAGE_TAG as its own DOCKER container/image names
# -- so leaking them makes it operate on the wrong container entirely. With a
# custom CONTAINER it will `docker rm -f' whatever docker object happens to carry
# the enroot container's name (observed: it deleted the live search sidecar, which
# then failed the run script's 8421 check inside enroot ~40s later) and then try to
# bind 8420 for a duplicate sandbox, dying with "port is already allocated".
# Unset both so the sidecar keeps its documented default names.
env -u CONTAINER -u IMAGE bash "$SCRIPT_DIR/tools/run_sandbox.sh"

# --- 0b. webshop/alfworld data prep on the HOST (agentic family only) --------
# build_webshop_data.py / build_alfworld_data.py MUST run inside their own
# sidecar's container (gym / alfworld -- deliberately absent from the training
# image, see tools/{webshop,alfworld}/docker/Dockerfile) -- but the training
# session started below has no `docker` binary to shell out to the sidecar
# with. So generate the jsonl files here, on the HOST, via `docker exec` into
# the (idempotently-started) sidecar containers, straight onto $DATA_DIR --
# the same host path the enroot session below mounts as /root/data -- BEFORE
# entering enroot. Every run script's own [-f ...] || python -m ...
# build_*_data invocation then finds the files already present and no-ops.
if [ "${SDPO_REACT_RUN_FAMILY:-native}" = "agentic" ]; then
    bash "$SCRIPT_DIR/tools/webshop/run_webshop_sidecar.sh"
    bash "$SCRIPT_DIR/tools/alfworld/run_alfworld_sidecar.sh"
    WEBSHOP_CONTAINER="${WEBSHOP_CONTAINER:-sdpo-react-webshop}"
    ALFWORLD_CONTAINER="${ALFWORLD_CONTAINER:-sdpo-react-alfworld}"
    mkdir -p "$DATA_DIR/webshop_data" "$DATA_DIR/alfworld_data"
    if [ ! -f "$DATA_DIR/webshop_data/webshop_train.jsonl" ]; then
        docker cp "$SCRIPT_DIR/data/build_webshop_data.py" "$WEBSHOP_CONTAINER:/tmp/build_webshop_data.py"
        # build_webshop_data.py imports examples.SRD.prompt.{system,agentic} (pure
        # prompt-string modules), but the standalone webshop image has no repo on
        # its path. Stage a minimal importable package tree and point PYTHONPATH at
        # it, rather than mounting the whole repo into the sidecar.
        docker exec "$WEBSHOP_CONTAINER" mkdir -p /tmp/pyroot/examples/SRD /tmp/webshop_out
        docker cp "$SCRIPT_DIR/prompt" "$WEBSHOP_CONTAINER:/tmp/pyroot/examples/SRD/prompt"
        docker exec "$WEBSHOP_CONTAINER" touch /tmp/pyroot/examples/__init__.py /tmp/pyroot/examples/SRD/__init__.py
        # APPEND to (not replace) the image's PYTHONPATH -- it already carries
        # /app/webshop where web_agent_site lives; -e PYTHONPATH= would clobber it.
        docker exec -e NTRAIN="${SDPO_REACT_WEBSHOP_N_TRAIN:-400}" -e NEVAL="${SDPO_REACT_WEBSHOP_N_EVAL:-100}" \
            "$WEBSHOP_CONTAINER" sh -c 'PYTHONPATH=/tmp/pyroot:$PYTHONPATH python3 /tmp/build_webshop_data.py \
                --out-dir /tmp/webshop_out --n-train "$NTRAIN" --n-eval "$NEVAL"'
        docker cp "$WEBSHOP_CONTAINER:/tmp/webshop_out/webshop_train.jsonl" "$DATA_DIR/webshop_data/webshop_train.jsonl"
        docker cp "$WEBSHOP_CONTAINER:/tmp/webshop_out/webshop_eval.jsonl" "$DATA_DIR/webshop_data/webshop_eval.jsonl"
    fi
    if [ ! -f "$DATA_DIR/alfworld_data/alfworld_train.jsonl" ]; then
        docker cp "$SCRIPT_DIR/data/build_alfworld_data.py" "$ALFWORLD_CONTAINER:/tmp/build_alfworld_data.py"
        # Same as webshop above: make examples.SRD.prompt.* importable in the
        # standalone alfworld sidecar via a minimal staged package + PYTHONPATH.
        docker exec "$ALFWORLD_CONTAINER" mkdir -p /tmp/pyroot/examples/SRD /tmp/alfworld_out
        docker cp "$SCRIPT_DIR/prompt" "$ALFWORLD_CONTAINER:/tmp/pyroot/examples/SRD/prompt"
        docker exec "$ALFWORLD_CONTAINER" touch /tmp/pyroot/examples/__init__.py /tmp/pyroot/examples/SRD/__init__.py
        # APPEND to the image's PYTHONPATH (carries alfworld's own package path).
        docker exec -e NTRAIN="${SDPO_REACT_ALFWORLD_N_TRAIN:-400}" \
            -e NID="${SDPO_REACT_ALFWORLD_N_EVAL_ID:-100}" -e NOOD="${SDPO_REACT_ALFWORLD_N_EVAL_OOD:-100}" \
            "$ALFWORLD_CONTAINER" sh -c 'PYTHONPATH=/tmp/pyroot:$PYTHONPATH python3 /tmp/build_alfworld_data.py \
                --out-dir /tmp/alfworld_out --n-train "$NTRAIN" --n-eval-id "$NID" --n-eval-ood "$NOOD"'
        docker cp "$ALFWORLD_CONTAINER:/tmp/alfworld_out/alfworld_train.jsonl" "$DATA_DIR/alfworld_data/alfworld_train.jsonl"
        docker cp "$ALFWORLD_CONTAINER:/tmp/alfworld_out/alfworld_eval_id.jsonl" "$DATA_DIR/alfworld_data/alfworld_eval_id.jsonl"
        docker cp "$ALFWORLD_CONTAINER:/tmp/alfworld_out/alfworld_eval_ood.jsonl" "$DATA_DIR/alfworld_data/alfworld_eval_ood.jsonl"
    fi
fi

# --- 1. import image -> squashfs on NVMe (skip if already imported) ----------
if [ ! -f "$SQSH" ]; then
    enroot import -o "$SQSH" "docker://${IMAGE}"
fi

# --- 2. create container rootfs (unsquashfs, no fuse) ------------------------
if ! enroot list 2>/dev/null | grep -qx "$CONTAINER"; then
    enroot create --name "$CONTAINER" "$SQSH"
fi

# --- 3. run ------------------------------------------------------------------
enroot start --rw \
    --mount "$REPO_ROOT":/root/miles \
    --mount "$ASSETS":/root/assets \
    --mount "$ASSETS/hf_cache":/root/hf_cache \
    --mount "$CACHES":/root/caches \
    --mount "$JIT_CACHE_DIR":/root/jit \
    --mount "$DATA_DIR":/root/data \
    --env PREP_ONLY="${PREP_ONLY:-0}" \
    --env SDPO_REACT_MODEL="${SDPO_REACT_MODEL:-qwen3.5-4B}" \
    --env SDPO_REACT_RUN_FAMILY="${SDPO_REACT_RUN_FAMILY:-}" \
    --env SDPO_REACT_DOMAIN="${SDPO_REACT_DOMAIN:-}" \
    --env SDPO_REACT_WEBSHOP_SIDECAR_URL="${SDPO_REACT_WEBSHOP_SIDECAR_URL:-}" \
    --env SDPO_REACT_ALFWORLD_SIDECAR_URL="${SDPO_REACT_ALFWORLD_SIDECAR_URL:-}" \
    --env SDPO_REACT_AGENTIC_PER_DOMAIN="${SDPO_REACT_AGENTIC_PER_DOMAIN:-}" \
    --env SDPO_REACT_WEBSHOP_N_TRAIN="${SDPO_REACT_WEBSHOP_N_TRAIN:-}" \
    --env SDPO_REACT_WEBSHOP_N_EVAL="${SDPO_REACT_WEBSHOP_N_EVAL:-}" \
    --env SDPO_REACT_ALFWORLD_N_TRAIN="${SDPO_REACT_ALFWORLD_N_TRAIN:-}" \
    --env SDPO_REACT_ALFWORLD_N_EVAL_ID="${SDPO_REACT_ALFWORLD_N_EVAL_ID:-}" \
    --env SDPO_REACT_ALFWORLD_N_EVAL_OOD="${SDPO_REACT_ALFWORLD_N_EVAL_OOD:-}" \
    --env SDPO_REACT_ARM="${SDPO_REACT_ARM:-}" \
    --env SDPO_ABLATION_ALGO="${SDPO_ABLATION_ALGO:-}" \
    --env SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:-}" \
    --env SDPO_ABLATION_SKILL_KD_MODE="${SDPO_ABLATION_SKILL_KD_MODE:-}" \
    --env SDPO_ABLATION_SKILL_KD_COEF="${SDPO_ABLATION_SKILL_KD_COEF:-}" \
    --env SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-}" \
    --env SDPO_REACT_NUM_ROLLOUT="${SDPO_REACT_NUM_ROLLOUT:-}" \
    --env SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-}" \
    --env SDPO_REACT_PURE_DISTILL="${SDPO_REACT_PURE_DISTILL:-}" \
    --env SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-}" \
    --env SDPO_REACT_MT_PER_DOMAIN="${SDPO_REACT_MT_PER_DOMAIN:-}" \
    --env SDPO_REACT_MIN_CORRECT="${SDPO_REACT_MIN_CORRECT:-}" \
    --env SDPO_REACT_DYNAMIC_SAMPLE="${SDPO_REACT_DYNAMIC_SAMPLE:-}" \
    --env SDPO_REACT_ROLLOUT_BATCH="${SDPO_REACT_ROLLOUT_BATCH:-}" \
    --env SDPO_REACT_GLOBAL_BATCH="${SDPO_REACT_GLOBAL_BATCH:-}" \
    --env SDPO_REACT_TP="${SDPO_REACT_TP:-}" \
    --env SDPO_REACT_MAX_TOKENS_PER_GPU="${SDPO_REACT_MAX_TOKENS_PER_GPU:-}" \
    --env SDPO_ABLATION_MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-}" \
    --env SDPO_ABLATION_SGLANG_MEM_FRACTION="${SDPO_ABLATION_SGLANG_MEM_FRACTION:-}" \
    --env SDPO_REACT_EP_SIZE="${SDPO_REACT_EP_SIZE:-}" \
    --env SDPO_REACT_R3="${SDPO_REACT_R3:-}" \
    --env SDPO_REACT_OPT_CPU_OFFLOAD="${SDPO_REACT_OPT_CPU_OFFLOAD:-}" \
    --env SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-}" \
    --env SDPO_REACT_LOGPROBS_CHUNK="${SDPO_REACT_LOGPROBS_CHUNK:-}" \
    --env SDPO_REACT_DIST_TIMEOUT_MIN="${SDPO_REACT_DIST_TIMEOUT_MIN:-}" \
    --env SDPO_REACT_SAVE_INTERVAL="${SDPO_REACT_SAVE_INTERVAL:-}" \
    --env SDPO_REACT_ASYNC_SAVE="${SDPO_REACT_ASYNC_SAVE:-}" \
    --env SDPO_REACT_SKIP_EVAL0="${SDPO_REACT_SKIP_EVAL0:-}" \
    --env SDPO_REACT_SAVE_CKPT="${SDPO_REACT_SAVE_CKPT:-}" \
    --env SDPO_REACT_NOTE="${SDPO_REACT_NOTE:-}" \
    --env SDPO_REACT_TRAIN_MAX_TURNS="${SDPO_REACT_TRAIN_MAX_TURNS:-}" \
    --env SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-}" \
    --env SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-}" \
    --env SDPO_REACT_EVAL_CONFIG="${SDPO_REACT_EVAL_CONFIG:-}" \
    --env SDPO_REACT_MAX_RESPONSE_LEN="${SDPO_REACT_MAX_RESPONSE_LEN:-}" \
    --env SDPO_REACT_TAG_SUFFIX="${SDPO_REACT_TAG_SUFFIX:-}" \
    --env HF_HOME=/root/hf_cache \
    --env TRITON_CACHE_DIR=/root/jit/triton \
    --env TORCHINDUCTOR_CACHE_DIR=/root/jit/inductor \
    --env TORCH_EXTENSIONS_DIR=/root/jit/torch_extensions \
    --env CUDA_CACHE_PATH=/root/jit/nv \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env LLM_GATEWAY_KEY="${LLM_GATEWAY_KEY:-}" \
    --env OPENAI_API_URL="${OPENAI_API_URL:-}" \
    --env SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    "$CONTAINER" \
    bash -euxc '
        cd /root/miles

        # Pick model by $SDPO_REACT_MODEL: local dir name, HF repo id, megatron
        # model-arg script, and the run-script targets. Only Qwen3.5 4B/9B/
        # 35B-A3B are wired -- the publication model set. Each branch sets
        # NATIVE_RUN_SH (mathcodesearch, the two-axis SDPO_ABLATION_ALGO x
        # SDPO_ABLATION_ARM matrix in examples/SRD/ablation/) and
        # AGENTIC_RUN_SH (its alfworld-webshop sibling); the RUN_SH switch below
        # picks between them via SDPO_REACT_RUN_FAMILY. 4B is the default.
        case "${SDPO_REACT_MODEL:-qwen3.5-4B}" in
            qwen3.5-9B)
                MODEL_DIR=Qwen3.5-9B
                HF_REPO=Qwen/Qwen3.5-9B
                MODEL_SH=scripts/models/qwen3.5-9B.sh
                NATIVE_RUN_SH=examples/SRD/ablation/run-qwen3.5-9B-sdpo-react-ablation-mathcodesearch.sh
                AGENTIC_RUN_SH=examples/SRD/ablation/run-qwen3.5-9B-sdpo-react-ablation-alfworld-webshop.sh
                ;;
            qwen3.5-35B-A3B)
                # MoE (256 experts, top-8, ~3B active). The large-model launcher
                # sets EP=8 + R3 rollout-routing-replay (train/inference router
                # alignment). MODEL_SH drives the HF->torch_dist convert below.
                MODEL_DIR=Qwen3.5-35B-A3B
                HF_REPO=Qwen/Qwen3.5-35B-A3B
                MODEL_SH=scripts/models/qwen3.5-35B-A3B.sh
                NATIVE_RUN_SH=examples/SRD/ablation/run-qwen3.5-35B-A3B-sdpo-react-ablation-mathcodesearch.sh
                AGENTIC_RUN_SH=examples/SRD/ablation/run-qwen3.5-35B-A3B-sdpo-react-ablation-alfworld-webshop.sh
                export SDPO_REACT_MODEL=qwen3.5-35B-A3B
                ;;
            qwen3.5-4B)
                # Qwen3.5-4B (default). Hybrid linear-attention arch; SGLang
                # rollout engine runs tp=1 (4B fits 1 GPU).
                MODEL_DIR=Qwen3.5-4B
                HF_REPO=Qwen/Qwen3.5-4B
                MODEL_SH=scripts/models/qwen3.5-4B.sh
                NATIVE_RUN_SH=examples/SRD/ablation/run-qwen3.5-4B-sdpo-react-ablation-mathcodesearch.sh
                AGENTIC_RUN_SH=examples/SRD/ablation/run-qwen3.5-4B-sdpo-react-ablation-alfworld-webshop.sh
                ;;
            *)
                echo "SDPO_REACT_MODEL=${SDPO_REACT_MODEL} not recognized; expected one of: qwen3.5-4B | qwen3.5-9B | qwen3.5-35B-A3B" >&2
                exit 1
                ;;
        esac
        case "${SDPO_REACT_RUN_FAMILY:-native}" in
            agentic) RUN_SH="$AGENTIC_RUN_SH" ;;
            *) RUN_SH="$NATIVE_RUN_SH" ;;
        esac

        for name in "$MODEL_DIR" "${MODEL_DIR}_torch_dist" "${MODEL_DIR}_miles" dapo-math-17k math_eval; do
            mkdir -p /root/assets/$name
            ln -sfn /root/assets/$name /root/$name
        done

        # Idempotent sglang tolist patch (shared, algorithm-agnostic infra --
        # see examples/SRD/patch-sglang-tolist.sh).
        bash examples/SRD/patch-sglang-tolist.sh

        python -c "import miles; print(\"Miles import OK\")"

        [ -n "$(ls -A /root/$MODEL_DIR 2>/dev/null)" ] || \
            hf download "$HF_REPO" --local-dir /root/$MODEL_DIR

        if [ -z "$(ls -A /root/${MODEL_DIR}_torch_dist 2>/dev/null)" ]; then
            source "$MODEL_SH"
            PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
                "${MODEL_ARGS[@]}" \
                --hf-checkpoint /root/$MODEL_DIR \
                --save /root/${MODEL_DIR}_torch_dist
        fi

        if [ "$PREP_ONLY" = "1" ]; then
            echo "PREP_ONLY=1 -> assets ready under /root/assets, skipping training."
            exit 0
        fi

        bash "$RUN_SH"
    '
