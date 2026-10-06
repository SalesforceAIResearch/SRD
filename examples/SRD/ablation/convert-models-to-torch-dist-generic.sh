#!/bin/bash
# Convert arbitrary HF checkpoints (trained arms, not just the Qwen3.5-<size>
# bases) to Megatron torch_dist, which is what --ref-load needs. Takes a list of
# directory NAMES under MODEL_STORE and derives the scripts/models/*.sh arg file
# from the name the same way the ablation/eval scripts do.
#
# Env:
#   MODEL_NAMES  space-separated dir names under MODEL_STORE (required)
#   MODEL_STORE  (default /root/data/hf_models)
#   REPO_ROOT    (default /root/miles)
#
# Run inside the enroot container with 8 GPUs visible.
set -e

export PATH="/usr/local/bin:/usr/bin:/usr/sbin:/bin:/sbin:$PATH"

REPO_ROOT="${REPO_ROOT:-/root/miles}"
MODEL_STORE="${MODEL_STORE:-/root/data/hf_models}"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${MEGATRON_PATH:-/root/Megatron-LM}/"
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ -z "${MODEL_NAMES:-}" ]; then
    echo "ERROR: set MODEL_NAMES to a space-separated list of dirs under ${MODEL_STORE}" >&2
    exit 1
fi

# Same name -> arg-script mapping the eval scripts use. Most-specific patterns
# first (35B-A3B must beat 35B).
model_arg_sh() {
    case "$1" in
        *[Qq]wen3.5-35B-A3B*) echo scripts/models/qwen3.5-35B-A3B.sh ;;
        *[Qq]wen3.5-27B*|*[Qq]wen3.6-27B*) echo scripts/models/qwen3.5-27B.sh ;;
        *[Qq]wen3.5-9B*) echo scripts/models/qwen3.5-9B.sh ;;
        *[Qq]wen3.5-4B*) echo scripts/models/qwen3.5-4B.sh ;;
        *[Qq]wen3.5-2B*) echo scripts/models/qwen3.5-2B.sh ;;
        *[Qq]wen3-4B*) echo scripts/models/qwen3-4B.sh ;;
        *) echo "" ;;
    esac
}

FAILED=()
for NAME in ${MODEL_NAMES}; do
    HF_PATH="${MODEL_STORE}/${NAME}"
    DIST_PATH="${MODEL_STORE}/${NAME}_torch_dist"

    if [ -f "${DIST_PATH}/latest_checkpointed_iteration.txt" ]; then
        echo "SKIP ${NAME}: torch_dist already exists"
        continue
    fi
    if [ ! -f "${HF_PATH}/config.json" ]; then
        echo "SKIP ${NAME}: no HF checkpoint at ${HF_PATH}" >&2
        FAILED+=("${NAME}(no-hf)")
        continue
    fi
    ARG_SH="$(model_arg_sh "$NAME")"
    if [ -z "$ARG_SH" ] || [ ! -f "$REPO_ROOT/$ARG_SH" ]; then
        echo "SKIP ${NAME}: no scripts/models/*.sh matches this name" >&2
        FAILED+=("${NAME}(no-args)")
        continue
    fi

    echo "============================================================"
    echo "Converting ${NAME} (${ARG_SH}) -> ${DIST_PATH}"
    echo "============================================================"
    # Subshell: each model script re-defines MODEL_ARGS, so don't let them leak.
    (
        source "$REPO_ROOT/$ARG_SH"
        # Write to a staging dir and move on success, so a half-written dir is
        # never picked up as valid by the eval scripts' --ref-load probe.
        rm -rf "${DIST_PATH}.partial"
        torchrun --nproc_per_node=8 tools/convert_hf_to_torch_dist.py \
            ${MODEL_ARGS[@]} \
            --hf-checkpoint "${HF_PATH}" \
            --save "${DIST_PATH}.partial" \
            --tensor-model-parallel-size 1 \
            --pipeline-model-parallel-size 1 \
            --context-parallel-size 1 \
            --expert-model-parallel-size 1 \
            --expert-tensor-parallel-size 1
    ) && mv "${DIST_PATH}.partial" "${DIST_PATH}" && echo "DONE ${NAME}" || {
        echo "FAILED ${NAME}" >&2
        FAILED+=("${NAME}(convert)")
        rm -rf "${DIST_PATH}.partial"
    }
done

echo "============================================================"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}" >&2
    exit 1
fi
echo "ALL CONVERSIONS COMPLETE"
