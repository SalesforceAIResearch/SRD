#!/bin/bash
# Standalone AMO-Bench evaluation (50 IMO-level math problems,
# meituan-longcat/AMO-Bench) via the same multi-turn native tool-calling
# rollout as training step-0 eval. Reports pass@1..8, turns, tool stats.
# Grading: 39/50 via DAPO math grader; 11/50 (answer_type='description') via
# LLM-judge (--sdpo-judge). Runs train.py --num-rollout 1 without
# --skip-eval-before-train so the step-0 eval fires and exits.
#
# Required env vars:
#   MODEL_PATH        full path to the HF checkpoint dir (e.g. /root/Qwen3.5-9B)
#   WANDB_API_KEY     for logging to wandb
#
# Optional env vars:
#   AMO_BENCH_DATA_DIR       (default /root/data/amo_bench) where the eval jsonl lives
#   SDPO_REACT_EVAL_N_SAMPLES (default 8) samples per prompt for pass@k
#   SDPO_REACT_EVAL_MAX_TURNS (default 20) max tool-calling turns per sample
#   SDPO_REACT_TP            (default 2) tensor parallel size
#   SDPO_REACT_TRAIN_GPUS    (default 8) total GPUs
#   MEGATRON_PATH            (default /root/Megatron-LM)
#   OPENAI_API_URL           gateway URL for LLM judge
#   LLM_GATEWAY_KEY          API key for LLM judge
#   SDPO_REACT_JUDGE_MODEL   (default gpt-5.6-luna) judge model
#   SGLANG_MEM_FRACTION      (default 0.75) sglang memory fraction
#   AMO_BENCH_WANDB_PROJECT  (default miles-sdpo) wandb project
#   AMO_BENCH_WANDB_GROUP    (default amo-bench-eval) wandb group
#   SDPO_EVAL_CONFIG         (default eval_amo_bench.yaml) bare filename resolved
#                            against data/, or an absolute path
#   SDPO_EVAL_REF_LOAD       megatron dist-ckpt to load the WEIGHTS from; lets
#                            this script eval a TRAINED checkpoint directly
#                            (no HF round-trip), with MODEL_PATH only supplying
#                            tokenizer/config
#   SDPO_EVAL_TAG            suffix for dump-dir / log naming, so a base-model
#                            MODEL_PATH + trained SDPO_EVAL_REF_LOAD is not filed
#                            under the base model's name
#
# usage:
#   MODEL_PATH=/root/Qwen3.5-9B \
#     bash examples/SRD/ablation/eval-amo-bench.sh
# usage (trained checkpoint, aime24+aime25+amo in one boot):
#   MODEL_PATH=/root/data/home-static/data/hf_models/Qwen3.5-9B \
#   SDPO_EVAL_REF_LOAD=/root/data/sdpo_ckpts/<run>_ckpt \
#   SDPO_EVAL_CONFIG=eval_math_final.yaml SDPO_EVAL_TAG=grpo-e-step29 \
#     bash examples/SRD/ablation/eval-amo-bench.sh
set -exf

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the HF checkpoint directory}"
MODEL_NAME="$(basename "$MODEL_PATH")"
# SDPO_EVAL_TAG renames dump dir / wandb suffix when MODEL_PATH is a base HF dir
# used only for tokenizer/config while weights come from SDPO_EVAL_REF_LOAD.
MODEL_TAG="${MODEL_NAME}${SDPO_EVAL_TAG:+-${SDPO_EVAL_TAG}}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
AMO_BENCH_DATA_DIR="${AMO_BENCH_DATA_DIR:-/root/data/amo_bench}"
export AMO_BENCH_DATA_DIR

export PYTHONBUFFERED=16
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
export SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-20}"
# No real training here; set to 1 so the single no-op rollout is cheap.
export SDPO_REACT_TRAIN_MAX_TURNS=1

SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
N_SAMPLES_PER_PROMPT=8
SDPO_REACT_TP="${SDPO_REACT_TP:-2}"
DP_SIZE=$((SDPO_REACT_TRAIN_GPUS / SDPO_REACT_TP))
ROLLOUT_BATCH_SIZE=$((DP_SIZE * 4))
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
echo "BATCH: train_gpus=${SDPO_REACT_TRAIN_GPUS} tp=${SDPO_REACT_TP} dp=${DP_SIZE} rollout_batch=${ROLLOUT_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REACT_DIR="$REPO_ROOT/examples/SRD"

# --- Detect model arch, source its model args script ---
MODEL_ARG_SH=""
if echo "$MODEL_NAME" | grep -qi "qwen3.5-9B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-9B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3.5-4B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-4B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3.5-35B\|Qwen3.5-35B-A3B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-35B-A3B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3.5-27B\|Qwen3.6-27B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-27B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3-4B"; then
    MODEL_ARG_SH=scripts/models/qwen3-4B.sh
fi

if [ -n "$MODEL_ARG_SH" ] && [ -f "$REPO_ROOT/$MODEL_ARG_SH" ]; then
    echo "MODEL: ${MODEL_NAME} (sourcing ${MODEL_ARG_SH})"
    source "$REPO_ROOT/${MODEL_ARG_SH}"
else
    echo "ERROR: Could not find model args script for '${MODEL_NAME}'." >&2
    echo "Set MODEL_ARG_SH env var to point to the correct scripts/models/*.sh" >&2
    exit 1
fi

TOOL_PARSER=qwen3_coder
TOOL_GRAMMAR=qwen3_coder

# --- 0. sandbox sidecar (code_interpreter) ---
bash "$REACT_DIR/tools/run_sandbox.sh"

# --- 0a. data prep: download + build AMO-Bench eval jsonl ---
mkdir -p "$AMO_BENCH_DATA_DIR"
[ -f "${AMO_BENCH_DATA_DIR}/amo_bench_eval.jsonl" ] || \
    (cd "$REPO_ROOT" && python -m examples.SRD.data.build_amo_bench \
        --out-dir "$AMO_BENCH_DATA_DIR")

# aime24/25/26 for data/eval_math_final.yaml (harmless under the default
# SDPO_EVAL_CONFIG=eval_amo_bench.yaml, which doesn't reference aime).
AIME_DATA_DIR="${AIME_DATA_DIR:-/root/data/aime}"
mkdir -p "$AIME_DATA_DIR"
[ -f "${AIME_DATA_DIR}/aime24_eval.jsonl" ] || \
    (cd "$REPO_ROOT" && python -m examples.SRD.data.build_aime --out-dir "$AIME_DATA_DIR")

# Bare filename (resolved against data/) or absolute path. Lets this script (the
# only eval that passes --sdpo-judge) be reused for any math eval set, e.g.
# data/eval_math_final.yaml (aime24 + aime25 + amo_bench in one boot).
EVAL_CFG="${SDPO_EVAL_CONFIG:-eval_amo_bench.yaml}"
case "$EVAL_CFG" in
    /*) ;;
    *)  EVAL_CFG="$REACT_DIR/data/$EVAL_CFG" ;;
esac

# Thinking on by default (Qwen3.5 reasoning mode).
export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
export SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-minimal}"

DUMP_DIR="${AMO_BENCH_DUMP_DIR:-/root/data/sdpo_dumps/amo-bench-eval-${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)}"
echo "AMO-Bench eval dump dir: ${DUMP_DIR}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

# Megatron loads the actor from a torch_dist ckpt, not the HF dir, so pass
# --ref-load. Resolve the sibling _torch_dist dir, then per-node asset copies.
#   SDPO_EVAL_REF_LOAD=<dir>  a TRAINED megatron dist-ckpt (a run's --save dir):
#   loaded as weights-only (no HF round-trip), MODEL_PATH then only supplies
#   tokenizer/config, so point it at the base model and set SDPO_EVAL_TAG.
REF_LOAD="${SDPO_EVAL_REF_LOAD:-}"
if [ -z "$REF_LOAD" ]; then
    for cand in "${MODEL_PATH}_torch_dist" "/root/assets/${MODEL_NAME}_torch_dist" "/root/${MODEL_NAME}_torch_dist"; do
        if [ -f "${cand}/latest_checkpointed_iteration.txt" ]; then
            REF_LOAD="$cand"
            break
        fi
    done
fi
if [ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: no torch_dist checkpoint for ${MODEL_NAME} (REF_LOAD='${REF_LOAD}')." >&2
    echo "  Convert it first: bash examples/SRD/ablation/convert-models-to-torch-dist-generic.sh" >&2
    echo "  or point SDPO_EVAL_REF_LOAD at a trained run's --save dir." >&2
    exit 1
fi
echo "REF_LOAD: ${REF_LOAD}"
CKPT_ARGS+=(--ref-load "${REF_LOAD}")

# The training loop needs a --prompt-data even for eval-only runs; reuse the
# eval data (num-rollout=1 exits after the step-0 eval, never trains on it).
ROLLOUT_ARGS=(
   --prompt-data "${AMO_BENCH_DATA_DIR}/amo_bench_eval.jsonl"
   --input-key prompt
   --label-key label
   --tool-key tools
   --apply-chat-template
   --apply-chat-template-kwargs "{\"enable_thinking\":${SDPO_REACT_THINKING}}"
   --num-rollout 1
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len 16384
   --rollout-max-context-len 81920
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}"
)

TOOL_SPECS_PATH="examples.SRD.tools.registry.all_tool_specs"
EXECUTE_TOOL_PATH="examples.SRD.tools.registry.execute_tool"
ROLLOUT_ARGS+=(--tool-specs-resolver-path "$TOOL_SPECS_PATH")
CUSTOM_GENERATE_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
   --generate-tool-specs-path "$TOOL_SPECS_PATH"
   --generate-execute-tool-function-path "$EXECUTE_TOOL_PATH"
   --generate-tool-call-parser "$TOOL_PARSER"
   --generate-max-turns "${SDPO_REACT_TRAIN_MAX_TURNS}"
)

# Plain GRPO reward for the no-op training step.
RM_ARGS=(
   --custom-rm-path examples.SRD.sdpo_react.sdpo_react_plain_grpo_reward
   --sdpo-grader dapo
   --sdpo-answer-tag answer
   --sdpo-judge
   --sdpo-judge-base-url "${OPENAI_API_URL:-}"
   --sdpo-judge-model "${SDPO_REACT_JUDGE_MODEL:-gpt-5.6-luna}"
   --sdpo-judge-api-key-env LLM_GATEWAY_KEY
)

EVAL_ARGS=(
   --eval-interval 1
   --eval-config "$EVAL_CFG"
   --eval-tool-key tools
   --n-samples-per-eval-prompt "${SDPO_REACT_EVAL_N_SAMPLES}"
   --log-passrate
)
# Do NOT add --skip-eval-before-train: we WANT the step-0 eval to fire.

PERF_ARGS=(
   --tensor-model-parallel-size "${SDPO_REACT_TP}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-6144}"
   --sequence-parallel
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters 0
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${AMO_BENCH_WANDB_PROJECT:-miles-sdpo}"
   --wandb-group "${AMO_BENCH_WANDB_GROUP:-amo-bench-eval}"
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.75}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

if [ "$SDPO_REACT_TRAIN_GPUS" -lt 8 ]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((SDPO_REACT_TRAIN_GPUS-1)))"
    echo "TRAIN GPUs: ${SDPO_REACT_TRAIN_GPUS} (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
fi

ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 2

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${SDPO_REACT_TRAIN_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# AIME_DATA_DIR is read by data/eval_math_final.yaml; defaulted to the same
# literal the yaml falls back to (NOT ""), since oc.env treats a set-but-empty
# var as a value and would resolve the aime paths to "/aime24_eval.jsonl".
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${MEGATRON_PATH}/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"0\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\",
        \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
        \"OPENAI_API_URL\": \"${OPENAI_API_URL:-}\",
        \"LLM_GATEWAY_KEY\": \"${LLM_GATEWAY_KEY:-}\",
        \"AMO_BENCH_DATA_DIR\": \"${AMO_BENCH_DATA_DIR}\",
        \"AIME_DATA_DIR\": \"${AIME_DATA_DIR:-/root/data/aime}\",
        \"SDPO_REACT_EVAL_N_SAMPLES\": \"${SDPO_REACT_EVAL_N_SAMPLES}\",
        \"SDPO_REACT_EVAL_MAX_TURNS\": \"${SDPO_REACT_EVAL_MAX_TURNS}\",
        \"SDPO_REACT_TRAIN_MAX_TURNS\": \"${SDPO_REACT_TRAIN_MAX_TURNS}\",
        \"SDPO_REACT_PROMPT\": \"${SDPO_REACT_PROMPT}\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${SDPO_REACT_TRAIN_GPUS}" \
   --rollout-num-gpus "${SDPO_REACT_TRAIN_GPUS}" \
   --colocate \
   --update-weights-interval 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CUSTOM_GENERATE_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

ray stop --force
echo "AMO-Bench eval complete for ${MODEL_TAG}. Check wandb group '${AMO_BENCH_WANDB_GROUP:-amo-bench-eval}'."
echo "Traces dumped to: ${DUMP_DIR}"
