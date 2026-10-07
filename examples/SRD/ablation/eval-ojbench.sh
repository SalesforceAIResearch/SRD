#!/bin/bash
# Standalone OJBench (NOI + ICPC competition programming) evaluation -- runs HF
# checkpoints through the same multi-turn native tool-calling rollout as training
# step-0 eval. Sibling of eval-lcb-functional.sh / eval-amo-bench.sh; only the
# dataset differs.
#
# DEFAULT SLICE: medium only (77 problems: 52 NOI + 25 ICPC), where arms
# separate -- hard yields near-zero solves, easy saturates on tool protocol. Full
# set is one env var away (OJBENCH_DATA_DIR + build_ojbench --difficulty
# easy,medium,hard).
#
# Grading: judge.py's STDIN harness, all-or-nothing over up to 10 test cases per
# problem. TOOL-MANDATORY by default (--sdpo-code-require-tool), matching
# training: grades the code last RAN through code_interpreter.
# OJBENCH_REQUIRE_TOOL=0 grades the final ```python fence (official OJBench
# protocol; scores base models much higher).
#
# Scaffold matches the training step-0 eval scaffold deliberately (16384 per-turn
# budget, 20 turns, thinking on, minimal prompt, 8 samples @ temp 1), so numbers
# are comparable to the training curves. Runs train.py --num-rollout 1 without
# --skip-eval-before-train; see EVAL_ARGS for why --eval-interval is 2.
#
# Required env vars:
#   WANDB_API_KEY     for logging to wandb
#
# Optional env vars:
#   MODEL_PATHS       space-separated HF checkpoint dirs to evaluate SEQUENTIALLY
#                     (default: Qwen3.5-4B + Qwen3.5-9B base under MODEL_STORE)
#   MODEL_PATH        single checkpoint dir; overrides MODEL_PATHS when set
#   MODEL_STORE       (default /root/data/home-static/data/hf_models) base-model dir
#   OJBENCH_DATA_DIR  (default /root/data/home-static/data/ojbench_medium)
#   OJBENCH_DIFFICULTY (default medium) comma-separated, only used if the eval
#                     jsonl has to be BUILT (needs network for the HF download)
#   OJBENCH_DUMP_DIR  (default /root/data/sdpo_dumps) trace dump root
#   OJBENCH_MAX_RESPONSE_LEN (default 16384) PER-TURN generation budget
#   OJBENCH_REQUIRE_TOOL (default 1) 0 -> grade the ```python fence
#   SDPO_REACT_EVAL_N_SAMPLES (default 8) samples per prompt for pass@k
#   SDPO_REACT_EVAL_MAX_TURNS (default 20) max tool-calling turns per sample
#   SDPO_REACT_TP     (default 2) tensor parallel size
#   SDPO_REACT_TRAIN_GPUS (default 8) total GPUs
#   MEGATRON_PATH     (default /root/Megatron-LM)
#   SGLANG_MEM_FRACTION (default 0.75)
#   OJBENCH_WANDB_PROJECT (default miles-sdpo)
#   OJBENCH_WANDB_GROUP   (default ojbench-medium-eval)
#   SDPO_EVAL_REF_LOAD    megatron dist-ckpt to load the WEIGHTS from; lets this
#                         script eval a TRAINED checkpoint directly (no HF
#                         round-trip), MODEL_PATH then only gives tokenizer/config
#   SDPO_EVAL_TAG         suffix for dump-dir / log naming, so a base-model
#                         MODEL_PATH + trained SDPO_EVAL_REF_LOAD is not filed
#                         under the base model's name
#
# usage (both base models, the default):
#   bash examples/SRD/ablation/eval-ojbench.sh
# usage (one trained arm):
#   MODEL_PATH=/root/data/home-static/data/hf_models/Qwen3.5-9B-...-arm-e \
#     bash examples/SRD/ablation/eval-ojbench.sh
set -exf

# Paths are CONTAINER paths: enroot mounts host $DATA_DIR as /root/data.
MODEL_STORE="${MODEL_STORE:-/root/data/home-static/data/hf_models}"
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATHS="$MODEL_PATH"
fi
MODEL_PATHS="${MODEL_PATHS:-${MODEL_STORE}/Qwen3.5-4B ${MODEL_STORE}/Qwen3.5-9B}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

OJBENCH_DATA_DIR="${OJBENCH_DATA_DIR:-/root/data/home-static/data/ojbench_medium}"
export OJBENCH_DATA_DIR
OJBENCH_DIFFICULTY="${OJBENCH_DIFFICULTY:-medium}"
# PER-TURN response budget; 16384 matches the training eval configs.
OJBENCH_MAX_RESPONSE_LEN="${OJBENCH_MAX_RESPONSE_LEN:-16384}"
export OJBENCH_MAX_RESPONSE_LEN

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

TOOL_PARSER=qwen3_coder
TOOL_GRAMMAR=qwen3_coder

# --- 0. sandbox sidecar (code_interpreter AND the judge's harness) ---
# OJBench test cases are CPU-heavy: an oversubscribed sandbox (see
# docker/sandbox_server.py::MAX_CONCURRENCY) turns correct submissions into
# wall-clock timeouts.
bash "$REACT_DIR/tools/run_sandbox.sh"

# --- 0a. data prep ---
EVAL_JSONL="${OJBENCH_DATA_DIR}/ojbench_eval.jsonl"
mkdir -p "$OJBENCH_DATA_DIR"
[ -f "$EVAL_JSONL" ] || \
    (cd "$REPO_ROOT" && python -m examples.SRD.data.build_ojbench \
        --out-dir "$OJBENCH_DATA_DIR" \
        --difficulty "${OJBENCH_DIFFICULTY}" \
        --max-tests-per-problem 10)
echo "OJBench eval rows: $(wc -l < "$EVAL_JSONL")"

EVAL_CFG="$REACT_DIR/data/eval_ojbench.yaml"

# Thinking on by default (Qwen3.5 reasoning mode), same as training.
export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
export SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-minimal}"

for MODEL_PATH in ${MODEL_PATHS}; do
MODEL_NAME="$(basename "$MODEL_PATH")"
# SDPO_EVAL_TAG renames dump dir / log lines when MODEL_PATH is a base HF dir
# used only for tokenizer/config while weights come from SDPO_EVAL_REF_LOAD.
# Only meaningful for a single-model MODEL_PATH.
MODEL_TAG="${MODEL_NAME}${SDPO_EVAL_TAG:+-${SDPO_EVAL_TAG}}"
echo "============================================================"
echo "OJBench ${OJBENCH_DIFFICULTY} eval: ${MODEL_TAG}"
echo "============================================================"

# --- Detect model arch for model args ---
MODEL_ARG_SH=""
if echo "$MODEL_NAME" | grep -qi "qwen3.5-9B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-9B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3.5-4B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-4B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3.5-2B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-2B.sh
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

DUMP_DIR="${OJBENCH_DUMP_DIR:-/root/data/sdpo_dumps}/ojbench-${OJBENCH_DIFFICULTY}-eval-${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)"
echo "OJBench eval dump dir: ${DUMP_DIR}"

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
    echo "ERROR: no torch_dist checkpoint for ${MODEL_NAME}. Convert it first:" >&2
    echo "  bash examples/SRD/ablation/convert-models-to-torch-dist-generic.sh" >&2
    exit 1
fi
echo "REF_LOAD: ${REF_LOAD}"
CKPT_ARGS+=(--ref-load "${REF_LOAD}")

# The training loop needs a --prompt-data even for an eval-only run. Use the
# eval data itself; with --eval-interval 2 (see EVAL_ARGS) the single no-op
# rollout step never feeds an eval that follows it.
ROLLOUT_ARGS=(
   --prompt-data "${EVAL_JSONL}"
   --input-key prompt
   --label-key label
   --tool-key tools
   --apply-chat-template
   --apply-chat-template-kwargs "{\"enable_thinking\":${SDPO_REACT_THINKING}}"
   --num-rollout 1
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${OJBENCH_MAX_RESPONSE_LEN}"
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

# Plain GRPO reward; domain-routes code to the test-case judge, which the eval
# reuses (no --eval-custom-rm-path). No LLM judge -- code grading is deterministic.
RM_ARGS=(
   --custom-rm-path examples.SRD.sdpo_react.sdpo_react_plain_grpo_reward
   --sdpo-grader dapo
   --sdpo-answer-tag answer
)
if [ "${OJBENCH_REQUIRE_TOOL:-1}" = "0" ]; then
    RM_ARGS+=(--no-sdpo-code-require-tool)
fi

EVAL_ARGS=(
   # eval-interval 2, NOT 1: with interval 1 + --num-rollout 1 a second eval runs
   # after the step trained on --prompt-data (= the eval set here), overwriting
   # the step-0 dump with a leaked number. interval 2 skips the periodic check so
   # exactly one eval runs, on the loaded weights.
   --eval-interval 2
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
   --wandb-project "${OJBENCH_WANDB_PROJECT:-miles-sdpo}"
   --wandb-group "${OJBENCH_WANDB_GROUP:-ojbench-medium-eval}"
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

cd "$REPO_ROOT"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${SDPO_REACT_TRAIN_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# `ray start` returns when the GCS is up, but the dashboard job-submission API
# (8265) can take ~20s more to bind; wait for it to answer before submitting.
for i in $(seq 1 60); do
    curl -sf -o /dev/null "http://127.0.0.1:8265/api/version" && break
    if [ "$i" = 60 ]; then
        echo "ERROR: ray dashboard API never came up on 127.0.0.1:8265" >&2
        exit 1
    fi
    sleep 2
done

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${REPO_ROOT}:${MEGATRON_PATH}/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"0\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\",
        \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
        \"OJBENCH_DATA_DIR\": \"${OJBENCH_DATA_DIR}\",
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
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 3
echo "DONE ${MODEL_TAG} -- traces: ${DUMP_DIR}"
done

echo "OJBench ${OJBENCH_DIFFICULTY} eval complete for: ${MODEL_PATHS}"
echo "Check wandb group '${OJBENCH_WANDB_GROUP:-ojbench-medium-eval}' (eval/ojbench*)."
