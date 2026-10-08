#!/bin/bash
# SDPO/RLSD/GRPO ablation matrix -- Qwen3.5-4B on ALFWorld+WebShop combined
# (SDPO_REACT_DOMAIN=agentic, one shared train/eval set), 8x A100 (80GB),
# COLOCATE variant. Arm/algo -> flag mapping is identical to the
# mathcodesearch sibling; only the domain/data-prep differs.
#
# Path knobs + env overrides (defaults in parens): see the arm/algo tables
# in RM_ARGS/GRPO_ARGS below.
#   SDPO_ABLATION_DATA_ROOT / _MODEL_ROOT (/root)
#   SDPO_ABLATION_DUMP_ROOT (/root/data/sdpo_dumps)
#   SDPO_ABLATION_CKPT_ROOT (/root/data/sdpo_ckpts)
#   SDPO_ABLATION_MEGATRON_PATH (/root/Megatron-LM)
#   SDPO_ABLATION_ALGO   (required) grpo | sdpo | rlsd
#   SDPO_ABLATION_ARM    (required) a | b | c | d | e | f
#   SDPO_REACT_TRAIN_MAX_TURNS (8) / SDPO_REACT_EVAL_MAX_TURNS (20)
#   SDPO_ABLATION_MAX_TOKENS_PER_GPU (6144, 3072 for arms e/f)
#   SDPO_ABLATION_SGLANG_MEM_FRACTION (0.85)
#
# usage:
#   SDPO_ABLATION_ALGO=sdpo SDPO_ABLATION_ARM=b \
#     bash examples/SRD/ablation/run-qwen3.5-4B-sdpo-react-ablation-alfworld-webshop.sh
set -exf

SDPO_ABLATION_DATA_ROOT="${SDPO_ABLATION_DATA_ROOT:-/root}"
SDPO_ABLATION_MODEL_ROOT="${SDPO_ABLATION_MODEL_ROOT:-/root}"
SDPO_ABLATION_DUMP_ROOT="${SDPO_ABLATION_DUMP_ROOT:-/root/data/sdpo_dumps}"
SDPO_ABLATION_CKPT_ROOT="${SDPO_ABLATION_CKPT_ROOT:-/root/data/sdpo_ckpts}"
SDPO_ABLATION_MEGATRON_PATH="${SDPO_ABLATION_MEGATRON_PATH:-/root/Megatron-LM}"

export PYTHONBUFFERED=16
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
SDPO_ABLATION_ALGO="${SDPO_ABLATION_ALGO:?Set SDPO_ABLATION_ALGO to one of: grpo sdpo rlsd}"
SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:?Set SDPO_ABLATION_ARM to one of: a b c d e f}"
if [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    case "$SDPO_ABLATION_ARM" in
        a|e|f) ;;
        *) echo "GRPO only supports arms a/e/f (got '${SDPO_ABLATION_ARM}')" >&2; exit 1 ;;
    esac
fi
SDPO_REACT_NUM_ROLLOUT="${SDPO_REACT_NUM_ROLLOUT:-51}"

SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
N_SAMPLES_PER_PROMPT=8
SDPO_REACT_TP="${SDPO_REACT_TP:-2}"
DP_SIZE=$((SDPO_REACT_TRAIN_GPUS / SDPO_REACT_TP))
ROLLOUT_BATCH_SIZE="${SDPO_REACT_ROLLOUT_BATCH:-$((DP_SIZE * 4))}"
if [ $((ROLLOUT_BATCH_SIZE % DP_SIZE)) -ne 0 ]; then
    ROLLOUT_BATCH_SIZE=$((DP_SIZE * 4))
    echo "WARN: rollout batch not divisible by dp=${DP_SIZE}; falling back to ${ROLLOUT_BATCH_SIZE}"
fi
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
echo "BATCH: train_gpus=${SDPO_REACT_TRAIN_GPUS} tp=${SDPO_REACT_TP} dp=${DP_SIZE} rollout_batch=${ROLLOUT_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REACT_DIR="$REPO_ROOT/examples/SRD"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

MODEL_NAME=Qwen3.5-4B
MODEL_ARG_SH=scripts/models/qwen3.5-4B.sh
TOOL_PARSER=qwen3_coder
TOOL_GRAMMAR=qwen3_coder
# Reshardable optimizer ckpt (--dist-ckpt-optim-fully-reshardable) + mem-efficient
# save-time all-gather (--distrib-optim-fully-reshardable-mem-efficient, avoids the
# per-rank full-copy OOM). Both needed once ckpt saving is enabled.
MODEL_EXTRA_ARGS=(--dist-ckpt-optim-fully-reshardable --distrib-optim-fully-reshardable-mem-efficient)
MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-6144}"
if [ "$SDPO_ABLATION_ARM" = "e" ] || [ "$SDPO_ABLATION_ARM" = "f" ]; then
    MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-3072}"
fi
echo "MODEL: ${MODEL_NAME} | tool-parser=${TOOL_PARSER} | max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
source "$REPO_ROOT/${MODEL_ARG_SH}"

SDPO_REACT_DOMAIN=agentic
echo "SDPO_REACT_DOMAIN: ${SDPO_REACT_DOMAIN} (fixed -- ALFWorld+WebShop combined, per instruction not split)"

DEFAULT_TRAIN_MAX_TURNS=8
DEFAULT_EVAL_MAX_TURNS=20
export SDPO_REACT_TRAIN_MAX_TURNS="${SDPO_REACT_TRAIN_MAX_TURNS:-$DEFAULT_TRAIN_MAX_TURNS}"
export SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-$DEFAULT_EVAL_MAX_TURNS}"
echo "TURN BUDGET: train=${SDPO_REACT_TRAIN_MAX_TURNS} eval=${SDPO_REACT_EVAL_MAX_TURNS}"

# --- 0. domain sidecars (idempotent, ONE container / ONE port each) ---
bash "$REACT_DIR/tools/webshop/run_webshop_sidecar.sh"
bash "$REACT_DIR/tools/alfworld/run_alfworld_sidecar.sh"

export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"

# --- AGENTIC data prep: webshop + alfworld combined (paths parameterized) ---
MT_DIR="${SDPO_ABLATION_DATA_ROOT}/data/agentic"
TRAIN_DATA="$MT_DIR/train.jsonl"
EVAL_CFG="$REACT_DIR/data/eval_agentic.yaml"
mkdir -p "$MT_DIR" "${SDPO_ABLATION_DATA_ROOT}/data/webshop_data" "${SDPO_ABLATION_DATA_ROOT}/data/alfworld_data"
[ -f "${SDPO_ABLATION_DATA_ROOT}/data/webshop_data/webshop_train.jsonl" ] || \
    (cd "$REPO_ROOT" && python -m examples.SRD.data.build_webshop_data \
        --out-dir "${SDPO_ABLATION_DATA_ROOT}/data/webshop_data" --n-train "${SDPO_REACT_WEBSHOP_N_TRAIN:-400}" --n-eval "${SDPO_REACT_WEBSHOP_N_EVAL:-100}")
[ -f "${SDPO_ABLATION_DATA_ROOT}/data/alfworld_data/alfworld_train.jsonl" ] || \
    (cd "$REPO_ROOT" && python -m examples.SRD.data.build_alfworld_data \
        --out-dir "${SDPO_ABLATION_DATA_ROOT}/data/alfworld_data" --n-train "${SDPO_REACT_ALFWORLD_N_TRAIN:-400}" \
        --n-eval-id "${SDPO_REACT_ALFWORLD_N_EVAL_ID:-100}" --n-eval-ood "${SDPO_REACT_ALFWORLD_N_EVAL_OOD:-100}")
MT_SOURCES=(--source "webshop:${SDPO_ABLATION_DATA_ROOT}/data/webshop_data/webshop_train.jsonl"
            --source "alfworld:${SDPO_ABLATION_DATA_ROOT}/data/alfworld_data/alfworld_train.jsonl")
[ -f "$TRAIN_DATA" ] || \
    (cd "$REPO_ROOT" && python -m examples.SRD.data.build_multitask_data \
        --out "$TRAIN_DATA" "${MT_SOURCES[@]}" --per-domain "${SDPO_REACT_AGENTIC_PER_DOMAIN:-400}")
python - <<PYCK
import json
from collections import Counter
c=Counter(json.loads(l)["metadata"]["domain"] for l in open("$TRAIN_DATA"))
print(f"agentic train rows OK: {dict(c)}")
PYCK

# --- 0c. one-time template check ---
python - "$REPO_ROOT" "$MODEL_NAME" "$SDPO_ABLATION_MODEL_ROOT" <<'PYCHECK'
import sys
from transformers import AutoTokenizer
sys.path.insert(0, sys.argv[1])
model_name = sys.argv[2]
model_root = sys.argv[3]
from examples.SRD.tools.webshop.spec import webshop_specs
from examples.SRD.tools.alfworld.spec import alfworld_specs
tok = AutoTokenizer.from_pretrained(f"{model_root}/{model_name}", trust_remote_code=True)
checks = [
    ("webshop", [{"role": "user", "content": "find me a mouse"}], webshop_specs, "webshop_step"),
    ("alfworld", [{"role": "user", "content": "go to the kitchen"}], alfworld_specs, "alfworld_step"),
]
for name, msgs, tspecs, tool_name in checks:
    r = tok.apply_chat_template(msgs, tools=tspecs, tokenize=False, add_generation_prompt=True)
    assert "<tools>" in r, f"{name}: native <tools> block missing"
    assert tool_name in r, f"{name}: {tool_name} not in rendered <tools> block"
_SENT = "SDPOSENTINEL"
probe = tok.apply_chat_template([{"role":"user","content":_SENT}], tokenize=False, add_generation_prompt=True)
assert _SENT in probe and probe.split(_SENT,1)[1], "gen_suffix empty -> SDPO splice undefined"
print(f"Native template check OK for {model_name}")
PYCHECK

[ "$SDPO_REACT_THINKING" = "true" ] && _THINK=think || _THINK=nothink
if [ "$SDPO_REACT_THINKING" = "true" ]; then
    REMOVE_THINKING_ARG=()
else
    REMOVE_THINKING_ARG=(--sdpo-remove-thinking-from-demonstration)
fi
ARM_TAG="$SDPO_ABLATION_ARM"
CFG_TAG="${MODEL_NAME}-alfworld-webshop-${SDPO_ABLATION_ALGO}-${ARM_TAG}-${_THINK}"

SDPO_REACT_EXP="${SDPO_REACT_EXP:-sdpo-react-ablation-${CFG_TAG}_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="${SDPO_ABLATION_DUMP_ROOT}/${SDPO_REACT_EXP}"
echo "SDPO_ReAct dump dir: ${DUMP_DIR}"
CKPT_DIR="${SDPO_REACT_CKPT_DIR:-${SDPO_ABLATION_CKPT_ROOT}/sdpo-react-ablation-${CFG_TAG}_ckpt}"

EXPLOG="${SDPO_REACT_EXPLOG:-$REPO_ROOT/examples/SRD/ablation/explog.jsonl}"
GIT_COMMIT="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(cd "$REPO_ROOT" && [ -n "$(git status --porcelain 2>/dev/null)" ] && echo dirty || echo clean)"
python - <<PYLOG || true
import json, time, os
row = {
    "exp": "${SDPO_REACT_EXP}", "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "algo": "${SDPO_ABLATION_ALGO}", "arm": "${SDPO_ABLATION_ARM}", "domain": "alfworld_webshop",
    "model": "${MODEL_NAME}", "train_data": "${TRAIN_DATA}", "num_rollout": "${SDPO_REACT_NUM_ROLLOUT}",
    "dump_dir": "${DUMP_DIR}", "ckpt_dir": "${CKPT_DIR}", "git_commit": "${GIT_COMMIT}", "git_state": "${GIT_DIRTY}",
    "note": os.environ.get("SDPO_REACT_NOTE", ""),
}
with open("${EXPLOG}", "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("explog appended ->", "${EXPLOG}")
print(json.dumps(row, ensure_ascii=False, indent=2))
PYLOG

CKPT_ARGS=(
   --hf-checkpoint "${SDPO_ABLATION_MODEL_ROOT}/${MODEL_NAME}"
   --ref-load "${SDPO_ABLATION_MODEL_ROOT}/${MODEL_NAME}_torch_dist"
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)
if [ "${SDPO_REACT_SAVE_CKPT:-0}" = "1" ]; then
    CKPT_ARGS+=(--save "${CKPT_DIR}" --load "${CKPT_DIR}" --save-interval "${SDPO_REACT_SAVE_INTERVAL:-10}" --override-opt-param-scheduler)
fi

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_DATA"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs "{\"enable_thinking\":${SDPO_REACT_THINKING}}"
   --num-rollout "${SDPO_REACT_NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len 8192
   --rollout-max-context-len 81920
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}"
   # agentic -> preserve balanced per-batch domain mix (no --rollout-shuffle)
)
if [ "$SDPO_ABLATION_ARM" = "a" ] && [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
else
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix)
    ROLLOUT_ARGS+=(--sdpo-dynamic-filter-min-correct "${SDPO_REACT_MIN_CORRECT:-1}")
fi

TOOL_SPECS_PATH="examples.SRD.tools.registry.agentic_tool_specs"
EXECUTE_TOOL_PATH="examples.SRD.tools.registry.execute_tool"
ROLLOUT_ARGS+=(--tool-specs-resolver-path "$TOOL_SPECS_PATH")
CUSTOM_GENERATE_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
   --generate-tool-specs-path "$TOOL_SPECS_PATH"
   --generate-execute-tool-function-path "$EXECUTE_TOOL_PATH"
   --generate-tool-call-parser "$TOOL_PARSER"
   --generate-max-turns "${SDPO_REACT_TRAIN_MAX_TURNS}"
)

# ============================================================================ #
# ARM -> skill-related RM_ARGS (shared across all 3 algorithms) -- see the
# mathcodesearch sibling script for the full mapping table/rationale.
# ============================================================================ #
RM_ARGS=(
   --sdpo-answer-tag answer
   "${REMOVE_THINKING_ARG[@]}"
   --sdpo-reframe-multiturn-prefix
   --sdpo-tool-grammar "${TOOL_GRAMMAR}"
   --sdpo-max-prefix-chars 20000
)

if [ "$SDPO_ABLATION_ARM" = "a" ] && [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    RM_ARGS=(
       --custom-rm-path examples.SRD.sdpo_react.sdpo_react_plain_grpo_reward
       --sdpo-grader dapo
       --sdpo-answer-tag answer
    )
else
    RM_ARGS+=(
       --group-rm
       --custom-rm-path examples.SRD.sdpo_react.sdpo_react_group_reward
       --eval-custom-rm-path examples.SRD.sdpo_react.sdpo_react_eval_reward
       --sdpo-grader dapo
    )
    case "$SDPO_ABLATION_ARM" in
        a) : ;;
        b)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source correct --sdpo-skill-max-new-tokens 2048 --sdpo-response-prefix skill)
            ;;
        c)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source incorrect --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-env-feedback-max-chars 2000)
            ;;
        d)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill --sdpo-env-feedback-max-chars 2000)
            ;;
        e)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill --sdpo-env-feedback-max-chars 2000 \
                      --sdpo-skill-kd --sdpo-skill-kd-coef 0.01 --sdpo-skill-kd-mode both)
            ;;
        f)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill --sdpo-env-feedback-max-chars 2000 \
                      --sdpo-skill-kd --sdpo-skill-kd-coef 0.01 --sdpo-skill-kd-mode both-blind)
            ;;
    esac
fi

# Skill-augmented eval (--sdpo-eval-skill-mode) was removed with
# eval_agentic_skill.yaml; agentic arms eval on eval_agentic.yaml only.
EVAL_SKILL_ARGS=()

# ============================================================================ #
# ALGORITHM -> GRPO_ARGS
# ============================================================================ #
case "$SDPO_ABLATION_ALGO" in
    grpo)
        if [ "$SDPO_ABLATION_ARM" = "a" ]; then
            GRPO_ARGS=(
               --advantage-estimator grpo
               --entropy-coef 0.00
            )
        else
            # Arms e/f use the mixed GRPO(task reward) + skill-KD target;
            # --no-sdpo-pure-distill keeps the response-level advantage alongside
            # the skill-KD loss.
            RM_ARGS+=(--no-sdpo-pure-distill)
            GRPO_ARGS=(
               --advantage-estimator grpo
               --sdpo-teacher-backend megatron
               --sdpo-ema-teacher
               --sdpo-ema-teacher-rate 0.05
               --sdpo-logprob-mode sampled
               --sdpo-self-teacher
               --sdpo-prefer-tool-use-peer
               --entropy-coef 0.00
               --calculate-per-token-loss
            )
        fi
        ;;
    sdpo)
        RM_ARGS+=(--no-sdpo-pure-distill)
        GRPO_ARGS=(
           --advantage-estimator grpo
           --sdpo-teacher-backend megatron
           --sdpo-ema-teacher
           --sdpo-ema-teacher-rate 0.05
           --sdpo-logprob-mode topk
           --opd-log-prob-top-k 100
           --sdpo-divergence jsd
           --sdpo-is-clip 2.0
           --sdpo-kd-loss
           --sdpo-kd-coef 1.0
           --sdpo-kd-max-tokens 8192
           --sdpo-self-teacher
           --sdpo-prefer-tool-use-peer
           --entropy-coef 0.00
           --calculate-per-token-loss
        )
        ;;
    rlsd)
        RM_ARGS+=(--no-sdpo-pure-distill)
        GRPO_ARGS=(
           --advantage-estimator grpo
           --sdpo-teacher-backend megatron
           --sdpo-ema-teacher
           --sdpo-ema-teacher-rate 0.05
           --sdpo-logprob-mode sampled
           --sdpo-rlsd
           --sdpo-rlsd-clip-eps 0.2
           --sdpo-rlsd-lambda-init 1.0
           --sdpo-rlsd-lambda-warmup-steps 0
           --use-tis
           --sdpo-self-teacher
           --sdpo-prefer-tool-use-peer
           --entropy-coef 0.00
           --calculate-per-token-loss
        )
        ;;
    *)
        echo "Unknown SDPO_ABLATION_ALGO='${SDPO_ABLATION_ALGO}' (expected: grpo | sdpo | rlsd)" >&2
        exit 1 ;;
esac

EVAL_ARGS=(
   --eval-interval 10
   --eval-config "$EVAL_CFG"
   --n-samples-per-eval-prompt "${SDPO_REACT_EVAL_N_SAMPLES}"
   --log-passrate
)
if [ "${SDPO_REACT_SKIP_EVAL0:-0}" = "1" ]; then
    EVAL_ARGS+=(--skip-eval-before-train)
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${SDPO_REACT_TP:-2}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${SDPO_REACT_LOGPROBS_CHUNK:-4096}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters 10
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-sdpo
   --wandb-group "sdpo-react-ablation-${CFG_TAG}"
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SDPO_ABLATION_SGLANG_MEM_FRACTION:-0.85}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
if [ "$SDPO_REACT_TRAIN_GPUS" -lt 8 ]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((SDPO_REACT_TRAIN_GPUS-1)))"
    echo "TRAIN GPUs: ${SDPO_REACT_TRAIN_GPUS} (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
fi

ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 2

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${SDPO_REACT_TRAIN_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [ "${SDPO_REACT_SAVE_CKPT:-0}" = "1" ]; then
(
    set +f
    while true; do
        sleep 60
        latest_file="${CKPT_DIR}/latest_checkpointed_iteration.txt"
        [ -f "$latest_file" ] || continue
        latest_num="$(cat "$latest_file" 2>/dev/null)"
        [ -n "$latest_num" ] || continue
        case "$latest_num" in ''|*[!0-9]*) continue;; esac
        for d in "${CKPT_DIR}"/iter_*; do
            [ -d "$d" ] || continue
            n="$(basename "$d")"; n="${n#iter_}"
            case "$n" in ''|*[!0-9]*) continue;; esac
            if [ "$((10#$n))" -lt "$((10#$latest_num))" ]; then
                rm -rf "$d"
            fi
        done
    done
) &
PRUNER_PID=$!
trap 'kill "$PRUNER_PID" 2>/dev/null || true' EXIT
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${SDPO_ABLATION_MEGATRON_PATH}/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\",
        \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
        \"SDPO_REACT_EVAL_MAX_TURNS\": \"${SDPO_REACT_EVAL_MAX_TURNS}\",
        \"SDPO_REACT_EVAL_N_SAMPLES\": \"${SDPO_REACT_EVAL_N_SAMPLES}\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${SDPO_REACT_TRAIN_GPUS}" \
   --rollout-num-gpus "${SDPO_REACT_TRAIN_GPUS}" \
   --colocate \
   --update-weights-interval 1 \
   ${MODEL_ARGS[@]} \
   ${MODEL_EXTRA_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CUSTOM_GENERATE_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${EVAL_SKILL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
