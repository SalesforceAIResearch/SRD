# SRD — Ablation arms, algorithms & hyperparameters

This page is the full parameter reference behind the two one-click commands in the
top-level `README.md`. It documents the two orthogonal ablation axes, the exact
flag set each produces, the two reference configurations we highlight, and every
environment override.

All runs go through one of the `examples/SRD/ablation/run-qwen3.5-*-sdpo-react-ablation-*.sh`
scripts (selected by the launcher `examples/SRD/enroot-run-sdpo-react.sh`). The
two axes are read from the environment:

```
SDPO_ABLATION_ALGO   grpo | sdpo | rlsd    # how the teacher–student signal is consumed
SDPO_ABLATION_ARM    a | b | c | d | e | f | j | z   # the retrospection / skill machinery layered on top
```

**SRD is arm `e`.** `grpo e` = *GRPO + SRD*; `sdpo e` = *SDPO + SRD*.

---

## 1. Ablation arms (`SDPO_ABLATION_ARM`)

Each arm layers a different amount of self-retrospection machinery on top of the
base RL objective. The model rolls out a group of attempts per prompt; a **skill**
is a short natural-language note the model writes *after* seeing an attempt's
outcome (post-hoc experience) and is spliced back in as a **prefix** that
conditions the next attempt (prior foresight).

| arm | name | flags added |
|---|---|---|
| `a` | Baseline (no skill) | *(none — `--sdpo-response-prefix` defaults to `trace`)* |
| `b` | + correct-skill prefix | `--sdpo-self-skill --sdpo-skill-source correct --sdpo-response-prefix skill` |
| `c` | + pitfall-skill prefix | `--sdpo-self-skill --sdpo-skill-source incorrect --sdpo-pitfall-summary-backend self` |
| `d` | + all-skill prefix | `--sdpo-self-skill --sdpo-skill-source all --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill` |
| **`e`** | **SRD** (self-success + pitfall skill distillation) | `d` + `--sdpo-skill-kd --sdpo-skill-kd-mode both` |
| `f` | skill-SD both-blind | `d` + `--sdpo-skill-kd --sdpo-skill-kd-mode both-blind` |
| `j` | group-skills teacher | `f` + `--sdpo-blind-correct-info group-skills` |
| `z` | skill-SD only (pure distill, GRPO-only) | `e`'s machinery with `--sdpo-pure-distill` left ON, no dynamic filter; `--sdpo-skill-kd-coef` defaults to `1.0` |

Shared skill flags (arms `b`–`z`): `--sdpo-skill-max-new-tokens 2048`,
`--sdpo-env-feedback-max-chars 2000`, `--sdpo-max-prefix-chars 20000`.
For arms `e`/`f`/`j` the skill-KD coefficient defaults to `0.01`
(`--sdpo-skill-kd-coef`, override with `SDPO_ABLATION_SKILL_KD_COEF`).

**Support matrix:** `grpo` supports `a/e/f/j/z`; `sdpo` and `rlsd` support `a`–`f`;
`z` is GRPO-only. (The agentic scripts expose `a/b/c/d/e/f/z` — no `j`.)

---

## 2. Algorithms (`SDPO_ABLATION_ALGO`)

How the divergence between the skill-conditioned teacher and the policy is turned
into a loss. Mutually exclusive (asserted in `miles/utils/arguments.py`).

| algo | mechanism | key flags |
|---|---|---|
| `grpo` | Task reward drives the GRPO advantage; SRD arms add an orthogonal skill-KD loss. | `--advantage-estimator grpo`; arms `e/f/z` also set `--sdpo-teacher-backend megatron --sdpo-logprob-mode sampled --calculate-per-token-loss` |
| `sdpo` | Additive distribution-KD loss on response tokens. | `--sdpo-kd-loss --sdpo-divergence jsd --sdpo-logprob-mode topk --opd-log-prob-top-k 100 --sdpo-is-clip 2.0 --sdpo-kd-coef 1.0 --sdpo-kd-max-tokens 8192` |
| `rlsd` | Multiplicative advantage reweighting. | `--sdpo-rlsd --sdpo-rlsd-clip-eps 0.2 --sdpo-rlsd-lambda-init 1.0 --sdpo-rlsd-lambda-warmup-steps 0 --use-tis --sdpo-logprob-mode sampled` |

All non-baseline runs share: `--sdpo-teacher-backend megatron --sdpo-ema-teacher
--sdpo-ema-teacher-rate 0.05 --sdpo-self-teacher --calculate-per-token-loss
--entropy-coef 0.00`. `sdpo`/`rlsd` additionally set `--sdpo-prefer-tool-use-peer`
(prevents tool-call collapse). The plain `grpo a` baseline uses none of the SDPO
machinery (single-sample domain-routed reward via `sdpo_react_plain_grpo_reward`).

---

## 3. The two reference configurations

These are the two commands highlighted in the README.

### GRPO + SRD — Qwen3.5-4B — math + code + search

```bash
SDPO_ABLATION_ALGO=grpo SDPO_ABLATION_ARM=e \
  bash examples/SRD/ablation/run-qwen3.5-4B-sdpo-react-ablation-mathcodesearch.sh
# or via the launcher:
SDPO_REACT_MODEL=qwen3.5-4B SDPO_REACT_RUN_FAMILY=native \
SDPO_ABLATION_ALGO=grpo SDPO_ABLATION_ARM=e \
  bash examples/SRD/enroot-run-sdpo-react.sh
```

Resolved extra flags (on top of §4 shared hyperparameters): `--advantage-estimator
grpo --sdpo-teacher-backend megatron --sdpo-ema-teacher --sdpo-ema-teacher-rate
0.05 --sdpo-logprob-mode sampled --sdpo-self-teacher --sdpo-prefer-tool-use-peer
--calculate-per-token-loss --no-sdpo-pure-distill --sdpo-self-skill
--sdpo-skill-source all --sdpo-pitfall-summary-backend self --sdpo-response-prefix
skill --sdpo-skill-kd --sdpo-skill-kd-coef 0.01 --sdpo-skill-kd-mode both`.

Training data = shuffled `math` (DAPO-Math-17k) + `code` (LiveCodeBench) + `search`
(HotpotQA / 2Wiki, FlashRAG), `--per-domain 400`. Needs the code-interpreter
sandbox and the search retriever sidecars (started automatically by the script).

### SDPO + SRD — Qwen3.5-4B — agentic (ALFWorld + WebShop)

```bash
SDPO_ABLATION_ALGO=sdpo SDPO_ABLATION_ARM=e \
  bash examples/SRD/ablation/run-qwen3.5-4B-sdpo-react-ablation-alfworld-webshop.sh
# or via the launcher:
SDPO_REACT_MODEL=qwen3.5-4B SDPO_REACT_RUN_FAMILY=agentic \
SDPO_ABLATION_ALGO=sdpo SDPO_ABLATION_ARM=e \
  bash examples/SRD/enroot-run-sdpo-react.sh
```

Resolved extra flags: `--sdpo-kd-loss --sdpo-divergence jsd --sdpo-logprob-mode
topk --opd-log-prob-top-k 100 --sdpo-is-clip 2.0 --sdpo-kd-coef 1.0
--sdpo-kd-max-tokens 8192 --sdpo-teacher-backend megatron --sdpo-ema-teacher
--sdpo-ema-teacher-rate 0.05 --sdpo-self-teacher --sdpo-prefer-tool-use-peer
--calculate-per-token-loss --no-sdpo-pure-distill --sdpo-self-skill
--sdpo-skill-source all --sdpo-pitfall-summary-backend self --sdpo-response-prefix
skill --sdpo-skill-kd --sdpo-skill-kd-coef 0.01 --sdpo-skill-kd-mode both`.

Training data = combined ALFWorld + WebShop (`--per-domain 400`); eval on
`data/eval_agentic.yaml`. Needs the WebShop and ALFWorld sidecars (started
automatically). Swap `4B` → `9B` / `35B-A3B` for the other scales.

---

## 4. Shared hyperparameters

Identical across both reference configs (single 8-GPU node, colocated).

| group | value |
|---|---|
| parallelism | `--tensor-model-parallel-size 2` (TP=2, DP=4 on 8 GPUs), PP=1, CP=1, colocate |
| batch | rollout-batch = DP×4 = 16, `--n-samples-per-prompt 8` → global batch = 128 |
| rollout | `--num-rollout 51`, `--rollout-max-response-len 8192` (per turn), `--rollout-max-context-len 81920`, `--rollout-temperature 1` |
| turns | train max turns = 8, eval max turns = 20 |
| optimizer | Adam, `--lr 1e-6` constant, `--lr-warmup-iters 10`, `--weight-decay 0.1`, β=(0.9, 0.98), CPU-offload + precision-aware |
| memory | `--max-tokens-per-gpu 6144` (3072 for arms `e`/`f`/`j`/`z`), `--recompute-granularity full`, `--use-dynamic-batch-size` |
| eval | `--eval-interval 10`, avg@8 (`SDPO_REACT_EVAL_N_SAMPLES=8`), `--log-passrate` |
| sglang | `--rollout-num-gpus-per-engine 1`, `--sglang-mem-fraction-static 0.85` |
| misc | `--entropy-coef 0.00`, `--attention-backend flash`, dropout 0, fp32 grad all-reduce |

---

## 5. Environment overrides

| variable | default | meaning |
|---|---|---|
| `SDPO_ABLATION_ALGO` | *(required)* | `grpo` \| `sdpo` \| `rlsd` |
| `SDPO_ABLATION_ARM` | *(required)* | `a`–`f`, `j`, `z` (see support matrix) |
| `SDPO_REACT_MODEL` | — | `qwen3.5-4B` \| `qwen3.5-9B` \| `qwen3.5-35B-A3B` (launcher) |
| `SDPO_REACT_RUN_FAMILY` | `native` | `native` (math+code+search) \| `agentic` (ALFWorld+WebShop) |
| `SDPO_REACT_EVAL_CONFIG` | `eval_multitask_min.yaml` | `eval_multitask_full.yaml` for the full held-out suite |
| `SDPO_REACT_TRAIN_MAX_TURNS` / `_EVAL_MAX_TURNS` | 8 / 20 | per-rollout turn budget |
| `SDPO_REACT_EVAL_N_SAMPLES` | 8 | samples per eval prompt (avg@k) |
| `SDPO_REACT_NUM_ROLLOUT` | 51 | number of rollout steps |
| `SDPO_REACT_MAX_RESPONSE_LEN` | 8192 | train rollout cap; set 16384 to match eval |
| `SDPO_ABLATION_MAX_TOKENS_PER_GPU` | 6144 (3072 for `e`/`f`/`j`) | microbatch token budget |
| `SDPO_ABLATION_SGLANG_MEM_FRACTION` | 0.85 | SGLang static memory fraction |
| `SDPO_ABLATION_SKILL_KD_COEF` | 0.01 (`e`/`f`/`j`), 1.0 (`z`) | skill-KD loss weight |
| `SDPO_REACT_SAVE_CKPT` / `_SAVE_INTERVAL` | 0 / 10 | enable checkpointing + interval |
| `SDPO_REACT_TRAIN_GPUS` / `SDPO_REACT_TP` | 8 / 2 | GPU count / tensor-parallel size |
| `SDPO_ABLATION_{DATA,MODEL,DUMP,CKPT}_ROOT` | `/root`, `/root`, `/root/data/sdpo_dumps`, `/root/data/sdpo_ckpts` | path roots |
| `SDPO_REACT_TAG_SUFFIX` | *(empty)* | appended to the wandb group + checkpoint dir (use for a rerun under a changed scaffold) |

Secrets (`WANDB_API_KEY`, `HF_TOKEN`, and the optional LLM-judge gateway
`LLM_GATEWAY_KEY` / `OPENAI_API_URL` / `SDPO_REACT_JUDGE_MODEL`) are read from the
gitignored `examples/SRD/.env`.
