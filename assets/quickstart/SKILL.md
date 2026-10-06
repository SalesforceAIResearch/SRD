---
name: srd-quickstart
description: One-click launch of an SRD (Self-Retrospection Distillation) training run on a single 8-GPU node. Use when asked to start, launch, or reproduce an SRD run (GRPO+SRD or OPSD+SRD, Qwen3.5-4B/9B/35B-A3B) from this repository.
---

# SRD quick-start

Goal: from a fresh checkout, bring up one SRD training run end-to-end. Follow the
steps in order; each has a check so you know it worked before moving on. All paths
are relative to the repository root.

## 0. Preconditions (check, don't assume)

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader   # expect 8 GPUs (H200/H100/A100-80GB)
enroot version                                                   # expect >= 3.5.0 — if missing, see README "Install enroot"
test -f examples/SRD/enroot-run-sdpo-react.sh && echo launcher-ok
```

If `enroot` is missing, install it (one-off):

```bash
arch=$(dpkg --print-architecture)
curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot_3.5.0-1_${arch}.deb
sudo apt install -y ./enroot_3.5.0-1_${arch}.deb
```

## 1. Secrets

Create `examples/SRD/.env` (gitignored) if it does not exist. WANDB + HF are
required; the LLM-judge block is optional (search-domain grading fallback only):

```bash
cat > examples/SRD/.env <<'EOF'
WANDB_API_KEY=__from https://wandb.ai/authorize__
HF_TOKEN=__from https://huggingface.co/settings/tokens__
# optional (search-domain LLM-judge fallback):
OPENAI_API_URL=              # e.g. https://api.openai.com/v1
LLM_GATEWAY_KEY=             # API key for OPENAI_API_URL (your OpenAI key if that is the endpoint)
SDPO_REACT_JUDGE_MODEL=gpt-5.6-luna
EOF
```

Do NOT commit this file. Stop and ask the user for the keys if they are not
available — never invent them.

## 2. Pick a scratch root (big disk — holds model, data, checkpoints)

```bash
export SDPO_REACT_LOCAL_ROOT=/path/on/a/big/disk      # ask the user if unsure
```

## 3. Launch (one command)

Pick ONE of the two reference configurations and run it. The launcher pulls the
`radixark/miles:v0.1.0-cu12` image into an enroot squashfs, starts the required
sidecars (code sandbox / search retriever / WebShop+ALFWorld), builds any missing
data from public upstreams, and starts training — all from this single command.

```bash
# (A) GRPO + SRD — Qwen3.5-4B — math + code + search
SDPO_REACT_MODEL=qwen3.5-4B SDPO_REACT_RUN_FAMILY=native \
SDPO_ABLATION_ALGO=grpo SDPO_ABLATION_ARM=e \
  bash examples/SRD/enroot-run-sdpo-react.sh

# (B) OPSD + SRD — Qwen3.5-4B — agentic (ALFWorld + WebShop)
SDPO_REACT_MODEL=qwen3.5-4B SDPO_REACT_RUN_FAMILY=agentic \
SDPO_ABLATION_ALGO=sdpo SDPO_ABLATION_ARM=e \
  bash examples/SRD/enroot-run-sdpo-react.sh
```

`SDPO_ABLATION_ARM=e` is SRD; `SDPO_ABLATION_ARM=a` is the no-skill control.
Swap `qwen3.5-4B` → `qwen3.5-9B` / `qwen3.5-35B-A3B` for larger scales.

## 4. Confirm it is training

- Console prints `BATCH: ...`, `MODEL: Qwen3.5-4B ...`, `eval config OK`, then a
  wandb run URL in group `sdpo-react-ablation-<config-tag>`.
- Dumps appear under `$SDPO_REACT_LOCAL_ROOT/data/$USER/sdpo_dumps/<exp>/`; per-
  trajectory traces under `<exp>/agentic_traces/{rollout_id}.jsonl`.

## 5. Common knobs (only if needed)

| symptom / want | fix |
|---|---|
| fewer than 8 GPUs | `export SDPO_REACT_TRAIN_GPUS=<n>` (TP divides it; default TP=2) |
| CUDA OOM on H100 / A100-80GB | lower `export SDPO_ABLATION_MAX_TOKENS_PER_GPU=3072` (or 2048) |
| save checkpoints | `export SDPO_REACT_SAVE_CKPT=1 SDPO_REACT_SAVE_INTERVAL=10` |
| full held-out eval suite | `export SDPO_REACT_EVAL_CONFIG=eval_multitask_full.yaml` |
| custom image / squashfs | `export IMAGE=... SQSH=...` |

The full arm × algorithm matrix and every override live in
[`docs/srd/ablations.md`](../../docs/srd/ablations.md).
