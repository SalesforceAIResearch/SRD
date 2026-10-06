# Running an SDPO_ReAct ablation on a fresh host

This is the exact, ordered sequence to bring up and run one SDPO_ReAct
mathcodesearch ablation (e.g. **4B / 9B, native, SDPO, arm e**) on a host that
is **not** the original training cluster — i.e. no `enroot`, no `/fsx`, no
`/opt/dlami/nvme`. It was written from an actual bring-up on an 8×H200 box
(Ubuntu 24.04). Adapt paths to your machine.

The single launcher is `examples/SRD/enroot-run-sdpo-react.sh`; it starts the
host-side docker sidecars, imports the training image into enroot, downloads +
converts the model, and runs the ablation matrix
(`examples/SRD/ablation/run-qwen3.5-*-sdpo-react-ablation-mathcodesearch.sh`).

---

## 0. Host prerequisites (one-time)

1. **enroot** (the launcher uses `enroot import/create/start`; it is NOT in the
   Ubuntu repos — install the release `.deb`s):

   ```bash
   arch=$(dpkg --print-architecture); ver=3.5.0
   curl -fSL -O https://github.com/NVIDIA/enroot/releases/download/v${ver}/enroot_${ver}-1_${arch}.deb
   curl -fSL -O https://github.com/NVIDIA/enroot/releases/download/v${ver}/enroot+caps_${ver}-1_${arch}.deb
   sudo apt-get install -y ./enroot_${ver}-1_${arch}.deb ./enroot+caps_${ver}-1_${arch}.deb pv
   enroot version   # -> 3.5.0
   ```
   Requires unprivileged user namespaces (`cat /proc/sys/kernel/unprivileged_userns_clone` → `1`).

2. **Docker access.** The launcher starts the code-interpreter / search sidecars
   with a bare `docker` command, so the invoking user must reach the daemon:

   ```bash
   sudo usermod -aG docker "$USER"    # persistent; log out/in, OR use `sg docker -c '...'` this session
   ```

3. **Local scratch/asset/data trees** on a big disk (the launcher's defaults
   point at the cluster's `/opt/dlami/nvme` + `/fsx`). Pick one root:

   ```bash
   export SDPO_REACT_LOCAL_ROOT=/mnt/bigdisk/miles-run     # <-- your big disk
   mkdir -p "$SDPO_REACT_LOCAL_ROOT"/{enroot,assets,data/$USER}
   ```
   `SDPO_REACT_LOCAL_ROOT` is the one knob that relocates `ENROOT_NVME`,
   `ASSETS`, and `DATA_DIR` together. Each is still individually overridable
   (`ENROOT_NVME=`, `ASSETS=`, `DATA_DIR=`); leave `SDPO_REACT_LOCAL_ROOT` unset
   to keep the original cluster paths.

---

## 1. Secrets — `examples/SRD/.env` (gitignored)

```
WANDB_API_KEY=...
HF_TOKEN=...
OPENAI_API_KEY=...
# LLM-as-judge (search-domain grading fallback). Default judge backend is now
# 'openai' (OpenAI-compatible HTTP). The run scripts pass
#   --sdpo-judge-base-url "$OPENAI_API_URL"  --sdpo-judge-api-key-env LLM_GATEWAY_KEY
OPENAI_API_URL=https://api.openai.com/v1
LLM_GATEWAY_KEY=${OPENAI_API_KEY}
```

Notes:
- The LLM judge default backend is **openai** (`--sdpo-judge-backend`, in
  `miles/utils/arguments.py`); `bedrock` (AWS Converse, IAM-role auth) is the
  opt-in alternative.
- The run scripts default `--sdpo-judge-model <gateway-judge-model>`, which is a gateway
  model, **not** an OpenAI model. If you point the judge at real OpenAI, set a
  real model or the search-judge calls 4xx and fall back to deterministic
  grading (training is unaffected, only search-domain reward quality):
  `SDPO_REACT_JUDGE_MODEL=gpt-4o-mini`.

---

## 2. Image tag  (IMPORTANT — the original was pruned)

The launcher now defaults to **`<ORG>/miles:v0.1.0-cu12`**. Why this one:
- The validated `dev-cu12-202607040446` (torch **2.11.0+cu129**, sglang dev13799)
  was **pruned upstream** (404).
- **Do NOT use `latest-cu12`**: it drifted to torch **2.13** with a mismatched
  TransformerEngine 2.17 → `ImportError: ... undefined symbol` at
  `import megatron.core` (crashes the convert *and* training). This is a
  compiled-binary ABI mismatch inside the image; **no code change on our side
  can fix it** (it fails before any of our code runs).
- **Do NOT rely on dated `dev-cu12-YYYYMMDD` tags**: they get garbage-collected.
- `v0.1.0-cu12` is a **release** tag (not pruned), on the **same torch
  2.11.0+cu129** as the original, and was verified here end-to-end (import →
  `megatron.core`+TE → Qwen3.5 plugin `get_qwen3_5_spec` → real 4B HF→torch_dist
  convert).

The launcher imports it to `$ENROOT_NVME/miles-v0.1.0-cu12.sqsh` on first run.
List published tags: `curl -s https://registry.hub.docker.com/v2/repositories/<ORG>/miles/tags?page_size=100 | jq -r '.results[].name'`.
To confirm any candidate image before committing a run, probe it:
`enroot start <ctr> python -c "import torch; print(torch.__version__)"` (want 2.11.x)
and `... bash -c "cd /root/miles && python -c 'from megatron.core.enums import ModelType'"`.

---

## 3. Run

The mathcodesearch matrix trains on **math + code + search**, so it needs the
host-side **search stack up first** (the launcher auto-starts only the
code_interpreter sandbox). This is two commands:

### 3a. Bring up the search stack (once per host/boot)

```bash
cd <repo>
export SDPO_REACT_LOCAL_ROOT=/mnt/bigdisk/miles-run
export SQSH="$SDPO_REACT_LOCAL_ROOT/enroot/miles-v0.1.0-cu12.sqsh"   # imported on first training run
sg docker -c 'bash examples/SRD/tools/search/prepare-search-host.sh'
```
This idempotently: stages the wiki-18 corpus (`PeterJinGo/wiki-18-corpus`,
~5 GB — **not** the 40 GB e5 index) + BM25 Lucene index, starts the CPU-only
BM25 **retriever** on :8000 in a **dedicated** enroot container `miles-bm25`
(isolated so its torch/dep surgery can't corrupt the training container), and
starts the search **sidecar** on :8421. Re-running is a no-op once both are
healthy. (If you haven't imported the image yet, do one `PREP_ONLY=1` training
launch first, or point `SQSH`/`IMAGE` so this script can create the container.)

### 3b. Launch training

4B, mathcodesearch (native), SDPO, arm e:

```bash
export SDPO_REACT_LOCAL_ROOT=/mnt/bigdisk/miles-run
sg docker -c 'SDPO_REACT_MODEL=qwen3.5-4B SDPO_REACT_RUN_FAMILY=native \
  SDPO_ABLATION_ALGO=sdpo SDPO_ABLATION_ARM=e \
  bash examples/SRD/enroot-run-sdpo-react.sh'
```

9B — identical, `SDPO_REACT_MODEL=qwen3.5-9B`. **Run it after 4B**: each run
takes all 8 GPUs (`SDPO_REACT_TRAIN_GPUS=8`), so they cannot run concurrently
on a single node. The image + imported `.sqsh` and the search stack are reused
(no re-import, no re-stage). `sg docker -c '...'` is only needed until your
shell picks up docker-group membership (a fresh login drops it).

What the launcher does, in order: start sandbox sidecar (port 8420) → (search
domain) start search sidecar (8421) + wiki-18 retriever (8000) → `enroot import`
image → `enroot create` → `enroot start` → patch sglang → `hf download` the
model → convert HF→torch_dist → `bash run-...-mathcodesearch.sh` (training).

Algorithm axis (`SDPO_ABLATION_ALGO`): `grpo | sdpo | rlsd`.
Arm axis (`SDPO_ABLATION_ARM`): `a b c d e f j z` (arm `e` = `+skill-sd both`;
GRPO supports `a/e/f/j/z`; `z` is GRPO-only). See the header of
`examples/SRD/ablation/run-qwen3.5-4B-sdpo-react-ablation-mathcodesearch.sh`.

---

## 4. Where outputs land

With `DATA_DIR = $SDPO_REACT_LOCAL_ROOT/data/$USER` (mounted into the container
as `/root/data`):

- **Rollout/eval dumps**: `$DATA_DIR/sdpo_dumps/<exp>/`
- **Checkpoints**: `$DATA_DIR/sdpo_ckpts/` (only when `SDPO_REACT_SAVE_CKPT=1`)
- **wandb**: group `sdpo-react-ablation-<CFG_TAG>`

> These are **not** the repo's top-level `sdpo_dumps/` (which only holds older
> historical runs). Override the dump/ckpt roots directly if desired:
> `SDPO_ABLATION_DUMP_ROOT`, `SDPO_ABLATION_CKPT_ROOT` (paths inside the
> container, i.e. under the mounted `/root/data`).

---

## 5. Gotchas fixed during bring-up

**a. Sandbox image file perms.** The code-interpreter sandbox image runs its
server as an unprivileged `sandbox` user. If your checkout gives
`tools/docker/sandbox_server.py` a restrictive mode (e.g. `0600` from an odd
umask/ACL), the `COPY` carries it into the image and the container crashes at
import with `PermissionError: '/app/sandbox_server.py'`.
`tools/docker/Dockerfile` now uses `COPY --chmod=0644` so the build is
perm-robust regardless of the source file's mode.

**b. JIT caches must be on a LOCAL fs (not the shared cache mount).** SGLang
runs one rollout engine per GPU, and all of them JIT-compile the same Triton
kernels (Qwen3.5's `chunk_gated_delta_rule`, `_fused_sigmoid_mul`) concurrently
into one `TRITON_CACHE_DIR`. On a networked/virtualized `$CACHES` (this host:
**virtiofs**) the cache's write-then-rename is not atomic across processes, so an
engine opens a `.cubin` that is not there yet and dies:
`FileNotFoundError: .../<kernel>.cubin` → `Server process terminated
unexpectedly`. The launcher now puts all JIT caches (Triton/inductor/torch-ext/
nvrtc) on tmpfs `/dev/shm/miles-jit` (mounted `/root/jit`), overridable via
`SDPO_REACT_JIT_CACHE_DIR`. Symptom to recognize: the run reaches the first
rollout, all GPUs load, then one SGLang scheduler throws the `.cubin` error.

**c. Retriever JDK version drift.** `tools/search/run_retrieval.sh` hardcoded
`jdk-21.0.12+8`, but `install-jdk` fetches whatever current 21.x build it wants
(seen: `jdk-21.0.12.1+1`), leaving `JAVA_HOME` pointing at a nonexistent dir and
pyserini failing with `Unable to find libjvm.so`. It now auto-detects the
extracted `jdk-*` dir and exports `JVM_PATH` explicitly.

**d. Judge / image / path notes** are in §1–§2 above (judge default → openai;
image pinned to the un-prunable release `v0.1.0-cu12`; `SDPO_REACT_LOCAL_ROOT`
relocates all host paths off the cluster's `/opt/dlami/nvme` + `/fsx`).
