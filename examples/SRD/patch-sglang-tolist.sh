#!/bin/bash
# Idempotent patch for a sglang bug in the latest-cu12 image that crashes the
# SDPO top-k logprob path (--opd-log-prob-top-k > 0).
#
# Bug: batch_result_processor.py unconditionally calls `v.tolist()` over
# next_token_token_ids_logprobs_val, but upstream (managers/utils.py) leaves
# non-tensor entries as plain lists (it guards with torch.is_tensor). A list
# entry then hits `list.tolist()` -> AttributeError, killing the scheduler.
#
# Fix: make the two call sites tensor-guarded, matching utils.py. Runs inside the
# container; safe to run repeatedly (grep-guarded), so it survives container
# rebuilds when invoked from the launcher.
set -eu

F=/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py

if [ ! -f "$F" ]; then
    echo "patch-sglang-tolist: $F not found, skipping" >&2
    exit 0
fi

# Already patched? (guard token we introduce below)
if grep -q "# sdpo-patch" "$F"; then
    echo "patch-sglang-tolist: already applied"
    exit 0
fi

# Make every `v.tolist()` over next_token_token_ids_logprobs_val tensor-guarded.
# This sglang file has TWO crash sites with DIFFERENT formatting:
#   (a) single line:  `v.tolist() for v in logits_output.next_token_token_ids_logprobs_val`
#   (b) multi line:    `v.tolist()` on its own line, then `for v in ...` on the next.
# A regex over `v.tolist()` immediately followed (any whitespace/newline) by
# `for v in logits_output.next_token_token_ids_logprobs_val` covers both.
python - "$F" <<'PY'
import sys, re
path = sys.argv[1]
src = open(path).read()
pat = re.compile(r"v\.tolist\(\)(\s+)for v in logits_output\.next_token_token_ids_logprobs_val")
def repl(m):
    return f"(v.tolist() if torch.is_tensor(v) else v){m.group(1)}for v in logits_output.next_token_token_ids_logprobs_val  # sdpo-patch"
src2, n = pat.subn(repl, src)
if n == 0:
    print("patch-sglang-tolist: target pattern not found (sglang version changed?) — leaving file untouched", file=sys.stderr)
    sys.exit(0)
if "import torch" not in src2:
    src2 = "import torch\n" + src2
open(path, "w").write(src2)
print(f"patch-sglang-tolist: patched {n} site(s)")
PY
